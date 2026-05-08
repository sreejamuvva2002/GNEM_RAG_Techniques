"""Hybrid retriever — orchestrates the full retrieval flow.

Flow (matching the architecture diagram exactly):
    1. Get all child documents from the vector store.
    2. BM25 search → normalise scores (min-max → [0, 1]).
    3. Vector search → invert distances & normalise.
    4. Union candidate sets, combine scores with weighted sum.
    5. Rank by combined score.
    6. Deduplicate by ``parent_id`` (keep best-scoring child per parent).
    7. Return top K parent documents.

This module is **pure domain logic** — it calls interfaces, not
concrete adapters.  No I/O happens here; I/O is delegated to the
injected vector store and keyword search backends.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.score_fusion import (
    combine_scores,
    normalize_bm25_scores,
    normalize_vector_scores,
)
from src.interfaces.embedding_interface import EmbeddingInterface
from src.interfaces.keyword_search_interface import KeywordSearchInterface
from src.interfaces.vector_store_interface import VectorStoreInterface
from src.exceptions import RetrievalError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """Orchestrates hybrid (semantic + keyword) retrieval.

    Args:
        embedding: An ``EmbeddingInterface`` implementation.
        vector_store: A ``VectorStoreInterface`` implementation.
        keyword_search: A ``KeywordSearchInterface`` implementation.
        top_k: Number of final parent documents to return.
        alpha: Semantic weight in ``[0, 1]`` for score fusion.
        candidate_pool_size: Number of candidates to fetch from
                             *each* retriever before fusion.
        parent_lookup: Mapping ``parent_id → parent_text``.
    """

    def __init__(
        self,
        embedding: EmbeddingInterface,
        vector_store: VectorStoreInterface,
        keyword_search: KeywordSearchInterface,
        top_k: int,
        alpha: float,
        candidate_pool_size: int,
        parent_lookup: dict[str, dict[str, Any]],
    ) -> None:
        self._embedding = embedding
        self._vector_store = vector_store
        self._keyword_search = keyword_search
        self._top_k = top_k
        self._alpha = alpha
        self._pool_size = candidate_pool_size
        self._parent_lookup = parent_lookup

    # -------------------------------------------------------------- #
    # Public API                                                      #
    # -------------------------------------------------------------- #

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """Run the full hybrid retrieval flow for a single query.

        Args:
            query: Natural-language user query.

        Returns:
            A list of up to ``top_k`` result dicts, each containing:
            ``parent_id``, ``company``, ``parent_text``,
            ``similarity_score``, ``keyword_score``, ``combined_score``.

        Raises:
            RetrievalError: On any unrecoverable failure in the
                            retrieval pipeline.
        """
        try:
            return self._retrieve_impl(query)
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(
                f"Hybrid retrieval failed for query: {query!r}"
            ) from exc

    # -------------------------------------------------------------- #
    # Internal implementation                                         #
    # -------------------------------------------------------------- #

    def _retrieve_impl(self, query: str) -> list[dict[str, Any]]:
        """Core retrieval logic — no error wrapping."""

        # Step 1: Embed the query.
        query_embedding = self._embedding.embed_query(query)

        # Step 2a: Perform vector search.
        vector_results = self._vector_store.search(
            query_embedding=query_embedding,
            top_k=self._pool_size,
        )

        # Step 2b: Perform BM25 search.
        bm25_results = self._keyword_search.search(
            query=query,
            top_k=self._pool_size,
        )

        if not vector_results and not bm25_results:
            logger.warning("Both retrievers returned zero results for query: %s", query)
            return []

        # Step 3: Build a unified candidate map.
        # Key = child chunk id,  value = {parent_id, vector_distance, bm25_score, metadata}
        candidates: dict[str, dict[str, Any]] = {}

        for vr in vector_results:
            cid = vr["id"]
            candidates[cid] = {
                "parent_id": vr["metadata"]["parent_id"],
                "company": vr["metadata"].get("company", ""),
                "vector_distance": vr["distance"],
                "bm25_score": 0.0,
                "metadata": vr["metadata"],
            }

        for br in bm25_results:
            cid = br["id"]
            if cid in candidates:
                candidates[cid]["bm25_score"] = br["score"]
            else:
                candidates[cid] = {
                    "parent_id": br["metadata"]["parent_id"],
                    "company": br["metadata"].get("company", ""),
                    "vector_distance": float("inf"),  # not in vector results
                    "bm25_score": br["score"],
                    "metadata": br["metadata"],
                }

        # Step 4: Normalise and combine scores.
        cids = list(candidates.keys())
        vector_distances = np.array(
            [candidates[c]["vector_distance"] for c in cids], dtype=np.float64
        )
        bm25_raw = np.array(
            [candidates[c]["bm25_score"] for c in cids], dtype=np.float64
        )

        # FAISS returns L2 distances by default — lower is better.
        # normalize_vector_scores inverts distances to similarity
        # via sim = 1/(1+distance), then min-max normalises to [0,1].
        vector_norm = normalize_vector_scores(
            vector_distances, already_similarity=False
        )
        bm25_norm = normalize_bm25_scores(bm25_raw)
        combined = combine_scores(vector_norm, bm25_norm, self._alpha)

        # Attach normalised scores back to candidates.
        for i, cid in enumerate(cids):
            candidates[cid]["vector_norm"] = float(vector_norm[i])
            candidates[cid]["bm25_norm"] = float(bm25_norm[i])
            candidates[cid]["combined"] = float(combined[i])

        # Step 5: Rank by combined score (descending).
        ranked = sorted(cids, key=lambda c: candidates[c]["combined"], reverse=True)

        # Step 6: Deduplicate by parent_id — keep best-scoring child.
        seen_parents: set[str] = set()
        deduped: list[str] = []
        for cid in ranked:
            pid = candidates[cid]["parent_id"]
            if pid not in seen_parents:
                seen_parents.add(pid)
                deduped.append(cid)

        # Step 7: Return top K parent documents.
        top_cids = deduped[: self._top_k]

        results: list[dict[str, Any]] = []
        for cid in top_cids:
            c = candidates[cid]
            pid = c["parent_id"]
            parent_info = self._parent_lookup.get(pid, {})
            results.append(
                {
                    "parent_id": pid,
                    "company": c["company"],
                    "parent_text": parent_info.get("text", ""),
                    "similarity_score": round(c["vector_norm"], 6),
                    "keyword_score": round(c["bm25_norm"], 6),
                    "combined_score": round(c["combined"], 6),
                }
            )

        logger.info(
            "Query retrieved %d candidates → %d after dedup → returning top %d",
            len(candidates),
            len(deduped),
            len(results),
        )
        return results
