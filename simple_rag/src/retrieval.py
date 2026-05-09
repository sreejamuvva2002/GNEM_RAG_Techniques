from __future__ import annotations
"""Score normalisation and fusion for hybrid retrieval.

This module implements the exact normalisation and combination logic
described in the architecture specification:

1. **BM25 normalisation** — min-max scaling to [0, 1].  Higher is better.
2. **Vector normalisation** — convert distances (lower = better) to
   similarity via ``sim = 1 / (1 + distance)``, then min-max to [0, 1].
   If the store already returns cosine similarity (higher = better),
   the inversion step is a no-op but min-max still runs.
3. **Weighted combination** —
   ``combined = alpha * vector_norm + (1 - alpha) * bm25_norm``

This module is **pure domain logic** — stdlib + numpy only, no I/O.
"""

from src.utils import get_logger, RetrievalError

import numpy as np


logger = get_logger(__name__)


def normalize_bm25_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalise BM25 scores to [0, 1].

    BM25 scores are already "higher = better".

    Args:
        scores: 1-D array of raw BM25 scores.

    Returns:
        Normalised scores as a 1-D numpy array.  If all scores are
        equal, returns an array of ones (all candidates are equivalent).
    """
    if scores.size == 0:
        return scores.copy()

    min_s = scores.min()
    max_s = scores.max()
    span = max_s - min_s

    if span == 0.0:
        # All scores identical — treat every candidate as equally good.
        return np.ones_like(scores, dtype=np.float64)

    return (scores - min_s) / span


def normalize_vector_scores(
    distances: np.ndarray,
    already_similarity: bool = False,
) -> np.ndarray:
    """Invert distances to similarities, then min-max normalise to [0, 1].

    The inversion formula is ``sim = 1 / (1 + distance)`` which maps
    distance 0 → similarity 1 and large distance → near 0.

    If the vector store already returns cosine *similarity* (higher is
    better), set ``already_similarity=True`` to skip the inversion —
    the min-max normalisation still runs so the output is in [0, 1].

    Args:
        distances: 1-D array of raw distance (or similarity) values.
        already_similarity: If ``True``, treat values as similarities
                           (higher = better) and skip inversion.

    Returns:
        Normalised scores in [0, 1], higher = better.  Returns ones
        when all values are equal.
    """
    if distances.size == 0:
        return distances.copy()

    if already_similarity:
        # No inversion needed — higher is already better.
        # Still min-max normalise for consistency.
        similarities = distances.astype(np.float64)
    else:
        # Invert: distance → similarity.
        # sim = 1 / (1 + distance).  Lower distance → higher similarity.
        similarities = 1.0 / (1.0 + distances.astype(np.float64))

    min_s = similarities.min()
    max_s = similarities.max()
    span = max_s - min_s

    if span == 0.0:
        return np.ones_like(similarities, dtype=np.float64)

    return (similarities - min_s) / span


def combine_scores(
    vector_norm: np.ndarray,
    bm25_norm: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Compute the weighted combination of normalised scores.

    Formula::

        combined = alpha * vector_norm + (1 - alpha) * bm25_norm

    Args:
        vector_norm: Min-max normalised vector similarity scores.
        bm25_norm: Min-max normalised BM25 scores.
        alpha: Semantic (vector) weight in ``[0, 1]``.

    Returns:
        Combined scores as a 1-D numpy array.
    """
    return alpha * vector_norm + (1.0 - alpha) * bm25_norm


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


from typing import Any

import numpy as np

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
        fusion_mode: One of ``"dense"``, ``"sparse"``, or ``"hybrid"``.
                     ``"dense"``  uses vector similarity only.
                     ``"sparse"`` uses BM25 keyword scores only.
                     ``"hybrid"`` combines both via weighted sum.
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
        fusion_mode: str = "hybrid",
    ) -> None:
        valid_modes = {"dense", "sparse", "hybrid"}
        if fusion_mode.lower() not in valid_modes:
            raise ValueError(
                f"Invalid fusion_mode {fusion_mode!r}. Must be one of {sorted(valid_modes)}."
            )
        self._embedding = embedding
        self._vector_store = vector_store
        self._keyword_search = keyword_search
        self._top_k = top_k
        self._alpha = alpha
        self._pool_size = candidate_pool_size
        self._parent_lookup = parent_lookup
        self._fusion_mode = fusion_mode.lower()

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
        """Core retrieval logic — no error wrapping.

        Behaviour differs by ``fusion_mode``:
        - ``dense``  : embed query → vector search only → rank by vector score.
        - ``sparse`` : BM25 search only → rank by keyword score.
        - ``hybrid`` : both → weighted combination of normalised scores.
        """

        # Step 1: Embed the query (needed for dense/hybrid; cheap to skip for sparse
        # but we call it unconditionally so the interface contract stays simple).
        query_embedding = self._embedding.embed_query(query)

        # Step 2: Fetch candidates according to the fusion mode.
        candidates: dict[str, dict[str, Any]] = {}

        if self._fusion_mode in ("dense", "hybrid"):
            vector_results = self._vector_store.search(
                query_embedding=query_embedding,
                top_k=self._pool_size,
            )
            for vr in vector_results:
                cid = vr["id"]
                candidates[cid] = {
                    "parent_id": vr["metadata"]["parent_id"],
                    "company": vr["metadata"].get("company", ""),
                    "vector_distance": vr["distance"],
                    "bm25_score": 0.0,
                    "metadata": vr["metadata"],
                }

        if self._fusion_mode in ("sparse", "hybrid"):
            bm25_results = self._keyword_search.search(
                query=query,
                top_k=self._pool_size,
            )
            for br in bm25_results:
                cid = br["id"]
                if cid in candidates:
                    candidates[cid]["bm25_score"] = br["score"]
                else:
                    candidates[cid] = {
                        "parent_id": br["metadata"]["parent_id"],
                        "company": br["metadata"].get("company", ""),
                        "vector_distance": float("inf"),
                        "bm25_score": br["score"],
                        "metadata": br["metadata"],
                    }

        if not candidates:
            logger.warning(
                "[%s] Both retrievers returned zero results for query: %s",
                self._fusion_mode,
                query,
            )
            return []

        # Step 3: Normalise scores.
        cids = list(candidates.keys())
        vector_distances = np.array(
            [candidates[c]["vector_distance"] for c in cids], dtype=np.float64
        )
        bm25_raw = np.array(
            [candidates[c]["bm25_score"] for c in cids], dtype=np.float64
        )

        vector_norm = normalize_vector_scores(vector_distances, already_similarity=False)
        bm25_norm = normalize_bm25_scores(bm25_raw)

        # Step 4: Combine scores according to fusion mode.
        if self._fusion_mode == "dense":
            combined = vector_norm
        elif self._fusion_mode == "sparse":
            combined = bm25_norm
        else:  # hybrid
            combined = combine_scores(vector_norm, bm25_norm, self._alpha)

        for i, cid in enumerate(cids):
            candidates[cid]["vector_norm"] = float(vector_norm[i])
            candidates[cid]["bm25_norm"] = float(bm25_norm[i])
            candidates[cid]["combined"] = float(combined[i])

        # Step 5: Rank by combined score (descending).
        ranked = sorted(cids, key=lambda c: candidates[c]["combined"], reverse=True)

        # Step 6: Deduplicate by parent_id — keep best-scoring child per parent.
        seen_parents: set[str] = set()
        deduped: list[str] = []
        for cid in ranked:
            pid = candidates[cid]["parent_id"]
            if pid not in seen_parents:
                seen_parents.add(pid)
                deduped.append(cid)

        # Step 7: Return top K, including rich metadata for downstream use.
        top_cids = deduped[: self._top_k]

        results: list[dict[str, Any]] = []
        for cid in top_cids:
            c = candidates[cid]
            pid = c["parent_id"]
            meta = c["metadata"]
            parent_info = self._parent_lookup.get(pid, {})
            results.append(
                {
                    "parent_id": pid,
                    "company": c["company"],
                    "county": meta.get("county", ""),
                    "ev_supply_chain_role": meta.get("ev_supply_chain_role", ""),
                    "industry_name": meta.get("industry_name", ""),
                    "tier_level": meta.get("tier_level", ""),
                    "primary_oems": meta.get("primary_oems", ""),
                    "facility_type": meta.get("facility_type", ""),
                    "employment": meta.get("employment", ""),
                    "product_service": meta.get("product_service", ""),
                    "ev_relevant": meta.get("ev_relevant", ""),
                    "supplier_type": meta.get("supplier_type", ""),
                    "parent_text": parent_info.get("text", ""),
                    "similarity_score": round(c["vector_norm"], 6),
                    "keyword_score": round(c["bm25_norm"], 6),
                    "combined_score": round(c["combined"], 6),
                }
            )

        logger.info(
            "[%s] Query retrieved %d candidates → %d after dedup → returning top %d",
            self._fusion_mode,
            len(candidates),
            len(deduped),
            len(results),
        )
        return results



