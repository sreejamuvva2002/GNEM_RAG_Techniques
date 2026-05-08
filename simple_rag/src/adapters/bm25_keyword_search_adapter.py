"""BM25 keyword search adapter using ``rank_bm25``.

Wraps the ``BM25Okapi`` implementation behind
``KeywordSearchInterface`` for clean swappability.
"""

from __future__ import annotations

from typing import Any

from rank_bm25 import BM25Okapi

from src.interfaces.keyword_search_interface import KeywordSearchInterface
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BM25KeywordSearchAdapter(KeywordSearchInterface):
    """Keyword search using BM25 (Okapi variant).

    Documents are tokenised via simple whitespace + lower-case splitting.
    For production use, a more sophisticated tokeniser could be swapped
    in without changing the interface contract.
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._documents: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # KeywordSearchInterface implementation                               #
    # ------------------------------------------------------------------ #

    def index(self, documents: list[dict[str, Any]]) -> None:
        """Build the BM25 index from documents.

        Args:
            documents: Each dict must contain ``id``, ``text``, ``metadata``.
        """
        self._documents = documents
        corpus = [self._tokenize(doc["text"]) for doc in documents]
        self._bm25 = BM25Okapi(corpus)
        logger.info("BM25 index built over %d documents", len(documents))

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Retrieve the top-K documents by BM25 score.

        Args:
            query: Raw query string.
            top_k: Maximum results.

        Returns:
            List of result dicts ordered by descending score.
        """
        if self._bm25 is None or not self._documents:
            return []

        tokens = self._tokenize(query)
        raw_scores = self._bm25.get_scores(tokens)

        # Get top-k indices by score.
        top_indices = sorted(
            range(len(raw_scores)),
            key=lambda i: raw_scores[i],
            reverse=True,
        )[:top_k]

        results: list[dict[str, Any]] = []
        for idx in top_indices:
            if raw_scores[idx] <= 0.0:
                continue  # Skip zero-score documents.
            doc = self._documents[idx]
            results.append(
                {
                    "id": doc["id"],
                    "score": float(raw_scores[idx]),
                    "metadata": doc["metadata"],
                }
            )
        return results

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace tokenizer with lowercasing."""
        return text.lower().split()
