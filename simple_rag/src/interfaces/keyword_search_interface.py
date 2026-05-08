"""Interface for keyword / sparse search backends.

Implementations may use BM25, TF-IDF, Elasticsearch, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class KeywordSearchInterface(ABC):
    """Contract for keyword-based (sparse) document retrieval."""

    @abstractmethod
    def index(self, documents: list[dict[str, Any]]) -> None:
        """Build the keyword index from a list of documents.

        Args:
            documents: Each dict must contain at least
                       ``{"id": str, "text": str, "metadata": dict}``.
        """

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Search the index for keyword-relevant documents.

        Args:
            query: Raw query string.
            top_k: Maximum number of results.

        Returns:
            A list of result dicts, each containing:
            ``{"id": str, "score": float, "metadata": dict}``.
            Results are ordered by descending score (higher = better).
        """
