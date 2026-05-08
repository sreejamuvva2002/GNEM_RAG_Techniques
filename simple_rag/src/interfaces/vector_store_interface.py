"""Interface for vector store backends.

Implementations may use FAISS, Chroma, Pinecone, etc.  Core and
service layers depend only on this contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class VectorStoreInterface(ABC):
    """Contract for storing and searching dense vector embeddings."""

    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add documents to the index.

        Args:
            ids: Unique document identifiers (one per document).
            embeddings: Dense vectors corresponding to each document.
            metadatas: Metadata dicts carried alongside each vector.
                       Must include at least ``"text"`` and
                       ``"parent_id"`` keys.
        """

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Search the index for nearest neighbours.

        Args:
            query_embedding: Dense vector for the query.
            top_k: Maximum number of results to return.

        Returns:
            A list of result dicts, each containing at minimum:
            ``{"id": str, "distance": float, "metadata": dict}``.
            Results are ordered by ascending distance (lower = better).
        """

    @abstractmethod
    def save(self, directory: Path) -> None:
        """Persist the index to disk.

        Args:
            directory: Target directory (created if it does not exist).
        """

    @abstractmethod
    def load(self, directory: Path) -> None:
        """Load a previously persisted index from disk.

        Args:
            directory: Directory containing the saved index files.
        """

    @abstractmethod
    def size(self) -> int:
        """Return the number of vectors currently in the index."""

    @abstractmethod
    def get_all_documents(self) -> list[dict[str, Any]]:
        """Return all stored documents with their metadata.

        Returns:
            A list of dicts, each containing at minimum:
            ``{"id": str, "metadata": dict}`` where metadata
            includes ``"text"`` and ``"parent_id"``.
        """
