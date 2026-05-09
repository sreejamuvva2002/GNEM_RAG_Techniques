from __future__ import annotations
"""ChromaDB vector store adapter.

ChromaDB is a lightweight, open-source embedding database that stores
embeddings alongside metadata and supports efficient similarity search.
It persists to disk automatically and allows retrieval of stored
embeddings — making it easy to inspect and export vectors.
"""

from src.utils import get_logger

import json
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings


logger = get_logger(__name__)


class ChromaVectorStore:
    """ChromaDB-backed vector store.

    Args:
        persist_directory: Directory for ChromaDB's persistent storage.
        collection_name: Name of the ChromaDB collection.
        distance_metric: Distance function — ``"cosine"``, ``"l2"``,
                         or ``"ip"`` (inner product).
    """

    def __init__(
        self,
        persist_directory: str | Path = ".cache/chroma_db",
        collection_name: str = "gnem_chunks",
        distance_metric: str = "cosine",
        upsert_batch_size: int = 500,
    ) -> None:
        self._persist_dir = str(Path(persist_directory).resolve())
        self._collection_name = collection_name
        self._distance_metric = distance_metric
        self._upsert_batch_size = upsert_batch_size

        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": self._distance_metric},
        )
        logger.info(
            "ChromaDB collection '%s' ready (persist=%s, metric=%s)",
            self._collection_name,
            self._persist_dir,
            self._distance_metric,
        )

    # ------------------------------------------------------------------ #
    # VectorStoreInterface implementation                                 #
    # ------------------------------------------------------------------ #

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        documents: list[str],
    ) -> None:
        """Add vectors and metadata to the ChromaDB collection.

        Args:
            ids: Unique document identifiers.
            embeddings: Dense float vectors.
            metadatas: Metadata dicts.
            documents: Raw document texts.
        """
        # ChromaDB metadata values must be str, int, float, or bool.
        sanitised_metas = [self._sanitise_metadata(m) for m in metadatas]

        for start in range(0, len(ids), self._upsert_batch_size):
            end = start + self._upsert_batch_size
            self._collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                metadatas=sanitised_metas[start:end],
                documents=documents[start:end],
            )
        logger.info(
            "Added %d vectors to ChromaDB (total: %d)",
            len(ids),
            self._collection.count(),
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Search for nearest neighbours in the collection.

        Args:
            query_embedding: Dense query vector.
            top_k: Maximum results to return.

        Returns:
            List of result dicts with ``id``, ``distance``, ``metadata``.
            Ordered by ascending distance.
        """
        if self._collection.count() == 0:
            return []

        effective_k = min(top_k, self._collection.count())
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_k,
            include=["distances", "metadatas", "documents"],
        )

        output: list[dict[str, Any]] = []
        for i in range(len(results["ids"][0])):
            output.append(
                {
                    "id": results["ids"][0][i],
                    "distance": results["distances"][0][i],
                    "metadata": results["metadatas"][0][i],
                }
            )
        return output

    def save(self, directory: Path) -> None:
        """ChromaDB auto-persists — this is a no-op.

        Args:
            directory: Ignored (ChromaDB manages its own persistence).
        """
        logger.debug("ChromaDB auto-persists; explicit save is a no-op.")

    def load(self, directory: Path) -> None:
        """Reload collection from the persisted directory.

        Args:
            directory: Ignored — uses the directory from ``__init__``.
        """
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": self._distance_metric},
        )
        logger.info(
            "ChromaDB collection loaded (%d vectors)",
            self._collection.count(),
        )

    def size(self) -> int:
        """Return the number of vectors in the collection."""
        return self._collection.count()

    def get_all_documents(self) -> list[dict[str, Any]]:
        """Return all stored documents with metadata.

        Returns:
            List of dicts with ``id`` and ``metadata`` keys.
        """
        if self._collection.count() == 0:
            return []

        results = self._collection.get(include=["metadatas", "documents"])
        return [
            {
                "id": results["ids"][i],
                "metadata": results["metadatas"][i],
            }
            for i in range(len(results["ids"]))
        ]

    def get_embeddings(self, ids: list[str] | None = None) -> dict[str, list[float]]:
        """Retrieve stored embedding vectors from ChromaDB.

        Args:
            ids: Optional list of IDs.  If ``None``, returns all.

        Returns:
            Dict mapping document ID → embedding vector.
        """
        if self._collection.count() == 0:
            return {}

        if ids is not None:
            results = self._collection.get(
                ids=ids,
                include=["embeddings"],
            )
        else:
            results = self._collection.get(include=["embeddings"])

        return {
            results["ids"][i]: results["embeddings"][i]
            for i in range(len(results["ids"]))
        }

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sanitise_metadata(meta: dict[str, Any]) -> dict[str, Any]:
        """Ensure all metadata values are ChromaDB-compatible types.

        ChromaDB accepts: ``str``, ``int``, ``float``, ``bool``.
        Convert anything else to string; replace ``None`` with ``""``.
        """
        clean: dict[str, Any] = {}
        for key, value in meta.items():
            if value is None:
                clean[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                clean[key] = value
            else:
                clean[key] = str(value)
        return clean
