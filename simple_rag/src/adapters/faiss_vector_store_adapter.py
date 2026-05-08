"""FAISS vector store adapter.

Chosen for simplicity and zero-server overhead — FAISS runs entirely
in-process.  This adapter wraps ``faiss-cpu`` behind the
``VectorStoreInterface`` so it can be swapped for Chroma, Pinecone, etc.
without touching core or service code.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.interfaces.vector_store_interface import VectorStoreInterface
from src.utils.logger import get_logger

logger = get_logger(__name__)

_INDEX_FILE = "faiss_index.bin"
_META_FILE = "faiss_meta.pkl"


class FAISSVectorStoreAdapter(VectorStoreInterface):
    """In-process FAISS vector store.

    Uses a flat L2 index (``IndexFlatL2``) for exact nearest-neighbour
    search.  For datasets of ~5 k vectors this is fast enough and
    avoids the complexity of approximate indices.

    Args:
        dimension: Dimensionality of the embedding vectors.  Must
                   match the embedding model output (768 for
                   ``nomic-embed-text``).
    """

    def __init__(self, dimension: int = 768) -> None:
        self._dimension = dimension
        self._index: faiss.IndexFlatL2 = faiss.IndexFlatL2(dimension)
        self._ids: list[str] = []
        self._metadatas: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # VectorStoreInterface implementation                                 #
    # ------------------------------------------------------------------ #

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add vectors and their metadata to the FAISS index.

        Args:
            ids: Unique document identifiers.
            embeddings: Dense float vectors.
            metadatas: Metadata dicts (must include ``text``, ``parent_id``).
        """
        vectors = np.array(embeddings, dtype=np.float32)
        self._index.add(vectors)
        self._ids.extend(ids)
        self._metadatas.extend(metadatas)
        logger.debug("Added %d vectors (total: %d)", len(ids), self._index.ntotal)

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Search for nearest neighbours by L2 distance.

        Args:
            query_embedding: Dense query vector.
            top_k: Maximum results to return.

        Returns:
            List of dicts with ``id``, ``distance``, ``metadata``.
            Ordered by ascending distance (lower = more similar).
        """
        if self._index.ntotal == 0:
            return []

        effective_k = min(top_k, self._index.ntotal)
        query_vec = np.array([query_embedding], dtype=np.float32)
        distances, indices = self._index.search(query_vec, effective_k)

        results: list[dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append(
                {
                    "id": self._ids[idx],
                    "distance": float(dist),
                    "metadata": self._metadatas[idx],
                }
            )
        return results

    def save(self, directory: Path) -> None:
        """Persist the FAISS index and metadata to disk.

        Args:
            directory: Target directory (created if absent).
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / _INDEX_FILE))
        with open(directory / _META_FILE, "wb") as fh:
            pickle.dump(
                {"ids": self._ids, "metadatas": self._metadatas}, fh
            )
        logger.info("FAISS index saved to %s (%d vectors)", directory, self._index.ntotal)

    def load(self, directory: Path) -> None:
        """Load a previously saved FAISS index from disk.

        Args:
            directory: Directory containing the saved files.
        """
        directory = Path(directory)
        self._index = faiss.read_index(str(directory / _INDEX_FILE))
        with open(directory / _META_FILE, "rb") as fh:
            data = pickle.load(fh)  # noqa: S301
        self._ids = data["ids"]
        self._metadatas = data["metadatas"]
        self._dimension = self._index.d
        logger.info("FAISS index loaded from %s (%d vectors)", directory, self._index.ntotal)

    def size(self) -> int:
        """Return the number of vectors in the index."""
        return self._index.ntotal

    def get_all_documents(self) -> list[dict[str, Any]]:
        """Return all stored documents with their metadata.

        Returns:
            List of dicts with ``id`` and ``metadata`` keys.
        """
        return [
            {"id": self._ids[i], "metadata": self._metadatas[i]}
            for i in range(len(self._ids))
        ]
