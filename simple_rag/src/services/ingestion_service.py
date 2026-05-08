"""Ingestion service — load → chunk → embed → index.

This service orchestrates the data ingestion pipeline.  It depends
only on interfaces, so swapping data sources, embedding providers,
or vector stores requires zero changes here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.parent_child_chunker import Chunk, ParentChildChunker
from src.interfaces.data_loader_interface import DataLoaderInterface
from src.interfaces.embedding_interface import EmbeddingInterface
from src.interfaces.keyword_search_interface import KeywordSearchInterface
from src.interfaces.vector_store_interface import VectorStoreInterface
from src.utils.logger import get_logger

logger = get_logger(__name__)


class IngestionService:
    """Orchestrates data ingestion: load → chunk → embed → index.

    Args:
        data_loader: Adapter for reading the source data.
        chunker: Domain-logic chunker that produces parent + child chunks.
        embedding: Adapter for generating dense embeddings.
        vector_store: Adapter for storing and searching embeddings.
        keyword_search: Adapter for keyword-based search (BM25).
    """

    def __init__(
        self,
        data_loader: DataLoaderInterface,
        chunker: ParentChildChunker,
        embedding: EmbeddingInterface,
        vector_store: VectorStoreInterface,
        keyword_search: KeywordSearchInterface,
    ) -> None:
        self._data_loader = data_loader
        self._chunker = chunker
        self._embedding = embedding
        self._vector_store = vector_store
        self._keyword_search = keyword_search
        self._parent_lookup: dict[str, dict[str, Any]] = {}

    @property
    def parent_lookup(self) -> dict[str, dict[str, Any]]:
        """Return the parent lookup built during ingestion.

        Keys are parent IDs; values are dicts with ``text`` and
        ``company`` keys.
        """
        return self._parent_lookup

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def ingest(
        self,
        data_path: Path,
        sheet_name: str,
        columns: list[str],
        index_dir: Path,
    ) -> None:
        """Run the full ingestion pipeline.

        1. Load rows from the data source.
        2. Chunk rows into parents + children.
        3. Embed child chunks via the embedding adapter.
        4. Index children in the vector store (FAISS).
        5. Index children in the keyword search backend (BM25).
        6. Persist the FAISS index to disk.

        Args:
            data_path: Path to the source data file.
            sheet_name: Worksheet name.
            columns: Column names to retain.
            index_dir: Directory for persisting the FAISS index.
        """
        # Step 1: Load data.
        logger.info("Step 1/5: Loading data from %s", data_path)
        rows = self._data_loader.load(
            path=data_path,
            sheet_name=sheet_name,
            columns=columns,
        )
        logger.info("Loaded %d rows", len(rows))

        # Step 2: Chunk rows.
        logger.info("Step 2/5: Chunking rows into parent + child chunks")
        parents, children = self._chunker.chunk_rows(rows)

        # Build parent lookup for retrieval.
        self._parent_lookup = {
            p.chunk_id: {"text": p.text, "company": p.metadata.get("company", "")}
            for p in parents
        }

        # Step 3: Embed child chunks.
        logger.info("Step 3/5: Embedding %d child chunks", len(children))
        child_texts = [c.text for c in children]
        embeddings = self._embedding.embed_texts(child_texts)

        # Step 4: Index children in the vector store.
        logger.info("Step 4/5: Indexing children in vector store")
        child_ids = [c.chunk_id for c in children]
        child_metadatas = [
            {
                "text": c.text,
                "parent_id": c.parent_id,
                "company": c.metadata.get("company", ""),
                "group": c.group_name or "",
            }
            for c in children
        ]
        self._vector_store.add(
            ids=child_ids,
            embeddings=embeddings,
            metadatas=child_metadatas,
        )

        # Save FAISS index to disk.
        self._vector_store.save(index_dir)

        # Step 5: Index children in the keyword search backend.
        logger.info("Step 5/5: Building BM25 keyword index")
        keyword_docs = [
            {
                "id": c.chunk_id,
                "text": c.text,
                "metadata": {
                    "text": c.text,
                    "parent_id": c.parent_id,
                    "company": c.metadata.get("company", ""),
                    "group": c.group_name or "",
                },
            }
            for c in children
        ]
        self._keyword_search.index(keyword_docs)

        logger.info(
            "Ingestion complete: %d parents, %d children indexed "
            "(vector store: %d, BM25: %d docs)",
            len(parents),
            len(children),
            self._vector_store.size(),
            len(keyword_docs),
        )
