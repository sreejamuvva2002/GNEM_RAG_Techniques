from __future__ import annotations
"""Ingestion service — load → chunk → embed → index → export.

This service orchestrates the data ingestion pipeline.  It depends
only on interfaces, so swapping data sources, embedding providers,
or vector stores requires zero changes here.
"""

from src.utils import get_logger

from pathlib import Path
from typing import Any

import pandas as pd

from src.chunking import Chunk, ParentChildChunker

logger = get_logger(__name__)


class IngestionService:
    """Orchestrates data ingestion: load → chunk → embed → index.

    Args:
        data_loader: Adapter for reading the source data.
        chunker: Domain-logic chunker that produces chunks.
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
        self._chunks: list[Chunk] = []

    @property
    def chunks(self) -> list[Chunk]:
        """Return the chunks built during ingestion."""
        return self._chunks

    @property
    def parent_lookup(self) -> dict[str, dict[str, Any]]:
        """Build a parent lookup from chunks.

        Since each chunk represents one company record, the lookup
        maps ``record_id`` → ``{text, company}``.
        """
        return {
            c.record_id: {
                "text": c.embedding_text,
                "company": c.metadata.get("Company_Clean", ""),
            }
            for c in self._chunks
        }

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
        2. Chunk rows into structured chunks with metadata.
        3. Embed chunk texts via the embedding adapter.
        4. Index chunks in the vector store (ChromaDB).
        5. Index chunks in the keyword search backend (BM25).

        Args:
            data_path: Path to the source data file.
            sheet_name: Worksheet name.
            columns: Column names to retain.
            index_dir: Directory for persisting the vector index.
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
        logger.info("Step 2/5: Chunking rows into structured chunks")
        self._chunks = self._chunker.chunk_rows(rows)

        # Step 3: Embed chunks.
        logger.info("Step 3/5: Embedding %d chunks", len(self._chunks))
        chunk_texts = [c.embedding_text for c in self._chunks]
        embeddings = self._embedding.embed_texts(chunk_texts)

        # Step 4: Index chunks in the vector store.
        logger.info("Step 4/5: Indexing chunks in vector store")
        chunk_ids = [c.chunk_id for c in self._chunks]
        chunk_metadatas = [
            {
                "text": c.embedding_text,
                "parent_id": c.record_id,
                "company": c.metadata.get("Company_Clean", ""),
                "county": c.metadata.get("County", ""),
                "tier_level": c.metadata.get("Tier_Level", ""),
                "is_oem": c.metadata.get("Is_OEM", False),
                "ev_relevant": c.metadata.get("EV_Relevant", ""),
                "industry_name": c.metadata.get("Industry_Name", ""),
                "ev_supply_chain_role": c.metadata.get("EV_Supply_Chain_Role", ""),
                "primary_oems": c.metadata.get("Primary_OEMs", ""),
                "facility_type": c.metadata.get("Facility_Type", ""),
                "employment": c.metadata.get("Employment_Formatted", ""),
                "product_service": c.metadata.get("Product_Service", ""),
                "supplier_type": c.metadata.get("Supplier_Type", ""),
            }
            for c in self._chunks
        ]
        self._vector_store.add(
            ids=chunk_ids,
            embeddings=embeddings,
            metadatas=chunk_metadatas,
            documents=chunk_texts,
        )

        # Save vector index to disk.
        self._vector_store.save(index_dir)

        # Step 5: Index chunks in the keyword search backend.
        logger.info("Step 5/5: Building BM25 keyword index")
        keyword_docs = [
            {
                "id": c.chunk_id,
                "text": c.embedding_text,
                "metadata": {
                    "text": c.embedding_text,
                    "parent_id": c.record_id,
                    "company": c.metadata.get("Company_Clean", ""),
                    "county": c.metadata.get("County", ""),
                    "tier_level": c.metadata.get("Tier_Level", ""),
                    "is_oem": c.metadata.get("Is_OEM", False),
                    "ev_relevant": c.metadata.get("EV_Relevant", ""),
                    "industry_name": c.metadata.get("Industry_Name", ""),
                    "ev_supply_chain_role": c.metadata.get("EV_Supply_Chain_Role", ""),
                    "primary_oems": c.metadata.get("Primary_OEMs", ""),
                    "facility_type": c.metadata.get("Facility_Type", ""),
                    "employment": c.metadata.get("Employment_Formatted", ""),
                    "product_service": c.metadata.get("Product_Service", ""),
                    "supplier_type": c.metadata.get("Supplier_Type", ""),
                },
            }
            for c in self._chunks
        ]
        self._keyword_search.index(keyword_docs)

        logger.info(
            "Ingestion complete: %d chunks indexed "
            "(vector store: %d, BM25: %d docs)",
            len(self._chunks),
            self._vector_store.size(),
            len(keyword_docs),
        )

    def export_chunks(self, output_path: Path) -> None:
        """Export all chunks to an Excel file.

        Each row contains: Chunk_ID, Record_ID, metadata fields,
        Embedding_Text, Char_Count, Token_Estimate.

        Args:
            output_path: Destination ``.xlsx`` path.
        """
        if not self._chunks:
            logger.warning("No chunks to export.")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows = [c.to_flat_dict() for c in self._chunks]
        df = pd.DataFrame(rows)
        df.to_excel(output_path, index=False, engine="openpyxl")
        logger.info("Chunks exported to %s (%d rows)", output_path, len(df))
