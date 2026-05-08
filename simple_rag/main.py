#!/usr/bin/env python3
"""Simple RAG — composition root and CLI entry point.

A single invocation of ``python main.py`` performs the full pipeline:

    1. Load configuration from ``config/config.yaml``.
    2. Load GNEM data → chunk → embed via Ollama → build FAISS + BM25 indices.
    3. Query 50 questions through the hybrid retrieval flow.
    4. Write ``output/contexts.xlsx``.

**Dependency wiring happens ONLY here** (DIP).  Services and core import
from ``interfaces/``, never from ``adapters/``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that ``src`` is importable
# regardless of the working directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logging, get_logger

# Adapters (concrete implementations) — imported only here.
from src.adapters.nomic_ollama_embedding_adapter import NomicOllamaEmbeddingAdapter
from src.adapters.faiss_vector_store_adapter import FAISSVectorStoreAdapter
from src.adapters.bm25_keyword_search_adapter import BM25KeywordSearchAdapter
from src.adapters.excel_data_loader_adapter import ExcelDataLoaderAdapter
from src.adapters.excel_result_writer_adapter import ExcelResultWriterAdapter

# Core / domain logic.
from src.core.parent_child_chunker import ParentChildChunker
from src.core.hybrid_retriever import HybridRetriever

# Services.
from src.services.ingestion_service import IngestionService
from src.services.rag_service import RAGService


def main() -> None:
    """Run the end-to-end Simple RAG pipeline."""

    # ------------------------------------------------------------------ #
    # 1. Load configuration.
    # ------------------------------------------------------------------ #
    cfg = load_config()
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    logger = get_logger("main")
    logger.info("=" * 60)
    logger.info("Simple RAG Pipeline — starting")
    logger.info("=" * 60)
    start_time = time.time()

    # ------------------------------------------------------------------ #
    # 2. Wire adapters (Dependency Injection via composition root).
    # ------------------------------------------------------------------ #
    embedding_cfg = cfg["embedding"]
    retrieval_cfg = cfg["retrieval"]
    data_cfg = cfg["data"]
    chunking_cfg = cfg["chunking"]

    data_loader = ExcelDataLoaderAdapter()
    embedding = NomicOllamaEmbeddingAdapter(
        model=embedding_cfg["model"],
        ollama_host=embedding_cfg["ollama_host"],
        batch_size=embedding_cfg["batch_size"],
        timeout=embedding_cfg["request_timeout_seconds"],
    )
    vector_store = FAISSVectorStoreAdapter(dimension=768)
    keyword_search = BM25KeywordSearchAdapter()
    result_writer = ExcelResultWriterAdapter()

    chunker = ParentChildChunker(
        text_columns=data_cfg["text_columns"],
        child_groups=chunking_cfg["child_groups"],
    )

    # ------------------------------------------------------------------ #
    # 3. Ingestion: load → chunk → embed → index.
    # ------------------------------------------------------------------ #
    ingestion = IngestionService(
        data_loader=data_loader,
        chunker=chunker,
        embedding=embedding,
        vector_store=vector_store,
        keyword_search=keyword_search,
    )

    ingestion.ingest(
        data_path=cfg["paths"]["input_data"],
        sheet_name=data_cfg["sheet_name"],
        columns=data_cfg["text_columns"],
        index_dir=cfg["paths"]["index_dir"],
    )

    # ------------------------------------------------------------------ #
    # 4. Build the hybrid retriever.
    # ------------------------------------------------------------------ #
    retriever = HybridRetriever(
        embedding=embedding,
        vector_store=vector_store,
        keyword_search=keyword_search,
        top_k=retrieval_cfg["top_k"],
        alpha=retrieval_cfg["semantic_weight"],
        candidate_pool_size=retrieval_cfg["candidate_pool_size"],
        parent_lookup=ingestion.parent_lookup,
    )

    questions_path: Path = cfg["paths"]["questions"]

    # ------------------------------------------------------------------ #
    # 5. Query phase: 50 questions → hybrid retrieval → write output.
    # ------------------------------------------------------------------ #
    rag_service = RAGService(
        retriever=retriever,
        result_writer=result_writer,
    )

    rag_service.run(
        questions_path=questions_path,
        output_path=cfg["paths"]["output"],
    )

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Pipeline finished in %.1f seconds", elapsed)
    logger.info("Output: %s", cfg["paths"]["output"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
