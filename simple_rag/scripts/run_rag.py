#!/usr/bin/env python3
"""Minimal, unified RAG Pipeline for GNEM.

Performs:
  1. Ingestion (Parent-Child Chunking -> Chroma/BM25 & chunks.xlsx).
  2. Retrieval using each fusion mode from config (dense / sparse / hybrid).
  3. Generation (LLM) with all retrieved context passed verbatim.
  4. Saves results in separate Excel sheets — one per fusion mode.

Usage:
    cd simple_rag/
    python scripts/run_rag.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import (
    get_logger,
    setup_logging,
    load_config,
    load_questions,
    write_multi_sheet_excel,
)
from src.data_loader import ExcelDataLoader
from src.embedding import NomicOllamaEmbedding
from src.vector_store import ChromaVectorStore
from src.keyword_search import BM25KeywordSearch
from src.chunking import ParentChildChunker
from src.retrieval import HybridRetriever
from src.ingestion import IngestionService
from src.llm import LLMClient, build_evidence_prompt, parse_llm_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_retrieved_context(evidence_list: list[dict[str, Any]]) -> str:
    """Serialise the full retrieved context to a readable string for Excel.

    Mirrors exactly what is sent to the LLM so the analyst can inspect
    every KB entry and its retrieval scores side-by-side with the answer.
    """
    parts: list[str] = []
    for ev in evidence_list:
        eid          = ev.get("evidence_id", "")
        company      = ev.get("company", "")          or "Not available"
        county       = ev.get("county", "")           or "Not available"
        role         = ev.get("ev_supply_chain_role", "") or "Not available"
        tier         = ev.get("tier_level", "")       or "Not available"
        employment   = ev.get("employment", "")       or "Not available"
        primary_oems = ev.get("primary_oems", "")     or "Not available"
        facility     = ev.get("facility_type", "")    or "Not available"
        industry     = ev.get("industry_name", "")    or "Not available"
        ev_relevant  = ev.get("ev_relevant", "")      or "Not available"
        product_svc  = ev.get("product_service", "")  or "Not available"
        supplier_type = ev.get("supplier_type", "")   or "Not available"
        sim      = ev.get("similarity_score", 0.0)
        kw       = ev.get("keyword_score", 0.0)
        combined = ev.get("combined_score", 0.0)

        parts.append(
            f"[{eid}]  sim={sim:.4f}  kw={kw:.4f}  combined={combined:.4f}\n"
            f"Company: {company}\n"
            f"Location (County): {county}\n"
            f"EV Supply Chain Role: {role}\n"
            f"Tier: {tier}\n"
            f"Employment: {employment}\n"
            f"Primary OEMs: {primary_oems}\n"
            f"Facility Type: {facility}\n"
            f"Industry Group: {industry}\n"
            f"EV / Battery Relevant: {ev_relevant}\n"
            f"Product / Service: {product_svc}\n"
            f"Supplier / Affiliation Type: {supplier_type}"
        )

    return "\n\n---\n\n".join(parts)


def _run_for_mode(
    retriever: HybridRetriever,
    llm_client: LLMClient,
    questions_df: pd.DataFrame,
    fusion_mode: str,
    logger: Any,
) -> list[dict[str, Any]]:
    """Run the full RAG pipeline for one fusion mode.

    For every question:
      1. Retrieve top_k contexts.
      2. Build evidence prompt (all contexts passed verbatim to LLM).
      3. Generate answer.
      4. Parse LLM citations.
      5. Return a result row including the exact retrieved context.
    """
    results: list[dict[str, Any]] = []

    for _, row in questions_df.iterrows():
        qid = row["question_id"]
        question = row["question"]
        logger.info("[%s] Q%s: %s", fusion_mode, qid, question[:80])

        try:
            # --- Retrieve ---
            retrieved_items = retriever.retrieve(question)

            evidence_list: list[dict[str, Any]] = []
            for i, item in enumerate(retrieved_items):
                evidence_list.append(
                    {
                        "evidence_id": f"E{i + 1:03d}",
                        "company":             item.get("company", ""),
                        "county":              item.get("county", ""),
                        "ev_supply_chain_role":item.get("ev_supply_chain_role", ""),
                        "tier_level":          item.get("tier_level", ""),
                        "employment":          item.get("employment", ""),
                        "primary_oems":        item.get("primary_oems", ""),
                        "facility_type":       item.get("facility_type", ""),
                        "industry_name":       item.get("industry_name", ""),
                        "ev_relevant":         item.get("ev_relevant", ""),
                        "product_service":     item.get("product_service", ""),
                        "supplier_type":       item.get("supplier_type", ""),
                        "text":                item.get("parent_text", ""),
                        "similarity_score":    item.get("similarity_score", 0.0),
                        "keyword_score":       item.get("keyword_score", 0.0),
                        "combined_score":      item.get("combined_score", 0.0),
                    }
                )

            retrieved_context = _format_retrieved_context(evidence_list)

            # --- Generate (all top_k contexts sent to LLM) ---
            messages = build_evidence_prompt(question, evidence_list)
            answer_text = llm_client.chat(messages)

            # --- Parse citations ---
            evidence_ids = [e["evidence_id"] for e in evidence_list]
            parsed = parse_llm_response(answer_text, evidence_ids)

            results.append(
                {
                    "question_id": qid,
                    "question": question,
                    "answer": answer_text,
                    "retrieved_context": retrieved_context,
                    "used_evidence_ids": parsed.get("used_evidence_ids", ""),
                    "llm_selected_evidence_summary": parsed.get(
                        "llm_selected_evidence_summary", ""
                    ),
                    "insufficient_evidence_flag": parsed.get(
                        "insufficient_evidence_flag", ""
                    ),
                    "evidence_count": len(retrieved_items),
                    "fusion_mode": fusion_mode,
                    "model": llm_client.model,
                    "error": "",
                }
            )

        except Exception as exc:
            logger.error("[%s] Failed on Q%s: %s", fusion_mode, qid, exc)
            results.append(
                {
                    "question_id": qid,
                    "question": question,
                    "answer": "",
                    "retrieved_context": "",
                    "used_evidence_ids": "",
                    "llm_selected_evidence_summary": "",
                    "insufficient_evidence_flag": "",
                    "evidence_count": 0,
                    "fusion_mode": fusion_mode,
                    "model": llm_client.model,
                    "error": str(exc),
                }
            )

    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Setup
    cfg = load_config()
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    logger = get_logger("run_rag")

    logger.info("=" * 60)
    logger.info("Starting RAG Pipeline")
    logger.info("=" * 60)
    start_time = time.time()

    # 2. Init Adapters & Chunker
    data_loader = ExcelDataLoader()
    embedding = NomicOllamaEmbedding(
        model=cfg["embedding"]["model"],
        ollama_host=cfg["embedding"]["ollama_host"],
        batch_size=cfg["embedding"]["batch_size"],
    )
    vector_store = ChromaVectorStore(
        persist_directory=cfg["paths"]["index_dir"],
        collection_name=cfg["vector_store"]["collection_name"],
        upsert_batch_size=cfg["vector_store"].get("upsert_batch_size", 500),
    )
    keyword_search = BM25KeywordSearch()
    chunker = ParentChildChunker(chunking_config=cfg["chunking"])

    # 3. Ingestion
    ingestion = IngestionService(
        data_loader=data_loader,
        chunker=chunker,
        embedding=embedding,
        vector_store=vector_store,
        keyword_search=keyword_search,
    )
    ingestion.ingest(
        data_path=cfg["paths"]["input_data"],
        sheet_name=cfg["data"]["sheet_name"],
        columns=cfg["data"]["text_columns"],
        index_dir=cfg["paths"]["index_dir"],
    )

    # 4. LLM client
    llm_defaults = cfg.get("llm", {}).get("defaults", {})
    llm_client = LLMClient(
        base_url=llm_defaults.get("base_url"),
        api_key=llm_defaults.get("api_key"),
        model=llm_defaults.get("model"),
        temperature=llm_defaults.get("temperature", 0.2),
    )

    # 5. Load questions
    questions_df = load_questions(cfg["paths"]["questions"])

    # 6. Retrieve & generate for each fusion mode
    retrieval_cfg = cfg["retrieval"]
    top_k = retrieval_cfg["top_k"]
    alpha = retrieval_cfg["semantic_weight"]
    pool_size = retrieval_cfg["candidate_pool_size"]
    fusion_modes: list[str] = retrieval_cfg.get("fusion_modes", ["hybrid"])

    logger.info(
        "Configuration: top_k=%d, pool_size=%d, alpha=%.2f, modes=%s",
        top_k,
        pool_size,
        alpha,
        fusion_modes,
    )

    output_sheets: dict[str, pd.DataFrame] = {}

    for mode in fusion_modes:
        logger.info("=" * 60)
        logger.info("Fusion mode: %s", mode)
        logger.info("=" * 60)

        retriever = HybridRetriever(
            embedding=embedding,
            vector_store=vector_store,
            keyword_search=keyword_search,
            top_k=top_k,
            alpha=alpha,
            candidate_pool_size=pool_size,
            parent_lookup=ingestion.parent_lookup,
            fusion_mode=mode,
        )

        mode_results = _run_for_mode(
            retriever=retriever,
            llm_client=llm_client,
            questions_df=questions_df,
            fusion_mode=mode,
            logger=logger,
        )

        sheet_name = f"RAG_{mode.capitalize()}"
        output_sheets[sheet_name] = pd.DataFrame(mode_results)
        logger.info("[%s] Done. %d results.", mode, len(mode_results))

    # 7. Save all sheets to a single Excel file
    output_path = Path(cfg["paths"]["generated_answers"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_multi_sheet_excel(path=str(output_path), sheets=output_sheets)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1f seconds", elapsed)
    logger.info("Output: %s", output_path)
    logger.info("Sheets: %s", list(output_sheets.keys()))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
