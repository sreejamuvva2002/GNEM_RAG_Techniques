#!/usr/bin/env python3
"""Parametric RAG experiment runner for the six-experiment comparison.

Runs one experiment at a time and saves outputs to a dedicated folder:
    outputs/experiments/{chunking}_{retrieval}/

Usage:
    cd simple_rag/
    python scripts/run_experiment.py --chunking normal --retrieval dense
    python scripts/run_experiment.py --chunking normal --retrieval sparse
    python scripts/run_experiment.py --chunking normal --retrieval hybrid
    python scripts/run_experiment.py --chunking parent_child --retrieval dense
    python scripts/run_experiment.py --chunking parent_child --retrieval sparse
    python scripts/run_experiment.py --chunking parent_child --retrieval hybrid

Environment variables:
    LLM_BASE_URL   Ollama/OpenAI-compatible base URL (default: http://localhost:11434/v1)
    LLM_API_KEY    API key (default: ollama)
    LLM_MODEL      Model name (default: gemma3:27b)

Outputs per experiment (written to outputs/experiments/{name}/):
    generated_answers.xlsx   50 rows, one per question
    retrieved_contexts.xlsx  Up to top_k rows per question
    run_config.json          All parameters for reproducibility
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

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
from src.chunking import ParentChildChunker, TrueParentChildChunker
from src.retrieval import HybridRetriever
from src.ingestion import IngestionService
from src.llm import LLMClient, build_evidence_prompt, parse_llm_response


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_CHUNKING = {"normal", "parent_child"}
VALID_RETRIEVAL = {"dense", "sparse", "hybrid"}

NORMAL_COLLECTION = "gnem_chunks"
PARENT_CHILD_COLLECTION = "gnem_child_chunks"


# ---------------------------------------------------------------------------
# Retrieval output helpers (identical for both chunking types)
# ---------------------------------------------------------------------------

def _format_retrieved_context(evidence_list: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for ev in evidence_list:
        eid           = ev.get("evidence_id", "")
        company       = ev.get("company", "")           or "Not available"
        county        = ev.get("county", "")            or "Not available"
        role          = ev.get("ev_supply_chain_role", "") or "Not available"
        tier          = ev.get("tier_level", "")        or "Not available"
        employment    = ev.get("employment", "")        or "Not available"
        primary_oems  = ev.get("primary_oems", "")      or "Not available"
        facility      = ev.get("facility_type", "")     or "Not available"
        industry      = ev.get("industry_name", "")     or "Not available"
        ev_relevant   = ev.get("ev_relevant", "")       or "Not available"
        product_svc   = ev.get("product_service", "")   or "Not available"
        supplier_type = ev.get("supplier_type", "")     or "Not available"
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
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []

    for _, row in questions_df.iterrows():
        qid = row["question_id"]
        question = row["question"]
        logger.info("[%s] Q%s: %s", fusion_mode, qid, question[:80])

        try:
            retrieved_items = retriever.retrieve(question)

            evidence_list: list[dict[str, Any]] = []
            for i, item in enumerate(retrieved_items):
                evidence = {
                    "evidence_id": f"E{i + 1:03d}",
                    "company":             item.get("company", ""),
                    "county":              item.get("county", ""),
                    "ev_supply_chain_role": item.get("ev_supply_chain_role", ""),
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
                evidence_list.append(evidence)
                context_rows.append({"question_id": qid, "question": question,
                                     "fusion_mode": fusion_mode, "evidence_rank": i + 1,
                                     **evidence})

            messages = build_evidence_prompt(question, evidence_list)
            answer_text = llm_client.chat(messages)
            evidence_ids = [e["evidence_id"] for e in evidence_list]
            parsed = parse_llm_response(answer_text, evidence_ids)

            results.append({
                "question_id": qid,
                "question": question,
                "answer": answer_text,
                "retrieved_context": _format_retrieved_context(evidence_list),
                "used_evidence_ids": parsed.get("used_evidence_ids", ""),
                "llm_selected_evidence_summary": parsed.get("llm_selected_evidence_summary", ""),
                "insufficient_evidence_flag": parsed.get("insufficient_evidence_flag", ""),
                "evidence_count": len(retrieved_items),
                "fusion_mode": fusion_mode,
                "model": llm_client.model,
                "error": "",
            })

        except Exception as exc:
            logger.error("[%s] Failed on Q%s: %s", fusion_mode, qid, exc)
            results.append({
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
            })

    return results, context_rows


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------

def _ingest_normal(
    cfg: dict[str, Any],
    embedding: NomicOllamaEmbedding,
    vector_store: ChromaVectorStore,
    keyword_search: BM25KeywordSearch,
    logger: logging.Logger,
) -> dict[str, dict[str, Any]]:
    """Standard row-level ingestion. Returns parent_lookup."""
    chunker = ParentChildChunker(chunking_config=cfg["chunking"])
    ingestion = IngestionService(
        data_loader=ExcelDataLoader(),
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
    return ingestion.parent_lookup


def _ingest_parent_child(
    cfg: dict[str, Any],
    embedding: NomicOllamaEmbedding,
    vector_store: ChromaVectorStore,
    keyword_search: BM25KeywordSearch,
    logger: logging.Logger,
) -> dict[str, dict[str, Any]]:
    """True parent-child ingestion. Indexes child chunks, returns parent_lookup."""
    chunker = TrueParentChildChunker(chunking_config=cfg["chunking"])
    data_loader = ExcelDataLoader()

    logger.info("Step 1/5: Loading data")
    rows = data_loader.load(
        path=cfg["paths"]["input_data"],
        sheet_name=cfg["data"]["sheet_name"],
        columns=cfg["data"]["text_columns"],
    )
    logger.info("Loaded %d rows", len(rows))

    logger.info("Step 2/5: Creating child chunks")
    child_chunks, parent_lookup = chunker.chunk_rows(rows)
    logger.info("%d child chunks, %d parent records", len(child_chunks), len(parent_lookup))

    logger.info("Step 3/5: Embedding %d child chunks", len(child_chunks))
    child_texts = [c.embedding_text for c in child_chunks]
    embeddings = embedding.embed_texts(child_texts)

    logger.info("Step 4/5: Indexing child chunks in vector store")
    chunk_ids = [c.chunk_id for c in child_chunks]
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
            "child_group": c.metadata.get("child_group", ""),
        }
        for c in child_chunks
    ]
    vector_store.add(
        ids=chunk_ids,
        embeddings=embeddings,
        metadatas=chunk_metadatas,
        documents=child_texts,
    )
    vector_store.save(cfg["paths"]["index_dir"])

    logger.info("Step 5/5: Indexing child chunks in BM25")
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
                "child_group": c.metadata.get("child_group", ""),
            },
        }
        for c in child_chunks
    ]
    keyword_search.index(keyword_docs)

    logger.info(
        "Ingestion complete: %d child chunks indexed (vector store + BM25)",
        len(child_chunks),
    )
    return parent_lookup


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one RAG experiment (chunking × retrieval)."
    )
    parser.add_argument(
        "--chunking",
        required=True,
        choices=sorted(VALID_CHUNKING),
        help="Chunking strategy: 'normal' (row-level) or 'parent_child' (field groups).",
    )
    parser.add_argument(
        "--retrieval",
        required=True,
        choices=sorted(VALID_RETRIEVAL),
        help="Retrieval mode: dense, sparse, or hybrid.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs (default: abort if outputs exist).",
    )
    args = parser.parse_args()

    chunking_mode: str = args.chunking
    retrieval_mode: str = args.retrieval
    experiment_name = f"{chunking_mode}_{retrieval_mode}"

    cfg = load_config()
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    logger = get_logger("run_experiment")

    out_dir = _PROJECT_ROOT / "outputs" / "experiments" / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    answers_path = out_dir / "generated_answers.xlsx"
    if answers_path.exists() and not args.force:
        logger.error(
            "Output already exists: %s\n"
            "Use --force to overwrite, or choose a different experiment.",
            answers_path,
        )
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Experiment: %s", experiment_name)
    logger.info("Chunking:   %s", chunking_mode)
    logger.info("Retrieval:  %s", retrieval_mode)
    logger.info("Output dir: %s", out_dir)
    logger.info("=" * 60)

    start_time = time.time()
    run_ts = datetime.now().isoformat()

    # Select ChromaDB collection based on chunking type
    collection_name = NORMAL_COLLECTION if chunking_mode == "normal" else PARENT_CHILD_COLLECTION

    # Init adapters
    embedding = NomicOllamaEmbedding(
        model=cfg["embedding"]["model"],
        ollama_host=cfg["embedding"]["ollama_host"],
        batch_size=cfg["embedding"]["batch_size"],
    )
    vector_store = ChromaVectorStore(
        persist_directory=cfg["paths"]["index_dir"],
        collection_name=collection_name,
        upsert_batch_size=cfg["vector_store"].get("upsert_batch_size", 500),
    )
    keyword_search = BM25KeywordSearch()

    # Run ingestion
    logger.info("Starting ingestion (%s chunking, collection=%s)", chunking_mode, collection_name)
    if chunking_mode == "normal":
        parent_lookup = _ingest_normal(cfg, embedding, vector_store, keyword_search, logger)
    else:
        parent_lookup = _ingest_parent_child(cfg, embedding, vector_store, keyword_search, logger)

    logger.info("Ingestion done. Parent records: %d", len(parent_lookup))

    # LLM client
    llm_defaults = cfg.get("llm", {}).get("defaults", {})
    llm_client = LLMClient(
        base_url=llm_defaults.get("base_url"),
        api_key=llm_defaults.get("api_key"),
        model=llm_defaults.get("model"),
        temperature=llm_defaults.get("temperature", 0.2),
    )

    # Questions
    questions_df = load_questions(cfg["paths"]["questions"])
    logger.info("Loaded %d questions", len(questions_df))

    # Retrieval parameters
    retrieval_cfg = cfg["retrieval"]
    top_k = retrieval_cfg["top_k"]
    alpha = retrieval_cfg["semantic_weight"]
    pool_size = retrieval_cfg["candidate_pool_size"]

    logger.info("Config: top_k=%d, pool_size=%d, alpha=%.2f", top_k, pool_size, alpha)

    # Build retriever
    retriever = HybridRetriever(
        embedding=embedding,
        vector_store=vector_store,
        keyword_search=keyword_search,
        top_k=top_k,
        alpha=alpha,
        candidate_pool_size=pool_size,
        parent_lookup=parent_lookup,
        fusion_mode=retrieval_mode,
    )

    # Run RAG for all 50 questions
    logger.info("=" * 60)
    logger.info("Running retrieval + generation (%s mode)", retrieval_mode)
    logger.info("=" * 60)
    results, context_rows = _run_for_mode(
        retriever=retriever,
        llm_client=llm_client,
        questions_df=questions_df,
        fusion_mode=retrieval_mode,
        logger=logger,
    )

    # Save generated answers
    answers_df = pd.DataFrame(results)
    answers_df.to_excel(answers_path, index=False, engine="openpyxl")
    logger.info("Saved %d answers → %s", len(answers_df), answers_path)

    # Save retrieved contexts
    ctx_path = out_dir / "retrieved_contexts.xlsx"
    ctx_df = pd.DataFrame(context_rows)
    ctx_df.to_excel(ctx_path, index=False, engine="openpyxl")
    logger.info("Saved %d context rows → %s", len(ctx_df), ctx_path)

    # Save run config
    elapsed = time.time() - start_time
    num_errors = int((answers_df["error"] != "").sum()) if "error" in answers_df.columns else 0
    run_config = {
        "experiment_name": experiment_name,
        "chunking_type": chunking_mode,
        "retrieval_type": retrieval_mode,
        "chroma_collection": collection_name,
        "embedding_model": cfg["embedding"]["model"],
        "embedding_host": cfg["embedding"]["ollama_host"],
        "llm_model": llm_client.model,
        "llm_temperature": llm_defaults.get("temperature", 0.2),
        "top_k": top_k,
        "candidate_pool_size": pool_size,
        "semantic_weight": alpha,
        "keyword_weight": retrieval_cfg.get("keyword_weight", 0.4),
        "sparse_method": "BM25Okapi",
        "prompt_version": "v1",
        "num_questions": len(results),
        "num_errors": num_errors,
        "elapsed_seconds": round(elapsed, 1),
        "run_timestamp": run_ts,
        "outputs": {
            "generated_answers": str(answers_path),
            "retrieved_contexts": str(ctx_path),
        },
    }
    config_path = out_dir / "run_config.json"
    config_path.write_text(json.dumps(run_config, indent=2))
    logger.info("Saved run config → %s", config_path)

    logger.info("=" * 60)
    logger.info("Experiment complete: %s", experiment_name)
    logger.info("Elapsed: %.1f seconds", elapsed)
    logger.info("Questions: %d, Errors: %d", len(results), num_errors)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
