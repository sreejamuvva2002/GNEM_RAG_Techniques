#!/usr/bin/env python3
"""RAGAS-based evaluation of dense, sparse, and hybrid retrieval modes.

Evaluates the generated answers from three retrieval strategies using official
RAGAS metrics:
  - faithfulness: Does the answer contradict the context?
  - answer_relevancy: Is the answer relevant to the question?
  - context_precision: Are all retrieved contexts relevant?
  - context_recall: Do the retrieved contexts contain all information needed?
  - answer_correctness: Does the answer match the golden answer?
  - factual_correctness: Are facts in the answer factually correct?

Usage:
    cd simple_rag/
    export LLM_BASE_URL="http://localhost:11434/v1"
    export LLM_API_KEY="ollama"
    export LLM_MODEL="gemma3:27b"
    python scripts/evaluate_retrieval_modes_ragas.py
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

# Set up project path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import get_logger, setup_logging

# RAGAS imports
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
    factual_correctness,
)
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

setup_logging("INFO")
logger = get_logger(__name__)

# Configuration defaults
DEFAULT_GENERATED_ANSWERS_FILE = "outputs/generated_answers/generated_answers_current_complete_20260509_015013.xlsx"
DEFAULT_GOLDEN_ANSWERS_FILE = "data/Human validated 50 questions.xlsx"

# Environment variable names
ENV_GENERATED_ANSWERS = "GENERATED_ANSWERS_FILE"
ENV_RAGAS_LLM_BASE_URL = "RAGAS_LLM_BASE_URL"
ENV_RAGAS_LLM_API_KEY = "RAGAS_LLM_API_KEY"
ENV_RAGAS_LLM_MODEL = "RAGAS_LLM_MODEL"
ENV_RAGAS_EMBEDDING_BASE_URL = "RAGAS_EMBEDDING_BASE_URL"
ENV_RAGAS_EMBEDDING_API_KEY = "RAGAS_EMBEDDING_API_KEY"
ENV_RAGAS_EMBEDDING_MODEL = "RAGAS_EMBEDDING_MODEL"
ENV_LLM_BASE_URL = "LLM_BASE_URL"
ENV_LLM_API_KEY = "LLM_API_KEY"
ENV_LLM_MODEL = "LLM_MODEL"


def load_config() -> dict[str, str]:
    """Load configuration from environment variables with sensible defaults."""
    # Determine paths
    generated_answers_file = os.environ.get(ENV_GENERATED_ANSWERS)
    if not generated_answers_file:
        path = _PROJECT_ROOT / DEFAULT_GENERATED_ANSWERS_FILE
        if path.exists():
            generated_answers_file = str(path)
        else:
            # Fallback: search for most recent generated_answers_current_*.xlsx
            outputs_dir = _PROJECT_ROOT / "outputs" / "generated_answers"
            if outputs_dir.exists():
                files = sorted(outputs_dir.glob("generated_answers_current_*.xlsx"), reverse=True)
                if files:
                    generated_answers_file = str(files[0])
                    logger.info(f"Using most recent file: {files[0].name}")
            if not generated_answers_file:
                generated_answers_file = str(_PROJECT_ROOT / DEFAULT_GENERATED_ANSWERS_FILE)

    golden_answers_file = _PROJECT_ROOT / DEFAULT_GOLDEN_ANSWERS_FILE

    # LLM configuration (RAGAS evaluation LLM)
    ragas_llm_base_url = os.environ.get(
        ENV_RAGAS_LLM_BASE_URL,
        os.environ.get(ENV_LLM_BASE_URL, "http://localhost:11434/v1"),
    )
    ragas_llm_api_key = os.environ.get(ENV_RAGAS_LLM_API_KEY, os.environ.get(ENV_LLM_API_KEY, "ollama"))
    ragas_llm_model = os.environ.get(ENV_RAGAS_LLM_MODEL, os.environ.get(ENV_LLM_MODEL, "gemma3:27b"))

    # Embedding configuration
    ragas_embedding_base_url = os.environ.get(
        ENV_RAGAS_EMBEDDING_BASE_URL,
        os.environ.get(ENV_LLM_BASE_URL, "http://localhost:11434/v1"),
    )
    ragas_embedding_api_key = os.environ.get(ENV_RAGAS_EMBEDDING_API_KEY, os.environ.get(ENV_LLM_API_KEY, "ollama"))
    ragas_embedding_model = os.environ.get(ENV_RAGAS_EMBEDDING_MODEL, "nomic-embed-text")

    return {
        "generated_answers_file": generated_answers_file,
        "golden_answers_file": str(golden_answers_file),
        "ragas_llm_base_url": ragas_llm_base_url,
        "ragas_llm_api_key": ragas_llm_api_key,
        "ragas_llm_model": ragas_llm_model,
        "ragas_embedding_base_url": ragas_embedding_base_url,
        "ragas_embedding_api_key": ragas_embedding_api_key,
        "ragas_embedding_model": ragas_embedding_model,
    }


def load_golden_answers(path: str | Path) -> pd.DataFrame:
    """Load golden answers and add question_id column.

    Args:
        path: Path to Human validated 50 questions.xlsx

    Returns:
        DataFrame with columns: question_id, question, golden_answer
    """
    df = pd.read_excel(path, engine="openpyxl")
    # Align by row order: Num 1 -> Q001, etc.
    df["question_id"] = [f"Q{i+1:03d}" for i in range(len(df))]
    # Extract relevant columns
    result = pd.DataFrame({
        "question_id": df["question_id"],
        "question": df["Question"],
        "golden_answer": df["Human validated answers"],
    })
    logger.info(f"Loaded {len(result)} golden answers from {Path(path).name}")
    return result


def load_generated_answers(path: str | Path, sheet_name: str) -> pd.DataFrame:
    """Load generated answers from a specific sheet.

    Args:
        path: Path to generated answers Excel file
        sheet_name: Sheet name (RAG_Dense, RAG_Sparse, RAG_Hybrid)

    Returns:
        DataFrame with columns: question_id, question, answer, error
    """
    df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    result = pd.DataFrame({
        "question_id": df["question_id"],
        "question": df["question"],
        "answer": df["answer"],
        "error": df.get("error", ""),
    })
    logger.info(f"Loaded {len(result)} generated answers from sheet: {sheet_name}")
    return result


def load_contexts(path: str | Path, sheet_name: str) -> dict[str, list[str]]:
    """Load contexts and group by question_id.

    Args:
        path: Path to generated answers Excel file
        sheet_name: Sheet name (Context_Dense, Context_Sparse, Context_Hybrid)

    Returns:
        Dict mapping question_id -> list of context strings
    """
    df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    # Group text column by question_id
    contexts_dict = {}
    for qid, group in df.groupby("question_id"):
        contexts_dict[qid] = group["text"].tolist()
    logger.info(f"Loaded contexts from {sheet_name}: {len(contexts_dict)} questions, {len(df)} total rows")
    return contexts_dict


def build_ragas_dataset(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    golden_answers: list[str],
) -> Dataset:
    """Build a HuggingFace Dataset compatible with RAGAS.

    Args:
        questions: List of questions
        answers: List of generated answers
        contexts: List of context lists
        golden_answers: List of golden answers

    Returns:
        HuggingFace Dataset with columns: question, answer, contexts, ground_truths
    """
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truths": [[ga] for ga in golden_answers],  # RAGAS expects list of lists
    })
    logger.info(f"Built RAGAS dataset with {len(dataset)} samples")
    return dataset


def run_evaluation_for_mode(
    generated_answers: pd.DataFrame,
    contexts_dict: dict[str, list[str]],
    golden_answers: pd.DataFrame,
    ragas_llm,
    ragas_embeddings,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run RAGAS evaluation for one retrieval mode.

    Args:
        generated_answers: DataFrame with question_id, question, answer, error columns
        contexts_dict: Dict mapping question_id -> list of context strings
        golden_answers: DataFrame with question_id, question, golden_answer columns
        ragas_llm: Configured RAGAS LLM
        ragas_embeddings: Configured RAGAS embeddings

    Returns:
        Tuple of (per-question scores DataFrame, summary stats dict)
    """
    # Merge data
    merged = generated_answers.merge(golden_answers, on="question_id", how="left")
    merged = merged.merge(
        pd.DataFrame([
            {"question_id": qid, "contexts": ctx}
            for qid, ctx in contexts_dict.items()
        ]),
        on="question_id",
        how="left",
    )

    # Handle errors: rows with errors get NaN scores
    has_error = merged["error"].notna() & (merged["error"] != "")
    logger.info(f"Found {has_error.sum()} rows with errors")

    # Build RAGAS dataset (only for non-error rows)
    valid_idx = ~has_error
    questions_list = merged[valid_idx]["question"].tolist()
    answers_list = merged[valid_idx]["answer"].tolist()
    contexts_list = merged[valid_idx]["contexts"].tolist()
    golden_list = merged[valid_idx]["golden_answer"].tolist()

    dataset = build_ragas_dataset(questions_list, answers_list, contexts_list, golden_list)

    # Run evaluation
    logger.info("Running RAGAS evaluation...")
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
            factual_correctness,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,  # Return NaN instead of crashing on failures
        show_progress=True,
    )

    # Extract scores
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness", "factual_correctness"]
    scores_dict = {name: [] for name in metric_names}

    for metric_name in metric_names:
        if hasattr(result, metric_name):
            scores_dict[metric_name] = result[metric_name].tolist()
        else:
            logger.warning(f"Metric {metric_name} not found in results")

    # Build output DataFrame
    output_rows = []
    valid_count = 0
    for idx, row in merged.iterrows():
        output_row = {
            "question_id": row["question_id"],
            "question": row["question"],
            "generated_answer": row["answer"],
            "golden_answer": row["golden_answer"],
            "error": row.get("error", ""),
        }

        # Add metric scores
        if valid_idx.iloc[idx]:
            for metric_name in metric_names:
                if metric_name in scores_dict and valid_count < len(scores_dict[metric_name]):
                    score = scores_dict[metric_name][valid_count]
                    output_row[metric_name] = score if not np.isnan(score) else None
                else:
                    output_row[metric_name] = None
            valid_count += 1
        else:
            for metric_name in metric_names:
                output_row[metric_name] = None

        output_rows.append(output_row)

    output_df = pd.DataFrame(output_rows)

    # Compute summary stats
    summary_stats = {}
    for metric_name in metric_names:
        scores = output_df[metric_name].dropna()
        if len(scores) > 0:
            summary_stats[metric_name] = float(scores.mean())
        else:
            summary_stats[metric_name] = None
    summary_stats["num_questions"] = len(output_df)
    summary_stats["num_errors"] = has_error.sum()

    return output_df, summary_stats


def main() -> None:
    """Main evaluation pipeline."""
    logger.info("=" * 80)
    logger.info("RAGAS Evaluation Pipeline")
    logger.info("=" * 80)
    start_time = time.time()

    # Load config
    config = load_config()
    logger.info(f"Generated answers file: {config['generated_answers_file']}")
    logger.info(f"Golden answers file: {config['golden_answers_file']}")
    logger.info(f"RAGAS LLM: {config['ragas_llm_model']} at {config['ragas_llm_base_url']}")
    logger.info(f"RAGAS Embeddings: {config['ragas_embedding_model']}")

    # Load golden answers
    golden_answers_df = load_golden_answers(config["golden_answers_file"])

    # Setup RAGAS LLM and embeddings
    logger.info("Setting up RAGAS LLM and embeddings...")
    ragas_llm = LangchainLLMWrapper(
        ChatOllama(
            base_url=config["ragas_llm_base_url"].replace("/v1", ""),
            model=config["ragas_llm_model"],
            temperature=0.2,
        )
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(
            base_url=config["ragas_embedding_base_url"].replace("/v1", ""),
            model=config["ragas_embedding_model"],
        )
    )

    # Evaluate each mode
    modes = ["dense", "sparse", "hybrid"]
    all_results_per_mode = {}
    all_summaries = {}

    for mode in modes:
        logger.info("=" * 80)
        logger.info(f"Evaluating {mode.upper()} retrieval")
        logger.info("=" * 80)

        sheet_name_rag = f"RAG_{mode.capitalize()}"
        sheet_name_ctx = f"Context_{mode.capitalize()}"

        # Load generated answers and contexts
        generated_answers_df = load_generated_answers(config["generated_answers_file"], sheet_name_rag)
        contexts_dict = load_contexts(config["generated_answers_file"], sheet_name_ctx)

        # Run evaluation
        results_df, summary_stats = run_evaluation_for_mode(
            generated_answers_df,
            contexts_dict,
            golden_answers_df,
            ragas_llm,
            ragas_embeddings,
        )

        all_results_per_mode[mode] = results_df
        all_summaries[mode] = summary_stats

        logger.info(f"{mode.upper()} evaluation complete")
        logger.info(f"  Faithfulness: {summary_stats.get('faithfulness', 'N/A'):.3f}" if summary_stats.get('faithfulness') else f"  Faithfulness: N/A")
        logger.info(f"  Answer Relevancy: {summary_stats.get('answer_relevancy', 'N/A'):.3f}" if summary_stats.get('answer_relevancy') else f"  Answer Relevancy: N/A")
        logger.info(f"  Context Precision: {summary_stats.get('context_precision', 'N/A'):.3f}" if summary_stats.get('context_precision') else f"  Context Precision: N/A")
        logger.info(f"  Context Recall: {summary_stats.get('context_recall', 'N/A'):.3f}" if summary_stats.get('context_recall') else f"  Context Recall: N/A")
        logger.info(f"  Answer Correctness: {summary_stats.get('answer_correctness', 'N/A'):.3f}" if summary_stats.get('answer_correctness') else f"  Answer Correctness: N/A")
        logger.info(f"  Factual Correctness: {summary_stats.get('factual_correctness', 'N/A'):.3f}" if summary_stats.get('factual_correctness') else f"  Factual Correctness: N/A")

    # Save per-question scores
    output_per_question_path = _PROJECT_ROOT / "outputs" / "evaluation" / "ragas_per_question_scores.xlsx"
    output_per_question_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_per_question_path, engine="openpyxl") as writer:
        for mode, results_df in all_results_per_mode.items():
            results_df.to_excel(writer, sheet_name=mode, index=False)

    logger.info(f"Saved per-question scores to {output_per_question_path}")

    # Save summary
    output_summary_path = _PROJECT_ROOT / "outputs" / "evaluation" / "ragas_summary.xlsx"
    summary_rows = []
    for mode in modes:
        stats = all_summaries[mode]
        summary_rows.append({
            "retrieval_mode": mode,
            "num_questions": stats.get("num_questions"),
            "num_errors": stats.get("num_errors"),
            "mean_faithfulness": stats.get("faithfulness"),
            "mean_answer_relevancy": stats.get("answer_relevancy"),
            "mean_context_precision": stats.get("context_precision"),
            "mean_context_recall": stats.get("context_recall"),
            "mean_answer_correctness": stats.get("answer_correctness"),
            "mean_factual_correctness": stats.get("factual_correctness"),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_excel(output_summary_path, index=False, engine="openpyxl")
    logger.info(f"Saved summary to {output_summary_path}")

    # Save markdown report
    output_md_path = _PROJECT_ROOT / "outputs" / "evaluation" / "ragas_summary.md"
    with open(output_md_path, "w") as f:
        f.write("# RAGAS Evaluation Summary\n\n")
        f.write(f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Generated Answers File:** {config['generated_answers_file']}\n")
        f.write(f"**RAGAS LLM:** {config['ragas_llm_model']}\n")
        f.write(f"**RAGAS Embeddings:** {config['ragas_embedding_model']}\n\n")

        # Overall comparison
        f.write("## Overall Comparison\n\n")
        f.write("| Retrieval Mode | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Answer Correctness | Factual Correctness |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for mode in modes:
            stats = all_summaries[mode]
            f.write(f"| {mode} |")
            f.write(f" {stats.get('faithfulness', 'N/A'):.3f} |" if stats.get('faithfulness') else " N/A |")
            f.write(f" {stats.get('answer_relevancy', 'N/A'):.3f} |" if stats.get('answer_relevancy') else " N/A |")
            f.write(f" {stats.get('context_precision', 'N/A'):.3f} |" if stats.get('context_precision') else " N/A |")
            f.write(f" {stats.get('context_recall', 'N/A'):.3f} |" if stats.get('context_recall') else " N/A |")
            f.write(f" {stats.get('answer_correctness', 'N/A'):.3f} |" if stats.get('answer_correctness') else " N/A |")
            f.write(f" {stats.get('factual_correctness', 'N/A'):.3f} |\n" if stats.get('factual_correctness') else " N/A |\n")

        # Detailed analysis
        f.write("\n## Detailed Analysis\n\n")

        # Best performing metrics
        f.write("### Best Performing Modes by Metric\n\n")
        metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness", "factual_correctness"]
        for metric in metrics:
            scores = [(mode, all_summaries[mode].get(metric)) for mode in modes]
            scores = [(m, s) for m, s in scores if s is not None]
            if scores:
                best_mode, best_score = max(scores, key=lambda x: x[1])
                f.write(f"- **{metric}:** {best_mode} ({best_score:.3f})\n")

        # Error rates
        f.write("\n### Error Rates\n\n")
        for mode in modes:
            stats = all_summaries[mode]
            num_errors = stats.get("num_errors", 0)
            num_total = stats.get("num_questions", 50)
            error_rate = (num_errors / num_total * 100) if num_total > 0 else 0
            f.write(f"- **{mode}:** {num_errors}/{num_total} errors ({error_rate:.1f}%)\n")

        # Key findings
        f.write("\n## Key Findings\n\n")
        f.write("1. Compare the best-performing retrieval modes for each metric.\n")
        f.write("2. Identify which mode achieves the best overall balance across metrics.\n")
        f.write("3. Note any modes with significantly higher error rates.\n")
        f.write("4. Consider the trade-offs between dense (semantic) and sparse (keyword) retrieval.\n\n")

        # Limitations
        f.write("## Limitations of RAGAS/LLM-as-Judge Evaluation\n\n")
        f.write("- **LLM Variability:** Judge LLM (gemma3:27b) may not perfectly align with human judgment.\n")
        f.write("- **Hallucination:** The judge LLM may hallucinate facts when evaluating faithfulness.\n")
        f.write("- **Context Length:** Large context windows (top_k=80) may reduce LLM discrimination ability.\n")
        f.write("- **Embedding Model:** nomic-embed-text may not capture all semantic nuances.\n")
        f.write("- **Golden Answers:** Human-validated answers are a single reference; multiple valid answers may exist.\n")
        f.write("- **Score Interpretation:** RAGAS scores are normalized 0-1 but interpretation may vary by metric.\n\n")

        # Files generated
        f.write("## Output Files\n\n")
        f.write("- `ragas_per_question_scores.xlsx` — Per-question metric scores (sheets: dense, sparse, hybrid)\n")
        f.write("- `ragas_summary.xlsx` — Aggregate scores by retrieval mode\n")
        f.write("- `ragas_summary.md` — This markdown report\n")

    logger.info(f"Saved markdown report to {output_md_path}")

    # Elapsed time
    elapsed = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"Evaluation complete in {elapsed:.1f} seconds")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
