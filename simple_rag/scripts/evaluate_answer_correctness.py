#!/usr/bin/env python3
"""Human-gold answer comparison evaluation.

Computes multiple answer correctness metrics:
  - exact_match: Exact string match (case-sensitive)
  - normalized_exact_match: Lowercase + punctuation normalization
  - semantic_similarity: Cosine similarity via nomic-embed-text embeddings
  - judge_correctness_score: LLM judge rating (0.0-1.0), if configured
  - missing_answer_flag: Whether answer is empty or unavailable

Usage:
    cd simple_rag/
    export LLM_BASE_URL="http://localhost:11434/v1"
    export LLM_API_KEY="ollama"
    export LLM_MODEL="gemma3:27b"
    export JUDGE_MODEL="gemma3:27b"  # Optional, for LLM-based correctness judge
    python scripts/evaluate_answer_correctness.py
"""

from __future__ import annotations

import os
import re
import sys
import time
import string
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

# Set up project path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import get_logger, setup_logging
from src.llm import LLMClient

setup_logging("INFO")
logger = get_logger(__name__)

# Configuration defaults
DEFAULT_GENERATED_ANSWERS_FILE = "outputs/generated_answers/generated_answers_current_complete_20260509_015013.xlsx"
DEFAULT_GOLDEN_ANSWERS_FILE = "data/Human validated 50 questions.xlsx"

# Environment variable names
ENV_GENERATED_ANSWERS = "GENERATED_ANSWERS_FILE"
ENV_EMBEDDING_BASE_URL = "EMBEDDING_BASE_URL"
ENV_EMBEDDING_MODEL = "EMBEDDING_MODEL"
ENV_JUDGE_BASE_URL = "JUDGE_BASE_URL"
ENV_JUDGE_API_KEY = "JUDGE_API_KEY"
ENV_JUDGE_MODEL = "JUDGE_MODEL"
ENV_LLM_BASE_URL = "LLM_BASE_URL"
ENV_LLM_API_KEY = "LLM_API_KEY"
ENV_LLM_MODEL = "LLM_MODEL"


def load_config() -> dict[str, str | bool]:
    """Load configuration from environment variables."""
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

    # Embedding configuration
    embedding_base_url = os.environ.get(ENV_EMBEDDING_BASE_URL, os.environ.get(ENV_LLM_BASE_URL, "http://localhost:11434/v1"))
    embedding_model = os.environ.get(ENV_EMBEDDING_MODEL, "nomic-embed-text")

    # Judge configuration (optional)
    judge_base_url = os.environ.get(ENV_JUDGE_BASE_URL, os.environ.get(ENV_LLM_BASE_URL, "http://localhost:11434/v1"))
    judge_api_key = os.environ.get(ENV_JUDGE_API_KEY, os.environ.get(ENV_LLM_API_KEY, "ollama"))
    judge_model = os.environ.get(ENV_JUDGE_MODEL)
    judge_configured = judge_model is not None

    return {
        "generated_answers_file": generated_answers_file,
        "golden_answers_file": str(golden_answers_file),
        "embedding_base_url": embedding_base_url,
        "embedding_model": embedding_model,
        "judge_base_url": judge_base_url,
        "judge_api_key": judge_api_key,
        "judge_model": judge_model,
        "judge_configured": judge_configured,
    }


def load_golden_answers(path: str | Path) -> pd.DataFrame:
    """Load golden answers and add question_id column."""
    df = pd.read_excel(path, engine="openpyxl")
    df["question_id"] = [f"Q{i+1:03d}" for i in range(len(df))]
    result = pd.DataFrame({
        "question_id": df["question_id"],
        "question": df["Question"],
        "golden_answer": df["Human validated answers"],
    })
    logger.info(f"Loaded {len(result)} golden answers from {Path(path).name}")
    return result


def load_generated_answers(path: str | Path, sheet_name: str) -> pd.DataFrame:
    """Load generated answers from a specific sheet."""
    df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    result = pd.DataFrame({
        "question_id": df["question_id"],
        "question": df["question"],
        "answer": df["answer"],
        "error": df.get("error", ""),
    })
    logger.info(f"Loaded {len(result)} generated answers from sheet: {sheet_name}")
    return result


def get_embedding(text: str, base_url: str, model: str) -> np.ndarray | None:
    """Get embedding for text via Ollama API.

    Args:
        text: Text to embed
        base_url: Ollama base URL (e.g., http://localhost:11434/v1)
        model: Model name (e.g., nomic-embed-text)

    Returns:
        Embedding as numpy array, or None if embedding fails
    """
    try:
        # Use Ollama REST API directly
        endpoint = f"{base_url.rstrip('/v1').rstrip('/')}/api/embeddings"
        response = requests.post(
            endpoint,
            json={"model": model, "prompt": text},
            timeout=30,
        )
        response.raise_for_status()
        embedding = response.json().get("embedding")
        if embedding:
            return np.array(embedding, dtype=np.float32)
    except Exception as e:
        logger.warning(f"Failed to get embedding for text: {e}")
    return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_exact_match(answer: str, golden: str) -> bool:
    """Exact string match (case-sensitive)."""
    return answer.strip() == golden.strip()


def compute_normalized_exact_match(answer: str, golden: str) -> bool:
    """Normalized exact match: lowercase + remove punctuation + collapse whitespace."""
    def normalize(text: str) -> str:
        text = text.lower().strip()
        # Remove punctuation except spaces and hyphens within words
        text = re.sub(r"[^\w\s-]", "", text)
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)
        return text

    return normalize(answer) == normalize(golden)


def compute_semantic_similarity(answer: str, golden: str, embedding_base_url: str, embedding_model: str) -> float | None:
    """Compute semantic similarity via embeddings."""
    if not answer.strip() or not golden.strip():
        return None

    answer_emb = get_embedding(answer, embedding_base_url, embedding_model)
    golden_emb = get_embedding(golden, embedding_base_url, embedding_model)

    if answer_emb is None or golden_emb is None:
        return None

    return cosine_similarity(answer_emb, golden_emb)


def compute_judge_correctness(
    question: str,
    answer: str,
    golden: str,
    judge_base_url: str,
    judge_api_key: str,
    judge_model: str,
) -> float | None:
    """Use LLM judge to rate answer correctness (0.0-1.0).

    Args:
        question: The question
        answer: Generated answer
        golden: Golden answer
        judge_base_url: Judge LLM base URL
        judge_api_key: Judge LLM API key
        judge_model: Judge LLM model name

    Returns:
        Score 0.0-1.0, or None if judge fails
    """
    try:
        client = LLMClient(
            base_url=judge_base_url,
            api_key=judge_api_key,
            model=judge_model,
            temperature=0.0,  # Deterministic
        )

        prompt = f"""Rate how well the following generated answer matches the golden answer.

Question: {question}

Golden Answer: {golden}

Generated Answer: {answer}

Rate on a scale of 0.0 to 1.0 where:
- 0.0 = completely wrong or unrelated
- 0.5 = partially correct
- 1.0 = fully correct and complete

Respond with ONLY a number between 0.0 and 1.0."""

        response = client.chat([{"role": "user", "content": prompt}])

        # Extract float from response
        match = re.search(r"\b\d+\.\d+\b|\b\d+\b", response.strip())
        if match:
            score = float(match.group())
            return min(1.0, max(0.0, score))  # Clamp to [0, 1]
    except Exception as e:
        logger.debug(f"Judge failed for question: {e}")

    return None


def is_missing_answer(answer: str) -> bool:
    """Check if answer is missing or indicates unavailability."""
    answer_lower = answer.lower().strip()
    missing_patterns = [
        "not available",
        "insufficient",
        "not found",
        "no information",
        "cannot answer",
        "no context",
    ]
    return any(pattern in answer_lower for pattern in missing_patterns) or answer.strip() == ""


def evaluate_mode(
    generated_answers: pd.DataFrame,
    golden_answers: pd.DataFrame,
    embedding_base_url: str,
    embedding_model: str,
    judge_configured: bool,
    judge_base_url: str,
    judge_api_key: str,
    judge_model: str,
) -> pd.DataFrame:
    """Evaluate all answers in a retrieval mode."""
    merged = generated_answers.merge(golden_answers, on="question_id", how="left")

    results = []
    for idx, row in merged.iterrows():
        answer = str(row["answer"]) if pd.notna(row["answer"]) else ""
        golden = str(row["golden_answer"]) if pd.notna(row["golden_answer"]) else ""

        result_row = {
            "question_id": row["question_id"],
            "question_x": row["question_x"],  # from generated_answers
            "generated_answer": answer,
            "golden_answer": golden,
            "error": row.get("error", ""),
            "exact_match": compute_exact_match(answer, golden),
            "normalized_exact_match": compute_normalized_exact_match(answer, golden),
            "semantic_similarity": compute_semantic_similarity(answer, golden, embedding_base_url, embedding_model),
            "judge_correctness_score": None,
            "missing_answer_flag": is_missing_answer(answer),
        }

        # Compute judge score if configured
        if judge_configured:
            result_row["judge_correctness_score"] = compute_judge_correctness(
                row["question_x"], answer, golden, judge_base_url, judge_api_key, judge_model
            )

        results.append(result_row)

    return pd.DataFrame(results)


def main() -> None:
    """Main evaluation pipeline."""
    logger.info("=" * 80)
    logger.info("Answer Correctness Evaluation")
    logger.info("=" * 80)
    start_time = time.time()

    # Load config
    config = load_config()
    logger.info(f"Generated answers file: {config['generated_answers_file']}")
    logger.info(f"Embedding model: {config['embedding_model']}")
    if config["judge_configured"]:
        logger.info(f"Judge model: {config['judge_model']}")
    else:
        logger.info("Judge model: NOT CONFIGURED (judge_correctness_score will be N/A)")

    # Load golden answers
    golden_answers_df = load_golden_answers(config["golden_answers_file"])

    # Evaluate each mode
    modes = ["dense", "sparse", "hybrid"]
    all_results_per_mode = {}
    all_summaries = {}

    for mode in modes:
        logger.info("=" * 80)
        logger.info(f"Evaluating {mode.upper()} retrieval")
        logger.info("=" * 80)

        sheet_name_rag = f"RAG_{mode.capitalize()}"
        generated_answers_df = load_generated_answers(config["generated_answers_file"], sheet_name_rag)

        # Evaluate
        results_df = evaluate_mode(
            generated_answers_df,
            golden_answers_df,
            config["embedding_base_url"],
            config["embedding_model"],
            config["judge_configured"],
            config["judge_base_url"],
            config["judge_api_key"],
            config["judge_model"],
        )

        all_results_per_mode[mode] = results_df

        # Summary stats
        exact_matches = results_df["exact_match"].sum()
        normalized_matches = results_df["normalized_exact_match"].sum()
        semantic_sims = results_df["semantic_similarity"].dropna()
        judge_scores = results_df["judge_correctness_score"].dropna()
        missing_answers = results_df["missing_answer_flag"].sum()

        summary = {
            "mode": mode,
            "total_questions": len(results_df),
            "exact_match_count": exact_matches,
            "exact_match_pct": exact_matches / len(results_df) * 100,
            "normalized_match_count": normalized_matches,
            "normalized_match_pct": normalized_matches / len(results_df) * 100,
            "semantic_similarity_mean": semantic_sims.mean() if len(semantic_sims) > 0 else None,
            "semantic_similarity_std": semantic_sims.std() if len(semantic_sims) > 1 else None,
            "judge_score_mean": judge_scores.mean() if len(judge_scores) > 0 else None,
            "missing_answer_count": missing_answers,
            "missing_answer_pct": missing_answers / len(results_df) * 100,
        }

        all_summaries[mode] = summary

        logger.info(f"{mode.upper()} evaluation complete")
        logger.info(f"  Exact match: {summary['exact_match_count']}/{summary['total_questions']} ({summary['exact_match_pct']:.1f}%)")
        logger.info(f"  Normalized match: {summary['normalized_match_count']}/{summary['total_questions']} ({summary['normalized_match_pct']:.1f}%)")
        if summary["semantic_similarity_mean"] is not None:
            logger.info(f"  Semantic similarity: {summary['semantic_similarity_mean']:.3f} ± {summary['semantic_similarity_std']:.3f}")
        if summary["judge_score_mean"] is not None:
            logger.info(f"  Judge score: {summary['judge_score_mean']:.3f}")
        logger.info(f"  Missing answers: {summary['missing_answer_count']}/{summary['total_questions']} ({summary['missing_answer_pct']:.1f}%)")

    # Save per-question results
    output_dir = _PROJECT_ROOT / "outputs" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "answer_correctness_50q.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for mode, results_df in all_results_per_mode.items():
            results_df.to_excel(writer, sheet_name=mode, index=False)

        # Summary sheet
        summary_rows = []
        for mode in modes:
            summary = all_summaries[mode]
            summary_rows.append(summary)
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    logger.info(f"Saved answer correctness results to {output_path}")

    # Elapsed time
    elapsed = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"Evaluation complete in {elapsed:.1f} seconds")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
