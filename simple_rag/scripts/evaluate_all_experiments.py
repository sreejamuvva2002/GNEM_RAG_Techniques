#!/usr/bin/env python3
"""Unified evaluation for all six RAG experiments.

Evaluates experiments found in outputs/experiments/ using:
  - Official RAGAS metrics (faithfulness, answer_relevancy, context_precision,
    context_recall, answer_correctness, factual_correctness)
  - Non-RAGAS correctness metrics (exact_match, normalized_exact_match,
    semantic_similarity, judge_correctness, missing_answer_flag)

Usage:
    cd simple_rag/
    python scripts/evaluate_all_experiments.py

Required environment variables:
    LLM_BASE_URL             Ollama/OpenAI base URL for RAGAS judge
    LLM_MODEL                Model for RAGAS evaluation
    RAGAS_LLM_BASE_URL       Override LLM base URL for RAGAS (optional)
    RAGAS_LLM_API_KEY        Override API key for RAGAS (optional)
    RAGAS_LLM_MODEL          Override model name for RAGAS (optional)
    RAGAS_EMBEDDING_BASE_URL Ollama base URL for embeddings (default: http://localhost:11434/v1)
    RAGAS_EMBEDDING_API_KEY  API key for embeddings (default: ollama)
    RAGAS_EMBEDDING_MODEL    Embedding model (default: nomic-embed-text)

Optional environment variables:
    JUDGE_BASE_URL           LLM judge base URL (for judge_correctness_score)
    JUDGE_API_KEY            LLM judge API key
    JUDGE_MODEL              LLM judge model

Outputs:
    outputs/evaluation/ragas_per_question_scores.xlsx  — per-question RAGAS scores
    outputs/evaluation/ragas_summary.xlsx              — aggregate RAGAS means
    outputs/evaluation/answer_correctness_50q.xlsx     — per-question correctness metrics
    outputs/evaluation/final_experiment_comparison.xlsx — full comparison table
    outputs/evaluation/final_experiment_comparison.md  — narrative report
"""

from __future__ import annotations

import os
import re
import sys
import json
import string
import logging
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import get_logger, setup_logging

logger = get_logger("evaluate_all")


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

EXPERIMENTS_DIR = _PROJECT_ROOT / "outputs" / "experiments"
GOLDEN_ANSWERS_FILE = _PROJECT_ROOT / "data" / "Human validated 50 questions.xlsx"
EVAL_DIR = _PROJECT_ROOT / "outputs" / "evaluation"

EXPERIMENT_ORDER = [
    "normal_dense",
    "normal_sparse",
    "normal_hybrid",
    "parent_child_dense",
    "parent_child_sparse",
    "parent_child_hybrid",
]

RAGAS_METRIC_COLS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]

CORRECTNESS_METRIC_COLS = [
    "exact_match",
    "normalized_exact_match",
    "semantic_similarity",
    "judge_correctness_score",
    "missing_answer_flag",
]


# ---------------------------------------------------------------------------
# Environment / configuration
# ---------------------------------------------------------------------------

def _get_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _ragas_llm_config() -> dict[str, str]:
    base_url = (
        _get_env("RAGAS_LLM_BASE_URL")
        or _get_env("LLM_BASE_URL", "http://localhost:11434/v1")
    )
    api_key = _get_env("RAGAS_LLM_API_KEY") or _get_env("LLM_API_KEY", "ollama")
    # RAGAS judge is intentionally separate from the generation model (LLM_MODEL).
    # Override with RAGAS_LLM_MODEL env var if needed.
    model = _get_env("RAGAS_LLM_MODEL", "qwen2.5:14b")
    return {"base_url": base_url, "api_key": api_key, "model": model}


def _ragas_embed_config() -> dict[str, str]:
    return {
        "base_url": _get_env("RAGAS_EMBEDDING_BASE_URL", "http://localhost:11434/v1"),
        "api_key": _get_env("RAGAS_EMBEDDING_API_KEY", "ollama"),
        "model": _get_env("RAGAS_EMBEDDING_MODEL", "nomic-embed-text"),
    }


def _judge_config() -> dict[str, str] | None:
    base_url = _get_env("JUDGE_BASE_URL")
    api_key = _get_env("JUDGE_API_KEY")
    model = _get_env("JUDGE_MODEL")
    if not model:
        return None
    return {
        "base_url": base_url or _get_env("LLM_BASE_URL", "http://localhost:11434/v1"),
        "api_key": api_key or _get_env("LLM_API_KEY", "ollama"),
        "model": model,
    }


# ---------------------------------------------------------------------------
# Golden answer loading
# ---------------------------------------------------------------------------

def _load_golden_answers() -> dict[str, str]:
    """Load golden answers and align to Q001-Q050 by row order."""
    if not GOLDEN_ANSWERS_FILE.exists():
        raise FileNotFoundError(f"Golden answers file not found: {GOLDEN_ANSWERS_FILE}")
    df = pd.read_excel(GOLDEN_ANSWERS_FILE)
    logger.info("Golden answers columns: %s", list(df.columns))
    # Find the answer column (flexible matching)
    ans_col = None
    for col in df.columns:
        if "validated" in str(col).lower() or "answer" in str(col).lower():
            ans_col = col
            break
    if ans_col is None:
        raise ValueError(f"Cannot find golden answer column in {list(df.columns)}")
    golden = {}
    for i, row in df.iterrows():
        qid = f"Q{i + 1:03d}"
        val = str(row[ans_col]).strip()
        golden[qid] = val if val != "nan" else ""
    logger.info("Loaded %d golden answers (col=%r)", len(golden), ans_col)
    return golden


# ---------------------------------------------------------------------------
# Experiment data loading
# ---------------------------------------------------------------------------

def _load_experiment(experiment_name: str) -> dict[str, Any] | None:
    """Load generated answers and retrieved contexts for one experiment."""
    exp_dir = EXPERIMENTS_DIR / experiment_name
    answers_file = exp_dir / "generated_answers.xlsx"
    contexts_file = exp_dir / "retrieved_contexts.xlsx"

    if not answers_file.exists():
        logger.warning("[%s] No generated_answers.xlsx — skipping", experiment_name)
        return None
    if not contexts_file.exists():
        logger.warning("[%s] No retrieved_contexts.xlsx — skipping", experiment_name)
        return None

    answers_df = pd.read_excel(answers_file)
    ctx_df = pd.read_excel(contexts_file)

    # Build contexts dict: question_id → list[str]
    text_col = "text"
    if text_col not in ctx_df.columns:
        logger.warning("[%s] No 'text' column in contexts file", experiment_name)
        text_col = None

    contexts_by_qid: dict[str, list[str]] = {}
    if text_col:
        for qid, grp in ctx_df.groupby("question_id"):
            texts = [t for t in grp[text_col].tolist() if isinstance(t, str) and t.strip()]
            contexts_by_qid[str(qid)] = texts

    # Parse chunking/retrieval type from name
    parts = experiment_name.split("_")
    if len(parts) >= 2:
        retrieval_type = parts[-1]
        chunking_type = "_".join(parts[:-1])
    else:
        chunking_type = experiment_name
        retrieval_type = "unknown"

    # Load run_config if available
    config_path = exp_dir / "run_config.json"
    run_config = {}
    if config_path.exists():
        run_config = json.loads(config_path.read_text())

    return {
        "name": experiment_name,
        "chunking_type": run_config.get("chunking_type", chunking_type),
        "retrieval_type": run_config.get("retrieval_type", retrieval_type),
        "answers_df": answers_df,
        "contexts_by_qid": contexts_by_qid,
        "run_config": run_config,
    }


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

def _build_ragas_evaluator():
    """Build RAGAS metrics using ragas.metrics singletons (the only type accepted by evaluate()).

    ragas.metrics.collections classes use SimpleBaseMetric and are NOT accepted by
    ragas.evaluate() which requires isinstance(m, Metric). The singleton objects in
    ragas.metrics._* do pass that check and are the correct API to use.
    """
    try:
        from langchain_community.chat_models import ChatOllama
        from langchain_community.embeddings import OllamaEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        )
    except ImportError as e:
        raise ImportError(
            f"RAGAS or langchain_community not installed: {e}\n"
            "Install with: pip install ragas langchain-community"
        ) from e

    llm_cfg = _ragas_llm_config()
    embed_cfg = _ragas_embed_config()

    llm = LangchainLLMWrapper(ChatOllama(
        base_url=llm_cfg["base_url"].replace("/v1", ""),
        model=llm_cfg["model"],
        temperature=0,
    ))
    embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(
        base_url=embed_cfg["base_url"].replace("/v1", ""),
        model=embed_cfg["model"],
    ))

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall,
               answer_correctness]
    for m in metrics:
        m.llm = llm
        if hasattr(m, "embeddings"):
            m.embeddings = embeddings

    logger.info(
        "RAGAS configured: llm=%s, embedding=%s",
        llm_cfg["model"],
        embed_cfg["model"],
    )
    return metrics


def _run_ragas_for_experiment(
    exp: dict[str, Any],
    golden: dict[str, str],
    metrics: list,
) -> pd.DataFrame:
    """Run RAGAS metrics for one experiment. Returns per-question DataFrame."""
    try:
        from ragas import evaluate, EvaluationDataset, SingleTurnSample
    except ImportError as e:
        raise ImportError(f"RAGAS not available: {e}") from e

    answers_df = exp["answers_df"]
    contexts_by_qid = exp["contexts_by_qid"]
    name = exp["name"]

    questions_col, answer_col = _detect_answer_columns(answers_df)

    meta_rows: list[dict[str, Any]] = []
    samples: list = []

    for _, row in answers_df.iterrows():
        qid = str(row.get("question_id", "")).strip()
        question = str(row.get(questions_col, "")).strip()
        answer = str(row.get(answer_col, "")).strip()
        golden_ans = golden.get(qid, "")
        contexts = contexts_by_qid.get(qid, [])

        meta_rows.append({
            "question_id": qid,
            "question": question,
            "generated_answer": answer,
            "golden_answer": golden_ans,
        })
        # Use SingleTurnSample so field names (user_input, response, retrieved_contexts,
        # reference) are explicit — avoids ground_truths→reference mapping failures
        # when RAGAS converts old Dataset format internally.
        samples.append(SingleTurnSample(
            user_input=question,
            retrieved_contexts=contexts if contexts else ["No context retrieved."],
            response=answer,
            reference=golden_ans if golden_ans else "",
        ))

    if not samples:
        logger.warning("[%s] No rows to evaluate", name)
        return pd.DataFrame()

    dataset = EvaluationDataset(samples=samples)
    meta_df = pd.DataFrame(meta_rows)

    logger.info("[%s] Running RAGAS on %d samples", name, len(samples))
    try:
        from ragas import RunConfig
        run_cfg = RunConfig(timeout=300, max_retries=2, max_workers=1)
        result = evaluate(dataset, metrics=metrics, raise_exceptions=False, run_config=run_cfg)
        scores_df = result.to_pandas()
    except Exception as exc:
        logger.error("[%s] RAGAS evaluation failed: %s", name, exc)
        scores_df = pd.DataFrame()

    if scores_df.empty:
        for col in RAGAS_METRIC_COLS:
            meta_df[col] = "error"
        meta_df["error"] = "RAGAS evaluation failed"
        return meta_df

    if len(scores_df) != len(meta_df):
        logger.error(
            "[%s] RAGAS row count mismatch: scores=%d meta=%d — filling NaN",
            name, len(scores_df), len(meta_df),
        )
        for col in RAGAS_METRIC_COLS:
            meta_df[col] = float("nan")
        meta_df["error"] = "row_count_mismatch"
        return meta_df

    for col in RAGAS_METRIC_COLS:
        if col in scores_df.columns:
            meta_df[col] = scores_df[col].values
        else:
            meta_df[col] = "not_available"

    meta_df["error"] = ""
    return meta_df


# ---------------------------------------------------------------------------
# Non-RAGAS correctness metrics
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_missing_answer(answer: str) -> bool:
    patterns = [
        r"not available in the provided context",
        r"not available",
        r"no information",
        r"cannot be answered",
        r"not mentioned",
        r"^$",
    ]
    ans_lower = answer.strip().lower()
    return any(re.search(p, ans_lower) for p in patterns)


def _embed_text_ollama(text: str, ollama_host: str, model: str) -> np.ndarray | None:
    """Embed a single text via Ollama embeddings API."""
    try:
        resp = requests.post(
            f"{ollama_host.rstrip('/')}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=120,
        )
        resp.raise_for_status()
        return np.array(resp.json()["embedding"], dtype=np.float64)
    except Exception as exc:
        logger.warning("Embedding failed for text snippet: %s", exc)
        return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _call_llm_judge(
    question: str, generated: str, golden: str, judge_cfg: dict[str, str]
) -> float | str:
    prompt = (
        "You are an evaluation judge. Score how well the Generated Answer matches "
        "the Golden Answer for the given Question.\n\n"
        f"Question: {question}\n\n"
        f"Golden Answer: {golden}\n\n"
        f"Generated Answer: {generated}\n\n"
        "Score from 0.0 (completely wrong) to 1.0 (fully correct). "
        "Respond with only a number like 0.7 or 1.0."
    )
    try:
        resp = requests.post(
            f"{judge_cfg['base_url'].rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {judge_cfg['api_key']}"},
            json={
                "model": judge_cfg["model"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 10,
            },
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        match = re.search(r"(\d+\.?\d*)", content)
        if match:
            return float(match.group(1))
        return "parse_error"
    except Exception as exc:
        logger.warning("Judge call failed: %s", exc)
        return "judge_error"


def _run_correctness_for_experiment(
    exp: dict[str, Any],
    golden: dict[str, str],
    embed_host: str,
    embed_model: str,
    judge_cfg: dict[str, str] | None,
) -> pd.DataFrame:
    """Compute non-RAGAS correctness metrics for one experiment."""
    answers_df = exp["answers_df"]
    name = exp["name"]

    questions_col, answer_col = _detect_answer_columns(answers_df)

    rows: list[dict[str, Any]] = []
    for _, row in answers_df.iterrows():
        qid = str(row.get("question_id", "")).strip()
        question = str(row.get(questions_col, "")).strip()
        generated = str(row.get(answer_col, "")).strip()
        golden_ans = golden.get(qid, "")

        exact = generated.strip() == golden_ans.strip()
        norm_exact = _normalize_text(generated) == _normalize_text(golden_ans)
        missing = _is_missing_answer(generated)

        # Semantic similarity via embeddings
        sem_sim: float | str = "not_computed"
        if generated and golden_ans:
            gen_emb = _embed_text_ollama(generated, embed_host, embed_model)
            gold_emb = _embed_text_ollama(golden_ans, embed_host, embed_model)
            if gen_emb is not None and gold_emb is not None:
                sem_sim = round(_cosine_similarity(gen_emb, gold_emb), 4)
            else:
                sem_sim = "embedding_error"

        # Judge correctness
        if judge_cfg is None:
            judge_score: float | str = "not_configured"
        else:
            judge_score = _call_llm_judge(question, generated, golden_ans, judge_cfg)

        rows.append({
            "question_id": qid,
            "question": question,
            "generated_answer": generated,
            "golden_answer": golden_ans,
            "exact_match": exact,
            "normalized_exact_match": norm_exact,
            "semantic_similarity": sem_sim,
            "judge_correctness_score": judge_score,
            "missing_answer_flag": missing,
        })

    logger.info("[%s] Correctness metrics computed for %d questions", name, len(rows))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Column detection helpers
# ---------------------------------------------------------------------------

def _detect_answer_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Return (question_col, answer_col) names from a DataFrame."""
    q_col = "question"
    a_col = "answer"
    for col in df.columns:
        if col.lower() == "question":
            q_col = col
        if col.lower() in ("answer", "generated_answer"):
            a_col = col
    return q_col, a_col


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _build_summary_row(
    exp_name: str,
    chunking_type: str,
    retrieval_type: str,
    ragas_df: pd.DataFrame,
    num_questions: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "experiment_name": exp_name,
        "chunking_type": chunking_type,
        "retrieval_type": retrieval_type,
        "num_questions": num_questions,
    }
    for col in RAGAS_METRIC_COLS:
        mean_col = f"mean_{col}"
        if not ragas_df.empty and col in ragas_df.columns:
            numeric = pd.to_numeric(ragas_df[col], errors="coerce")
            row[mean_col] = round(float(numeric.mean(skipna=True)), 4) if numeric.notna().any() else "not_available"
        else:
            row[mean_col] = "not_available"
    num_errors = int((ragas_df["error"] != "").sum()) if (not ragas_df.empty and "error" in ragas_df.columns) else 0
    row["num_errors"] = num_errors
    return row


def _build_comparison_md(
    summary_df: pd.DataFrame,
    correctness_summary: dict[str, pd.DataFrame],
) -> str:
    lines = [
        "# Final Experiment Comparison",
        "",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## RAGAS Summary Table",
        "",
    ]
    lines.append(summary_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Answer Correctness Summary")
    lines.append("")

    # Collect semantic similarity means per experiment
    for exp_name, df in correctness_summary.items():
        if df.empty:
            continue
        sem = pd.to_numeric(df["semantic_similarity"], errors="coerce")
        missing_rate = df["missing_answer_flag"].mean() if not df.empty else float("nan")
        lines.append(
            f"- **{exp_name}**: "
            f"sem_sim={sem.mean():.3f}, "
            f"missing_rate={missing_rate:.1%}"
        )
    lines.append("")

    # Best experiment analysis
    metric_col = "mean_answer_correctness"
    if metric_col in summary_df.columns:
        numeric = pd.to_numeric(summary_df[metric_col], errors="coerce")
        if numeric.notna().any():
            best_idx = numeric.idxmax()
            best = summary_df.loc[best_idx]
            lines.append(f"## Best Overall Experiment (by answer_correctness)")
            lines.append(f"")
            lines.append(f"**{best['experiment_name']}**")
            lines.append(f"- Chunking: {best.get('chunking_type', 'N/A')}")
            lines.append(f"- Retrieval: {best.get('retrieval_type', 'N/A')}")
            lines.append(f"- Mean answer_correctness: {best.get(metric_col, 'N/A')}")
            lines.append("")

    lines.append("## Best Retrieval Under Normal Chunking")
    normal_rows = summary_df[summary_df["chunking_type"] == "normal"]
    if not normal_rows.empty and metric_col in normal_rows.columns:
        numeric = pd.to_numeric(normal_rows[metric_col], errors="coerce")
        if numeric.notna().any():
            best_normal = normal_rows.loc[numeric.idxmax()]
            lines.append(f"**{best_normal['experiment_name']}** (retrieval: {best_normal.get('retrieval_type', 'N/A')})")
    else:
        lines.append("_Not yet available — run all experiments first._")
    lines.append("")

    lines.append("## Best Retrieval Under Parent-Child Chunking")
    pc_rows = summary_df[summary_df["chunking_type"] == "parent_child"]
    if not pc_rows.empty and metric_col in pc_rows.columns:
        numeric = pd.to_numeric(pc_rows[metric_col], errors="coerce")
        if numeric.notna().any():
            best_pc = pc_rows.loc[numeric.idxmax()]
            lines.append(f"**{best_pc['experiment_name']}** (retrieval: {best_pc.get('retrieval_type', 'N/A')})")
    else:
        lines.append("_Not yet available — parent-child experiments pending._")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- RAGAS metrics are LLM-as-judge evaluations and inherit LLM biases.\n"
        "- Sparse retrieval returned fewer than 80 contexts for some questions (BM25 "
        "`score > 0.0` filter). This may affect sparse performance metrics.\n"
        "- Semantic similarity uses nomic-embed-text embeddings and may not capture all "
        "aspects of answer quality.\n"
        "- Parent-child chunking uses 3× more indexed chunks (~3510 vs ~1170), which may "
        "affect retrieval recall independently of chunk quality."
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging("INFO")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Unified RAG Evaluation — All 6 Experiments")
    logger.info("=" * 60)

    # Load golden answers
    golden = _load_golden_answers()

    # Discover experiments
    available = []
    for name in EXPERIMENT_ORDER:
        exp = _load_experiment(name)
        if exp is not None:
            available.append(exp)
            logger.info("Found experiment: %s (%d questions)", name, len(exp["answers_df"]))
        else:
            logger.warning("Experiment not ready: %s", name)

    if not available:
        logger.error("No experiments found in %s", EXPERIMENTS_DIR)
        sys.exit(1)

    logger.info("Evaluating %d experiments", len(available))

    # Build RAGAS evaluator
    logger.info("Setting up RAGAS evaluator...")
    try:
        metrics = _build_ragas_evaluator()
    except ImportError as e:
        logger.error("RAGAS setup failed: %s", e)
        logger.error("Install with: pip install ragas langchain-community")
        sys.exit(1)

    embed_cfg = _ragas_embed_config()
    judge_cfg = _judge_config()
    if judge_cfg:
        logger.info("Judge configured: model=%s", judge_cfg["model"])
    else:
        logger.info("Judge not configured — judge_correctness_score will be 'not_configured'")

    # Evaluate each experiment
    ragas_sheets: dict[str, pd.DataFrame] = {}
    correctness_sheets: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []

    for exp in available:
        name = exp["name"]
        logger.info("-" * 40)
        logger.info("Evaluating: %s", name)

        # RAGAS evaluation
        ragas_df = _run_ragas_for_experiment(exp, golden, metrics)
        ragas_sheets[name] = ragas_df

        # Non-RAGAS correctness
        correctness_df = _run_correctness_for_experiment(
            exp, golden,
            embed_host=embed_cfg["base_url"].replace("/v1", ""),
            embed_model=embed_cfg["model"],
            judge_cfg=judge_cfg,
        )
        correctness_sheets[name] = correctness_df

        # Summary row
        summary_rows.append(_build_summary_row(
            exp_name=name,
            chunking_type=exp["chunking_type"],
            retrieval_type=exp["retrieval_type"],
            ragas_df=ragas_df,
            num_questions=len(exp["answers_df"]),
        ))

    summary_df = pd.DataFrame(summary_rows)

    # ---------------------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------------------

    # 1. RAGAS per-question scores (6 sheets)
    ragas_path = EVAL_DIR / "ragas_per_question_scores.xlsx"
    with pd.ExcelWriter(ragas_path, engine="openpyxl") as writer:
        for name, df in ragas_sheets.items():
            sheet_name = name[:31]  # Excel sheet name limit
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    logger.info("Saved: %s", ragas_path)

    # 2. RAGAS summary
    summary_path = EVAL_DIR / "ragas_summary.xlsx"
    summary_df.to_excel(summary_path, index=False, engine="openpyxl")
    logger.info("Saved: %s", summary_path)

    # 3. Answer correctness (7 sheets: one per experiment + summary)
    correctness_path = EVAL_DIR / "answer_correctness_50q.xlsx"
    with pd.ExcelWriter(correctness_path, engine="openpyxl") as writer:
        for name, df in correctness_sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
        # Summary sheet: mean sem_sim, missing rate, etc.
        summary_rows_corr = []
        for name, df in correctness_sheets.items():
            if df.empty:
                continue
            sem = pd.to_numeric(df["semantic_similarity"], errors="coerce")
            missing = df["missing_answer_flag"].mean() if not df.empty else float("nan")
            norm_exact = df["normalized_exact_match"].mean() if not df.empty else float("nan")
            summary_rows_corr.append({
                "experiment_name": name,
                "mean_semantic_similarity": round(float(sem.mean(skipna=True)), 4) if sem.notna().any() else "not_available",
                "mean_normalized_exact_match": round(float(norm_exact), 4) if not pd.isna(norm_exact) else "not_available",
                "missing_answer_rate": round(float(missing), 4) if not pd.isna(missing) else "not_available",
            })
        pd.DataFrame(summary_rows_corr).to_excel(writer, sheet_name="summary", index=False)
    logger.info("Saved: %s", correctness_path)

    # 4. Final comparison table
    comparison_rows = []
    for exp_name in [e["name"] for e in available]:
        srow = summary_df[summary_df["experiment_name"] == exp_name]
        crow = correctness_sheets.get(exp_name, pd.DataFrame())

        sem_mean = "not_available"
        missing_rate = "not_available"
        if not crow.empty:
            sem = pd.to_numeric(crow["semantic_similarity"], errors="coerce")
            sem_mean = round(float(sem.mean(skipna=True)), 4) if sem.notna().any() else "not_available"
            missing_rate = round(float(crow["missing_answer_flag"].mean()), 4)

        row: dict[str, Any] = {
            "experiment_name": exp_name,
            "chunking_type": srow.iloc[0]["chunking_type"] if not srow.empty else "",
            "retrieval_type": srow.iloc[0]["retrieval_type"] if not srow.empty else "",
        }
        for col in RAGAS_METRIC_COLS:
            mean_col = f"mean_{col}"
            row[col] = srow.iloc[0].get(mean_col, "not_available") if not srow.empty else "not_available"
        row["semantic_similarity"] = sem_mean
        row["missing_answer_rate"] = missing_rate
        row["num_questions"] = srow.iloc[0].get("num_questions", "") if not srow.empty else ""
        row["num_errors"] = srow.iloc[0].get("num_errors", "") if not srow.empty else ""
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_xlsx_path = EVAL_DIR / "final_experiment_comparison.xlsx"
    comparison_df.to_excel(comparison_xlsx_path, index=False, engine="openpyxl")
    logger.info("Saved: %s", comparison_xlsx_path)

    # 5. Markdown narrative
    md_text = _build_comparison_md(summary_df, correctness_sheets)
    md_path = EVAL_DIR / "final_experiment_comparison.md"
    md_path.write_text(md_text)
    logger.info("Saved: %s", md_path)

    logger.info("=" * 60)
    logger.info("Evaluation complete.")
    logger.info("Experiments evaluated: %d / %d", len(available), len(EXPERIMENT_ORDER))
    if len(available) < len(EXPERIMENT_ORDER):
        missing = set(EXPERIMENT_ORDER) - {e["name"] for e in available}
        logger.info("Pending experiments: %s", sorted(missing))
    logger.info("Outputs in: %s", EVAL_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
