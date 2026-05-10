#!/usr/bin/env python3
"""Extract existing normal chunking results into per-experiment folders.

Reads the complete generated-answers Excel file and writes separate files
for each retrieval mode into outputs/experiments/normal_{mode}/.

Usage:
    cd simple_rag/
    python scripts/organize_experiments.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


SOURCE_FILE = (
    _PROJECT_ROOT
    / "outputs"
    / "generated_answers"
    / "generated_answers_current_complete_20260509_015013.xlsx"
)

EXPERIMENTS_DIR = _PROJECT_ROOT / "outputs" / "experiments"

MODES = ["Dense", "Sparse", "Hybrid"]

RUN_CONFIG_TEMPLATE = {
    "chunking_type": "normal",
    "embedding_model": "nomic-embed-text",
    "embedding_host": "http://localhost:11434",
    "vector_store": "chromadb",
    "chroma_collection": "gnem_chunks",
    "sparse_method": "BM25Okapi",
    "top_k": 80,
    "candidate_pool_size": 250,
    "semantic_weight": 0.6,
    "keyword_weight": 0.4,
    "llm_model": "gemma3:27b",
    "llm_temperature": 0.2,
    "llm_max_tokens": 4096,
    "prompt_version": "v1",
    "num_questions": 50,
    "source_file": str(SOURCE_FILE),
    "organized_at": datetime.now().isoformat(),
    "notes": (
        "Extracted from combined run file. Original chunking labeled 'parent-child' "
        "in code but is row-level (1 company = 1 chunk). Relabeled as 'normal' for "
        "research clarity."
    ),
}


def main() -> None:
    if not SOURCE_FILE.exists():
        print(f"ERROR: Source file not found: {SOURCE_FILE}")
        print("Available files in generated_answers/:")
        ga_dir = SOURCE_FILE.parent
        for f in sorted(ga_dir.glob("*.xlsx")):
            print(f"  {f.name}")
        sys.exit(1)

    print(f"Reading source: {SOURCE_FILE.name}")
    xf = pd.ExcelFile(SOURCE_FILE)
    print(f"Sheets found: {xf.sheet_names}")

    for mode in MODES:
        rag_sheet = f"RAG_{mode}"
        ctx_sheet = f"Context_{mode}"

        if rag_sheet not in xf.sheet_names:
            print(f"WARNING: Sheet {rag_sheet!r} not found — skipping {mode}")
            continue

        experiment_name = f"normal_{mode.lower()}"
        out_dir = EXPERIMENTS_DIR / experiment_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- Generated answers ---
        answers_df = pd.read_excel(SOURCE_FILE, sheet_name=rag_sheet)
        answers_path = out_dir / "generated_answers.xlsx"
        if answers_path.exists():
            print(f"  [{experiment_name}] generated_answers.xlsx already exists — skipping")
        else:
            answers_df.to_excel(answers_path, index=False, engine="openpyxl")
            print(f"  [{experiment_name}] generated_answers.xlsx — {len(answers_df)} rows")

        # --- Retrieved contexts ---
        if ctx_sheet in xf.sheet_names:
            ctx_df = pd.read_excel(SOURCE_FILE, sheet_name=ctx_sheet)
            ctx_path = out_dir / "retrieved_contexts.xlsx"
            if ctx_path.exists():
                print(f"  [{experiment_name}] retrieved_contexts.xlsx already exists — skipping")
            else:
                ctx_df.to_excel(ctx_path, index=False, engine="openpyxl")
                print(f"  [{experiment_name}] retrieved_contexts.xlsx — {len(ctx_df)} rows")
        else:
            print(f"  [{experiment_name}] WARNING: {ctx_sheet} not found — no retrieved_contexts.xlsx")

        # --- run_config.json ---
        config_path = out_dir / "run_config.json"
        if config_path.exists():
            print(f"  [{experiment_name}] run_config.json already exists — skipping")
        else:
            config = dict(RUN_CONFIG_TEMPLATE)
            config["retrieval_type"] = mode.lower()
            config["experiment_name"] = experiment_name
            config_path.write_text(json.dumps(config, indent=2))
            print(f"  [{experiment_name}] run_config.json written")

    # Create empty placeholder directories for parent-child experiments
    for mode in MODES:
        pc_name = f"parent_child_{mode.lower()}"
        pc_dir = EXPERIMENTS_DIR / pc_name
        pc_dir.mkdir(parents=True, exist_ok=True)
        placeholder = pc_dir / "README.txt"
        if not placeholder.exists():
            placeholder.write_text(
                f"Experiment: {pc_name}\n"
                "Status: Pending\n\n"
                "This folder will be populated by:\n"
                f"    python scripts/run_experiment.py --chunking parent_child "
                f"--retrieval {mode.lower()}\n\n"
                "Requires:\n"
                "  - Ollama running with gemma3:27b and nomic-embed-text\n"
                "  - TrueParentChildChunker implemented in src/chunking.py\n"
            )
            print(f"  [{pc_name}] placeholder created")

    print("\nDone. Experiment structure:")
    for d in sorted(EXPERIMENTS_DIR.iterdir()):
        files = [f.name for f in d.iterdir() if f.is_file()]
        print(f"  {d.name}/  {files}")


if __name__ == "__main__":
    main()
