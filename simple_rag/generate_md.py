import pandas as pd
from pathlib import Path

EVAL_DIR = Path("outputs/evaluation")
summary_df = pd.read_excel(EVAL_DIR / "ragas_summary.xlsx")

correctness_sheets = {}
xl = pd.ExcelFile(EVAL_DIR / "answer_correctness_50q.xlsx")
for sheet in xl.sheet_names:
    if sheet != "summary":
        correctness_sheets[sheet] = xl.parse(sheet)

# Copied from the script
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

for exp_name, df in correctness_sheets.items():
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

md_text = "\n".join(lines)
md_path = EVAL_DIR / "final_experiment_comparison.md"
md_path.write_text(md_text)
print(f"Saved: {md_path}")
