# Experiment Fairness Check

**Date:** 2026-05-09  
**Purpose:** Verify all six experiments use consistent, controlled settings to enable fair comparison.

---

## Overview

A six-experiment comparison is fair only when all variables except the ones being studied are held constant. This document verifies fairness across two axes:

- **Retrieval axis** (dense vs. sparse vs. hybrid): must use the same chunking, top_k, embedding model, LLM, prompt
- **Chunking axis** (normal vs. parent-child): must use the same retrieval parameters, LLM, prompt, questions

---

## Experiment Configuration Table

| experiment_name | chunking_type | retrieval_type | num_questions | num_answers | num_contexts | top_k | pool_size | alpha | prompt_version | llm_model | embedding_model | chroma_collection | valid_for_comparison | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| normal_dense | normal | dense | 50 | 50 | 4000 | 80 | 250 | 0.6 | v1 | gemma3:27b | nomic-embed-text | gnem_chunks | YES | Existing run — validated |
| normal_sparse | normal | sparse | 50 | 50 | 3949 | 80 | 250 | 0.4 | v1 | gemma3:27b | nomic-embed-text | gnem_chunks | YES | 51 missing contexts — document |
| normal_hybrid | normal | hybrid | 50 | 50 | 4000 | 80 | 250 | 0.6 | v1 | gemma3:27b | nomic-embed-text | gnem_chunks | YES | Existing run — validated |
| parent_child_dense | parent_child | dense | 50 | TBD | TBD | 80 | 250 | 0.6 | v1 | gemma3:27b | nomic-embed-text | gnem_child_chunks | PENDING | New run required |
| parent_child_sparse | parent_child | sparse | 50 | TBD | TBD | 80 | 250 | 0.4 | v1 | gemma3:27b | nomic-embed-text | gnem_child_chunks | PENDING | New run required |
| parent_child_hybrid | parent_child | hybrid | 50 | TBD | TBD | 80 | 250 | 0.6 | v1 | gemma3:27b | nomic-embed-text | gnem_child_chunks | PENDING | New run required |

**Note on alpha:** alpha controls the dense/sparse weight in hybrid mode only (`combined = alpha * vector_norm + (1-alpha) * bm25_norm`). It has no effect on dense-only or sparse-only retrieval modes.

---

## Controlled Variables (Must Be Identical Across All Six)

### Questions ✓
- Source file: `data/questions.xlsx` (50 rows) or `data/questions.csv`
- question_id: auto-generated Q001–Q050 by row order (fragile but consistent across runs)
- Golden answers: `data/Human validated 50 questions.xlsx` — used only for evaluation, never during generation

### LLM Model ✓
- Model: `gemma3:27b` (configured via `LLM_MODEL` env var)
- Temperature: 0.2 (from config.yaml `llm.defaults.temperature`)
- Max tokens: 4096 (from config.yaml `llm.defaults.max_tokens`)
- Base URL: `http://localhost:11434/v1` (Ollama)

### Answer Generation Prompt ✓
- SYSTEM_PROMPT is a constant string in `src/llm.py` (lines 107–141)
- Same prompt used for all three retrieval modes within each chunking type
- Prompt enforces KB-only answers — no outside knowledge
- Evidence format is identical across all modes

### Embedding Model ✓
- Model: `nomic-embed-text` via Ollama (`http://localhost:11434/api/embeddings`)
- Batch size: 32 (from config.yaml)
- Used for: indexing chunks AND query embedding (dense and hybrid modes)

### Retrieval Parameters ✓
- `top_k`: 80 (final number of contexts returned to LLM)
- `candidate_pool_size`: 250 (candidates fetched before fusion/ranking)
- `semantic_weight` (alpha): 0.6 (weight for dense scores in hybrid fusion)

### Evaluation Script ✓ (planned)
- `scripts/evaluate_all_experiments.py` — single script for all six
- Same RAGAS version (0.4.3), same metrics, same golden answers alignment

---

## Intentional Differences (What Is Being Compared)

### Retrieval axis: dense vs. sparse vs. hybrid
| Factor | Dense | Sparse | Hybrid |
|---|---|---|---|
| ChromaDB vector search | YES | NO | YES |
| BM25 keyword search | NO | YES | YES |
| Score combination | vector_norm only | bm25_norm only | 0.6·vector + 0.4·bm25 |
| embed_query called | YES | YES (wasteful, no-op) | YES |

### Chunking axis: normal vs. parent-child
| Factor | Normal | Parent-Child |
|---|---|---|
| Chunks per company | 1 (full record) | 3 (field groups) |
| What is indexed | Full embedding_text | Child field chunks |
| What is retrieved | Full record text | Best child chunk |
| What LLM sees | Full record (same as chunk) | Full parent record (expanded from child) |
| ChromaDB collection | gnem_chunks | gnem_child_chunks |
| Total indexed chunks | ~1170 | ~3510 |

---

## Fairness Checks

### Dense vs. Sparse vs. Hybrid within same chunking type — FAIR ✓
- Same questions: ✓
- Same LLM: ✓
- Same prompt: ✓
- Same top_k: ✓
- Same embedding model: ✓
- Same ChromaDB collection per chunking type: ✓
- Same golden answers for evaluation: ✓
- Difference: retrieval scoring method only — this is the intended variable

### Normal vs. Parent-Child within same retrieval mode — FAIR ✓
- Same questions: ✓
- Same LLM: ✓
- Same prompt: ✓
- Same top_k: ✓
- Same embedding model: ✓
- Same retrieval code path (HybridRetriever): ✓
- Difference: chunking strategy + indexed collection — this is the intended variable
- Note: Normal indexes 1170 full-text chunks; parent-child indexes ~3510 smaller child chunks. This is a legitimate architectural difference being tested.

---

## Known Fairness Concerns

### 1. Sparse retrieval returns fewer contexts for some questions

Context_Sparse has 3949 rows vs. 4000 for dense/hybrid (51 missing). This means some questions received fewer than 80 sparse contexts. The LLM still received fewer evidence items for those questions in sparse mode.

**Recommendation:** Do not conclude that sparse retrieval is inferior without investigating whether the questions with fewer contexts are also the ones where sparse underperforms. Document in final report.

### 2. Vector similarity scores use incorrect normalization formula

The formula `1/(1+distance)` is used for cosine distance (should be `1 - distance/2`). This applies equally to all six experiments. Relative ranking within each experiment is not affected, but absolute similarity scores are not true cosine similarities.

**Recommendation:** Use relative rankings for comparison, not absolute scores. Note the formula discrepancy in the research paper. Do not fix the formula mid-experiment (would break comparability between normal_* and parent_child_* runs).

### 3. Parent-child experiments not yet run

Experiments 4–6 are pending. This table will be updated after runs complete.

### 4. Different number of indexed chunks per chunking type

Normal: ~1170 chunks. Parent-child: ~3510 child chunks. This means the ChromaDB pool for parent-child has ~3× more entries, which may affect retrieval behavior even with the same top_k=80 final cutoff. This is an inherent property of the chunking strategies being compared — not a fairness violation, but should be noted in the paper.

---

## Post-Run Update Instructions

After running parent-child experiments, update this table:
1. Fill in `num_answers` and `num_contexts` from the output Excel files
2. Note any errors or missing rows
3. Update `valid_for_comparison` to YES if runs completed cleanly
4. Document any deviations from the expected configuration

---

## Conclusion

The three existing normal_* experiments are fair to compare with each other. The six experiments as a whole will be fair to compare once the parent-child experiments are completed with the same parameter settings verified in `run_config.json`. No unfair variable mixing has been detected in the current design.

**Important:** Do not claim one method is better unless all six experiments are complete with consistent configurations and metrics are computed by the same evaluation script.
