# Code Mistake Review

**Date:** 2026-05-09  
**Scope:** All source modules in `src/` and `scripts/` relevant to the six-experiment RAG pipeline.

---

## Critical Bugs Found

### 1. `src/chunking.py` — `ParentChildChunker` is not true parent-child chunking

**File:** `src/chunking.py`, lines 83–340  
**Severity:** Critical (architectural misrepresentation)

**What the code does:**
```python
def chunk_rows(self, rows):
    for idx, row in enumerate(rows):
        chunk = self._build_chunk(row, idx)  # 1 chunk per row
        chunks.append(chunk)
```

Each spreadsheet row → exactly 1 chunk. `parent_id = record_id` of that same chunk. `parent_text` in the retrieval output is the same text as the chunk itself. There is no child decomposition, no multi-level hierarchy, and no meaningful parent-child expansion.

**What true parent-child should do:**
- Split each company record into multiple smaller child chunks (e.g., by field groups)
- Index child chunks for retrieval
- On retrieval: identify the parent record from the matched child, return full parent text to LLM

**Research impact:** The current three experiments are valid as "normal chunking" experiments. They cannot be called "parent-child" experiments without misrepresenting the methodology.

**Fix:** Add `TrueParentChildChunker` to `src/chunking.py`. See plan.

---

## Possible Bugs / Needs Manual Review

### 2. `src/retrieval.py`, line 84 — Vector normalization formula incorrect for cosine distance

**File:** `src/retrieval.py`, lines 52–93  
**Severity:** Medium (scores incorrect, relative ranking partially preserved)

**Code:**
```python
similarities = 1.0 / (1.0 + distances.astype(np.float64))
```

**Problem:** ChromaDB uses cosine distance (range [0, 2]). The formula `1/(1+d)` is designed for L2/Euclidean distances. For cosine:
- distance=0 (identical) → `1/(1+0)` = 1.0 ✓ correct
- distance=1 (perpendicular) → `1/(1+1)` = 0.5 ❌ should be 0.0
- distance=2 (opposite) → `1/(1+2)` = 0.33 ❌ should be -1.0 (or 0.0 if clamped)

The subsequent min-max normalization rescales the output to [0, 1], which partially hides this error by preserving relative ordering within the dense candidate set. However, absolute scores are not true cosine similarities.

**Correct formula:**
```python
similarities = 1.0 - (distances.astype(np.float64) / 2.0)
```

**Research impact:** Relative ranking within each experiment is preserved. Cross-experiment absolute score comparison is invalid. Do NOT fix during the six-experiment run — fix after all runs complete to avoid introducing inconsistency between experiments.

### 3. `src/retrieval.py`, line 230 — `embed_query` called for sparse-only mode

**File:** `src/retrieval.py`, line 228–230  
**Severity:** Low (performance waste, no correctness impact)

**Code:**
```python
# Embed query (needed for dense/hybrid; cheap to skip for sparse...)
query_embedding = self._embedding.embed_query(query)
```

The comment acknowledges this is wasteful for sparse mode but calls it unconditionally. For 50 questions this is negligible (one extra Ollama API call per question).

**Fix:**
```python
if self._fusion_mode in ("dense", "hybrid"):
    query_embedding = self._embedding.embed_query(query)
else:
    query_embedding = None
```

**Research impact:** None. Sparse results are unchanged.

### 4. `src/utils.py` — question_id auto-generated from row order

**File:** `src/utils.py`, lines 284–285  
**Severity:** Low (fragile but currently stable)

**Code:**
```python
if "question_id" not in df.columns:
    df["question_id"] = [f"Q{i+1:03d}" for i in range(len(df))]
```

If anyone sorts or reorders `data/questions.xlsx`, question IDs will shift and all evaluation joins will be wrong.

**Fix:** Add `question_id` column permanently to `data/questions.xlsx`.

### 5. `src/llm.py` — Decorative placeholders in SYSTEM_PROMPT

**File:** `src/llm.py`, lines 107–141  
**Severity:** Low (cosmetic, no functional impact)

**Code includes:**
```
QUESTION:
{question}

RETRIEVED CONTEXT:
{context}
```

These `{question}` and `{context}` placeholders are never filled in — they're printed literally as part of the static system prompt. The actual question and context are passed in the user message, not the system message.

**Research impact:** None — the prompt functions correctly. Confusing to read but does not affect output.

---

## Correct Implementations

### Dense retrieval ✓
- Uses ChromaDB `PersistentClient` with cosine distance metric
- Embeddings via `nomic-embed-text` through Ollama REST API
- Collection `gnem_chunks` with 1170 indexed company chunks
- `query_embedding` correctly used for vector search

### Sparse retrieval ✓
- Uses `BM25Okapi` from `rank_bm25` library — genuinely separate from ChromaDB
- Indexed text: `embedding_text` from each chunk
- Tokenizer: whitespace split + lowercase (simple but consistent)
- Returns documents with score > 0.0 only

### Hybrid retrieval ✓
- Correctly combines dense + sparse via weighted sum: `0.6 * vector_norm + 0.4 * bm25_norm`
- Fetches top-250 candidates from each retriever before fusion
- Min-max normalizes both score sets independently
- Deduplicates by `parent_id` (keeps best-scoring child per parent)
- Returns top_k=80 after dedup

### Score deduplication logic ✓ (important for parent-child)
- `HybridRetriever._retrieve_impl()` lines 304–314
- Dedup by `parent_id`: keeps only the highest-scoring chunk per unique parent
- Looks up `parent_text` from `parent_lookup` dict
- This logic correctly handles true parent-child: multiple child chunks → same `parent_id` → best child wins → full parent text returned

### Controlled experimental setup ✓
- Same 50 questions, same LLM, same prompt for all three existing retrieval modes
- `top_k=80`, `pool_size=250`, `alpha=0.6` consistent across modes
- Parameters from centralized `config/config.yaml`

### Golden answer isolation ✓
- `data/Human validated 50 questions.xlsx` never loaded by `run_rag.py`
- Golden answers used only in evaluation scripts
- No data leakage confirmed

### Per-question error handling ✓
- Each question in a try-except block
- On error: error message saved to `error` column, answer set to `""`
- Failed rows still written to output (not silently dropped)
- All 50 rows present even if some have errors

### Output preservation ✓
- Previous results backed up with timestamp before new write
- New results always get unique timestamp — no data loss

---

## Required Code Fixes

### Critical (blocks six-experiment comparison):

1. **Add `TrueParentChildChunker` to `src/chunking.py`**
   - New class, does not modify existing `ParentChildChunker`
   - Field-level grouping: identity / industry / profile child chunks
   - Returns `(list[Chunk], dict[str, dict])` — child chunks + parent lookup

2. **Create `scripts/run_experiment.py`**
   - Accepts `--chunking {normal,parent_child}` and `--retrieval {dense,sparse,hybrid}`
   - Routes to correct chunking class and ChromaDB collection
   - Saves outputs to `outputs/experiments/{name}/`

3. **Create `scripts/evaluate_all_experiments.py`**
   - Evaluates all six experiments using RAGAS + custom metrics
   - Reads from `outputs/experiments/` folder structure

### Recommended (after all six runs complete):

4. **Fix vector normalization formula in `src/retrieval.py`**
   - Change `1/(1+d)` to `1 - d/2`
   - Only fix after all six experiments are complete (consistency requirement)
   - Note the fix in paper limitations section

5. **Add `question_id` column to `data/questions.xlsx`**

---

## Suggested Code Improvements

- Persist BM25 index to disk (pickle) — avoids re-indexing on every run
- Skip `embed_query` when `fusion_mode == "sparse"`
- Add `--dry-run` flag to `run_experiment.py` to validate config without running LLM
- Store `run_config.json` with git commit hash for full reproducibility

---

## Research Impact of Each Issue

| Issue | Research Impact | Blocks Runs? | Blocks Paper? |
|---|---|---|---|
| ParentChildChunker is row-level | Critical — 3 of 6 experiments missing | YES | YES |
| Vector normalization formula | Medium — absolute scores wrong, relative ranking OK | NO | NO (note in limitations) |
| BM25 not persisted | None | NO | NO |
| embed_query in sparse mode | None | NO | NO |
| question_id auto-generated | Low — fragile alignment | NO | NO (stable with current xlsx) |
| Sparse has 3949 contexts | Low — affects some questions | NO | NO (document it) |
| SYSTEM_PROMPT placeholders | None | NO | NO |
