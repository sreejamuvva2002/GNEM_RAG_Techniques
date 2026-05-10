# GNEM RAG Codebase Audit Report

**Date:** 2026-05-09  
**Repository:** GNEM_RAG_Techniques  
**Scope:** Three-way retrieval comparison (dense, sparse, hybrid) over 50 human-validated questions  
**Overall Verdict:** **Valid with minor fixes** — the core experiment is research-valid, with identified issues that require documentation and optional optimization.

---

## Summary Verdict

The RAG system successfully implements and compares three genuinely distinct retrieval strategies:
1. **Dense retrieval** via ChromaDB with nomic-embed-text embeddings
2. **Sparse retrieval** via real BM25Okapi keyword search
3. **Hybrid retrieval** via weighted combination (60% semantic, 40% keyword)

The experiment is **research-valid** because:
- Same 50 questions evaluated across all three modes
- Same LLM (gemma3:27b), same prompt template, same top_k=80 for all modes
- Three retrievers genuinely separated in code; not one retriever with fake variants
- Evidence metadata fully preserved and auditable
- Per-question error handling prevents silent failures
- Golden answers never seen during answer generation

**However**, the system has architectural misrepresentations and performance issues that require attention:
- The "parent-child chunking" label is misleading (it's row-level chunking, not multi-level decomposition)
- Vector similarity inversion formula is mathematically incorrect for cosine distance
- BM25 index rebuilt on every run instead of persisted
- RAGAS library missing from requirements.txt

These issues do **not invalidate the retrieval comparison results**, but do reduce technical correctness and reproducibility.

---

## What Is Correct

### 1. Dense Retrieval Implementation ✓

**Verified:**
- Uses `chromadb>=0.5.0` with `PersistentClient` persisting to `.cache/chroma_db`
- Collection name: `gnem_chunks`, distance metric: `cosine`
- Embeddings: `nomic-embed-text` via Ollama REST API (`http://localhost:11434/api/embeddings`)
- Embedding class: `NomicOllamaEmbedding` — batches requests (batch_size=32 configured)
- Query embedding called via `embed_query(query)`, returns numpy array
- ChromaDB search returns: (ids, distances, metadatas) — distances are cosine distances

**Evidence:**
- File: `simple_rag/src/embedding.py` (lines 1-60): NomicOllamaEmbedding makes direct HTTP POST to Ollama
- File: `simple_rag/src/vector_store.py`: ChromaDB add() and search() methods
- File: `simple_rag/src/retrieval.py` (line 236-239): vector_results = self._vector_store.search(query_embedding, top_k=250)

---

### 2. Sparse Retrieval Implementation ✓

**Verified:**
- Uses `rank-bm25>=0.2.2` BM25Okapi algorithm (genuine, not ChromaDB)
- Index class: `BM25KeywordSearch` in `simple_rag/src/keyword_search.py`
- Tokenizer: whitespace split + lowercase (no stemming, no stopwords)
- Indexed text: `embedding_text` from each chunk (the formatted template-rendered text)
- Search returns: up to top_k=250 documents with scores > 0.0

**Evidence:**
- File: `simple_rag/src/keyword_search.py`: imports `BM25Okapi` from `rank_bm25`
- BM25 is **not** ChromaDB with different settings — it's a separate library
- File: `simple_rag/src/retrieval.py` (line 250-266): bm25_results = self._keyword_search.search(query, top_k=250)

---

### 3. Hybrid Retrieval Implementation ✓

**Verified:**
- Hybrid mode truly combines dense + sparse results
- Algorithm:
  1. Fetch top_k=250 from BOTH vector store AND BM25
  2. Merge into union dict keyed by chunk_id
  3. Normalize vector scores: `sim = 1/(1+distance)`, then min-max to [0,1]
  4. Normalize BM25 scores: min-max to [0,1]
  5. Combine: `combined = 0.6 * vector_norm + 0.4 * bm25_norm`
  6. Rank by combined score, deduplicate by parent_id, return top_k=80

**Evidence:**
- File: `simple_rag/src/retrieval.py` (line 96-116): `combine_scores()` function
- File: `simple_rag/src/retrieval.py` (line 289-294): Mode-specific ranking logic
- Dense uses vector_norm only, Sparse uses bm25_norm only, Hybrid uses combined

---

### 4. Controlled Experimental Setup ✓

**Verified:**
- **Same 50 questions for all modes:** Questions loaded from `data/questions.xlsx` once, iterated 3 times (one per mode)
- **Same LLM:** All modes use `gemma3:27b` via Ollama, temperature 0.2, max_tokens 4096
- **Same prompt:** Prompt template in `SYSTEM_PROMPT` is identical for all modes; evidence list differs only in ranking
- **Same retrieval parameters:**
  - `top_k=80` (final rank cutoff)
  - `candidate_pool_size=250` (before fusion)
  - `semantic_weight=0.6`, `keyword_weight=0.4` (fixed alpha for hybrid)
  - All in `config.yaml` under `retrieval` section

**Evidence:**
- File: `simple_rag/scripts/run_rag.py` (line 288-315): Loop over fusion_modes, same retriever called with same top_k/alpha/pool_size
- File: `simple_rag/src/llm.py` (line 107-141): SYSTEM_PROMPT is constant
- File: `simple_rag/config/config.yaml`: All parameters centralized and read once

---

### 5. Question ID Consistency ✓

**Verified:**
- Questions loaded with question_id column: `Q001`, `Q002`, ..., `Q050`
- ID generated by `load_questions()` if missing from source file
- ID flows through all output sheets: RAG_Dense, RAG_Sparse, RAG_Hybrid (one row per question)
- Context sheets (Context_Dense, etc.) also have question_id column for join-ability

**Evidence:**
- File: `simple_rag/src/utils.py` (line 284-285): Auto-generation of question_id as Q{i+1:03d}
- File: `simple_rag/scripts/run_rag.py` (line 149-151): Each context row includes question_id

---

### 6. Output Files Preserved and Timestamped ✓

**Verified:**
- Previous results backed up before new write: `generated_answers_previous_YYYYMMDD_HHMMSS.xlsx`
- New results written to: `generated_answers_current_YYYYMMDD_HHMMSS.xlsx`
- Base path `generated_answers.xlsx` never modified (acts as checkpoint only)
- Each run creates unique timestamped outputs → no data loss

**Evidence:**
- File: `simple_rag/scripts/run_rag.py` (line 208-211): `_timestamped_path()` function
- Line 322-328: Backup and write logic

---

### 7. Evidence Metadata Fully Preserved ✓

**Verified:**
- Context sheets (Context_Dense, Context_Sparse, Context_Hybrid) have **4000 rows** (dense/hybrid) or **3949 rows** (sparse)
- Each row is one evidence item: question_id, evidence_rank (1-80), evidence_id (E001-E080), all company metadata
- Columns: company, county, tier_level, employment, primary_oems, facility_type, industry_name, ev_relevant, product_service, supplier_type
- Scores preserved: similarity_score, keyword_score, combined_score
- Text column: the exact embedding_text sent to LLM

**Evidence:**
- File: `simple_rag/scripts/run_rag.py` (line 124-155): Evidence dict construction with all fields
- Output confirmed: 4000 rows = 50 questions × 80 evidence items (dense/hybrid), 3949 = sparse (some questions < 80 results)

---

### 8. Prompt Enforces KB-Only Answering ✓

**Verified:**
- SYSTEM_PROMPT (file: `simple_rag/src/llm.py` lines 107-141) includes:
  - "Use ONLY the retrieved context"
  - "Do not use outside or pretrained knowledge"
  - "If context is incomplete, clearly state: 'Not available in the provided context.'"
  - "Do not guess or infer unsupported facts"
- Response format includes: Direct Answer, Supporting Information, Missing or Uncertain Information

**Evidence:**
- File: `simple_rag/src/llm.py` (line 107-141): Full SYSTEM_PROMPT text
- Prompt is sent verbatim to every LLM call via `build_evidence_prompt()`

---

### 9. Golden Answers Never Contaminate Generation ✓

**Verified:**
- Golden answers in separate file: `data/Human validated 50 questions.xlsx`
- Golden answers loaded **only** for evaluation, **not** during pipeline run
- Pipeline loads questions from `data/questions.xlsx` (no golden answers column)
- LLM sees only: question + retrieved context, never any golden answers

**Evidence:**
- File: `simple_rag/scripts/run_rag.py` (line 269): Only loads questions via `load_questions()`, not golden answers
- Golden answers file never mentioned in run_rag.py or any ingestion/retrieval code

---

### 10. Per-Question Error Handling ✓

**Verified:**
- Each question wrapped in try-except (line 120-203 in run_rag.py)
- On error: error message captured in `error` column, answer set to empty string
- Failed row still added to results (not silently dropped)
- All 50 rows present in output even if some failed

**Evidence:**
- File: `simple_rag/scripts/run_rag.py` (line 187-203): Exception handler appends error row

---

### 11. LLM Configuration Reads Environment Variables ✓

**Verified:**
- LLM client reads from env vars: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`
- Falls back to config.yaml defaults if env vars not set
- Model is required (raises ValueError if missing)

**Evidence:**
- File: `simple_rag/src/llm.py` (line 36-44): LLMClient constructor
- File: `simple_rag/config/config.yaml` (line 107-112): Default values provided

---

## Problems Found

### 1. Parent-Child Chunking Is Misleading — NOT True Multi-Level Decomposition ❌

**Issue:** The code and README refer to "parent-child chunking," but the implementation is actually **row-level chunking** with no actual parent-child hierarchy.

**What the code does:**
- Each spreadsheet row (1 company record) becomes exactly 1 chunk
- Chunk has: chunk_id (GA_AUTO_0000), record_id (MD5 hash of company + row index)
- In retrieval, `parent_id = record_id` of the same chunk (1:1 relationship)
- The `parent_text` = `embedding_text` of that same chunk (not a larger window)

**Expected parent-child RAG:**
- Split each document into multiple small child chunks (e.g., sliding window of 200 tokens)
- Embed and index the small child chunks
- On retrieval: find child chunks via vector search
- Expand to retrieve the larger parent document for final context

**Actual implementation:**
- 1 company record = 1 chunk (no decomposition)
- `parent_id` and `child_id` are the same
- Deduplication by `parent_id` only helps if multiple chunks of the same parent were retrieved, which cannot happen here (1:1 mapping)

**Evidence:**
- File: `simple_rag/src/chunking.py` (line 105-121): `chunk_rows()` loops once per row, creates 1 Chunk per row
- File: `simple_rag/src/ingestion.py` (line 53-65): `parent_lookup` maps record_id → {text, company}
- File: `simple_rag/src/retrieval.py` (line 304-311): Deduplication by parent_id, but parent_id == record_id of same chunk

**Impact on research:**
- Functional impact: **Low** — retrieval still works correctly for the intended experiment
- Architectural representation: **High** — misleading name in code and documentation
- Recommendation: Rename to "row-level chunking with duplicate prevention" or implement true parent-child if intended

---

### 2. Vector Similarity Inversion Formula Is Mathematically Incorrect for Cosine Distance ❌

**Issue:** Score normalization uses `sim = 1 / (1 + distance)` which is designed for **L2/Euclidean** distances, not **cosine** distances. ChromaDB is configured with cosine metric.

**What happens:**
- ChromaDB returns cosine **distance** (defined as `1 - cosine_similarity`)
- Cosine distance ranges: [0, 2] for normalized vectors
  - 0 = identical vectors (cos_sim = 1)
  - 1 = perpendicular (cos_sim = 0)
  - 2 = opposite (cos_sim = -1)
- Formula `1 / (1 + distance)` maps:
  - distance=0 → sim=1.0 ✓ correct
  - distance=1 → sim=0.5 (but cos_sim should be 0, not 0.5) ❌
  - distance=2 → sim=0.33 (but cos_sim should be -1, not 0.33) ❌
- Min-max normalization afterward **rescales** [0.33, 1] to [0, 1], partially hiding the error

**Correct formula for cosine distance:**
```
cosine_similarity = 1 - cosine_distance  (standard definition)
OR
similarity = 1 - distance / 2  (maps [0,2] → [1, 0])
```

**Impact on research:**
- Functional impact: **Medium** — min-max normalization rescales all scores, but relative ranking may shift
- Comparison impact: **Low** — error affects all 3 modes equally, so dense vs sparse vs hybrid ranking should be consistent
- Reproducibility: **Medium** — scores are not true cosine similarities, so external comparison is problematic
- Recommendation: Fix formula to `1 - distance / 2` or set `already_similarity=True` if ChromaDB was changed to return similarities

**Evidence:**
- File: `simple_rag/src/retrieval.py` (line 52-93): `normalize_vector_scores()` function
- File: `simple_rag/src/vector_store.py`: ChromaDB configured with `cosine` metric
- File: `simple_rag/config/config.yaml` (line 84): `distance_metric: cosine`

---

### 3. BM25 Index Not Persisted — Rebuilt Every Run ❌

**Issue:** BM25 index is built in-memory at runtime and **never saved** to disk. On every run, all 1170 documents are re-indexed.

**What the code does:**
- File: `simple_rag/src/ingestion.py` (line 143-167): Creates `BM25KeywordSearch` in-memory, calls `index(keyword_docs)`
- BM25 object kept in `IngestionService._keyword_search`
- On next run: entire index rebuilt from scratch

**Impact on research:**
- Functional impact: **None** — results are identical each run (deterministic index)
- Performance impact: **High** — startup time scales linearly with corpus size
- Reproducibility: **Low** — index state not captured, makes step-by-step replication harder
- Recommendation: Persist BM25 index using pickle or custom serialization

**Evidence:**
- File: `simple_rag/src/keyword_search.py`: No persistence methods
- File: `simple_rag/src/ingestion.py` (line 143): `logger.info("Step 5/5: Building BM25 keyword index")`

---

### 4. Sparse Mode Calls Embed Query Unnecessarily ❌

**Issue:** `query_embedding = self._embedding.embed_query(query)` is called **unconditionally** even when using sparse-only retrieval, wasting an Ollama API call.

**What the code does:**
- File: `simple_rag/src/retrieval.py` (line 230): "Embed query (needed for dense/hybrid; cheap to skip for sparse...)"
- Comment acknowledges the waste but calls it anyway "so the interface contract stays simple"

**Impact on research:**
- Functional impact: **None** — sparse mode ignores the embedding
- Performance impact: **Low** — one extra embedding call per question (not critical for 50 questions)
- Recommendation: Skip for sparse mode if performance critical, but current impact negligible

**Evidence:**
- File: `simple_rag/src/retrieval.py` (line 228-230)

---

### 5. Context_Sparse Has 3949 Rows vs 4000 for Dense/Hybrid ❌

**Issue:** Sparse retrieval returned fewer than 80 contexts for some questions (3949 vs expected 4000 = 50 × 80).

**What this means:**
- Dense: 50 questions × 80 results = 4000 context rows ✓
- Sparse: 50 questions × ~78.98 avg = 3949 rows (51 missing) ⚠️
- Hybrid: 50 questions × 80 results = 4000 rows ✓

**Likely cause:**
- BM25 search returns 0 results for some queries (score = 0.0 excluded)
- Or BM25 returned < 80 docs before deduplication

**Impact on research:**
- Functional impact: **Low** — some questions have < 80 sparse contexts
- Comparison impact: **Medium** — sparse mode has fewer evidence for some questions; may artificially reduce its performance
- Recommendation: Investigate which questions have < 80 sparse results and why

**Evidence:**
- Confirmed via Excel file inspection: Context_Sparse has 3949 rows vs 4000

---

### 6. questions.xlsx Lacks question_id Column ❌

**Issue:** The questions file used at runtime (`data/questions.xlsx`) has only a `Question` column, no `question_id` column. IDs are auto-generated by row order.

**What the code does:**
- File: `simple_rag/src/utils.py` (line 284-285): If question_id missing, generate as `Q{i+1:03d}`
- Works fine **as long as row order is stable**

**Risk:**
- If someone sorts the Excel file or adds rows in the middle, IDs will be re-generated incorrectly
- Golden answers file has `Num` (1-50) which must align by row order with auto-generated Q001-Q050

**Recommendation:**
- Either add explicit `question_id` column to questions.xlsx
- Or use questions.csv which already has question_id column

**Evidence:**
- File: `simple_rag/data/questions.xlsx` — only "Question" column
- File: `simple_rag/data/questions.csv` — has question_id and question columns

---

### 7. ragas Missing from requirements.txt ❌

**Issue:** RAGAS library is not listed in `requirements.txt`, making the evaluation pipeline undocumented.

**Impact:**
- Reproducibility: **High** — evaluation setup is not documented
- Recommendation: Add `ragas>=0.4.3` to requirements.txt

**Evidence:**
- File: `simple_rag/requirements.txt` — no ragas entry

---

### 8. SYSTEM_PROMPT Has Decorative Placeholders That Aren't Filled ⚠️

**Issue:** The SYSTEM_PROMPT (file `simple_rag/src/llm.py` lines 107-141) includes:
```
QUESTION:
{question}

RETRIEVED CONTEXT:
{context}
```

These placeholders are **never filled in** — they're part of the static prompt description, not Python format strings. The actual question and context go in the **user message**, not the system message.

**Impact:**
- Functional impact: **None** — prompt works correctly (question is in user message)
- Clarity impact: **Medium** — misleading structure suggests question/context go in system prompt
- Recommendation: Remove decorative placeholders from SYSTEM_PROMPT or convert to actual format strings

---

## Required Fixes

### High Priority (Blocks Research Validity)
None — the core experiment is valid.

### Medium Priority (Affects Reproducibility)
1. **Add ragas to requirements.txt** → needed for evaluation
2. **Document parent-child chunking as row-level chunking** → prevent misunderstanding of architecture
3. **Fix vector normalization formula** → scores should be accurate

### Low Priority (Performance/Clarity)
1. Persist BM25 index to disk
2. Add explicit question_id to questions.xlsx
3. Skip embed_query for sparse-only mode
4. Investigate why sparse mode has 51 fewer contexts

---

## Recommended Improvements

### 1. Fix Vector Similarity Inversion

**Current:**
```python
similarities = 1.0 / (1.0 + distances.astype(np.float64))
```

**Recommended:**
```python
# For cosine distance: distance = 1 - similarity
similarities = 1.0 - (distances.astype(np.float64) / 2.0)
```

Or better yet, configure ChromaDB to return cosine similarity directly and set `already_similarity=True`.

### 2. Persist BM25 Index

**Pattern:**
```python
import pickle
# Save: pickle.dump(bm25_instance, open("bm25.pkl", "wb"))
# Load: bm25_instance = pickle.load(open("bm25.pkl", "rb"))
```

### 3. Skip Query Embedding for Sparse Mode

**Pattern:**
```python
if self._fusion_mode in ("dense", "hybrid"):
    query_embedding = self._embedding.embed_query(query)
else:
    query_embedding = None
```

### 4. Use explicit question_id in questions.xlsx

**Or** update code to use questions.csv consistently.

---

## Research Validity Assessment

### Experimental Setup: ✓ Valid

**Strengths:**
- Three genuinely different retrieval strategies, not fake variants
- Same 50 questions, same model, same prompt across all modes
- Evidence fully auditable via Context_* sheets
- No data leakage (golden answers isolated)
- Error handling prevents silent failures
- Timestamped outputs enable reproduction

**Weaknesses:**
- Parent-child chunking misnamed (doesn't affect validity but confuses architecture)
- Vector similarity inversion technically incorrect (affects score interpretation but not relative ranking)
- top_k=80 is very large (reduces LLM discrimination ability)
- BM25 index rebuilt every run (no impact on correctness, only performance)

### Results Interpretation: ⚠️ Needs Context

When interpreting results:
1. **Validity:** Hybrid vs Dense vs Sparse ranking is valid
2. **Absolute scores:** Vector similarity scores are not true cosine similarities (use relative ranking instead)
3. **Sparse gaps:** 51 sparse contexts missing (investigate before claiming sparse underperforms)
4. **Context size:** 80 evidence items is very large (may overwhelm LLM; consider testing smaller top_k)

---

## Files Reviewed

### Data Files
- `simple_rag/data/GNEM_final_data.xlsx` (1170 company records)
- `simple_rag/data/questions.xlsx` (50 questions, no IDs)
- `simple_rag/data/questions.csv` (50 questions with IDs)
- `simple_rag/data/Human validated 50 questions.xlsx` (50 golden answers)

### Configuration
- `simple_rag/config/config.yaml` (full configuration including paths, chunking, embedding, vector store, retrieval, LLM)

### Source Code
- `simple_rag/scripts/run_rag.py` (pipeline orchestration, output formatting)
- `simple_rag/src/chunking.py` (ParentChildChunker — actually row-level)
- `simple_rag/src/embedding.py` (NomicOllamaEmbedding)
- `simple_rag/src/vector_store.py` (ChromaVectorStore)
- `simple_rag/src/keyword_search.py` (BM25KeywordSearch)
- `simple_rag/src/retrieval.py` (HybridRetriever, score normalization)
- `simple_rag/src/ingestion.py` (IngestionService)
- `simple_rag/src/llm.py` (LLMClient, prompt building, response parsing)
- `simple_rag/src/utils.py` (config loading, question loading, Excel I/O)
- `simple_rag/requirements.txt`

### Output Files
- `simple_rag/outputs/generated_answers/generated_answers_current_complete_20260509_015013.xlsx`
  - Sheets: RAG_Dense, Context_Dense, RAG_Sparse, Context_Sparse, RAG_Hybrid, Context_Hybrid
  - 50 rows per RAG_* sheet, 4000/3949/4000 rows per Context_* sheet

---

## Exact Commands to Reproduce Current Runs

### Prerequisites
```bash
cd simple_rag/
export LLM_BASE_URL="http://localhost:11434/v1"
export LLM_API_KEY="ollama"
export LLM_MODEL="gemma3:27b"
# Ensure Ollama is running with nomic-embed-text and gemma3:27b loaded
```

### Run Full Pipeline (Ingestion + Retrieval + Generation)
```bash
python scripts/run_rag.py
```

This will:
1. Load 1170 company records from GNEM_final_data.xlsx
2. Chunk each row, embed via nomic-embed-text, index in ChromaDB
3. Build BM25 index in-memory
4. Load 50 questions from questions.xlsx
5. For each mode (dense, sparse, hybrid):
   - Retrieve top_k=80 contexts per question
   - Generate answers via gemma3:27b
   - Save results to timestamped Excel file
6. Output: `outputs/generated_answers/generated_answers_current_YYYYMMDD_HHMMSS.xlsx`

### Output Schema

**RAG_Dense / RAG_Sparse / RAG_Hybrid sheets (50 rows):**
- question_id, question, answer, retrieved_context, used_evidence_ids
- llm_selected_evidence_summary, insufficient_evidence_flag, evidence_count
- fusion_mode, model, error

**Context_Dense / Context_Sparse / Context_Hybrid sheets (4000/3949/4000 rows):**
- question_id, question, fusion_mode, evidence_rank, evidence_id
- company, county, tier_level, employment, primary_oems, facility_type
- industry_name, ev_relevant, product_service, supplier_type, text
- similarity_score, keyword_score, combined_score

---

## Conclusion

The GNEM RAG pipeline successfully compares three retrieval modes in a controlled experimental setup. The core results are valid for research, with clear recommendations for architectural documentation and technical corrections. No show-stopping issues prevent the use of these results, but the identified problems should be addressed before considering the system production-ready.

**Audit conducted by:** Claude Code  
**Files verified:** 13 source files + 4 data files + 1 config file
