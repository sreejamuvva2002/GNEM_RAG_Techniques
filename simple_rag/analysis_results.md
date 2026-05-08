# Simple RAG — First Run Analysis

## Commit Message

```
feat: add Simple RAG pipeline with hybrid retrieval (FAISS + BM25)

Implement a complete Retrieval-Augmented Generation pipeline for the
GNEM dataset using layered architecture and SOLID principles.

- Parent-child chunking: each row → 1 parent + 4 field-grouped children
- Embedding: nomic-embed-text via Ollama (/api/embeddings)
- Hybrid retrieval: FAISS vector search + BM25 keyword search with
  weighted score fusion (α=0.6 semantic, 0.4 keyword)
- Adapter pattern: 5 interfaces with concrete implementations for
  embedding, vector store, keyword search, data loading, result writing
- Output: 60 parent records per query across 50 questions
- Tests: 46 passing (chunker, score fusion, adapter contracts)

Tech: Python 3.10+, FAISS, rank-bm25, pandas, Ollama, numpy
```

---

## Pipeline Performance

| Metric | Value |
|---|---|
| Total runtime | **17.1 seconds** |
| Dataset | 205 rows → 205 parents + ~820 children |
| Questions processed | 50 |
| Results per question | 60 parent records |
| Output file | `output/contexts.xlsx` (50 rows × 7 columns) |

---

## Observations

### ✅ What's Working Well

1. **Full coverage** — 187 out of 205 unique companies (91%) surface at least once across all 50 queries. The retriever is not stuck on a small subset.

2. **Result diversity is strong** — Consecutive questions share only **1.7 out of 10** top companies on average. The hybrid retrieval is genuinely adapting to each query, not returning the same companies every time.

3. **Score fusion is functioning correctly** — Both BM25 and vector scores contribute meaningfully:
   - Top-1 combined scores average **0.855** (range 0.648–1.000)
   - The semantic and keyword signals are complementary, not redundant

4. **Deduplication works** — Every query returns exactly 60 *unique* parent records (no duplicates). The parent-child dedup logic (keep best-scoring child per parent) is working as designed.

5. **Candidate pool is healthy** — Queries generate 216–382 candidate chunks from the union of both retrievers, which then deduplicate down to 158–205 unique parents. The pool_size=200 setting per retriever is sufficient.

### ⚠️ Areas to Note

1. **20.8% of returned items have keyword_score=0** — These 625/3000 items were found *only* by vector search, not BM25. This is expected for semantic-only matches but worth monitoring. The hybrid approach is doing its job here — surfacing relevant results that pure keyword search would miss.

2. **Score drop-off varies by query** — Some queries have tight score ranges (Q25: gap of 0.075 between rank 1 and 60), while others have large gaps (Q50: gap of 0.332). Tight ranges suggest the query is broadly applicable; large gaps suggest a few highly relevant results mixed with padding.

3. **Context length is near-uniform** — ~32K chars per query (32,393–32,767). Since we return 60 records for every query regardless of relevance, even weak queries get padded to the same length. A dynamic top-K cutoff (e.g., stop when combined_score drops below a threshold) could improve precision.

4. **K=60 out of 205 total records = 29%** — Each query retrieves nearly a third of the entire dataset. For a small dataset like this, that's reasonable. For larger datasets, the ratio would naturally shrink.

### 📊 Sample Result Quality

**Q1**: *"Show all Tier 1/2 suppliers..."* — Top result is Daesol Material Georgia (combined=0.686). The retriever correctly surfaces supply-chain-relevant companies, with both semantic and keyword signals contributing.

**Q25**: *"Find Tier 2/3 suppliers with employment over 300..."* — Top result is Valeo (combined=0.741). Strong semantic match (0.825) boosted by decent keyword match (0.615). Note: the retriever can't *filter* by employment > 300 — it retrieves all relevant records and lets the downstream LLM do the filtering.
