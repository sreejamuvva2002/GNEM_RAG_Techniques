# Final Experiment Comparison

**Generated:** 2026-05-10 19:29

## RAGAS Summary Table

| experiment_name     | chunking_type   | retrieval_type   |   num_questions |   mean_faithfulness |   mean_answer_relevancy |   mean_context_precision |   mean_context_recall |   mean_answer_correctness |   num_errors |
|:--------------------|:----------------|:-----------------|----------------:|--------------------:|------------------------:|-------------------------:|----------------------:|--------------------------:|-------------:|
| normal_dense        | normal          | dense            |              50 |              0.5551 |                  0.5859 |                   0.0954 |                0.2732 |                    0.5307 |            0 |
| normal_sparse       | normal          | sparse           |              50 |              0.5951 |                  0.5471 |                   0.1122 |                0.388  |                    0.5195 |            0 |
| normal_hybrid       | normal          | hybrid           |              50 |              0.5249 |                  0.5939 |                   0.115  |                0.3855 |                    0.555  |            0 |
| parent_child_dense  | parent_child    | dense            |              50 |              0.6231 |                  0.576  |                   0.0967 |                0.28   |                    0.5253 |            0 |
| parent_child_sparse | parent_child    | sparse           |              50 |              0.5685 |                  0.6077 |                   0.1141 |                0.4302 |                    0.5609 |            0 |
| parent_child_hybrid | parent_child    | hybrid           |              50 |              0.5836 |                  0.5401 |                   0.0962 |                0.288  |                    0.4941 |            0 |

## Answer Correctness Summary

- **normal_dense**: sem_sim=0.810, missing_rate=12.0%
- **normal_sparse**: sem_sim=0.809, missing_rate=16.0%
- **normal_hybrid**: sem_sim=0.808, missing_rate=18.0%
- **parent_child_dense**: sem_sim=0.801, missing_rate=22.0%
- **parent_child_sparse**: sem_sim=0.806, missing_rate=22.0%
- **parent_child_hybrid**: sem_sim=0.801, missing_rate=18.0%

## Best Overall Experiment (by answer_correctness)

**parent_child_sparse**
- Chunking: parent_child
- Retrieval: sparse
- Mean answer_correctness: 0.5609

## Best Retrieval Under Normal Chunking
**normal_hybrid** (retrieval: hybrid)

## Best Retrieval Under Parent-Child Chunking
**parent_child_sparse** (retrieval: sparse)

## Limitations

- RAGAS metrics are LLM-as-judge evaluations and inherit LLM biases.
- Sparse retrieval returned fewer than 80 contexts for some questions (BM25 `score > 0.0` filter). This may affect sparse performance metrics.
- Semantic similarity uses nomic-embed-text embeddings and may not capture all aspects of answer quality.
- Parent-child chunking uses 3× more indexed chunks (~3510 vs ~1170), which may affect retrieval recall independently of chunk quality.
