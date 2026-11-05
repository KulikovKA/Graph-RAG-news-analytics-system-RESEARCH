# NER-V2-004 — test

Candidate-set evaluation; new spans are unlabelled and excluded from precision/F1. Coverage and correct / candidates use all original candidates, including NON_ENTITY. Typing accuracy uses matched gold entities only; metrics.csv also gives accuracy across all matches, where matched NON_ENTITY is an error. A missed NON_ENTITY is not a correctly detected and typed entity. NON_ENTITY predictions count as false positives for the predicted type, and appear as a separate confusion-matrix row; they are not an ontology type. Missed gold entities count as FN. Macro-F1 averages types with a nonzero F1 denominator. Top score chooses one type per exact span; ties use ontology order.

| Variant | Coverage | Typing accuracy on matched entities | Correct / candidates | Macro-F1 | New spans | Inference s | Peak RSS bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0 | 0.8159 | 0.5979 | 0.4854 | 0.3695 | 119 | 81.68 | 2118991872 |
| V1 | 0.7782 | 0.6486 | 0.5021 | 0.3732 | 119 | 165.81 | 2118991872 |
| V2 | 0.6695 | 0.6478 | 0.4310 | 0.3251 | 85 | 219.64 | 2118991872 |

Predictions touching GLiNER-added punctuation are excluded; counts are in metrics.csv and details in run.log.
Peak RSS is process-wide sampled working set (10 ms), cumulative within this invocation and includes model loading; resumed times sum saved per-document inference time. New spans cannot establish full NER recall. See per_type.csv and confusion.csv.
