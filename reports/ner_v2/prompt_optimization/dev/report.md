# NER-V2-004 — dev

Candidate-set evaluation; new spans are unlabelled and excluded from precision/F1. Coverage and correct / candidates use all original candidates, including NON_ENTITY. Typing accuracy uses matched gold entities only; metrics.csv also gives accuracy across all matches, where matched NON_ENTITY is an error. A missed NON_ENTITY is not a correctly detected and typed entity. NON_ENTITY predictions count as false positives for the predicted type, and appear as a separate confusion-matrix row; they are not an ontology type. Missed gold entities count as FN. Macro-F1 averages types with a nonzero F1 denominator. Top score chooses one type per exact span; ties use ontology order.

| Variant | Coverage | Typing accuracy on matched entities | Correct / candidates | Macro-F1 | New spans | Inference s | Peak RSS bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0 | 0.8086 | 0.6244 | 0.4959 | 0.4551 | 553 | 320.50 | 2124640256 |
| V1 | 0.8014 | 0.6458 | 0.5082 | 0.4399 | 548 | 795.75 | 2124640256 |
| V2 | 0.6996 | 0.7141 | 0.4907 | 0.4410 | 497 | 1055.76 | 2124640256 |

Predictions touching GLiNER-added punctuation are excluded; counts are in metrics.csv and details in run.log.
Peak RSS is process-wide sampled working set (10 ms), cumulative within this invocation and includes model loading; resumed times sum saved per-document inference time. New spans cannot establish full NER recall. See per_type.csv and confusion.csv.
