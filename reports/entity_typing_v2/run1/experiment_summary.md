# ENTITY-TYPING-V2-001

Frozen dataset: pilot; candidates: 240.

Agreement is descriptive and does not establish correctness. Qwen confidence estimates are subjective; GLiNER scores are used only for routing.

| Method | Complete | Missing |
|---|---:|---:|
| T1 | 240 | 0 |
| T21 | 0 | 240 |
| T22 | 0 | 240 |

Comparisons (denominator: common valid candidate IDs):

- T1_vs_T21: agreement=None, n=0.
- T1_vs_T22: agreement=None, n=0.
- T21_vs_T22: agreement=None, n=0.

Hybrid policies saved: 0. Savings are planned estimates from repacked batches and byte token bounds, not measured hybrid latency.

Runtime counts include all attempts/retries. Separate thinking token counts may be unavailable. See metrics.json for timings, missing IDs, conflicts, distributions and RSS.

Independent human labels are required for precision, recall and F1; use the evaluate command after annotation.
