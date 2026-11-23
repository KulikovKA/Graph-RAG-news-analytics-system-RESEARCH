# Stage 3A.5.8 — Neural & Hybrid Graph-RAG Reranking

Status **COMPLETE**. Frozen B, budget 20. Latency covers reranking only. Model load and warmup are separate.

| Model | Size | Representation | Policy | N | Recall μ | Recall macro | Factual μ | Δ factual μ paired | All docs | Overlap μ | p50 s | p95 s | Pairs/s inference | Peak RSS MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heuristic_C | Baseline | — | heuristic_C | 35/35 | 0.893 | 0.919 | 0.893 | — | 0.829 | 0.888 | 0.107 | 0.180 | — | — |
| ettin-68m | Small | fact_focused | neural_only | 35/35 | 0.696 | 0.767 | 0.696 | -0.196 | 0.543 | 0.710 | 6.249 | 8.981 | 3.342 | 1981.262 |
| ettin-68m | Small | fact_focused | hybrid_C_neural | 35/35 | 0.786 | 0.838 | 0.786 | -0.107 | 0.686 | 0.757 | 6.249 | 8.981 | 3.342 | 1981.262 |
| ettin-68m | Small | document_aware | neural_only | 35/35 | 0.679 | 0.752 | 0.679 | -0.214 | 0.514 | 0.673 | 6.301 | 9.098 | 3.271 | 1981.262 |
| ettin-68m | Small | document_aware | hybrid_C_neural | 35/35 | 0.768 | 0.824 | 0.768 | -0.125 | 0.657 | 0.748 | 6.301 | 9.098 | 3.271 | 1981.262 |
| minilm-l6 | Small | fact_focused | neural_only | 35/35 | 0.661 | 0.738 | 0.661 | -0.232 | 0.486 | 0.692 | 2.040 | 2.680 | 12.476 | 3290.738 |
| minilm-l6 | Small | fact_focused | hybrid_C_neural | 35/35 | 0.750 | 0.805 | 0.750 | -0.143 | 0.600 | 0.757 | 2.040 | 2.680 | 12.476 | 3290.738 |
| minilm-l6 | Small | document_aware | neural_only | 35/35 | 0.661 | 0.738 | 0.661 | -0.232 | 0.486 | 0.692 | 2.110 | 2.765 | 12.206 | 3290.738 |
| minilm-l6 | Small | document_aware | hybrid_C_neural | 35/35 | 0.732 | 0.790 | 0.732 | -0.161 | 0.571 | 0.748 | 2.110 | 2.765 | 12.206 | 3290.738 |
| ettin-400m | Base | fact_focused | neural_only | 35/35 | 0.679 | 0.748 | 0.679 | -0.214 | 0.486 | 0.710 | 38.474 | 51.992 | 0.517 | 3279.102 |
| ettin-400m | Base | fact_focused | hybrid_C_neural | 35/35 | 0.821 | 0.862 | 0.821 | -0.071 | 0.714 | 0.804 | 38.474 | 51.992 | 0.517 | 3279.102 |
| ettin-400m | Base | document_aware | neural_only | 35/35 | 0.679 | 0.748 | 0.679 | -0.214 | 0.486 | 0.710 | 45.339 | 55.771 | 0.460 | 3279.102 |
| ettin-400m | Base | document_aware | hybrid_C_neural | 35/35 | 0.786 | 0.833 | 0.786 | -0.107 | 0.657 | 0.785 | 45.339 | 55.771 | 0.460 | 3279.102 |
| mxbai-base | Base | fact_focused | neural_only | 35/35 | 0.679 | 0.748 | 0.679 | -0.214 | 0.486 | 0.710 | 47.153 | 59.077 | 0.423 | 4002.078 |
| mxbai-base | Base | fact_focused | hybrid_C_neural | 35/35 | 0.804 | 0.848 | 0.804 | -0.089 | 0.686 | 0.785 | 47.153 | 59.077 | 0.423 | 4002.078 |
| mxbai-base | Base | document_aware | neural_only | 35/35 | 0.696 | 0.762 | 0.696 | -0.196 | 0.514 | 0.720 | 50.453 | 56.067 | 0.415 | 4002.078 |
| mxbai-base | Base | document_aware | hybrid_C_neural | 35/35 | 0.821 | 0.862 | 0.821 | -0.071 | 0.714 | 0.804 | 50.453 | 56.067 | 0.415 | 4002.078 |
| bge-m3 | Large | fact_focused | neural_only | 35/35 | 0.696 | 0.767 | 0.696 | -0.196 | 0.543 | 0.720 | 39.357 | 54.270 | 0.505 | 2893.672 |
| bge-m3 | Large | fact_focused | hybrid_C_neural | 35/35 | 0.732 | 0.790 | 0.732 | -0.161 | 0.571 | 0.748 | 39.357 | 54.270 | 0.505 | 2893.672 |
| bge-m3 | Large | document_aware | neural_only | 35/35 | 0.696 | 0.767 | 0.696 | -0.196 | 0.543 | 0.720 | 40.397 | 51.138 | 0.508 | 2893.672 |
| bge-m3 | Large | document_aware | hybrid_C_neural | 35/35 | 0.750 | 0.805 | 0.750 | -0.143 | 0.600 | 0.757 | 40.397 | 51.138 | 0.508 | 2893.672 |
| qwen3-0.6b | Large | fact_focused | neural_only | 35/35 | 0.696 | 0.767 | 0.696 | -0.196 | 0.543 | 0.710 | 71.461 | 86.375 | 0.290 | 4981.887 |
| qwen3-0.6b | Large | fact_focused | hybrid_C_neural | 35/35 | 0.821 | 0.862 | 0.821 | -0.071 | 0.714 | 0.822 | 71.461 | 86.375 | 0.290 | 4981.887 |
| qwen3-0.6b | Large | document_aware | neural_only | 35/35 | 0.679 | 0.752 | 0.679 | -0.214 | 0.514 | 0.701 | 68.362 | 85.310 | 0.288 | 4981.887 |
| qwen3-0.6b | Large | document_aware | hybrid_C_neural | 35/35 | 0.821 | 0.862 | 0.821 | -0.071 | 0.714 | 0.822 | 68.362 | 85.310 | 0.288 | 4981.887 |

## Candidate oracle

| Group | Budget | Questions | Gold μ | Factual Gold μ |
| --- | --- | --- | --- | --- |
| ALL | 20 | 35 | 0.893 | 0.893 |
| ALL | 50 | 35 | 0.911 | 0.911 |
| ALL | 100 | 35 | 0.911 | 0.911 |
| ALL | 200 | 35 | 0.911 | 0.911 |
| APPROVED | 20 | 25 | 0.889 | 0.889 |
| APPROVED | 50 | 25 | 0.917 | 0.917 |
| APPROVED | 100 | 25 | 0.917 | 0.917 |
| APPROVED | 200 | 25 | 0.917 | 0.917 |
| A | 20 | 8 | 1.000 | 1.000 |
| A | 50 | 8 | 1.000 | 1.000 |
| A | 100 | 8 | 1.000 | 1.000 |
| A | 200 | 8 | 1.000 | 1.000 |
| B | 20 | 7 | 1.000 | 1.000 |
| B | 50 | 7 | 1.000 | 1.000 |
| B | 100 | 7 | 1.000 | 1.000 |
| B | 200 | 7 | 1.000 | 1.000 |
| C | 20 | 10 | 0.810 | 0.810 |
| C | 50 | 10 | 0.857 | 0.857 |
| C | 100 | 10 | 0.857 | 0.857 |
| C | 200 | 10 | 0.857 | 0.857 |
| HOLDOUT | 20 | 10 | 0.900 | 0.900 |
| HOLDOUT | 50 | 10 | 0.900 | 0.900 |
| HOLDOUT | 100 | 10 | 0.900 | 0.900 |
| HOLDOUT | 200 | 10 | 0.900 | 0.900 |

## C03, C05, C07 and C09

Approved C and historical HOLDOUT are shown under their respective keys. Rank changes refer to the first selected path with validated factual evidence for the document.

- **C/3A52-C03** — C factual hits ['W2076293586', 'W3138328568'], missing from full eligible pool ['W2039836112'], off-Gold sources ['W1995028339', 'W3123056951'].
- **C/3A52-C05** — C factual hits ['W4309030044'], missing from full eligible pool [], off-Gold sources [].
- **C/3A52-C07** — C factual hits ['W4381572413'], missing from full eligible pool ['W2326480042'], off-Gold sources ['W2761455282', 'W4391351014'].
- **C/3A52-C09** — C factual hits ['W3081936674'], missing from full eligible pool ['W2084709926'], off-Gold sources ['W2791492120', 'W3027457624'].
- **HOLDOUT/3A52-C03** — C factual hits ['W2084709926', 'W2096307598'], missing from full eligible pool [], off-Gold sources ['W3138328568'].
- **HOLDOUT/3A52-C05** — C factual hits ['W4381572413'], missing from full eligible pool ['W2326480042'], off-Gold sources [].
- **HOLDOUT/3A52-C07** — C factual hits ['W2761455282', 'W4391351014'], missing from full eligible pool [], off-Gold sources [].
- **HOLDOUT/3A52-C09** — C factual hits ['W2130236240', 'W2906067930'], missing from full eligible pool [], off-Gold sources ['W2317432785'].

| Question | Model / repr / policy | Promoted | Lost | Absent at budget | Gold ranks C → new | +/− paths |
| --- | --- | --- | --- | --- | --- | --- |
| C/3A52-C03 | ettin-68m / fact_focused / neural_only |  | W3138328568 | W2039836112 | W2039836112 —→—, W2076293586 1→1, W3138328568 2→— | +3/−3 |
| C/3A52-C03 | ettin-68m / fact_focused / hybrid_C_neural |  | W3138328568 | W2039836112 | W2039836112 —→—, W2076293586 1→1, W3138328568 2→— | +2/−2 |
| C/3A52-C05 | ettin-68m / fact_focused / neural_only |  |  | W2906067930 | W2906067930 —→—, W4309030044 1→1 | +1/−1 |
| C/3A52-C05 | ettin-68m / fact_focused / hybrid_C_neural |  |  | W2906067930 | W2906067930 —→—, W4309030044 1→1 | +1/−1 |
| C/3A52-C07 | ettin-68m / fact_focused / neural_only |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +2/−2 |
| C/3A52-C07 | ettin-68m / fact_focused / hybrid_C_neural |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +2/−2 |
| C/3A52-C09 | ettin-68m / fact_focused / neural_only |  |  | W2084709926 | W2084709926 —→—, W3081936674 1→1 | +5/−5 |
| C/3A52-C09 | ettin-68m / fact_focused / hybrid_C_neural |  |  | W2084709926 | W2084709926 —→—, W3081936674 1→1 | +3/−3 |
| HOLDOUT/3A52-C03 | ettin-68m / fact_focused / neural_only |  | W2096307598 |  | W2084709926 1→1, W2096307598 2→— | +3/−3 |
| HOLDOUT/3A52-C03 | ettin-68m / fact_focused / hybrid_C_neural |  |  |  | W2084709926 1→1, W2096307598 2→3 | +1/−1 |
| HOLDOUT/3A52-C05 | ettin-68m / fact_focused / neural_only |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +2/−2 |
| HOLDOUT/3A52-C05 | ettin-68m / fact_focused / hybrid_C_neural |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +1/−1 |
| HOLDOUT/3A52-C07 | ettin-68m / fact_focused / neural_only |  | W2761455282 |  | W2761455282 2→—, W4391351014 1→1 | +5/−5 |
| HOLDOUT/3A52-C07 | ettin-68m / fact_focused / hybrid_C_neural |  |  |  | W2761455282 2→2, W4391351014 1→1 | +2/−2 |
| HOLDOUT/3A52-C09 | ettin-68m / fact_focused / neural_only |  |  |  | W2130236240 2→1, W2906067930 1→2 | +3/−3 |
| HOLDOUT/3A52-C09 | ettin-68m / fact_focused / hybrid_C_neural |  |  |  | W2130236240 2→1, W2906067930 1→2 | +3/−3 |
| C/3A52-C03 | ettin-68m / document_aware / neural_only |  | W3138328568 | W2039836112 | W2039836112 —→—, W2076293586 1→1, W3138328568 2→— | +3/−3 |
| C/3A52-C03 | ettin-68m / document_aware / hybrid_C_neural |  | W3138328568 | W2039836112 | W2039836112 —→—, W2076293586 1→1, W3138328568 2→— | +2/−2 |
| C/3A52-C05 | ettin-68m / document_aware / neural_only |  |  | W2906067930 | W2906067930 —→—, W4309030044 1→1 | +1/−1 |
| C/3A52-C05 | ettin-68m / document_aware / hybrid_C_neural |  |  | W2906067930 | W2906067930 —→—, W4309030044 1→1 | +1/−1 |
| C/3A52-C07 | ettin-68m / document_aware / neural_only |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +2/−2 |
| C/3A52-C07 | ettin-68m / document_aware / hybrid_C_neural |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +2/−2 |
| C/3A52-C09 | ettin-68m / document_aware / neural_only |  |  | W2084709926 | W2084709926 —→—, W3081936674 1→1 | +4/−4 |
| C/3A52-C09 | ettin-68m / document_aware / hybrid_C_neural |  |  | W2084709926 | W2084709926 —→—, W3081936674 1→1 | +3/−3 |
| HOLDOUT/3A52-C03 | ettin-68m / document_aware / neural_only |  | W2096307598 |  | W2084709926 1→1, W2096307598 2→— | +2/−2 |
| HOLDOUT/3A52-C03 | ettin-68m / document_aware / hybrid_C_neural |  |  |  | W2084709926 1→1, W2096307598 2→3 | +1/−1 |
| HOLDOUT/3A52-C05 | ettin-68m / document_aware / neural_only |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +3/−3 |
| HOLDOUT/3A52-C05 | ettin-68m / document_aware / hybrid_C_neural |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +1/−1 |
| HOLDOUT/3A52-C07 | ettin-68m / document_aware / neural_only |  |  |  | W2761455282 2→4, W4391351014 1→1 | +4/−4 |
| HOLDOUT/3A52-C07 | ettin-68m / document_aware / hybrid_C_neural |  |  |  | W2761455282 2→1, W4391351014 1→2 | +2/−2 |
| HOLDOUT/3A52-C09 | ettin-68m / document_aware / neural_only |  |  |  | W2130236240 2→2, W2906067930 1→1 | +3/−3 |
| HOLDOUT/3A52-C09 | ettin-68m / document_aware / hybrid_C_neural |  |  |  | W2130236240 2→2, W2906067930 1→1 | +2/−2 |
| C/3A52-C03 | ettin-400m / fact_focused / neural_only |  |  | W2039836112 | W2039836112 —→—, W2076293586 1→1, W3138328568 2→2 | +3/−3 |
| C/3A52-C03 | ettin-400m / fact_focused / hybrid_C_neural |  |  | W2039836112 | W2039836112 —→—, W2076293586 1→1, W3138328568 2→2 | +3/−3 |
| C/3A52-C05 | ettin-400m / fact_focused / neural_only |  |  | W2906067930 | W2906067930 —→—, W4309030044 1→1 | +2/−2 |
| C/3A52-C05 | ettin-400m / fact_focused / hybrid_C_neural |  |  | W2906067930 | W2906067930 —→—, W4309030044 1→1 | +1/−1 |
| C/3A52-C07 | ettin-400m / fact_focused / neural_only |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +2/−2 |
| C/3A52-C07 | ettin-400m / fact_focused / hybrid_C_neural |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +2/−2 |
| C/3A52-C09 | ettin-400m / fact_focused / neural_only |  |  | W2084709926 | W2084709926 —→—, W3081936674 1→1 | +4/−4 |
| C/3A52-C09 | ettin-400m / fact_focused / hybrid_C_neural |  |  | W2084709926 | W2084709926 —→—, W3081936674 1→1 | +3/−3 |
| HOLDOUT/3A52-C03 | ettin-400m / fact_focused / neural_only |  | W2096307598 |  | W2084709926 1→1, W2096307598 2→— | +3/−3 |
| HOLDOUT/3A52-C03 | ettin-400m / fact_focused / hybrid_C_neural |  |  |  | W2084709926 1→1, W2096307598 2→4 | +1/−1 |
| HOLDOUT/3A52-C05 | ettin-400m / fact_focused / neural_only |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +1/−1 |
| HOLDOUT/3A52-C05 | ettin-400m / fact_focused / hybrid_C_neural |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +1/−1 |
| HOLDOUT/3A52-C07 | ettin-400m / fact_focused / neural_only |  | W2761455282 |  | W2761455282 2→—, W4391351014 1→1 | +5/−5 |
| HOLDOUT/3A52-C07 | ettin-400m / fact_focused / hybrid_C_neural |  |  |  | W2761455282 2→4, W4391351014 1→1 | +3/−3 |
| HOLDOUT/3A52-C09 | ettin-400m / fact_focused / neural_only |  |  |  | W2130236240 2→4, W2906067930 1→1 | +4/−4 |
| HOLDOUT/3A52-C09 | ettin-400m / fact_focused / hybrid_C_neural |  |  |  | W2130236240 2→2, W2906067930 1→1 | +2/−2 |
| C/3A52-C03 | ettin-400m / document_aware / neural_only |  |  | W2039836112 | W2039836112 —→—, W2076293586 1→1, W3138328568 2→5 | +3/−3 |
| C/3A52-C03 | ettin-400m / document_aware / hybrid_C_neural |  |  | W2039836112 | W2039836112 —→—, W2076293586 1→1, W3138328568 2→5 | +2/−2 |
| C/3A52-C05 | ettin-400m / document_aware / neural_only |  |  | W2906067930 | W2906067930 —→—, W4309030044 1→1 | +1/−1 |
| C/3A52-C05 | ettin-400m / document_aware / hybrid_C_neural |  |  | W2906067930 | W2906067930 —→—, W4309030044 1→1 | +1/−1 |
| C/3A52-C07 | ettin-400m / document_aware / neural_only |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +2/−2 |
| C/3A52-C07 | ettin-400m / document_aware / hybrid_C_neural |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +2/−2 |
| C/3A52-C09 | ettin-400m / document_aware / neural_only |  |  | W2084709926 | W2084709926 —→—, W3081936674 1→1 | +3/−3 |
| C/3A52-C09 | ettin-400m / document_aware / hybrid_C_neural |  |  | W2084709926 | W2084709926 —→—, W3081936674 1→1 | +3/−3 |
| HOLDOUT/3A52-C03 | ettin-400m / document_aware / neural_only |  | W2096307598 |  | W2084709926 1→1, W2096307598 2→— | +3/−3 |
| HOLDOUT/3A52-C03 | ettin-400m / document_aware / hybrid_C_neural |  | W2096307598 |  | W2084709926 1→1, W2096307598 2→— | +2/−2 |
| HOLDOUT/3A52-C05 | ettin-400m / document_aware / neural_only |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +3/−3 |
| HOLDOUT/3A52-C05 | ettin-400m / document_aware / hybrid_C_neural |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +1/−1 |
| HOLDOUT/3A52-C07 | ettin-400m / document_aware / neural_only |  | W2761455282 |  | W2761455282 2→—, W4391351014 1→1 | +5/−5 |
| HOLDOUT/3A52-C07 | ettin-400m / document_aware / hybrid_C_neural |  |  |  | W2761455282 2→5, W4391351014 1→1 | +3/−3 |
| HOLDOUT/3A52-C09 | ettin-400m / document_aware / neural_only |  |  |  | W2130236240 2→4, W2906067930 1→1 | +4/−4 |
| HOLDOUT/3A52-C09 | ettin-400m / document_aware / hybrid_C_neural |  |  |  | W2130236240 2→2, W2906067930 1→1 | +2/−2 |
| C/3A52-C03 | bge-m3 / fact_focused / neural_only |  | W3138328568 | W2039836112 | W2039836112 —→—, W2076293586 1→1, W3138328568 2→— | +3/−3 |
| C/3A52-C03 | bge-m3 / fact_focused / hybrid_C_neural |  |  | W2039836112 | W2039836112 —→—, W2076293586 1→1, W3138328568 2→5 | +2/−2 |
| C/3A52-C05 | bge-m3 / fact_focused / neural_only |  |  | W2906067930 | W2906067930 —→—, W4309030044 1→1 | +4/−4 |
| C/3A52-C05 | bge-m3 / fact_focused / hybrid_C_neural |  |  | W2906067930 | W2906067930 —→—, W4309030044 1→1 | +3/−3 |
| C/3A52-C07 | bge-m3 / fact_focused / neural_only |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +2/−2 |
| C/3A52-C07 | bge-m3 / fact_focused / hybrid_C_neural |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +2/−2 |
| C/3A52-C09 | bge-m3 / fact_focused / neural_only |  |  | W2084709926 | W2084709926 —→—, W3081936674 1→1 | +3/−3 |
| C/3A52-C09 | bge-m3 / fact_focused / hybrid_C_neural |  |  | W2084709926 | W2084709926 —→—, W3081936674 1→1 | +3/−3 |
| HOLDOUT/3A52-C03 | bge-m3 / fact_focused / neural_only |  |  |  | W2084709926 1→3, W2096307598 2→1 | +4/−4 |
| HOLDOUT/3A52-C03 | bge-m3 / fact_focused / hybrid_C_neural |  | W2096307598 |  | W2084709926 1→1, W2096307598 2→— | +2/−2 |
| HOLDOUT/3A52-C05 | bge-m3 / fact_focused / neural_only |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +1/−1 |
| HOLDOUT/3A52-C05 | bge-m3 / fact_focused / hybrid_C_neural |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +1/−1 |
| HOLDOUT/3A52-C07 | bge-m3 / fact_focused / neural_only |  | W2761455282 |  | W2761455282 2→—, W4391351014 1→1 | +4/−4 |
| HOLDOUT/3A52-C07 | bge-m3 / fact_focused / hybrid_C_neural |  | W2761455282 |  | W2761455282 2→—, W4391351014 1→1 | +2/−2 |
| HOLDOUT/3A52-C09 | bge-m3 / fact_focused / neural_only |  | W2130236240 |  | W2130236240 2→—, W2906067930 1→1 | +3/−3 |
| HOLDOUT/3A52-C09 | bge-m3 / fact_focused / hybrid_C_neural |  | W2130236240 |  | W2130236240 2→—, W2906067930 1→1 | +3/−3 |
| C/3A52-C03 | bge-m3 / document_aware / neural_only |  | W3138328568 | W2039836112 | W2039836112 —→—, W2076293586 1→1, W3138328568 2→— | +3/−3 |
| C/3A52-C03 | bge-m3 / document_aware / hybrid_C_neural |  |  | W2039836112 | W2039836112 —→—, W2076293586 1→1, W3138328568 2→5 | +2/−2 |
| C/3A52-C05 | bge-m3 / document_aware / neural_only |  |  | W2906067930 | W2906067930 —→—, W4309030044 1→1 | +4/−4 |
| C/3A52-C05 | bge-m3 / document_aware / hybrid_C_neural |  |  | W2906067930 | W2906067930 —→—, W4309030044 1→1 | +2/−2 |
| C/3A52-C07 | bge-m3 / document_aware / neural_only |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +3/−3 |
| C/3A52-C07 | bge-m3 / document_aware / hybrid_C_neural |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +2/−2 |
| C/3A52-C09 | bge-m3 / document_aware / neural_only |  |  | W2084709926 | W2084709926 —→—, W3081936674 1→1 | +3/−3 |
| C/3A52-C09 | bge-m3 / document_aware / hybrid_C_neural |  |  | W2084709926 | W2084709926 —→—, W3081936674 1→1 | +1/−1 |
| HOLDOUT/3A52-C03 | bge-m3 / document_aware / neural_only |  |  |  | W2084709926 1→4, W2096307598 2→1 | +3/−3 |
| HOLDOUT/3A52-C03 | bge-m3 / document_aware / hybrid_C_neural |  |  |  | W2084709926 1→2, W2096307598 2→1 | +2/−2 |
| HOLDOUT/3A52-C05 | bge-m3 / document_aware / neural_only |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +3/−3 |
| HOLDOUT/3A52-C05 | bge-m3 / document_aware / hybrid_C_neural |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +1/−1 |
| HOLDOUT/3A52-C07 | bge-m3 / document_aware / neural_only |  | W2761455282 |  | W2761455282 2→—, W4391351014 1→1 | +4/−4 |
| HOLDOUT/3A52-C07 | bge-m3 / document_aware / hybrid_C_neural |  | W2761455282 |  | W2761455282 2→—, W4391351014 1→1 | +2/−2 |
| HOLDOUT/3A52-C09 | bge-m3 / document_aware / neural_only |  | W2130236240 |  | W2130236240 2→—, W2906067930 1→1 | +3/−3 |
| HOLDOUT/3A52-C09 | bge-m3 / document_aware / hybrid_C_neural |  | W2130236240 |  | W2130236240 2→—, W2906067930 1→1 | +2/−2 |
| C/3A52-C03 | mxbai-base / fact_focused / neural_only |  |  | W2039836112 | W2039836112 —→—, W2076293586 1→1, W3138328568 2→5 | +3/−3 |
| C/3A52-C03 | mxbai-base / fact_focused / hybrid_C_neural |  |  | W2039836112 | W2039836112 —→—, W2076293586 1→1, W3138328568 2→5 | +2/−2 |
| C/3A52-C05 | mxbai-base / fact_focused / neural_only |  |  | W2906067930 | W2906067930 —→—, W4309030044 1→1 | +4/−4 |
| C/3A52-C05 | mxbai-base / fact_focused / hybrid_C_neural |  |  | W2906067930 | W2906067930 —→—, W4309030044 1→1 | +4/−4 |
| C/3A52-C07 | mxbai-base / fact_focused / neural_only |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +3/−3 |
| C/3A52-C07 | mxbai-base / fact_focused / hybrid_C_neural |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +2/−2 |
| C/3A52-C09 | mxbai-base / fact_focused / neural_only |  |  | W2084709926 | W2084709926 —→—, W3081936674 1→1 | +4/−4 |
| C/3A52-C09 | mxbai-base / fact_focused / hybrid_C_neural |  |  | W2084709926 | W2084709926 —→—, W3081936674 1→1 | +1/−1 |
| HOLDOUT/3A52-C03 | mxbai-base / fact_focused / neural_only |  | W2096307598 |  | W2084709926 1→1, W2096307598 2→— | +3/−3 |
| HOLDOUT/3A52-C03 | mxbai-base / fact_focused / hybrid_C_neural |  |  |  | W2084709926 1→1, W2096307598 2→5 | +2/−2 |
| HOLDOUT/3A52-C05 | mxbai-base / fact_focused / neural_only |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +4/−4 |
| HOLDOUT/3A52-C05 | mxbai-base / fact_focused / hybrid_C_neural |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +3/−3 |
| HOLDOUT/3A52-C07 | mxbai-base / fact_focused / neural_only |  | W2761455282 |  | W2761455282 2→—, W4391351014 1→1 | +2/−2 |
| HOLDOUT/3A52-C07 | mxbai-base / fact_focused / hybrid_C_neural |  |  |  | W2761455282 2→5, W4391351014 1→1 | +0/−0 |
| HOLDOUT/3A52-C09 | mxbai-base / fact_focused / neural_only |  |  |  | W2130236240 2→1, W2906067930 1→2 | +3/−3 |
| HOLDOUT/3A52-C09 | mxbai-base / fact_focused / hybrid_C_neural |  |  |  | W2130236240 2→1, W2906067930 1→2 | +2/−2 |
| C/3A52-C03 | mxbai-base / document_aware / neural_only |  |  | W2039836112 | W2039836112 —→—, W2076293586 1→1, W3138328568 2→5 | +3/−3 |
| C/3A52-C03 | mxbai-base / document_aware / hybrid_C_neural |  |  | W2039836112 | W2039836112 —→—, W2076293586 1→1, W3138328568 2→5 | +2/−2 |
| C/3A52-C05 | mxbai-base / document_aware / neural_only |  |  | W2906067930 | W2906067930 —→—, W4309030044 1→1 | +4/−4 |
| C/3A52-C05 | mxbai-base / document_aware / hybrid_C_neural |  |  | W2906067930 | W2906067930 —→—, W4309030044 1→1 | +3/−3 |
| C/3A52-C07 | mxbai-base / document_aware / neural_only |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +4/−4 |
| C/3A52-C07 | mxbai-base / document_aware / hybrid_C_neural |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +2/−2 |
| C/3A52-C09 | mxbai-base / document_aware / neural_only |  |  | W2084709926 | W2084709926 —→—, W3081936674 1→1 | +3/−3 |
| C/3A52-C09 | mxbai-base / document_aware / hybrid_C_neural |  |  | W2084709926 | W2084709926 —→—, W3081936674 1→1 | +1/−1 |
| HOLDOUT/3A52-C03 | mxbai-base / document_aware / neural_only |  | W2084709926 |  | W2084709926 1→—, W2096307598 2→1 | +4/−4 |
| HOLDOUT/3A52-C03 | mxbai-base / document_aware / hybrid_C_neural |  |  |  | W2084709926 1→3, W2096307598 2→1 | +2/−2 |
| HOLDOUT/3A52-C05 | mxbai-base / document_aware / neural_only |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +2/−2 |
| HOLDOUT/3A52-C05 | mxbai-base / document_aware / hybrid_C_neural |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +1/−1 |
| HOLDOUT/3A52-C07 | mxbai-base / document_aware / neural_only |  | W2761455282 |  | W2761455282 2→—, W4391351014 1→1 | +2/−2 |
| HOLDOUT/3A52-C07 | mxbai-base / document_aware / hybrid_C_neural |  | W2761455282 |  | W2761455282 2→—, W4391351014 1→1 | +1/−1 |
| HOLDOUT/3A52-C09 | mxbai-base / document_aware / neural_only |  |  |  | W2130236240 2→1, W2906067930 1→2 | +3/−3 |
| HOLDOUT/3A52-C09 | mxbai-base / document_aware / hybrid_C_neural |  |  |  | W2130236240 2→1, W2906067930 1→2 | +1/−1 |
| C/3A52-C03 | minilm-l6 / fact_focused / neural_only |  | W3138328568 | W2039836112 | W2039836112 —→—, W2076293586 1→1, W3138328568 2→— | +3/−3 |
| C/3A52-C03 | minilm-l6 / fact_focused / hybrid_C_neural |  |  | W2039836112 | W2039836112 —→—, W2076293586 1→1, W3138328568 2→4 | +2/−2 |
| C/3A52-C05 | minilm-l6 / fact_focused / neural_only |  |  | W2906067930 | W2906067930 —→—, W4309030044 1→1 | +2/−2 |
| C/3A52-C05 | minilm-l6 / fact_focused / hybrid_C_neural |  |  | W2906067930 | W2906067930 —→—, W4309030044 1→1 | +1/−1 |
| C/3A52-C07 | minilm-l6 / fact_focused / neural_only |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +2/−2 |
| C/3A52-C07 | minilm-l6 / fact_focused / hybrid_C_neural |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +2/−2 |
| C/3A52-C09 | minilm-l6 / fact_focused / neural_only |  |  | W2084709926 | W2084709926 —→—, W3081936674 1→1 | +3/−3 |
| C/3A52-C09 | minilm-l6 / fact_focused / hybrid_C_neural |  |  | W2084709926 | W2084709926 —→—, W3081936674 1→1 | +0/−0 |
| HOLDOUT/3A52-C03 | minilm-l6 / fact_focused / neural_only |  | W2096307598 |  | W2084709926 1→1, W2096307598 2→— | +2/−2 |
| HOLDOUT/3A52-C03 | minilm-l6 / fact_focused / hybrid_C_neural |  |  |  | W2084709926 1→1, W2096307598 2→4 | +1/−1 |
| HOLDOUT/3A52-C05 | minilm-l6 / fact_focused / neural_only |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +3/−3 |
| HOLDOUT/3A52-C05 | minilm-l6 / fact_focused / hybrid_C_neural |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +2/−2 |
| HOLDOUT/3A52-C07 | minilm-l6 / fact_focused / neural_only |  |  |  | W2761455282 2→1, W4391351014 1→2 | +1/−1 |
| HOLDOUT/3A52-C07 | minilm-l6 / fact_focused / hybrid_C_neural |  |  |  | W2761455282 2→1, W4391351014 1→2 | +0/−0 |
| HOLDOUT/3A52-C09 | minilm-l6 / fact_focused / neural_only |  | W2130236240 |  | W2130236240 2→—, W2906067930 1→1 | +4/−4 |
| HOLDOUT/3A52-C09 | minilm-l6 / fact_focused / hybrid_C_neural |  | W2130236240 |  | W2130236240 2→—, W2906067930 1→1 | +2/−2 |
| C/3A52-C03 | minilm-l6 / document_aware / neural_only |  | W3138328568 | W2039836112 | W2039836112 —→—, W2076293586 1→1, W3138328568 2→— | +3/−3 |
| C/3A52-C03 | minilm-l6 / document_aware / hybrid_C_neural |  |  | W2039836112 | W2039836112 —→—, W2076293586 1→1, W3138328568 2→4 | +2/−2 |
| C/3A52-C05 | minilm-l6 / document_aware / neural_only |  |  | W2906067930 | W2906067930 —→—, W4309030044 1→1 | +4/−4 |
| C/3A52-C05 | minilm-l6 / document_aware / hybrid_C_neural |  |  | W2906067930 | W2906067930 —→—, W4309030044 1→1 | +2/−2 |
| C/3A52-C07 | minilm-l6 / document_aware / neural_only |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +4/−4 |
| C/3A52-C07 | minilm-l6 / document_aware / hybrid_C_neural |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +2/−2 |
| C/3A52-C09 | minilm-l6 / document_aware / neural_only |  |  | W2084709926 | W2084709926 —→—, W3081936674 1→1 | +1/−1 |
| C/3A52-C09 | minilm-l6 / document_aware / hybrid_C_neural |  |  | W2084709926 | W2084709926 —→—, W3081936674 1→1 | +0/−0 |
| HOLDOUT/3A52-C03 | minilm-l6 / document_aware / neural_only |  | W2096307598 |  | W2084709926 1→1, W2096307598 2→— | +2/−2 |
| HOLDOUT/3A52-C03 | minilm-l6 / document_aware / hybrid_C_neural |  |  |  | W2084709926 1→1, W2096307598 2→4 | +1/−1 |
| HOLDOUT/3A52-C05 | minilm-l6 / document_aware / neural_only |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +5/−5 |
| HOLDOUT/3A52-C05 | minilm-l6 / document_aware / hybrid_C_neural |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +2/−2 |
| HOLDOUT/3A52-C07 | minilm-l6 / document_aware / neural_only |  |  |  | W2761455282 2→1, W4391351014 1→2 | +0/−0 |
| HOLDOUT/3A52-C07 | minilm-l6 / document_aware / hybrid_C_neural |  |  |  | W2761455282 2→1, W4391351014 1→2 | +0/−0 |
| HOLDOUT/3A52-C09 | minilm-l6 / document_aware / neural_only |  | W2130236240 |  | W2130236240 2→—, W2906067930 1→1 | +4/−4 |
| HOLDOUT/3A52-C09 | minilm-l6 / document_aware / hybrid_C_neural |  | W2130236240 |  | W2130236240 2→—, W2906067930 1→1 | +2/−2 |
| C/3A52-C03 | qwen3-0.6b / fact_focused / neural_only |  | W3138328568 | W2039836112 | W2039836112 —→—, W2076293586 1→1, W3138328568 2→— | +3/−3 |
| C/3A52-C03 | qwen3-0.6b / fact_focused / hybrid_C_neural |  |  | W2039836112 | W2039836112 —→—, W2076293586 1→1, W3138328568 2→3 | +2/−2 |
| C/3A52-C05 | qwen3-0.6b / fact_focused / neural_only |  |  | W2906067930 | W2906067930 —→—, W4309030044 1→1 | +5/−5 |
| C/3A52-C05 | qwen3-0.6b / fact_focused / hybrid_C_neural |  |  | W2906067930 | W2906067930 —→—, W4309030044 1→1 | +1/−1 |
| C/3A52-C07 | qwen3-0.6b / fact_focused / neural_only |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +3/−3 |
| C/3A52-C07 | qwen3-0.6b / fact_focused / hybrid_C_neural |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +2/−2 |
| C/3A52-C09 | qwen3-0.6b / fact_focused / neural_only |  |  | W2084709926 | W2084709926 —→—, W3081936674 1→1 | +5/−5 |
| C/3A52-C09 | qwen3-0.6b / fact_focused / hybrid_C_neural |  |  | W2084709926 | W2084709926 —→—, W3081936674 1→1 | +3/−3 |
| HOLDOUT/3A52-C03 | qwen3-0.6b / fact_focused / neural_only |  | W2084709926 |  | W2084709926 1→—, W2096307598 2→1 | +4/−4 |
| HOLDOUT/3A52-C03 | qwen3-0.6b / fact_focused / hybrid_C_neural |  |  |  | W2084709926 1→4, W2096307598 2→1 | +3/−3 |
| HOLDOUT/3A52-C05 | qwen3-0.6b / fact_focused / neural_only |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +3/−3 |
| HOLDOUT/3A52-C05 | qwen3-0.6b / fact_focused / hybrid_C_neural |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +2/−2 |
| HOLDOUT/3A52-C07 | qwen3-0.6b / fact_focused / neural_only |  |  |  | W2761455282 2→1, W4391351014 1→2 | +4/−4 |
| HOLDOUT/3A52-C07 | qwen3-0.6b / fact_focused / hybrid_C_neural |  |  |  | W2761455282 2→1, W4391351014 1→2 | +2/−2 |
| HOLDOUT/3A52-C09 | qwen3-0.6b / fact_focused / neural_only |  | W2130236240 |  | W2130236240 2→—, W2906067930 1→1 | +3/−3 |
| HOLDOUT/3A52-C09 | qwen3-0.6b / fact_focused / hybrid_C_neural |  | W2130236240 |  | W2130236240 2→—, W2906067930 1→1 | +2/−2 |
| C/3A52-C03 | qwen3-0.6b / document_aware / neural_only |  | W3138328568 | W2039836112 | W2039836112 —→—, W2076293586 1→1, W3138328568 2→— | +3/−3 |
| C/3A52-C03 | qwen3-0.6b / document_aware / hybrid_C_neural |  |  | W2039836112 | W2039836112 —→—, W2076293586 1→1, W3138328568 2→5 | +2/−2 |
| C/3A52-C05 | qwen3-0.6b / document_aware / neural_only |  |  | W2906067930 | W2906067930 —→—, W4309030044 1→1 | +2/−2 |
| C/3A52-C05 | qwen3-0.6b / document_aware / hybrid_C_neural |  |  | W2906067930 | W2906067930 —→—, W4309030044 1→1 | +1/−1 |
| C/3A52-C07 | qwen3-0.6b / document_aware / neural_only |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +5/−5 |
| C/3A52-C07 | qwen3-0.6b / document_aware / hybrid_C_neural |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +2/−2 |
| C/3A52-C09 | qwen3-0.6b / document_aware / neural_only |  |  | W2084709926 | W2084709926 —→—, W3081936674 1→1 | +5/−5 |
| C/3A52-C09 | qwen3-0.6b / document_aware / hybrid_C_neural |  |  | W2084709926 | W2084709926 —→—, W3081936674 1→1 | +3/−3 |
| HOLDOUT/3A52-C03 | qwen3-0.6b / document_aware / neural_only |  | W2084709926 |  | W2084709926 1→—, W2096307598 2→1 | +4/−4 |
| HOLDOUT/3A52-C03 | qwen3-0.6b / document_aware / hybrid_C_neural |  |  |  | W2084709926 1→5, W2096307598 2→1 | +3/−3 |
| HOLDOUT/3A52-C05 | qwen3-0.6b / document_aware / neural_only |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +5/−5 |
| HOLDOUT/3A52-C05 | qwen3-0.6b / document_aware / hybrid_C_neural |  |  | W2326480042 | W2326480042 —→—, W4381572413 1→1 | +2/−2 |
| HOLDOUT/3A52-C07 | qwen3-0.6b / document_aware / neural_only |  |  |  | W2761455282 2→3, W4391351014 1→1 | +4/−4 |
| HOLDOUT/3A52-C07 | qwen3-0.6b / document_aware / hybrid_C_neural |  |  |  | W2761455282 2→1, W4391351014 1→2 | +3/−3 |
| HOLDOUT/3A52-C09 | qwen3-0.6b / document_aware / neural_only |  | W2130236240 |  | W2130236240 2→—, W2906067930 1→1 | +3/−3 |
| HOLDOUT/3A52-C09 | qwen3-0.6b / document_aware / hybrid_C_neural |  | W2130236240 |  | W2130236240 2→—, W2906067930 1→1 | +2/−2 |

## Top-5 changes

Full ordered paths and source windows are in per_question.jsonl. Every paired change, off-Gold source and the Gold gaps at budget, full eligible pool and raw BFS are recorded in results.json.

- C/3A52-C01 · ettin-68m · fact_focused · neural_only — ordered path IDs 04340fa89e, 471989f60c, 02e3120d99, 266a993c60, 90c7580aca → c09c20d5b8, 90c7580aca, f275b61b7a, 266a993c60, bcb2735c3c. Factual Gold added [], lost ['W2791492120'].
- C/3A52-C02 · ettin-68m · fact_focused · neural_only — ordered path IDs 04340fa89e, 0648be3f7c, 027e4fd327, 266a993c60, 90c7580aca → f275b61b7a, c09c20d5b8, bb332b209f, 266a993c60, 90c7580aca. Factual Gold added [], lost ['W2761455282'].
- C/3A52-C02 · ettin-68m · fact_focused · hybrid_C_neural — ordered path IDs 04340fa89e, 0648be3f7c, 027e4fd327, 266a993c60, 90c7580aca → c09c20d5b8, 04340fa89e, 266a993c60, f275b61b7a, 90c7580aca. Factual Gold added [], lost ['W2761455282'].
- C/3A52-C04 · ettin-68m · fact_focused · neural_only — ordered path IDs 327d00423b, 53807f0601, 3c39f6ad24, 132115b8d8, b916f2b4db → dfdc961a7e, 99ebec99c6, 53807f0601, 93560ba879, f68cb9d7a2. Factual Gold added [], lost ['W1979727947'].
- C/3A52-C04 · ettin-68m · fact_focused · hybrid_C_neural — ordered path IDs 327d00423b, 53807f0601, 3c39f6ad24, 132115b8d8, b916f2b4db → 53807f0601, dfdc961a7e, 93560ba879, 3c39f6ad24, 8300dbff68. Factual Gold added [], lost ['W1979727947'].
- C/3A52-C06 · ettin-68m · fact_focused · neural_only — ordered path IDs 4f154c1b12, bb332b209f, 8828792cef, d6f9044d63, e7c6ae5660 → 41dcdcf033, 2fd2ef4a1c, 5ea83d2f16, 20ae3eb9fb, 66be8aeab2. Factual Gold added [], lost ['W3101922761'].
- C/3A52-C06 · ettin-68m · fact_focused · hybrid_C_neural — ordered path IDs 4f154c1b12, bb332b209f, 8828792cef, d6f9044d63, e7c6ae5660 → 2fd2ef4a1c, 41dcdcf033, 4f154c1b12, 5ea83d2f16, d6f9044d63. Factual Gold added [], lost ['W3101922761'].
- C/3A52-C08 · ettin-68m · fact_focused · neural_only — ordered path IDs 2470d494ab, 377102676e, 6dde2c838e, 8d99c3ddc8, e1b345899d → 377102676e, 733824de1a, 4b53747fd6, 75d06351c9, 16f2827489. Factual Gold added [], lost ['W4381572413'].

## Interpretation and quality/speed

Document retrieval metrics use the frozen reference set. Off-Gold sources may be relevant. Semantic Evidence Coverage is **NOT_EVALUATED**. Graph provenance and overlap establish source identity and offsets. Semantic correctness of extracted relations needs independent factual labels. Historical HOLDOUT can overlap in content with approved questions.

Retain C as the quality/speed reference. The completed configurations did not meet all predeclared follow-up criteria.

Follow-up criteria on the 25 approved questions are higher factual micro recall, AllRequiredDocs at least equal to C, reranking p95 ≤ 10 s and peak process RSS ≤ 8 GiB. The report makes no production promotion or statistical-significance decision.
