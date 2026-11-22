# Stage 3A.5.7 — Integrated retrieval benchmark

**COMPLETE** · 35/35 questions · 140/140 evaluations

- Started 2026-11-22T19:16:26.208119+00:00 UTC
- Git commit `ebf0aadd1a1113ee0b785905fbb30cdd54af565d`
- Python 3.11.9, PyTorch 2.11.0, Neo4j driver 6.1.0
- Namespace `relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v2_sem_name_t0830`
- Graph fingerprint `41901f76e83ee9ab65fedfba888790504c8f328066a319115fcb6f13f7c3525a`
- Models `fastino/gliner2.5-base-v1@54785d51df8d86a0d8f2eb212fec15871aad2c4d` and `intfloat/e5-small-v2` (weight hashes in [config.json](config.json))
- Semantic Evidence Coverage = NOT_EVALUATED

## Policies

| Variant | Reranker | Cutoff | Link top_k | Fallback | BFS candidates |
| --- | --- | --- | --- | --- | --- |
| A | A | 0.85 | 5 | baseline | 200 |
| B | C | 0.85 | 5 | baseline | 200 |
| C | C | 0.85 | 10 | baseline | 200 |
| D | C | 0.85 | 10 | baseline | 100 |

All policies use GLiNER 0.15, FIFO BFS (5 hops, 50 neighbors, 1,000 relationships, 32 seeds, 10-second transaction timeout), and five ranked paths.

## Quality and latency

| Group | Variant | Completed | Gold Recall micro | Gold Recall macro | Factual Recall | AllRequired | Window Overlap | Unique Docs | Latency (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | A | 35/35 | 0.786 | 0.833 | 0.786 | 0.657 | 0.776 | 1.629 | 3.861 |
| ALL | B | 35/35 | 0.893 | 0.919 | 0.893 | 0.829 | 0.888 | 2.143 | 3.886 |
| ALL | C | 35/35 | 0.893 | 0.919 | 0.893 | 0.829 | 0.897 | 2.286 | 4.208 |
| ALL | D | 35/35 | 0.875 | 0.905 | 0.875 | 0.800 | 0.879 | 2.229 | 3.216 |
| A | A | 8/8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 5.285 |
| A | B | 8/8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 5.322 |
| A | C | 8/8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.125 | 5.468 |
| A | D | 8/8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.125 | 4.757 |
| B | A | 7/7 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 3.944 |
| B | B | 7/7 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.571 | 3.986 |
| B | C | 7/7 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.571 | 4.235 |
| B | D | 7/7 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.429 | 2.885 |
| C | A | 10/10 | 0.619 | 0.617 | 0.619 | 0.200 | 0.591 | 2.300 | 2.995 |
| C | B | 10/10 | 0.810 | 0.817 | 0.810 | 0.600 | 0.795 | 3.000 | 3.019 |
| C | C | 10/10 | 0.810 | 0.817 | 0.810 | 0.600 | 0.818 | 3.200 | 3.779 |
| C | D | 10/10 | 0.810 | 0.817 | 0.810 | 0.600 | 0.818 | 3.200 | 2.740 |
| HOLDOUT | A | 10/10 | 0.800 | 0.800 | 0.800 | 0.600 | 0.793 | 1.900 | 3.529 |
| HOLDOUT | B | 10/10 | 0.900 | 0.900 | 0.900 | 0.800 | 0.897 | 2.600 | 3.536 |
| HOLDOUT | C | 10/10 | 0.900 | 0.900 | 0.900 | 0.800 | 0.897 | 2.800 | 3.611 |
| HOLDOUT | D | 10/10 | 0.850 | 0.850 | 0.850 | 0.700 | 0.828 | 2.700 | 2.691 |

Latency is the sum of the measured pipeline stages for each policy with shared stage costs included. It includes observed first-use model loading. See the CSV for actual wall time and additional stage computation. Partial summaries use completed evaluations.

## Component comparisons

| Group | Change | Component | Paired q | Δ Gold micro | Δ Factual micro | Δ Candidate micro | Δ Window micro | Δ Unique Docs | Δ latency (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | A → B | Document-aware reranking A → C | 35 | +0.107 | +0.107 | +0.000 | +0.112 | +0.514 | +0.026 |
| ALL | B → C | Entity Linking top_k 5 → 10 | 35 | +0.000 | +0.000 | -0.018 | +0.009 | +0.143 | +0.322 |
| ALL | C → D | FIFO BFS max_candidates 200 → 100 | 35 | -0.018 | -0.018 | -0.018 | -0.019 | -0.057 | -0.992 |
| ALL | A → D | Joint effect | 35 | +0.089 | +0.089 | -0.036 | +0.103 | +0.600 | -0.645 |
| A | A → B | Document-aware reranking A → C | 8 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.037 |
| A | B → C | Entity Linking top_k 5 → 10 | 8 | +0.000 | +0.000 | +0.000 | +0.000 | +0.125 | +0.146 |
| A | C → D | FIFO BFS max_candidates 200 → 100 | 8 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | -0.711 |
| A | A → D | Joint effect | 8 | +0.000 | +0.000 | +0.000 | +0.000 | +0.125 | -0.528 |
| B | A → B | Document-aware reranking A → C | 7 | +0.000 | +0.000 | +0.000 | +0.000 | +0.571 | +0.041 |
| B | B → C | Entity Linking top_k 5 → 10 | 7 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.249 |
| B | C → D | FIFO BFS max_candidates 200 → 100 | 7 | +0.000 | +0.000 | +0.000 | +0.000 | -0.143 | -1.350 |
| B | A → D | Joint effect | 7 | +0.000 | +0.000 | +0.000 | +0.000 | +0.429 | -1.059 |
| C | A → B | Document-aware reranking A → C | 10 | +0.190 | +0.190 | +0.000 | +0.205 | +0.700 | +0.023 |
| C | B → C | Entity Linking top_k 5 → 10 | 10 | +0.000 | +0.000 | -0.048 | +0.023 | +0.200 | +0.761 |
| C | C → D | FIFO BFS max_candidates 200 → 100 | 10 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | -1.039 |
| C | A → D | Joint effect | 10 | +0.190 | +0.190 | -0.048 | +0.227 | +0.900 | -0.255 |
| HOLDOUT | A → B | Document-aware reranking A → C | 10 | +0.100 | +0.100 | +0.000 | +0.103 | +0.700 | +0.008 |
| HOLDOUT | B → C | Entity Linking top_k 5 → 10 | 10 | +0.000 | +0.000 | +0.000 | +0.000 | +0.200 | +0.075 |
| HOLDOUT | C → D | FIFO BFS max_candidates 200 → 100 | 10 | -0.050 | -0.050 | -0.050 | -0.069 | -0.100 | -0.920 |
| HOLDOUT | A → D | Joint effect | 10 | +0.050 | +0.050 | -0.050 | +0.034 | +0.800 | -0.838 |

Deltas use the same questions on both sides. Adjacent comparisons hold the other settings fixed. A → D measures the joint effect.

## Questions

| Question | A | B | C | D | Best |
| --- | --- | --- | --- | --- | --- |
| A/3A52-A01 | 1.000 | 1.000 | 1.000 | 1.000 | A, B, C, D |
| A/3A52-A02 | 1.000 | 1.000 | 1.000 | 1.000 | A, B, C, D |
| A/3A52-A03 | 1.000 | 1.000 | 1.000 | 1.000 | A, B, C, D |
| A/3A52-A04 | 1.000 | 1.000 | 1.000 | 1.000 | A, B, C, D |
| A/3A52-A05 | 1.000 | 1.000 | 1.000 | 1.000 | A, B, C, D |
| A/3A52-A06 | 1.000 | 1.000 | 1.000 | 1.000 | A, B, C, D |
| A/3A52-A07 | 1.000 | 1.000 | 1.000 | 1.000 | A, B, C, D |
| A/3A52-A08 | 1.000 | 1.000 | 1.000 | 1.000 | A, B, C, D |
| B/3A52-B01 | 1.000 | 1.000 | 1.000 | 1.000 | A, B, C, D |
| B/3A52-B02 | 1.000 | 1.000 | 1.000 | 1.000 | A, B, C, D |
| B/3A52-B03 | 1.000 | 1.000 | 1.000 | 1.000 | A, B, C, D |
| B/3A52-B04 | 1.000 | 1.000 | 1.000 | 1.000 | A, B, C, D |
| B/3A52-B05 | 1.000 | 1.000 | 1.000 | 1.000 | A, B, C, D |
| B/3A52-B06 | 1.000 | 1.000 | 1.000 | 1.000 | A, B, C, D |
| B/3A52-B07 | 1.000 | 1.000 | 1.000 | 1.000 | A, B, C, D |
| C/3A52-C01 | 0.500 | 1.000 | 1.000 | 1.000 | B, C, D |
| C/3A52-C02 | 0.500 | 1.000 | 1.000 | 1.000 | B, C, D |
| C/3A52-C03 | 0.667 | 0.667 | 0.667 | 0.667 | A, B, C, D |
| C/3A52-C04 | 1.000 | 1.000 | 1.000 | 1.000 | A, B, C, D |
| C/3A52-C05 | 0.500 | 0.500 | 0.500 | 0.500 | C, D |
| C/3A52-C06 | 0.500 | 1.000 | 1.000 | 1.000 | B, C, D |
| C/3A52-C07 | 0.500 | 0.500 | 0.500 | 0.500 | A, B, C, D |
| C/3A52-C08 | 0.500 | 1.000 | 1.000 | 1.000 | B, C, D |
| C/3A52-C09 | 0.500 | 0.500 | 0.500 | 0.500 | A, B, C, D |
| C/3A52-C10 | 1.000 | 1.000 | 1.000 | 1.000 | A, B, C, D |
| HOLDOUT/3A52-C01 | 1.000 | 1.000 | 1.000 | 1.000 | A, B, C, D |
| HOLDOUT/3A52-C02 | 1.000 | 1.000 | 1.000 | 1.000 | A, B, C, D |
| HOLDOUT/3A52-C03 | 1.000 | 1.000 | 1.000 | 1.000 | A, B, C, D |
| HOLDOUT/3A52-C04 | 1.000 | 1.000 | 1.000 | 1.000 | A, B, C, D |
| HOLDOUT/3A52-C05 | 0.500 | 0.500 | 0.500 | 0.500 | A, B, C, D |
| HOLDOUT/3A52-C06 | 0.500 | 0.500 | 0.500 | 0.500 | A, B, C, D |
| HOLDOUT/3A52-C07 | 1.000 | 1.000 | 1.000 | 1.000 | A, B, C, D |
| HOLDOUT/3A52-C08 | 0.500 | 1.000 | 1.000 | 0.500 | B, C |
| HOLDOUT/3A52-C09 | 1.000 | 1.000 | 1.000 | 1.000 | A, B, C, D |
| HOLDOUT/3A52-C10 | 0.500 | 1.000 | 1.000 | 1.000 | B, C, D |

Cells show GoldDocRecall@5. Best uses Gold recall, factual recall and then window overlap, with ties listed. [Per-question data](per_question.jsonl) has document IDs, ranked paths and source windows for every completed variant. [Results](results.json) has all pairwise document and rank changes.

## C03, C05, C07 and C09

### C/3A52-C03

- **A** · Gold found W2076293586, W3138328568 · missing W2039836112 · candidate Gold W2076293586, W3138328568 · fact sources W1995028339, W2076293586, W3123056951, W3138328568 · paths 1. `path:5496c01`, 2. `path:778d322`, 3. `path:3800f35`, 4. `path:f4c12df`, 5. `path:5b040d6`. [Evidence](cache/rerank/564e47bd0dc0431591d75150a31fa006f3a592edb23d4c640fd31c4a12f96fd0.json)
- **B** · Gold found W2076293586, W3138328568 · missing W2039836112 · candidate Gold W2076293586, W3138328568 · fact sources W1995028339, W2076293586, W3123056951, W3138328568 · paths 1. `path:5496c01`, 2. `path:778d322`, 3. `path:3800f35`, 4. `path:f4c12df`, 5. `path:5b040d6`. [Evidence](cache/rerank/a0306abbc97669daa67189bfe39b30a2c78f741da0d30eb905638cab8f4ac019.json)
- **C** · Gold found W2076293586, W3138328568 · missing W2039836112 · candidate Gold W2076293586, W3138328568 · fact sources W1995028339, W2076293586, W3123056951, W3138328568 · paths 1. `path:5496c01`, 2. `path:2b50f91`, 3. `path:1884a82`, 4. `path:3eb3bb2`, 5. `path:5b040d6`. [Evidence](cache/rerank/bf42877c28d5d17fa853bb153a2f6ebf3495740c338ed4e149d25a46071fe66b.json)
- **D** · Gold found W2076293586, W3138328568 · missing W2039836112 · candidate Gold W2076293586, W3138328568 · fact sources W1995028339, W2076293586, W3123056951, W3138328568 · paths 1. `path:5496c01`, 2. `path:2b50f91`, 3. `path:1884a82`, 4. `path:3eb3bb2`, 5. `path:5b040d6`. [Evidence](cache/rerank/c4b0251f661a5481f0d66d5cdf56ee7da535e0a168483b7c53bda6f1e49cbdd8.json)
- A → B · Gold gained — · Gold lost — · new docs — · 0 path rank changes
- B → C · Gold gained — · Gold lost — · new docs — · 6 path rank changes
- C → D · Gold gained — · Gold lost — · new docs — · 0 path rank changes
- A → D · Gold gained — · Gold lost — · new docs — · 6 path rank changes

### C/3A52-C05

- **A** · Gold found W4309030044 · missing W2906067930 · candidate Gold W2906067930, W4309030044 · fact sources W4309030044 · paths 1. `path:4995d45`, 2. `path:959dbac`, 3. `path:c62c15d`, 4. `path:04aaa70`, 5. `path:132ed0f`. [Evidence](cache/rerank/820ef1be8b7d772a23c717ead2158c44e4c23f4dd862733d524713d7727d8181.json)
- **B** · Gold found W4309030044 · missing W2906067930 · candidate Gold W2906067930, W4309030044 · fact sources W4309030044 · paths 1. `path:4995d45`, 2. `path:959dbac`, 3. `path:c62c15d`, 4. `path:04aaa70`, 5. `path:132ed0f`. [Evidence](cache/rerank/89f9e639e4f1e1e6a3247b3f3823adbf6fc8d4bdfcc79909bb65432f5553e119.json)
- **C** · Gold found W2906067930 · missing W4309030044 · candidate Gold W2906067930 · fact sources W2906067930 · paths 1. `path:c0405ec`, 2. `path:1255b07`, 3. `path:68f555e`, 4. `path:6afdd5e`, 5. `path:75b2e29`. [Evidence](cache/rerank/80815c10badb6037e46e73b46d67c8c21bc276804f9af9ec0aa4a32666752a67.json)
- **D** · Gold found W2906067930 · missing W4309030044 · candidate Gold W2906067930 · fact sources W2906067930 · paths 1. `path:c0405ec`, 2. `path:1255b07`, 3. `path:68f555e`, 4. `path:6afdd5e`, 5. `path:75b2e29`. [Evidence](cache/rerank/e7f80c2ac9e31c93f3a88c67557250ea1fa6bdbe55c97c6806621fcfd6647009.json)
- A → B · Gold gained — · Gold lost — · new docs — · 0 path rank changes
- B → C · Gold gained W2906067930 · Gold lost W4309030044 · new docs W2906067930 · 10 path rank changes
- C → D · Gold gained — · Gold lost — · new docs — · 0 path rank changes
- A → D · Gold gained W2906067930 · Gold lost W4309030044 · new docs W2906067930 · 10 path rank changes

### C/3A52-C07

- **A** · Gold found W4381572413 · missing W2326480042 · candidate Gold W4381572413 · fact sources W2761455282, W4381572413 · paths 1. `path:2470d49`, 2. `path:1cac07d`, 3. `path:8d99c3d`, 4. `path:e1b3458`, 5. `path:ec350d5`. [Evidence](cache/rerank/75411b8ed537b8aea586eade7d55304248b7d07daca96f3611b8899f9c4ef76e.json)
- **B** · Gold found W4381572413 · missing W2326480042 · candidate Gold W4381572413 · fact sources W2761455282, W4381572413, W4391351014 · paths 1. `path:2470d49`, 2. `path:1cac07d`, 3. `path:c609257`, 4. `path:8d99c3d`, 5. `path:e1b3458`. [Evidence](cache/rerank/52678b315a427c6031ff6616e8117096a44ecb6b5e883d3115f3bc75f1218da4.json)
- **C** · Gold found W4381572413 · missing W2326480042 · candidate Gold W4381572413 · fact sources W2761455282, W4381572413, W4391351014 · paths 1. `path:2470d49`, 2. `path:1cac07d`, 3. `path:c609257`, 4. `path:8d99c3d`, 5. `path:e1b3458`. [Evidence](cache/rerank/6332c55a5c6e160066774184b6d94fad782a12afdff8866d92673e3f42895a80.json)
- **D** · Gold found W4381572413 · missing W2326480042 · candidate Gold W4381572413 · fact sources W2761455282, W4381572413, W4391351014 · paths 1. `path:2470d49`, 2. `path:1cac07d`, 3. `path:c609257`, 4. `path:8d99c3d`, 5. `path:e1b3458`. [Evidence](cache/rerank/d03ed0ca5b498f9f0ffced0494e4346fc1bfec98b4b265c210cd61b7b60e0e64.json)
- A → B · Gold gained — · Gold lost — · new docs W4391351014 · 4 path rank changes
- B → C · Gold gained — · Gold lost — · new docs — · 0 path rank changes
- C → D · Gold gained — · Gold lost — · new docs — · 0 path rank changes
- A → D · Gold gained — · Gold lost — · new docs W4391351014 · 4 path rank changes

### C/3A52-C09

- **A** · Gold found W3081936674 · missing W2084709926 · candidate Gold W3081936674 · fact sources W2364973104, W2791492120, W3027457624, W3081936674, W4230116451 · paths 1. `path:fa3735d`, 2. `path:1adc577`, 3. `path:32db63b`, 4. `path:06a3e35`, 5. `path:1c2e455`. [Evidence](cache/rerank/c2b9216dbca524625488f3193c24af196d7357d1d6af15782b75264615293a6c.json)
- **B** · Gold found W3081936674 · missing W2084709926 · candidate Gold W3081936674 · fact sources W2791492120, W3027457624, W3081936674 · paths 1. `path:fa3735d`, 2. `path:1adc577`, 3. `path:20e7a5c`, 4. `path:e94348f`, 5. `path:32db63b`. [Evidence](cache/rerank/d331d07b62718b317946dd9657cbdd0b69dfc7797a49d6b2889c1d56492d5e78.json)
- **C** · Gold found W3081936674 · missing W2084709926 · candidate Gold W3081936674 · fact sources W2791492120, W3027457624, W3081936674 · paths 1. `path:fa3735d`, 2. `path:1adc577`, 3. `path:20e7a5c`, 4. `path:e94348f`, 5. `path:32db63b`. [Evidence](cache/rerank/4204df1eadb3f1fa1cdf8fe23946f35b43d4b054db4090a9f4ac10c3e78b66d7.json)
- **D** · Gold found W3081936674 · missing W2084709926 · candidate Gold W3081936674 · fact sources W2791492120, W3027457624, W3081936674 · paths 1. `path:fa3735d`, 2. `path:1adc577`, 3. `path:20e7a5c`, 4. `path:e94348f`, 5. `path:32db63b`. [Evidence](cache/rerank/b639a1af6250d1e1db5c8d0228568afc2c5ce8805770ff27dacdd1bd4ee64c18.json)
- A → B · Gold gained — · Gold lost — · new docs — · 5 path rank changes
- B → C · Gold gained — · Gold lost — · new docs — · 0 path rank changes
- C → D · Gold gained — · Gold lost — · new docs — · 0 path rank changes
- A → D · Gold gained — · Gold lost — · new docs — · 5 path rank changes

### HOLDOUT/3A52-C03

- **A** · Gold found W2084709926, W2096307598 · missing — · candidate Gold W2084709926, W2096307598 · fact sources W2084709926, W2096307598 · paths 1. `path:2fa4ec0`, 2. `path:049941b`, 3. `path:5f44895`, 4. `path:bf8e303`, 5. `path:7071f14`. [Evidence](cache/rerank/7cd45838630afbd63ff5d51a5a5f7f66e123512969578a3fd08cdfa9a57ceda5.json)
- **B** · Gold found W2084709926, W2096307598 · missing — · candidate Gold W2084709926, W2096307598 · fact sources W2084709926, W2096307598, W3138328568 · paths 1. `path:2fa4ec0`, 2. `path:049941b`, 3. `path:11759b3`, 4. `path:5f44895`, 5. `path:bf8e303`. [Evidence](cache/rerank/8372fdcaab81a56bd80961bfffe17c8b96461b966f25d08b6fcc1c8d966a1b59.json)
- **C** · Gold found W2084709926, W2096307598 · missing — · candidate Gold W2084709926, W2096307598 · fact sources W1979727947, W2084709926, W2096307598 · paths 1. `path:2fa4ec0`, 2. `path:7721214`, 3. `path:327d004`, 4. `path:5f44895`, 5. `path:bf8e303`. [Evidence](cache/rerank/f1f5ce3fac340b34d9a843d60186710db3480659f6bcc68da6e3b92bfe4dc180.json)
- **D** · Gold found W2084709926, W2096307598 · missing — · candidate Gold W2084709926, W2096307598 · fact sources W1979727947, W2084709926, W2096307598 · paths 1. `path:2fa4ec0`, 2. `path:7721214`, 3. `path:327d004`, 4. `path:5f44895`, 5. `path:bf8e303`. [Evidence](cache/rerank/137168e5d12cd478c5849cbc57abba3efc4dfc3463e8bfb4a427cb9b98ceb0ba.json)
- A → B · Gold gained — · Gold lost — · new docs W3138328568 · 4 path rank changes
- B → C · Gold gained — · Gold lost — · new docs W1979727947 · 4 path rank changes
- C → D · Gold gained — · Gold lost — · new docs — · 0 path rank changes
- A → D · Gold gained — · Gold lost — · new docs W1979727947 · 6 path rank changes

### HOLDOUT/3A52-C05

- **A** · Gold found W4381572413 · missing W2326480042 · candidate Gold W4381572413 · fact sources W4381572413 · paths 1. `path:0439e34`, 2. `path:20c79d1`, 3. `path:2e223c5`, 4. `path:2470d49`, 5. `path:8d99c3d`. [Evidence](cache/rerank/c44a955f6118a136b6a51a01ad1115351c9c26fa3cd49e5d7b18400ca0cccb41.json)
- **B** · Gold found W4381572413 · missing W2326480042 · candidate Gold W4381572413 · fact sources W4381572413 · paths 1. `path:0439e34`, 2. `path:20c79d1`, 3. `path:2e223c5`, 4. `path:2470d49`, 5. `path:8d99c3d`. [Evidence](cache/rerank/28e6dd3aee49c7a02dd0b09a51fd7ccf2ef3786bb61c814da749ee51713a55e7.json)
- **C** · Gold found W4381572413 · missing W2326480042 · candidate Gold W4381572413 · fact sources W2791492120, W4381572413 · paths 1. `path:0439e34`, 2. `path:1105a1a`, 3. `path:20c79d1`, 4. `path:2e223c5`, 5. `path:2470d49`. [Evidence](cache/rerank/66c308da06fcc63339e85832386bbae642100b6ae414ba7629813a9a48bf46b4.json)
- **D** · Gold found W4381572413 · missing W2326480042 · candidate Gold W4381572413 · fact sources W2791492120, W4381572413 · paths 1. `path:0439e34`, 2. `path:1105a1a`, 3. `path:20c79d1`, 4. `path:2e223c5`, 5. `path:2470d49`. [Evidence](cache/rerank/10c44d959c3d376a5b405a8e47cf7beebff9608a3d7f4977828e3c9dc5469760.json)
- A → B · Gold gained — · Gold lost — · new docs — · 0 path rank changes
- B → C · Gold gained — · Gold lost — · new docs W2791492120 · 5 path rank changes
- C → D · Gold gained — · Gold lost — · new docs — · 0 path rank changes
- A → D · Gold gained — · Gold lost — · new docs W2791492120 · 5 path rank changes

### HOLDOUT/3A52-C07

- **A** · Gold found W2761455282, W4391351014 · missing — · candidate Gold W2761455282, W4391351014 · fact sources W2761455282, W4391351014 · paths 1. `path:1a33ae0`, 2. `path:41dcdcf`, 3. `path:5a7cac2`, 4. `path:65a4eef`, 5. `path:679d3d0`. [Evidence](cache/rerank/2c6cb11ed4b87a06221bbbb03e1c52af7e181b85aa1a87efdb004e5e4dfa8b0d.json)
- **B** · Gold found W2761455282, W4391351014 · missing — · candidate Gold W2761455282, W4391351014 · fact sources W2761455282, W4391351014 · paths 1. `path:1a33ae0`, 2. `path:41dcdcf`, 3. `path:5a7cac2`, 4. `path:65a4eef`, 5. `path:679d3d0`. [Evidence](cache/rerank/c93fe6e29138c7712c4c5cba89aa2e334cf389c5ce3086b0363d90b8a03a7ce5.json)
- **C** · Gold found W2761455282, W4391351014 · missing — · candidate Gold W2761455282, W4391351014 · fact sources W2761455282, W3101922761, W4391351014 · paths 1. `path:1a33ae0`, 2. `path:41dcdcf`, 3. `path:e266dd8`, 4. `path:5a7cac2`, 5. `path:65a4eef`. [Evidence](cache/rerank/63ae713c96b0c1d43414b28ecfb06a5cc4c2a8a86ad39344d6f0fdb2189518e7.json)
- **D** · Gold found W2761455282, W4391351014 · missing — · candidate Gold W2761455282, W4391351014 · fact sources W2761455282, W4391351014 · paths 1. `path:1a33ae0`, 2. `path:41dcdcf`, 3. `path:5a7cac2`, 4. `path:65a4eef`, 5. `path:679d3d0`. [Evidence](cache/rerank/e6da9f4f6150239276735c3d732696f0f9f78e3f9ee21d855293eb8f5d7dba86.json)
- A → B · Gold gained — · Gold lost — · new docs — · 0 path rank changes
- B → C · Gold gained — · Gold lost — · new docs W3101922761 · 4 path rank changes
- C → D · Gold gained — · Gold lost — · new docs — · 4 path rank changes
- A → D · Gold gained — · Gold lost — · new docs — · 0 path rank changes

### HOLDOUT/3A52-C09

- **A** · Gold found W2130236240, W2906067930 · missing — · candidate Gold W2130236240, W2906067930 · fact sources W2130236240, W2906067930 · paths 1. `path:0f1ff13`, 2. `path:5356867`, 3. `path:22970f2`, 4. `path:c827533`, 5. `path:ea8935c`. [Evidence](cache/rerank/9254d88f7ff678b23a37028b0c408f638b8f97d69157f769d90d490763560225.json)
- **B** · Gold found W2130236240, W2906067930 · missing — · candidate Gold W2130236240, W2906067930 · fact sources W2130236240, W2317432785, W2906067930 · paths 1. `path:0f1ff13`, 2. `path:5356867`, 3. `path:234c2d3`, 4. `path:22970f2`, 5. `path:c827533`. [Evidence](cache/rerank/3c0f210f6bfe5d0f30b4a2ff336c63eb9e35807c05b4654e738d10d02e3bcc0a.json)
- **C** · Gold found W2130236240, W2906067930 · missing — · candidate Gold W2130236240, W2906067930 · fact sources W2130236240, W2761455282, W2906067930 · paths 1. `path:0f1ff13`, 2. `path:5356867`, 3. `path:007d684`, 4. `path:22970f2`, 5. `path:c827533`. [Evidence](cache/rerank/a94059efc95c38635e7ce0c213f9abaa96b6e3cf1a1bb00d72b2b6f54807c882.json)
- **D** · Gold found W2130236240, W2906067930 · missing — · candidate Gold W2130236240, W2906067930 · fact sources W2130236240, W2761455282, W2906067930 · paths 1. `path:0f1ff13`, 2. `path:5356867`, 3. `path:007d684`, 4. `path:22970f2`, 5. `path:c827533`. [Evidence](cache/rerank/91459d7aa82c0f7060a42c68ba587a93e29abd98fee2e32496441e7a3bb0e380.json)
- A → B · Gold gained — · Gold lost — · new docs W2317432785 · 4 path rank changes
- B → C · Gold gained — · Gold lost — · new docs W2761455282 · 2 path rank changes
- C → D · Gold gained — · Gold lost — · new docs — · 0 path rank changes
- A → D · Gold gained — · Gold lost — · new docs W2761455282 · 4 path rank changes

## Interpretation

The dataset covers 35 questions over a frozen corpus. The historical C set shares IDs with approved C, and some question texts recur. Results are specific to the pinned snapshot and query set.
Top-5 counts paths. Factual documents require validated source provenance. Window overlap uses the same document and revision with intersecting offsets. Documents outside the reference set can be relevant. Navigation transitions and validated factual cross-document paths have separate counts.
BFS truncation follows the retrieval diagnostics. Timing depends on the host, Neo4j load, query order and caching. Execution attempts and errors are recorded in the manifest and errors.jsonl.

Across 35 paired questions, D changed Gold recall by +8.9 percentage points, factual recall by +8.9 points and mean standalone pipeline latency by -0.645 s relative to A.
