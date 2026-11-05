# e3_base

Original E3 entities are preserved; RelEx predicts only relations between those exact spans.

| Metric | Value |
|---|---:|
| experiment | e3_base |
| workers | 2 |
| documents | 30 |
| entities | 1211 |
| source_mentions | 1327 |
| extracted_relations | 211 |
| unique_triples | 211 |
| involved_entities | 231 |
| entity_coverage | 0.1907514450867052 |
| endpoint_violations | 0 |
| suspicious_type_hint_fraction | 0.037914691943127965 |
| confidence_mean | 0.7968106210514267 |
| document_inference_seconds | 56.71620420040563 |
| model_load_seconds | 19.759577200049534 |
| total_run_seconds | 82.7823447000701 |
| peak_sampled_rss_mib | 4827.08984375 |
| resumed_documents | 0 |

Neo4j: `{"documents": 30, "entities": 1211, "relations": 211, "mentions": 1211, "cross_namespace_edges": 0}`; namespace `relex_base`.

No gold labels or RE quality metrics. Type hints are nonbinding review candidates, never filters.
Every relation endpoint is validated against this experiment's entities in the same document.
All raw window predictions and evidence are retained. Deduplication uses exact document/head/type/tail.
Base and Large use a common token window plan within each mode. E3 windows protect known spans; joint windows use text only.
Wall time and peak RSS describe this invocation; per-document times include resumed checkpoints.
RSS is sampled every 0.25 seconds for the parent and its model workers; other experiments and Docker are excluded.

Browser: http://127.0.0.1:17474 ; Bolt: bolt://127.0.0.1:17687 ; query: graph.cypher.
