# e3_base

Original E3 entities are preserved; RelEx predicts only relations between those exact spans.

| Metric | Value |
|---|---:|
| experiment | e3_base |
| workers | 2 |
| documents | 30 |
| entities | 1211 |
| source_mentions | 1327 |
| extracted_relations | 2609 |
| unique_triples | 2609 |
| involved_entities | 956 |
| entity_coverage | 0.7894302229562346 |
| endpoint_violations | 0 |
| suspicious_type_hint_fraction | 0.11843618244538137 |
| confidence_mean | 0.46564392484481787 |
| document_inference_seconds | 60.04757319949567 |
| model_load_seconds | 21.39372429996729 |
| total_run_seconds | 77.62398899998516 |
| peak_sampled_rss_mib | 4726.7890625 |
| resumed_documents | 0 |

Neo4j: `{"status": "NOT_IMPORTED"}`; namespace `relex_base`.

No gold labels or RE quality metrics. Type hints are nonbinding review candidates, never filters.
Every relation endpoint is validated against this experiment's entities in the same document.
All raw window predictions and evidence are retained. Deduplication uses exact document/head/type/tail.
Base and Large use a common token window plan within each mode. E3 windows protect known spans; joint windows use text only.
Wall time and peak RSS describe this invocation; per-document times include resumed checkpoints.
RSS is sampled every 0.25 seconds for the parent and its model workers; other experiments and Docker are excluded.

Browser: http://127.0.0.1:17474 ; Bolt: bolt://127.0.0.1:17687 ; query: graph.cypher.
