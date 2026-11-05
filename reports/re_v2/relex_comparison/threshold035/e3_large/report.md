# e3_large

Original E3 entities are preserved; RelEx predicts only relations between those exact spans.

| Metric | Value |
|---|---:|
| experiment | e3_large |
| workers | 2 |
| documents | 30 |
| entities | 1211 |
| source_mentions | 1327 |
| extracted_relations | 570 |
| unique_triples | 570 |
| involved_entities | 516 |
| entity_coverage | 0.42609413707679605 |
| endpoint_violations | 0 |
| suspicious_type_hint_fraction | 0.13157894736842105 |
| confidence_mean | 0.5165668179591497 |
| document_inference_seconds | 147.93233709898777 |
| model_load_seconds | 32.51464410009794 |
| total_run_seconds | 145.34360729996115 |
| peak_sampled_rss_mib | 8535.35546875 |
| resumed_documents | 0 |

Neo4j: `{"documents": 30, "entities": 1211, "relations": 570, "mentions": 1211, "cross_namespace_edges": 0}`; namespace `relex_large_t035`.

No gold labels or RE quality metrics. Type hints are nonbinding review candidates, never filters.
Every relation endpoint is validated against this experiment's entities in the same document.
All raw window predictions and evidence are retained. Deduplication uses exact document/head/type/tail.
Base and Large use a common token window plan within each mode. E3 windows protect known spans; joint windows use text only.
Wall time and peak RSS describe this invocation; per-document times include resumed checkpoints.
RSS is sampled every 0.25 seconds for the parent and its model workers; other experiments and Docker are excluded.

Browser: http://127.0.0.1:17474 ; Bolt: bolt://127.0.0.1:17687 ; query: graph.cypher.
