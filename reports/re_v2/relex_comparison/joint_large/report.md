# joint_large

Native joint RelEx predicts entities and relations. E3 mentions are never read or supplied in this experiment.

| Metric | Value |
|---|---:|
| experiment | joint_large |
| workers | 2 |
| documents | 30 |
| entities | 1062 |
| source_mentions | 1852 |
| extracted_relations | 142 |
| unique_triples | 142 |
| involved_entities | 196 |
| entity_coverage | 0.18455743879472694 |
| endpoint_violations | 0 |
| suspicious_type_hint_fraction | 0.1267605633802817 |
| confidence_mean | 0.8219560502280652 |
| document_inference_seconds | 164.1104437992908 |
| model_load_seconds | 28.09152529994026 |
| total_run_seconds | 142.94547790009528 |
| peak_sampled_rss_mib | 8347.21484375 |
| resumed_documents | 0 |

Neo4j: `{"documents": 30, "entities": 1062, "relations": 142, "mentions": 1062, "cross_namespace_edges": 0}`; namespace `relex_joint_large`.

No gold labels or RE quality metrics. Type hints are nonbinding review candidates, never filters.
Every relation endpoint is validated against this experiment's entities in the same document.
All raw window predictions and evidence are retained. Deduplication uses exact document/head/type/tail.
Base and Large use a common token window plan within each mode. E3 windows protect known spans; joint windows use text only.
Wall time and peak RSS describe this invocation; per-document times include resumed checkpoints.
RSS is sampled every 0.25 seconds for the parent and its model workers; other experiments and Docker are excluded.

Browser: http://127.0.0.1:17474 ; Bolt: bolt://127.0.0.1:17687 ; query: graph.cypher.
