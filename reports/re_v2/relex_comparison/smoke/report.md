# RE-V2-001: original E3 -> GLiNER-RelEx -> Neo4j

| Metric | Base | Large |
|---|---:|---:|
| documents | 2 | 2 |
| entities | 42 | 42 |
| source_mentions | 47 | 47 |
| extracted_relations | 13 | 5 |
| unique_triples | 13 | 5 |
| involved_entities | 15 | 7 |
| entity_coverage | 0.35714285714285715 | 0.16666666666666666 |
| suspicious_type_hint_fraction | 0.0 | 0.0 |
| endpoint_violations | 0 | 0 |
| confidence_mean | 0.7777363566251901 | 0.802227520942688 |
| confidence_min | 0.7039808630943298 | 0.7167283296585083 |
| confidence_max | 0.9301209449768066 | 0.8998945355415344 |
| document_inference_seconds | 2.143245399929583 | 4.994613799965009 |
| worker_wall_seconds | 13.169412800110877 | 12.923794599948451 |
| resumed_documents | 2 | 2 |
| peak_rss_mib | 387.11328125 | 386.7890625 |
| total_run_seconds | 44.90599099989049 | 44.90599099989049 |
| peak_combined_rss_mib | 1174.51953125 | 1174.51953125 |

Triple agreement: `{"shared_triples": 4, "union_triples": 14, "jaccard": 0.2857142857142857, "base_only": 9, "large_only": 1}`.

Verified Neo4j counts: `{"base": {"documents": 2, "entities": 42, "relations": 13, "mentions": 42, "cross_namespace_edges": 0}, "large": {"documents": 2, "entities": 42, "relations": 5, "mentions": 42, "cross_namespace_edges": 0}}`.

Original E3 may contain several entity types per span: all source records are preserved; one node per document/start/end.
Predictions use trained encoder/span projections/relation head with exact E3 endpoints; both Base token-level and Large markerV0 NER selection are bypassed.
All above-threshold window predictions are preserved before overlap deduplication. Triple confidence is the maximum across windows.
Direction is preserved. Exact E3 endpoints and same-document membership are validated before import.
Type hints are nonbinding review candidates inferred from predicate definitions. Ontology v2 has no approved domain/range matrix; formal violation rate is unavailable.
No suspicious triples are removed. Model disagreement and type hints do not establish an error.
No gold relation labels were used; RE quality metrics are unavailable.
RAM is sampled process RSS (parent plus both workers), every 0.25s; Docker/server RAM is excluded. Values are measured samples, not OS high-water marks.
Resume retains per-document inference durations; worker/total wall time describe this invocation.

Browser: http://127.0.0.1:17474 ; connect to bolt://127.0.0.1:17687.
Use base.cypher / large.cypher to show isolated entities as well as directed relations.

API references: [Base model card](https://huggingface.co/knowledgator/gliner-relex-base-v1.0), [Large model card](https://huggingface.co/knowledgator/gliner-relex-large-v1.0). The actual installed API audit is saved in base.api.json / large.api.json.
