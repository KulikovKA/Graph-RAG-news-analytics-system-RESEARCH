CYPHER 25
MATCH (a:RELEXEntity {namespace:'relex_crossdoc_e5_small_t095'})
      -[r:RELEX_REL|SIMILAR_TO|CROSS_DOC_RE_CANDIDATE]->
      (b:RELEXEntity {namespace:'relex_crossdoc_e5_small_t095'})
WHERE r.namespace='relex_crossdoc_e5_small_t095'
RETURN a,r,b
ORDER BY CASE WHEN type(r)='CROSS_DOC_RE_CANDIDATE' THEN 0 ELSE 1 END
LIMIT 300;

CYPHER 25
WITH 'relex_crossdoc_e5_small_t095' AS ns
CALL (ns) { MATCH (d:RELEXDocument {namespace:ns}) RETURN count(d) AS documents }
CALL (ns) {
  MATCH (e:RELEXEntity {namespace:ns})
  RETURN count(e) AS entities,
    sum(CASE WHEN NOT EXISTS {
      MATCH (e)-[r:RELEX_REL|SIMILAR_TO|CROSS_DOC_RE_CANDIDATE]-(other:RELEXEntity {namespace:ns})
      WHERE r.namespace=ns
    } THEN 1 ELSE 0 END) AS isolated
}
CALL (ns) { MATCH ()-[r:RELEX_REL {namespace:ns}]->() RETURN count(r) AS factual_relations }
CALL (ns) { MATCH ()-[r:SIMILAR_TO {namespace:ns}]->() RETURN count(r) AS similar_to }
CALL (ns) {
  MATCH (a:RELEXEntity {namespace:ns})-[r:CROSS_DOC_RE_CANDIDATE {namespace:ns}]->(b:RELEXEntity {namespace:ns})
  RETURN count(r) AS cross_doc_candidates, min(r.confidence) AS min_confidence,
    max(r.confidence) AS max_confidence,
    sum(CASE WHEN r.suspicious_type_hint THEN 1 ELSE 0 END) AS type_warnings,
    sum(CASE WHEN r.absence_of_direct_evidence THEN 1 ELSE 0 END) AS without_direct_evidence,
    collect(DISTINCT a.document_id)+collect(DISTINCT b.document_id) AS linked_docs
}
RETURN ns AS namespace, documents, entities, factual_relations, similar_to,
  cross_doc_candidates, isolated, 100.0*isolated/entities AS isolated_pct,
  min_confidence, max_confidence, type_warnings, without_direct_evidence,
  size(reduce(unique=[], d IN linked_docs | CASE WHEN d IN unique THEN unique ELSE unique+[d] END)) AS linked_documents;
