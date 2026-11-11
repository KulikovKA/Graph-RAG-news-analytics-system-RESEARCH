// Stage 2E, Neo4j 2026.08.1 / Cypher 25, no APOC/GDS required.
// Namespace is explicit in every query. Run statements separately in Browser.
// Each SIMILAR_TO is stored once, with directed=false. Traverse it undirected.
// KG connectivity excludes documents and RELEX_MENTIONS. Original loops remain.

// 1. Physical namespace counts, including service relationships.
CYPHER 25
WITH 'relex_sem_e5_small_name_t095_ner_t046_max' AS ns
CALL (ns) {
  MATCH (d:RELEXDocument {namespace: ns}) RETURN count(d) AS documents
}
CALL (ns) {
  MATCH (e:RELEXEntity {namespace: ns}) RETURN count(e) AS entities
}
CALL (ns) {
  MATCH ()-[r:RELEX_REL {namespace: ns}]->() RETURN count(r) AS factual_relations
}
CALL (ns) {
  MATCH ()-[r:SIMILAR_TO {namespace: ns}]->() RETURN count(r) AS semantic_links
}
CALL (ns) {
  MATCH ()-[r:RELEX_MENTIONS {namespace: ns}]->() RETURN count(r) AS document_mentions
}
RETURN ns AS namespace, documents, entities, documents+entities AS total_nodes,
       factual_relations, semantic_links, document_mentions,
       factual_relations+semantic_links AS kg_edges,
       factual_relations+semantic_links+document_mentions AS total_relationships;

// 2. Semantic links by scope, counted once rather than in both directions.
CYPHER 25
WITH 'relex_sem_e5_small_name_t095_ner_t046_max' AS ns
MATCH (a:RELEXEntity {namespace: ns})-[r:SIMILAR_TO {namespace: ns}]->(b:RELEXEntity {namespace: ns})
RETURN count(r) AS semantic_links,
       sum(CASE WHEN a.document_id<>b.document_id THEN 1 ELSE 0 END) AS cross_document,
       sum(CASE WHEN a.document_id=b.document_id THEN 1 ELSE 0 END) AS within_document,
       min(r.similarity) AS minimum_cosine, max(r.similarity) AS maximum_cosine;

// 3. Concrete cross-document links with input metadata and 2C annotations.
CYPHER 25
WITH 'relex_sem_e5_small_name_t095_ner_t046_max' AS ns
MATCH (a:RELEXEntity {namespace: ns})-[r:SIMILAR_TO {namespace: ns}]->(b:RELEXEntity {namespace: ns})
WHERE a.document_id<>b.document_id
RETURN a.canonical_entity_id AS left_id, a.surface_text AS left_name,
       a.document_id AS left_document, b.canonical_entity_id AS right_id,
       b.surface_text AS right_name, b.document_id AS right_document,
       r.similarity AS cosine, r.model AS model, r.mode AS mode,
       r.threshold AS threshold, r.decision_2c AS identity_decision_2c
ORDER BY cosine DESC, left_id, right_id LIMIT 20;

// 4. Isolation and mean total degree within the entity KG. A loop contributes 2.
CYPHER 25
WITH 'relex_sem_e5_small_name_t095_ner_t046_max' AS ns
MATCH (e:RELEXEntity {namespace: ns})
WITH ns, count(e) AS entities,
     sum(CASE WHEN NOT EXISTS {
       MATCH (e)-[r:RELEX_REL|SIMILAR_TO]-(other:RELEXEntity {namespace: ns})
       WHERE r.namespace=ns
     } THEN 1 ELSE 0 END) AS isolated
CALL (ns) {
  MATCH (:RELEXEntity {namespace: ns})-[r:RELEX_REL|SIMILAR_TO]->(:RELEXEntity {namespace: ns})
  WHERE r.namespace=ns
  RETURN count(r) AS kg_edges
}
RETURN entities, isolated, 100.0*isolated/entities AS isolated_pct,
       kg_edges, 2.0*kg_edges/entities AS mean_total_degree;

// 5. Exact live weak components, without GDS/APOC or stored component properties.
// Each vertex starts with its own index. A round propagates the smallest label
// from its neighbors. After V-1 rounds every weak component has one label.
// All computation is local to this read-only query. No graph labels are changed.
CYPHER 25
WITH 'relex_sem_e5_small_name_t095_ner_t046_max' AS ns
MATCH (e:RELEXEntity {namespace: ns})
WITH ns,e ORDER BY e.canonical_entity_id
WITH ns,collect(e) AS entities
UNWIND range(0,size(entities)-1) AS vertex
WITH ns,entities,vertex,entities[vertex] AS e
CALL (e, ns, entities) {
  OPTIONAL MATCH (e)-[r:RELEX_REL|SIMILAR_TO]-(other:RELEXEntity {namespace: ns})
  WHERE r.namespace=ns
  WITH entities,collect(DISTINCT other) AS neighbors
  RETURN [i IN range(0,size(entities)-1) WHERE entities[i] IN neighbors] AS neighbor_indices
}
WITH entities,vertex,neighbor_indices ORDER BY vertex
WITH entities,collect(neighbor_indices) AS adjacency
WITH entities,reduce(labels=range(0,size(entities)-1), round IN range(1,size(entities)-1) |
  [i IN range(0,size(entities)-1) |
    reduce(best=labels[i], neighbor IN adjacency[i] |
      CASE WHEN labels[neighbor]<best THEN labels[neighbor] ELSE best END)
  ]) AS labels
UNWIND range(0,size(entities)-1) AS vertex
WITH labels[vertex] AS component_id,entities[vertex].document_id AS document_id
WITH component_id,count(*) AS component_size,count(DISTINCT document_id) AS document_count
ORDER BY component_size DESC, component_id
RETURN count(*) AS weak_components, sum(component_size) AS entities,
       max(component_size) AS largest_component_vertices,
       head(collect(document_count)) AS largest_component_documents;

// 6. Documents participating in direct semantic links and unique document pairs.
CYPHER 25
WITH 'relex_sem_e5_small_name_t095_ner_t046_max' AS ns
MATCH (a:RELEXEntity {namespace: ns})-[r:SIMILAR_TO {namespace: ns}]->(b:RELEXEntity {namespace: ns})
WHERE a.document_id<>b.document_id
WITH collect(DISTINCT a.document_id)+collect(DISTINCT b.document_id) AS endpoints,
     collect(DISTINCT CASE WHEN a.document_id<b.document_id
       THEN [a.document_id,b.document_id] ELSE [b.document_id,a.document_id] END) AS pairs
UNWIND endpoints AS document_id
RETURN count(DISTINCT document_id) AS linked_documents, size(pairs) AS directly_linked_document_pairs;

// 7. Top documents by incident cross-document semantic links.
CYPHER 25
WITH 'relex_sem_e5_small_name_t095_ner_t046_max' AS ns
MATCH (a:RELEXEntity {namespace: ns})-[r:SIMILAR_TO {namespace: ns}]->(b:RELEXEntity {namespace: ns})
WHERE a.document_id<>b.document_id
UNWIND [a.document_id,b.document_id] AS document_id
WITH ns, document_id, count(r) AS cross_document_links,
     count(DISTINCT CASE WHEN document_id=a.document_id
       THEN b.document_id ELSE a.document_id END) AS partner_documents
MATCH (d:RELEXDocument {namespace: ns, document_id: document_id})
RETURN document_id, d.title AS title, cross_document_links, partner_documents
ORDER BY cross_document_links DESC, document_id LIMIT 10;

// 8. Top entities by distinct incident KG relationships. A loop counts once here.
CYPHER 25
WITH 'relex_sem_e5_small_name_t095_ner_t046_max' AS ns
MATCH (e:RELEXEntity {namespace: ns})
CALL (e, ns) {
  OPTIONAL MATCH (e)-[r:RELEX_REL|SIMILAR_TO]-(other:RELEXEntity {namespace: ns})
  WHERE r.namespace=ns
  RETURN count(DISTINCT r) AS incident_relationships,
         count(DISTINCT CASE WHEN type(r)='SIMILAR_TO' THEN r END) AS semantic_links
}
RETURN e.canonical_entity_id AS entity_id, e.surface_text AS name,
       e.entity_types AS types, e.document_id AS document_id,
       incident_relationships, semantic_links
ORDER BY incident_relationships DESC, entity_id LIMIT 20;

// 9. Actual three-hop paths across documents: factual -- semantic -- factual.
// Undirected traversal illustrates weak connectivity, not causal/logical inference.
CYPHER 25
WITH 'relex_sem_e5_small_name_t095_ner_t046_max' AS ns
MATCH (s:RELEXEntity {namespace: ns})-[sem:SIMILAR_TO {namespace: ns}]->(t:RELEXEntity {namespace: ns})
WHERE s.document_id<>t.document_id
MATCH path = (a:RELEXEntity {namespace: ns})-[first:RELEX_REL]-(s)-[sem]-(t)-[last:RELEX_REL]-(b:RELEXEntity {namespace: ns})
WHERE first.namespace=ns AND last.namespace=ns AND a.document_id<>b.document_id
RETURN path, a.document_id AS source_document, b.document_id AS target_document,
       sem.similarity AS cosine, first.relation_type AS first_fact,
       last.relation_type AS last_fact
LIMIT 10;
