// Stage 2G (reused 2F.3 checks): read-only queries; run statements separately. No APOC/GDS.

CYPHER 25
WITH 'relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1' AS ns
CALL (ns) { MATCH (n:RELEXDocument {namespace:ns}) RETURN count(n) AS documents }
CALL (ns) { MATCH (n:RELEXEntity {namespace:ns}) RETURN count(n) AS entities }
CALL (ns) { MATCH (n:RELEXConcept {namespace:ns}) RETURN count(n) AS concepts }
CALL (ns) { MATCH ()-[r:RELEX_REL {namespace:ns}]->() RETURN count(r) AS factual_relations }
CALL (ns) { MATCH ()-[r:SIMILAR_TO {namespace:ns}]->() RETURN count(r) AS semantic_links }
CALL (ns) { MATCH ()-[r:HAS_SHARED_CONCEPT {namespace:ns}]->() RETURN count(r) AS concept_links }
CALL (ns) { MATCH ()-[r:RELEX_MENTIONS {namespace:ns}]->() RETURN count(r) AS document_mentions }
RETURN ns AS namespace,documents,entities,concepts,entities+concepts AS kg_vertices,
       documents+entities+concepts AS total_nodes,factual_relations,semantic_links,concept_links,document_mentions,
       factual_relations+semantic_links+concept_links AS kg_edges,
       factual_relations+semantic_links+concept_links+document_mentions AS total_relationships;

CYPHER 25
WITH 'relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1' AS ns
MATCH (a:RELEXEntity {namespace:ns})-[r:SIMILAR_TO {namespace:ns}]->(b:RELEXEntity {namespace:ns})
RETURN count(r) AS semantic_links,count(DISTINCT r.semantic_link_id) AS distinct_original_ids,
sum(CASE WHEN a.document_id<>b.document_id THEN 1 ELSE 0 END) AS cross_document,
sum(CASE WHEN a=b THEN 1 ELSE 0 END) AS self_loops,min(r.similarity) AS minimum_cosine,max(r.similarity) AS maximum_cosine;

CYPHER 25
WITH 'relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1' AS ns
MATCH (a:RELEXEntity {namespace:ns})-[r:RELEX_REL {namespace:ns}]->(b:RELEXEntity {namespace:ns})
WITH count(r) AS factual_relations,count(DISTINCT r.canonical_triple_id) AS distinct_original_ids,
sum(CASE WHEN a=b THEN 1 ELSE 0 END) AS self_loops,
sum(CASE WHEN a.document_id<>b.document_id THEN 1 ELSE 0 END) AS cross_document,
reduce(all_ids=[], ids IN collect(r.source_triple_ids) | all_ids+ids) AS source_ids
UNWIND source_ids AS source_id
RETURN factual_relations,distinct_original_ids,self_loops,cross_document,count(DISTINCT source_id) AS raw_source_triple_ids;

CYPHER 25
WITH 'relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1' AS ns
MATCH (e {namespace:ns}) WHERE e:RELEXEntity OR e:RELEXConcept
WITH ns,e,
 EXISTS { MATCH (e)-[r:RELEX_REL|SIMILAR_TO|HAS_SHARED_CONCEPT]-(o {namespace:ns}) WHERE r.namespace=ns } AS touched,
 EXISTS { MATCH (e)-[r:RELEX_REL|SIMILAR_TO|HAS_SHARED_CONCEPT]-(o {namespace:ns}) WHERE r.namespace=ns AND o<>e } AS external
RETURN count(e) AS kg_vertices,sum(CASE WHEN NOT touched THEN 1 ELSE 0 END) AS isolated,
sum(CASE WHEN touched AND NOT external THEN 1 ELSE 0 END) AS self_loop_only_vertices,
sum(CASE WHEN NOT external THEN 1 ELSE 0 END) AS without_external_edges;

CYPHER 25
WITH 'relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1' AS ns
MATCH (e {namespace:ns}) WHERE e:RELEXEntity OR e:RELEXConcept
WITH ns,e ORDER BY e.key
WITH ns,collect(e) AS entities
UNWIND range(0,size(entities)-1) AS vertex
WITH ns,entities,vertex,entities[vertex] AS e
CALL (e,ns,entities) {
 OPTIONAL MATCH (e)-[r:RELEX_REL|SIMILAR_TO|HAS_SHARED_CONCEPT]-(other {namespace:ns}) WHERE r.namespace=ns
 WITH entities,collect(DISTINCT other) AS neighbors
 RETURN [i IN range(0,size(entities)-1) WHERE entities[i] IN neighbors] AS neighbor_indices
}
WITH entities,vertex,neighbor_indices ORDER BY vertex
WITH entities,collect(neighbor_indices) AS adjacency
WITH entities,reduce(labels=range(0,size(entities)-1), round IN range(1,size(entities)-1) |
 [i IN range(0,size(entities)-1) | reduce(best=labels[i], neighbor IN adjacency[i] |
 CASE WHEN labels[neighbor]<best THEN labels[neighbor] ELSE best END)]) AS labels
UNWIND range(0,size(entities)-1) AS vertex
WITH labels[vertex] AS component_id,entities[vertex].document_id AS document_id
WITH component_id,count(*) AS component_size,count(DISTINCT document_id) AS document_count
ORDER BY component_size DESC,component_id
RETURN count(*) AS weak_components,sum(component_size) AS kg_vertices,
max(component_size) AS largest_component_vertices,head(collect(document_count)) AS largest_component_documents;

CYPHER 25
WITH 'relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1' AS ns
MATCH (a:RELEXEntity {namespace:ns})-[r:RELEX_REL {namespace:ns}]->(b:RELEXEntity {namespace:ns})
WITH a.key AS head,r.relation_type AS relation_type,b.key AS tail,count(r) AS records
RETURN count(*) AS unique_structures,sum(records) AS factual_records,
sum(CASE WHEN records>1 THEN 1 ELSE 0 END) AS parallel_groups,sum(records-1) AS parallel_record_excess;

CYPHER 25
WITH 'relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1' AS ns
MATCH (c:RELEXConcept {namespace:ns})
OPTIONAL MATCH (e:RELEXEntity {namespace:ns})-[r:HAS_SHARED_CONCEPT {namespace:ns}]->(c)
RETURN c.concept_id AS concept_id,c.name AS concept,count(r) AS contextual_links,count(DISTINCT e.document_id) AS source_documents
ORDER BY contextual_links DESC,concept_id;

CYPHER 25
WITH 'relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1' AS ns
MATCH (d:RELEXDocument {namespace:ns})-[r:RELEX_MENTIONS {namespace:ns}]->(e:RELEXEntity {namespace:ns})
RETURN count(r) AS memberships,sum(r.source_membership_count) AS restored_original_memberships,
sum(CASE WHEN d.document_id<>e.document_id THEN 1 ELSE 0 END) AS mismatched_documents;

CYPHER 25
WITH 'relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1' AS ns
MATCH (c:RELEXConcept {namespace:ns})
RETURN count(c) AS concepts,
sum(CASE WHEN c.document_id IS NOT NULL THEN 1 ELSE 0 END) AS concepts_with_document_owner,
sum(CASE WHEN EXISTS { MATCH (c)-[r:RELEX_REL]-() WHERE r.namespace=ns } THEN 1 ELSE 0 END) AS concepts_with_factual_edges,
sum(CASE WHEN EXISTS { MATCH ()-[r:RELEX_MENTIONS]->(c) WHERE r.namespace=ns } THEN 1 ELSE 0 END) AS concepts_with_memberships;

CYPHER 25
WITH 'relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1' AS ns
MATCH (e:RELEXEntity {namespace:ns})-[r:HAS_SHARED_CONCEPT {namespace:ns}]->(c:RELEXConcept {namespace:ns,name:'Patients'})
RETURN e.canonical_entity_id AS contextual_id,e.name AS name,e.document_id AS document_id,
e.source_canonical_ids AS original_canonical_ids,e.entity_types AS original_type_union,r.source_canonical_ids AS concept_evidence_ids
ORDER BY document_id,contextual_id;
