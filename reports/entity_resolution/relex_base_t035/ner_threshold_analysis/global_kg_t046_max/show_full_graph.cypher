// All 935 nodes and all 2704 relationships, including isolates and documents.
// Set Neo4j Browser Graph visualization node limit >= 1000 before running.
MATCH (n {namespace: 'relex_sem_e5_small_name_t095_ner_t046_max'})
OPTIONAL MATCH (n)-[r]->(m {namespace: 'relex_sem_e5_small_name_t095_ner_t046_max'})
WHERE r.namespace = 'relex_sem_e5_small_name_t095_ner_t046_max'
RETURN n, r, m;
