// Neo4j Browser: set Graph visualization node limit >= 1000.
WITH 'relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v2_sem_name_t0830' AS ns
MATCH (n {namespace: ns})
OPTIONAL MATCH (n)-[r]->(m {namespace: ns})
WHERE r.namespace = ns
RETURN n, r, m;
