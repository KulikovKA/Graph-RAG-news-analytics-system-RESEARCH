// Read-only: all nodes and edges, with cross-document edges in a separate column.
// Column aliases do NOT change stored relationship types or Browser colors.
// For visible captions select SIMILAR_TO > Caption > scope in Browser,
// or import visualization.grass through :style. Node limit >= 1000.
WITH 'relex_sem_e5_small_name_t095_ner_t046_max' AS ns
MATCH (n {namespace: ns})
OPTIONAL MATCH (n)-[r]->(m {namespace: ns})
WHERE r.namespace = ns
RETURN n, m,
       CASE WHEN r.scope = 'cross_document'
            THEN r END AS cross_document_link,
       CASE WHEN coalesce(r.scope, '') <> 'cross_document'
            THEN r END AS other_link;
