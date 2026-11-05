// Includes all entities, including isolated nodes.
MATCH (e:RELEXEntity {namespace: 'relex_large'})
OPTIONAL MATCH (e)-[r:RELEX_REL {namespace: 'relex_large'}]->(t:RELEXEntity {namespace: 'relex_large'})
RETURN e, r, t;

MATCH (d:RELEXDocument {namespace: 'relex_large'})-[r:RELEX_MENTIONS {namespace: 'relex_large'}]->(e:RELEXEntity {namespace: 'relex_large'})
RETURN d,r,e;
