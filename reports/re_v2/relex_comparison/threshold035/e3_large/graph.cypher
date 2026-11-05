MATCH (e:RELEXEntity {namespace:'relex_large_t035'})
OPTIONAL MATCH (e)-[r:RELEX_REL {namespace:'relex_large_t035'}]->(t:RELEXEntity {namespace:'relex_large_t035'})
RETURN e,r,t;

MATCH (d:RELEXDocument {namespace:'relex_large_t035'})-[m:RELEX_MENTIONS {namespace:'relex_large_t035'}]->(e) RETURN d,m,e;
