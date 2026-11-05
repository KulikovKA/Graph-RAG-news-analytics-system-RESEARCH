MATCH (e:RELEXEntity {namespace:'relex_base'})
OPTIONAL MATCH (e)-[r:RELEX_REL {namespace:'relex_base'}]->(t:RELEXEntity {namespace:'relex_base'})
RETURN e,r,t;

MATCH (d:RELEXDocument {namespace:'relex_base'})-[m:RELEX_MENTIONS {namespace:'relex_base'}]->(e) RETURN d,m,e;
