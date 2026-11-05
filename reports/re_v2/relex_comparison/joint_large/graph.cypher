MATCH (e:RELEXEntity {namespace:'relex_joint_large'})
OPTIONAL MATCH (e)-[r:RELEX_REL {namespace:'relex_joint_large'}]->(t:RELEXEntity {namespace:'relex_joint_large'})
RETURN e,r,t;

MATCH (d:RELEXDocument {namespace:'relex_joint_large'})-[m:RELEX_MENTIONS {namespace:'relex_joint_large'}]->(e) RETURN d,m,e;
