MATCH (e:RELEXEntity {namespace:'relex_joint_base'})
OPTIONAL MATCH (e)-[r:RELEX_REL {namespace:'relex_joint_base'}]->(t:RELEXEntity {namespace:'relex_joint_base'})
RETURN e,r,t;

MATCH (d:RELEXDocument {namespace:'relex_joint_base'})-[m:RELEX_MENTIONS {namespace:'relex_joint_base'}]->(e) RETURN d,m,e;
