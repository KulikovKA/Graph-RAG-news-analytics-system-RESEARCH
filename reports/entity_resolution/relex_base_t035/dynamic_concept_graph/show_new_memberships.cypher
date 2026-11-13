// Only newly accepted memberships; no implied factual relationship.
MATCH (e:RELEXEntity {namespace:'relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1'})-[r:HAS_SHARED_CONCEPT {namespace:'relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1'}]->(c:RELEXConcept {namespace:'relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1'})
WHERE r.resolver_version='dynamic-concepts-v1'
RETURN e,r,c;
