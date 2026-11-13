// Semantic entity--concept--entity bridges involving a new membership.
MATCH (a:RELEXEntity {namespace:'relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1'})-[r1:HAS_SHARED_CONCEPT {namespace:'relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1'}]->(c:RELEXConcept {namespace:'relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1'})<-[r2:HAS_SHARED_CONCEPT {namespace:'relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1'}]-(b:RELEXEntity {namespace:'relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1'})
WHERE a.document_id<b.document_id AND
 (coalesce(r1.resolver_version,'')='dynamic-concepts-v1' OR coalesce(r2.resolver_version,'')='dynamic-concepts-v1')
RETURN a.document_id AS document_a,b.document_id AS document_b,a,r1,c,r2,b;
