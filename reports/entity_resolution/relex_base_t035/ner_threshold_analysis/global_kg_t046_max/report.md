# Global KG: NER ≥ 0.46, max member confidence

Фактически импортирован и проверен в Neo4j.

Namespace: `relex_sem_e5_small_name_t095_ner_t046_max`. Исходный namespace `relex_sem_e5_small_name_t095` сохранён.

| Показатель | Значение |
|---|---:|
| Документы | 30 |
| Сущности | 905 |
| Удалено сущностей относительно baseline | 217 |
| RELEX_REL | 1307 |
| SIMILAR_TO | 492 |
| Междокументные / внутридокументные SIMILAR_TO | 121 / 371 |
| RELEX_MENTIONS | 905 |
| Все физические узлы / отношения | 935 / 2704 |
| Изолированные entity-вершины | 175 |
| Слабосвязные компоненты entity KG | 230 |
| Крупнейшая компонента: вершины / документы | 378 / 17 |
| Исходные фактические петли | 12 |

Фильтрация: максимум confidence всех members существующей canonical entity ≥ 0.46. Исходный cutoff NER 0.35, ReLEx 0.35, E5-small/name cosine threshold 0.95 сохранены. Ребро сохраняется только по наличию обоих концов; его confidence, similarity, тип и provenance не пересчитываются. Все 30 документов сохранены. Связность считается без документов и RELEX_MENTIONS.

Entity `confidence` и `max_member_confidence` — максимум по members, `confidence_source_entity_id` указывает его источник. `representative_confidence` — исходный confidence канонического представителя. `start/end`, `span_source_entity_id` и `representative_entity_id` по-прежнему относятся к представителю; score другого member не подменяет offsets. Полный исходный canonical объект и все member scores сохранены в source_json и source_confidences_json.

Полный граф в Browser (включая документы, междокументные связи и изолированные сущности):

```cypher
MATCH (n {namespace: 'relex_sem_e5_small_name_t095_ner_t046_max'})
OPTIONAL MATCH (n)-[r]->(m {namespace: 'relex_sem_e5_small_name_t095_ner_t046_max'})
WHERE r.namespace = 'relex_sem_e5_small_name_t095_ner_t046_max'
RETURN n, r, m;
```

OPTIONAL MATCH сохраняет строки вершин без исходящих рёбер. Начальный MATCH включает все документы и все сущности, поэтому изолированные entity-вершины тоже видны; они могут иметь служебную связь от документа. Возвращаются RELEX_MENTIONS, RELEX_REL и SIMILAR_TO. SIMILAR_TO хранится одним физическим ребром на пару, без удвоения. Для полного отображения установите Graph visualization node limit ≥ 1000 (или 1500); полный результат — 935 вершин, 2704 отношения.

Файлы: `show_full_graph.cypher`, `graph_stats.cypher`, `manifest.json`, `membership.json`, `entities.csv`, `documents.csv`, `edges.csv`. CSV — компактное представление; исходные полные свойства восстанавливаются неизменённым импортёром из JSONL/CSV предыдущих этапов.

Повторный импорт: `.venv/Scripts/python.exe scripts/import_ner_filtered_global_graph.py` с существующими NEO4J_* либо `--neo4j-env <existing-file>`. `--prepare-only` создаёт локальные артефакты без доступа к Neo4j. Другой fingerprint в целевом namespace приводит к отказу; удалений, смены пароля и изменения конфигурации сервера нет.

Модели NER/ReLEx/embeddings/LLM, Entity Merge и пересчёт cosine не выполнялись.

Fingerprint: `dd36a0f38729c87cf609a82dbb56cf7dd9fdedbef909d693ae2bb4bec7ee29a9`.

Проверки фактического импорта:

- Прочитаны все узлы и отношения с проверкой ID, типов, направлений, всех ожидаемых свойств и provenance; связность вычислена по прочитанному графу.
- Второй MERGE-проход не создал узлов или отношений, полное повторное чтение совпало.
- Все 9 read-only запросов graph_stats.cypher выполнены; WCC подтверждены независимым live Cypher, без GDS/APOC.
- Полный запрос визуализации выполнен: подтверждено покрытие всех 935 узлов и всех 2704 отношений.
- SHA-256 полного снимка узлов и отношений вне нового namespace совпал до и после; предыдущие графы не изменены.
- SHA-256 входных файлов и предыдущих отчётов не изменились.

Время проверки (Europe/Moscow): 2026-11-11T16:26:48.197125+03:00. Подробности: `verification.json`.
