# Этап 2E — фактический импорт Global KG

Namespace: `relex_sem_e5_small_name_t095`. База: `neo4j`. Конфигурация: `intfloat/e5-small-v2`, `text_name`, threshold `0.95`. Фактическая проверка UTC: `2026-11-10T13:50:11.913280+00:00`.

| Показатель | Фактически в Neo4j |
|---|---:|
| Документы RELEXDocument | 30 |
| Канонические сущности RELEXEntity | 1122 |
| Исходные фактические RELEX_REL | 1760 |
| SIMILAR_TO | 685 |
| Междокументные / внутридокументные SIMILAR_TO | 194 / 491 |
| Служебные RELEX_MENTIONS | 1122 |
| Всего физических узлов / отношений | 1152 / 3567 |
| Рёбра KG без служебных связей | 2445 |
| Изолированные сущности | 200 |
| Доля изолированных | 17.83% |
| Слабосвязные компоненты | 258 |
| Крупнейшая компонента: вершины / документы | 580 / 23 |

Все показатели связности совпали с 2D. Они рассчитаны по сущностям и рёбрам RELEX_REL/SIMILAR_TO, без документов и RELEX_MENTIONS. Исходные 13 фактических петель сохранены. Новых петель и фактических междокументных отношений нет.

SIMILAR_TO хранится одним физическим отношением на неупорядоченную пару, в стабильном порядке canonical IDs, с directed=false и undirected=true. Для обхода используется неориентированный шаблон. Два встречных ребра не создаются. На ребре сохранены точный cosine, модель, режим, порог, scope, оба документа, исходные ID, флаги и решения 2C. Эти решения касаются тождества и не доказывают корректность SIMILAR_TO.

Схема следует scripts/build_relex_graphs.py: RELEXDocument/RELEXEntity, namespace-prefixed key, фактические типы в relation_type у RELEX_REL, принадлежность документа через RELEX_MENTIONS. Старый валидатор namespace и старый код импорта не изменены. Полные исходные объекты сохранены как source_json; вложенные структуры дополнительно хранятся в *_json. Все исходные confidence, offsets, evidence, source triple/entity/mention IDs и provenance доступны без усреднения. Плоские confidence/start/end сущности относятся только к исходному canonical ID; полные списки упоминаний сохранены.

## Проверки

- Проход MERGE перед финальной проверкой добавил: `{"nodes": 0, "relationships": 0}`. Это счётчики текущего запуска, а не размер namespace; при уже выполненном импорте ожидается 0.
- Повторный импорт создал: `{"nodes": 0, "relationships": 0}`; полное чтение свойств подтвердило отсутствие изменений и дублей.
- Все 30 документов, 1122 сущности, 1760 фактических и 685 семантических отношений прочитаны из Neo4j и сопоставлены с подготовленным payload, включая направления, типы и все свойства/provenance.
- Связность по фактически прочитанным узлам/рёбрам подтверждена независимым живым Cypher-запросом: распространение минимального индекса вершины по соседям за V−1 раундов, без GDS/APOC и кэшированных компонент. Это локальный расчёт внутри read-only запроса, не изменение вершин/типов.
- Все 9 запросов graph_stats.cypher выполнены: `[{"notifications": 0, "query": 1, "rows": 1}, {"notifications": 0, "query": 2, "rows": 1}, {"notifications": 0, "query": 3, "rows": 20}, {"notifications": 0, "query": 4, "rows": 1}, {"notifications": 0, "query": 5, "rows": 1}, {"notifications": 0, "query": 6, "rows": 1}, {"notifications": 0, "query": 7, "rows": 10}, {"notifications": 0, "query": 8, "rows": 20}, {"notifications": 0, "query": 9, "rows": 10}]`.
- Данные вне целевого namespace не изменились: `{"nodes": 8221, "relationships": 11412, "sha256": "84552c94d0b58a06cc21924cbdf7522ba849b5680c96db4f814405f8e33f2883"}` (сравнивались ID, labels, endpoints, типы и все свойства узлов/рёбер до и после).
- SHA-256 входных файлов и прошлых артефактов совпали до и после. Никакие NER/RE/embedding/LLM-модели не запускались. Пароли и конфигурация сервера не менялись и не записывались в отчёт.

## Cypher и повторный импорт

graph_stats.cypher содержит 9 самостоятельных запросов для Neo4j 2026.08.1 / Cypher 25: общие количества, scope семантических связей, междокументные примеры, изоляция и степень, точные слабые компоненты, охват документов, топ документов, топ сущностей и реальные междокументные пути. Используется CALL (variables), без устаревшего importing WITH. Запрос компонент выполняет точное распространение минимальных индексов по спискам смежности, без перечисления путей и ограничений на диаметр. После V−1 раундов метка совпадает у всех и только у вершин одной компоненты. Данные читаются непосредственно из БД.

Синтаксис: [CALL scope](https://neo4j.com/docs/cypher-manual/current/subqueries/call-subquery/index.html), [reduce](https://neo4j.com/docs/cypher-manual/current/functions/list/).

Повторный импорт: `.venv/Scripts/python.exe scripts/import_global_semantic_graph.py` с действующими NEO4J_* в окружении, либо `--neo4j-env <existing-file>`. Учётные данные не встроены в скрипт. MERGE ограничен единственным namespace; другой fingerprint в нём приводит к отказу без перезаписи. Все записи одного прохода выполняются одной транзакцией с проверкой количеств перед commit.

Fingerprint: `79fbee0d4a3dd89d23e20da7c33d9939c7d02cb814e09ccb2effa01a3a69437b`. Neo4j: `[{"edition": "community", "name": "Neo4j Kernel", "versions": ["2026.08.1"]}, {"edition": "", "name": "Cypher", "versions": ["5", "25"]}]`.

Качество Graph-RAG и достоверность новых семантических переходов не оценивались. Импорт не означает подтверждения эквивалентности сущностей. Этап 2E завершён.

SHA-256 входов и защищённых артефактов:

- `reports/entity_resolution/relex_base_t035/conservative_resolution/canonical_mapping.jsonl`: `f97e8f920cf7cced3483caca39bdf87079f2b8a9c7d5240b812c886cc766c8b3`
- `reports/entity_resolution/relex_base_t035/conservative_resolution/report.md`: `d9ae051c9bb57019f215aa5db9ab74150c144358850324337460db4b3930f9cc`
- `reports/entity_resolution/relex_base_t035/conservative_resolution/review_groups.csv`: `6ea86012816fc3d04603fc61a778c6577351611fba91984611f2beacb060551b`
- `reports/entity_resolution/relex_base_t035/exact_matching_audit/ambiguous_groups.csv`: `fdf596b77ed5e0fd063b52a64ee814da14675085d1520d4eb70e1606ce3e0795`
- `reports/entity_resolution/relex_base_t035/exact_matching_audit/exact_groups.csv`: `fda84b2faa3fddf13103f711ab1cf6bd04b86b802691b6d9d5c005b1583e0972`
- `reports/entity_resolution/relex_base_t035/exact_matching_audit/exact_merge_preview.jsonl`: `d68c01c9d72622a29deb3378079d6887641baef4f590a949bf2db9918bacd855`
- `reports/entity_resolution/relex_base_t035/exact_matching_audit/report.md`: `92158a6653b224524cbc231f04470465db3542771a6521187bda3863e84b595b`
- `reports/entity_resolution/relex_base_t035/graph_comparison/report.md`: `cff7dc986261ff26ed376ae309443291b1381fda3b349b53b6a6f4261815f14c`
- `reports/entity_resolution/relex_base_t035/relation_remapping/canonical_relations.jsonl`: `febcd7b9e717575bed79e3e1483402fdd81aae54b59d72b81d6b91582fccda6a`
- `reports/entity_resolution/relex_base_t035/relation_remapping/remapped_relations.jsonl`: `4b0ea8f647aed02a4b9a42a274d1e87611a2158fffe10cf3362f1d83884025c6`
- `reports/entity_resolution/relex_base_t035/relation_remapping/report.md`: `b760c94584a5f7dbf27bf410e88f678b1fba711292206f622c52ce1b69e546a4`
- `reports/entity_resolution/relex_base_t035/semantic_resolution/candidate_verification/proposed_merge_rules.md`: `38b22237a006cce27907a3ded4d2f5045d82b1f9a64d183f2e3718095c78a161`
- `reports/entity_resolution/relex_base_t035/semantic_resolution/candidate_verification/report.md`: `01753a6f060a9d7891f25e25a0ddb1441dc241b9257ebe6e9999c6f9e2cb17d1`
- `reports/entity_resolution/relex_base_t035/semantic_resolution/candidate_verification/verified_pairs.csv`: `b181f6466634f639ed9cde9c3a625ce2a7f26154b31bbfdecfabc96f93c49220`
- `reports/entity_resolution/relex_base_t035/semantic_resolution/global_linking_preview/report.md`: `b941a0080c70c0229d07e7538e240faf6c96b5718f027f9c806574a2999a1231`
- `reports/entity_resolution/relex_base_t035/semantic_resolution/global_linking_preview/semantic_links.csv`: `6a87703a6114b2e52c90bd3a69f7f475ef68f8f26a5c9dd5ee3ed380d5570cc1`
- `reports/entity_resolution/relex_base_t035/semantic_resolution/input/canonical_entities.jsonl`: `34fe3fd42e110b5b7a8b054360eee2a6182c67782e7afa54d2298e0491bf9151`
- `reports/entity_resolution/relex_base_t035/semantic_resolution/input/report.md`: `bd99b9d37b9461fcf932ff37428d1cbefb0d7fabb2b40977ae73430a212714dc`
- `reports/entity_resolution/relex_base_t035/semantic_resolution/model_comparison/candidate_pairs.csv`: `5b5101945ce513243d3a28969b77425c4e1549a96bb31bb616546727d815ae2f`
- `reports/entity_resolution/relex_base_t035/semantic_resolution/model_comparison/model_comparison.csv`: `e89fa1b0d133f33c0710b6bf595e1bbddc7fcc6c21d77bc2c9c222bf454aac6c`
- `reports/entity_resolution/relex_base_t035/semantic_resolution/model_comparison/report.md`: `40f5f0d7d5b0100a02c83b774430ec6538b99e3fde668c030f75c43c81a881bc`
- `reports/entity_resolution/relex_base_t035/semantic_resolution/threshold_sweep/report.md`: `68d301a45cdb24334fc189e4253a455995a01966eaf718d425a33386fbe9f017`
- `reports/entity_resolution/relex_base_t035/semantic_resolution/threshold_sweep/threshold_results.csv`: `f3a26801a3a505355043111cdab8bb9d99a544ea49ee59324dcbe5ccf0200f2a`
- `reports/ner_v2/research_run2/input_abstracts.jsonl`: `684ca3a758a128c6382ab924f5c131f0e27faa9f7625b431ae0d6db975e543b1`
- `scripts/build_relex_graphs.py`: `52bba8372b64abc660acff82a47987e752f9f2caa85a618f7cddea52b046ab85`
