# Этап 1C — Relation Remapping

Источник: неизменённое отображение 1B для relex_base_t035 и исходные relations.jsonl. Использованы все 1327 mention-записей для построения однозначного отображения 1211 entity ID на 1122 канонических ID. Дополнительное Entity Resolution не выполнялось.

| Показатель | Значение |
|---|---:|
| Исходных сущностей | 1211 |
| Канонических сущностей | 1122 |
| Исходных записей отношений | 1971 |
| Переназначенных записей | 1971 |
| Уникальных канонических троек | 1760 |
| Устранённых структурных дублей | 211 |
| Групп структурных дублей | 136 |
| Исходных self-loops (записей) | 0 |
| Self-loops после переназначения (записей) | 15 |
| Из них созданы объединением сущностей | 15 |
| Уникальных self-loop троек | 13 |
| Из них созданы объединением сущностей | 13 |
| Исходных изолированных сущностей | 353 |
| Изолированных канонических сущностей | 324 |

Ключ дедупликации: `(document_id, canonical_head_id, relation_type, canonical_tail_id)`. Head/tail не переставляются, типы сохраняются. canonical_triple_id — ct:SHA-256 JSON-массива ключа; это идентификатор записи в артефакте, не новый узел графа.

remapped_relations.jsonl: одна строка на каждое исходное отношение в исходном порядке. Только head_id/tail_id заменены отображением 1B. original_head_id/original_tail_id сохраняют исходные конечные точки; source_relation содержит полную неизменённую исходную запись, включая все неизвестные скрипту поля. Остальные исходные поля продублированы сверху без изменений.

canonical_relations.jsonl: одна строка на уникальную тройку; source_triple_ids, source_confidences и source_relations сохраняют полный состав объединения и все исходные confidence, evidence и provenance. Единого агрегированного confidence нет; оценки не усреднялись и не пересчитывались. Записи упорядочены по ключу, источники внутри группы — по triple_id.

is_self_loop отмечает петлю, self_loop_created_by_resolution отличает петлю после объединения от исходной; created_self_loop_source_triple_ids перечисляет её источники. Все петли сохранены, их корректность здесь не оценивается. Изолированный узел не имеет ни входящего, ни исходящего предметного отношения; self-loop считается связью. Служебные связи документа не учитываются.

Evidence, включая содержащиеся в нём entity_ids, offsets и текст, сохранено в исходной системе ID как provenance. Типы конечных точек из исходных отношений также не заменяются и не исправляются. Переназначены только конечные точки отношений; Relation Verification и оценка семантической корректности не выполнялись.

Проверены уникальность/полнота исходных ID, однозначность и идемпотентность отображения, покрытие всех 1971 отношений, наличие всех конечных точек среди 1122 канонических сущностей, сохранность направления, типа и полных исходных записей. Источники уникальных троек образуют точное разбиение исходных triple ID: нет потерь, повторного учёта или непрослеживаемых отношений. Междокументных объединений/отношений нет. Повторный запуск проверен побайтовым сравнением всех трёх артефактов.

Запуск: `python scripts/remap_canonical_relations.py`. Только стандартная библиотека и функции чтения/хеширования из 1A; NER/RE/LLM не запускаются. Neo4j не подключается, граф не создаётся и не импортируется. Выполнен исключительно этап 1C.

SHA-256 входов и защищённых артефактов 1A/1B (проверяются до и после записи):

- `reports/entity_resolution/relex_base_t035/conservative_resolution/canonical_mapping.jsonl`: `f97e8f920cf7cced3483caca39bdf87079f2b8a9c7d5240b812c886cc766c8b3`
- `reports/entity_resolution/relex_base_t035/conservative_resolution/report.md`: `d9ae051c9bb57019f215aa5db9ab74150c144358850324337460db4b3930f9cc`
- `reports/entity_resolution/relex_base_t035/conservative_resolution/review_groups.csv`: `6ea86012816fc3d04603fc61a778c6577351611fba91984611f2beacb060551b`
- `reports/entity_resolution/relex_base_t035/exact_matching_audit/ambiguous_groups.csv`: `fdf596b77ed5e0fd063b52a64ee814da14675085d1520d4eb70e1606ce3e0795`
- `reports/entity_resolution/relex_base_t035/exact_matching_audit/exact_groups.csv`: `fda84b2faa3fddf13103f711ab1cf6bd04b86b802691b6d9d5c005b1583e0972`
- `reports/entity_resolution/relex_base_t035/exact_matching_audit/exact_merge_preview.jsonl`: `d68c01c9d72622a29deb3378079d6887641baef4f590a949bf2db9918bacd855`
- `reports/entity_resolution/relex_base_t035/exact_matching_audit/report.md`: `92158a6653b224524cbc231f04470465db3542771a6521187bda3863e84b595b`
- `reports/re_v2/relex_comparison/threshold035/e3_base/entities.jsonl`: `19039d87119130dcc37d2c472a7f2242c6a3ad5059e46a1ff431e6fa37e3b6ba`
- `reports/re_v2/relex_comparison/threshold035/e3_base/relations.jsonl`: `bbb4e602abccedcc776b59cfd526f8ffe7ffb46bdea7f7dc3c38df33174dd099`
