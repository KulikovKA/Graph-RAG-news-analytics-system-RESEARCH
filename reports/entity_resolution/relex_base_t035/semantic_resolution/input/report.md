# Этап 2A — входные данные для Semantic Entity Resolution

Подготовлен набор без запуска моделей, расчёта embeddings/cosine similarity и поиска пар. Использовано существующее отображение 1B; решения об идентичности и типы не пересматривались.

| Проверка покрытия | Результат |
|---|---:|
| Уникальные канонические сущности | 1122 |
| Исходные entity ID (каждый ровно в одной канонической записи) | 1211 |
| Исходные mention ID (каждый ровно в одной канонической записи) | 1327 |
| Документы | 30 |
| Названия / канонические представления с контекстом | 1122 / 1122 |
| Отдельные контексты исходных span | 1211 |

## Состав и воспроизводимое представление

Одна строка canonical_entities.jsonl соответствует одному canonical_entity_id. surface_text и entity_types взяты без изменений у существующего представителя 1B. source_entity_ids, source_mention_ids и source_offsets задают полный состав. members хранит каждую исходную сущность целиком в source_entity, включая confidence, source_mentions с их score/типами/provenance; отдельно сохранены контекст, два текста и исходное основание resolution_1b. provenance содержит SHA-256 трёх входов и метаданные исходного документа с revision_id и source_reference.

Рецепт `source-name-context-v1`: text_name — исходный surface_text без нормализации и удаления модификаторов. Для каждого исходного span извлекается точный срез документа [max(0,start−120), min(len(text),end+120)); offsets — Python Unicode-индексы [start,end). Контекст включает весь span; его локальные границы сохранены. text_name_context имеет вид `Entity: <исходное название>\nContext: <исходный срез>`. Префиксы конкретных embedding-моделей не добавляются.

Каноническая запись использует только собственный контекст упоминания-представителя 1B, явно указанного в context_source_entity_id. Контексты других исходных span той же сущности остаются раздельными в members. Тексты разных канонических сущностей не конкатенируются. Перекрытие исходных окон возможно и не означает объединения сущностей. Не выполняются суммаризация, исправление текста, переназначение типов или выбор контекста по семантической похожести.

Длина сохранённых контекстов: 123–310 символов; средняя 242.33. Длина ограничивается символьным окном, не tokenizer конкретной модели. Повреждения и исходные пробелы сохраняются; при отсутствии допустимого контекста text_name_context был бы null, без искусственной подстановки.

## Потенциальные проблемы входа

Числа в таблице — количество затронутых записей, не число отдельных повреждённых символов. Флаги могут пересекаться. Несколько исходных типов и partial_token сами по себе не доказывают ошибку.

| Флаг | Канонических записей | Исходных span |
|---|---:|---:|
| canonical_member_type_conflict | 0 | — |
| damaged_context | 40 | 40 |
| damaged_name | 15 | 15 |
| document_text_hash_mismatch | 0 | — |
| missing_context | 0 | 0 |
| multiple_original_types | 108 | 108 |
| original_partial_token | 10 | 10 |
| source_mention_type_set_mismatch | 0 | 0 |
| source_span_text_mismatch | 0 | 0 |

damaged_name/context отмечают U+FFFD либо управляющий символ, не являющийся пробельным. Повреждённые названия включают формулы Co–Fe–Zr в W2317432785; символы не заменялись. multiple_original_types сохраняет исходную множественную типизацию, canonical_member_type_conflict означает разные наборы типов у участников одной канонической сущности. Между разными каноническими сущностями типы не сопоставлялись и пары не искались. original_partial_token перенесён из исходной NER-разметки. Качество/смысл терминов не оценивались.

## Проверки и ограничения

Проверены все 1122 canonical ID, 1211 entity ID, 1327 mention ID и 30 документов; однозначность отображения, состав групп 1B, отсутствие междокументного смешения, сохранность всех исходных entity/mention полей и provenance. Контексты сверены с исходным документом по абсолютным offsets, а упоминания — по локальным offsets. Хеши текста документов и revision_id проверены, найденные несоответствия перечислялись бы флагами. Повторный запуск проверен побайтовым сравнением двух выходных файлов; восстановление исходных 1211 сущностей проверено по members.source_entity.

Выбор единственного контекста представителя детерминирован, но не гарантирует оптимальное снятие полисемии. Все остальные контексты доступны раздельно. Это входной набор, не embeddings, не список кандидатов и не оценка точности ER. Этап 2B не выполнялся.

Запуск: `python scripts/prepare_semantic_entity_input.py`. Только стандартная библиотека и функции чтения/хеширования 1A. Создаются только canonical_entities.jsonl и report.md в semantic_resolution/input. Neo4j и результаты 1A–1D не изменяются.

SHA-256 входов и защищённых результатов 1A–1D:

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
- `reports/ner_v2/research_run2/input_abstracts.jsonl`: `684ca3a758a128c6382ab924f5c131f0e27faa9f7625b431ae0d6db975e543b1`
- `reports/re_v2/relex_comparison/threshold035/e3_base/entities.jsonl`: `19039d87119130dcc37d2c472a7f2242c6a3ad5059e46a1ff431e6fa37e3b6ba`
