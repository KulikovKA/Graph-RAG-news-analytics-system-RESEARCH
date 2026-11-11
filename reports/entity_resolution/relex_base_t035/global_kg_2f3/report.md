# Stage 2F.3 — Global KG Reconstruction & Neo4j Import

**Фактический импорт выполнен и проверен.**

Target: `relex_sem_e5_small_name_t095_ner_t046_er2f3`. Source: `relex_sem_e5_small_name_t095_ner_t046_max`. Дата: 12 ноября 2026, Europe/Moscow.

Применён только утверждённый mapping 2F.2: 69 групп /163 canonical IDs, устранены 94 повтора. Новые решения ER не выполнялись. NER 0.46/max_member, E5-small/name cosine 0.95 сохранены.

| Метрика | 2F.1 | 2F.3 |
|---|---:|---:|
| Документы | 30 | 30 |
| Контекстные сущности | 905 | 811 |
| RELEXConcept | 0 | 32 |
| KG-вершины без документов | 905 | 843 |
| Физические узлы | 935 | 873 |
| RELEX_REL | 1307 | 1307 |
| SIMILAR_TO | 492 | 492 |
| HAS_SHARED_CONCEPT | 0 | 118 |
| RELEX_MENTIONS | 905 | 811 |
| Физические отношения | 2704 | 2728 |
| Изолированные KG-вершины (петля = связь) | 175 | 172 |
| Слабосвязные компоненты KG | 230 | 223 |
| Крупнейшая компонента: вершины / документы | 378 / 17 | 369 / 17 |
| Вершины только с self-loops, без внешних KG-связей | — | 5 |
| Все вершины без внешних KG-связей | — | 177 |

## Сохранность и представление

- Все 905 исходных canonical IDs однозначно отображены в 811 существующих representatives. Документы внутри групп не смешиваются. `canonical_mapping.jsonl` сохраняет это отображение.
- У entity `source_json.source_canonical_nodes` содержит полные снимки всех исходных entity-узлов, включая их original source_json, offsets, contexts, scores, types и provenance. Восстановимы 993 source entity IDs и 1101 typed mention ID. `entity_types` — объединение исходных наборов; proposed_primary_type из 2F.2 не применяется.
- Entity confidence — максимум сохранённых native member confidences; все индивидуальные source scores сохранены отдельно. Start/end относятся только к representative/span_source_entity_id. Новые NER scores не рассчитываются.
- **Все 1307 RELEX_REL физически сохранены раздельно**, ID не пересчитаны. Их 1490 raw source triple IDs, исходные confidence/evidence/source_json сохранены. Endpoints переназначены по mapping; original_head_id/original_tail_id и source_relationship_json сохраняют исходное состояние.
- **Все 492 SIMILAR_TO сохранены**, включая 37 REJECT и 70 REVIEW из 2F.2. Решения хранятся как диагностические свойства resolution_decision_2f2/preview_status_2f2 и не фильтруют граф. Не используется display-view из 2F.2 с сокращённым набором рёбер.
- У SIMILAR_TO сохраняется semantic_link_id, точный исходный cosine и original endpoints; current left_canonical_id/right_canonical_id переназначены. Имена/типы/contexts и source IDs evidence на ребре относятся к исходным endpoints; полное исходное состояние есть в source_relationship_json. Прямые междокументные SIMILAR_TO: 121, внутридокументные: 371.
- Parallel edges сохраняют разные relationship keys. Физических factual edges 1307, структурных (head,type,tail) 1076; 164 parallel groups, 231 дополнительных записей. Factual self-loops: 39 (12 прежних +27 вследствие mapping), semantic self-loops: 72. Они не удаляются и не требуют нового содержательного аудита этого этапа.
- У 811 RELEX_MENTIONS поле source_memberships_json хранит полные 905 прежних membership records, включая их keys, endpoints и свойства. Концепты не имеют single document_id и не получают MENTIONS или RELEX_REL.
- 32 concept IDs и ровно 118 HAS_SHARED_CONCEPT взяты из готовых файлов. Source IDs, исходные типы, контекстное evidence и provenance сохранены. Это слой абстракции, не наследование фактов.

## Связность и изоляция

KG состоит из RELEXEntity+RELEXConcept и RELEX_REL+SIMILAR_TO+HAS_SHARED_CONCEPT, без документов и RELEX_MENTIONS. Для слабых компонент направления игнорируются, параллельные записи сохраняются. Петля даёт ненулевую степень, поэтому вершина только с петлями не входит в 172 изолированные. Её считают отдельно в self_loop_only_vertices.json. Число документов в компоненте учитывает только contextual entities: concept не добавляет фиктивный документ.

Только self-loops: **5**; без каких-либо внешних связей: **177 = 172 + 5**. Рост связности или уменьшение числа вершин не интерпретируется как доказательство улучшения качества Graph-RAG.

## Примеры концептов и Patients

Концепты включают Patients, Convergence, Genes, Eyes, Intraocular pressure, Energy consumption и другие 32 утверждённых класса. Общего Results нет. Концепты не получают факты исходных когорт/результатов.

`concept:2f2:patients` имеет 16 contextual links в пяти документах; их evidence охватывает 20 прежних contextual IDs, включая qualified patient forms. Когорта 21 ESRD/TKA-пациент объединяет только утверждённые N398/N399/N400; подгруппы other 20, six with complications, nineteen improved outcomes и Most patients остаются раздельными.

Другие документные контексты: 10 CRSwNP patients, 40 GDD/PK patients, 16 esophageal-cancer/RT patients и общие популяции rHuEPO recipients/подгруппа с антителами. В rHuEPO-работе не домысливается собственная enrolled clinical cohort. Concept Patients — общая категория, не один коллектив; ни одно RELEX_REL не перенесено на неё. Полные source scopes и исходные типы остаются при contextual nodes.

## Импорт и воспроизведение

Импорт использует существующие connect_neo4j, DOCUMENT_QUERY, EDGE_QUERY и namespace-scoped MERGE patterns предыдущих импортёров. Создана только новая область TARGET; новые глобальные schema constraints не добавлялись. Ключи включают namespace и стабильный source/concept ID. Другой fingerprint в TARGET вызывает отказ до записи. Первичное создание узлов/рёбер выполняется одной транзакцией с проверкой counts до commit. Если TARGET уже содержит полный снимок с тем же fingerprint, импортёр полностью сверяет все свойства, ID и endpoints и выполняет no-op без повторных SET. Это обеспечивает идемпотентность без расходования transaction memory на повторную запись больших provenance-строк.

Настоящая библиотека tqdm показывает обработку 905 сущностей, 69 групп, 811 итоговых сущностей, 32 концептов, 118 class links, 1307 factual edges, 492 SIMILAR_TO и 811 memberships; импорт и read-back показывают реальные counts. Batch-progress обновляется после успешного server response. Неопределённые по проценту операции полного hash-снимка сопровождаются CAPS-сообщениями. Искусственных задержек нет. При ошибке выводятся stage, последний успешно обработанный batch и исключение; success-banner не печатается при prepare-only/ошибке.

```powershell
.venv/Scripts/python.exe scripts/reconstruct_global_kg_2f3.py
```

Нужны существующие NEO4J_* либо --neo4j-env <existing-file>. --prepare-only читает baseline и подготавливает файлы без записи в Neo4j. Credentials не записываются в артефакты. Зависимости neo4j/tqdm уже объявлены в extra relex; модели не загружаются.

Полный граф: выполнить show_full_graph.cypher; Node visualization limit ≥1000. Запрос возвращает все 873 узла и 2728 отношений, включая documents, concepts и KG-isolates с их служебными mentions. graph_stats.cypher содержит 10 самостоятельных read-only проверок, в том числе live WCC без GDS/APOC.

## Подтверждённые результаты проверок

Время: 2026-11-12T00:57:58.404120+03:00. База: `neo4j`.

- Первый проход: `{"nodes": 0, "relationships": 0}`; повторный проход: `{"nodes": 0, "relationships": 0}`.
- После обоих проходов все 873 nodes и 2728 relationships полностью прочитаны; exact keys/labels/types/endpoints/properties совпадают с payload. SHA-256 целевого графа одинаков после двух проходов.
- Из прочитанных свойств строго восстановлены все 905 source entity nodes и все 2704 source relationships, включая 905 original memberships; полные source property dictionaries совпали с baseline.
- Все 10 live Cypher-проверок прошли; независимый Cypher WCC подтвердил 843/172/223/369/17. Проверены parallel edges, loops, raw triples и отсутствие factual links у concepts.
- Проверка с чужим fingerprint была отклонена в read-only guard без записи.
- Полный внешний снимок узлов/отношений вне TARGET и schema metadata одинаковы до/после. Предыдущие namespaces и файлы не изменены.
- Модели, новый ER, cosine, очистка/дедупликация отношений и этап 2F.4 не запускались.

## Восстановление после лимита памяти повторной записи

Первый фактический импорт создал 873 узла и 2728 отношений и прошёл полное чтение/проверку provenance. При попытке второго полного write-прохода с теми же свойствами Neo4j отклонила транзакцию по dbms.memory.transaction.total.max (716.8 MiB). Драйвер исчерпал retry; операция зафиксирована в initial_attempt_failure.json с этапом и последним обработанным batch.

Идемпотентный путь исправлен: повторный запуск проверяет fingerprint, counts и точное содержимое всего TARGET, затем выполняет no-op. Настройки памяти, пароль, старые namespaces и утверждённая структура не менялись. Два завершившихся прохода подтверждены полным read-back и одинаковым target SHA-256. Подробности первоначального создания и исправления — recovery_notes.json; first_import/repeated_import в verification.json относятся к финальному успешному запуску, поэтому оба равны 0/0.

Артефакты: report.md, manifest.json, verification.json (при успешном импорте), graph_stats.cypher, show_full_graph.cypher, canonical_mapping.jsonl, self_loop_only_vertices.json.
