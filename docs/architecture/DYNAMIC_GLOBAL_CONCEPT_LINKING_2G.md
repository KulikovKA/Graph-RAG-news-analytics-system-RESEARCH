# Stage 2G — Dynamic Global Concept Linking

## Эксперимент и неизменяемый корпус

Эксперимент выполняется только на 30 документах проверенного namespace
`relex_sem_e5_small_name_t095_ner_t046_er2f3`. Его полный SHA-256 должен совпасть
с `global_kg_2f3/verification.json`:
`5daa32d6b8e389592db3e3d0a5326f04cf06706b198448950ee7d9250ca431cf`.
Это проверка всех properties, labels, IDs, направлений и endpoints, а не только
числа объектов. `document_lock.json` фиксирует 30 IDs, revisions, hashes текстов,
названия и исходные URLs.

Сохраняются 811 contextual entities, 1307 RELEX_REL, 492 SIMILAR_TO, 811
RELEX_MENTIONS и исходные 32 концепта /118 HAS_SHARED_CONCEPT. Принятый mapping
Stage 2F.2/2F.3 не пересчитывается. Ни один документ не добавляется, не заменяется
и не удаляется. NER, ReLEx, GLiREL, CrossDocRE, обучение и новые embeddings
не запускаются; модели не импортируются. Код старого импортёра параметризован
по ожидаемым totals для его повторного использования; отчёты 2F.2/2F.3 неизменны.

Результат размещается только в
`relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1`.

## Семантика и структура

```text
Document A -> RELEX_MENTIONS -> Entity A -- RELEX_REL ... -- Entity A2
                                                               |
                                                     HAS_SHARED_CONCEPT
                                                               |
                                                            Concept
                                                               |
                                                     HAS_SHARED_CONCEPT
                                                               |
Document B -> RELEX_MENTIONS -> Entity B -- RELEX_REL ... -- Entity B2
```

Entity A2 и Entity B2 остаются разными контекстными объектами. Между ними нет
MERGE, identity или новой фактической зависимости. У memberships обязательно
`factual=false`, `identity=false`, `fact_inheritance_allowed=false`. Концепт
не получает document owner, RELEX_MENTIONS или RELEX_REL. При поиске пути можно
обходить RELEX_REL в любом направлении, но исходное направление и predicate
сохраняются и явно показываются в evidence. Внутридокументный маршрут может
содержать произвольное число фактических рёбер.

Приложение Graph-RAG должно разделять семантический поиск и доказательство факта.
Общий Fish не переносит смертность популяции A на популяцию B. Cancer не
переносит особенности конкретного рака, лечения или cohort между публикациями.
Follow-up не делает разные длительности наблюдения одним измерением.

## Подготовка кандидатов

`src/cgraph/concepts.py` содержит чистый `ConceptResolver`. Экспериментальный
runner — `scripts/build_dynamic_concept_graph.py`. Он читает готовый снимок
2F.3 и проверяет каждое исходное упоминание и контекст по offsets/revision
в соответствующем документе. Сохранённые source entity и mention IDs участвуют
в provenance каждого решения.

Источники кандидатов:

1. Имена и все aliases утверждённых contextual entities; нормализация NFC,
   регистра и пробелов соответствует existing exact audit.
2. Ограниченная регулярная морфология и конструкция `Fish populations` → Fish.
   Это retrieval-эвристика, требующая дальнейшего подтверждения смысла.
3. Существующие abbreviation candidates, только если short/long spans точно
   подтверждены в исходном документе и относятся к данному упоминанию.
4. Имена и aliases 32 существующих концептов, восстановленные из их
   подтверждённых memberships.
5. Все существующие SIMILAR_TO как источник кандидатов, включая исходные
   REJECT/REVIEW annotations прежних этапов: физические связи не фильтруются.
6. Пересечение значимых токенов имён. Даже при отсутствии SIMILAR_TO повторное
   научное имя/явная аббревиатура может предложить новый концепт.

Повторяющееся имя/алиас минимум в двух документах формирует *предложение*,
но не автоматически принятое понятие. Один концепт на каждую entity не создаётся.
Предложения со совпадающим набором явных aliases дедуплицируются. Предпочитается
полная расшифровка, а не короткая аббревиатура. Existing concept IDs всегда
переиспользуются. Здесь не нужен пересчёт E5-small: новых векторных моделей нет,
существующие cosine scores остаются неизменными в SIMILAR_TO.

## Решение ACCEPT / REJECT

По последнему уточнению пользователя итоговых статусов ровно два:
**ACCEPT** и **REJECT**. Неопределённость/недостаточное evidence трактуется как
REJECT без создания ребра. Исходные REVIEW annotations Stage 2F.2 остаются
в неизменённом SIMILAR_TO; это исторические свойства другого этапа.

ACCEPT требует одновременно:

- Совпадение научного класса, документированного алиаса либо явного названия
  класса в квалифицированной сущности, например esophageal cancer → Cancer.
- Исходные проверенные mentions и source contexts у entity и reference.
- Общий контекстный смысл: минимум два явных термина в каждом контексте из
  соответствующего словаря признаков biological/clinical/genetic и других
  областей. Все сработавшие признаки сохранены в decision evidence.
- Совместимую абстракцию и исходные типы. Ограниченная совместимость
  TASK/PROCESS/ATTRIBUTE/CONCEPT учитывает шум исходного typing.

Словари признаков — воспроизводимые эвристики. Это дополнительное подтверждение
лексически указанного класса, а не основание связывать объекты общей темы.
Алгоритм не претендует на полное разрешение всех научных смыслов.

REJECT применяется к общим Study/Results и полисемичным Resolution/Reliability,
экспериментальным RESULT, отсутствию контекста, несовместимому смыслу/типам,
неподтверждённому совпадению темы/cosine и конфликтующим расшифровкам аббревиатур.
Отношение к объекту не означает принадлежность к его классу:
`co-investment of green banks` не является Green bank. Поэтому relational
модификаторы `of/with/for/from/by/to/in` не удаляются ради suffix-match.

Новый Concept получает стабильный ID из нормализованного полного названия.
Создание разрешается, только если ACCEPT memberships подтверждены минимум
в двух документах. Если это условие не выполнено, предварительные ACCEPT
переводятся в REJECT; orphan proposal node не импортируется. IDs memberships
зависят от entity/concept IDs, IDs решений — ещё и от версии resolver.

## Импорт, валидация и воспроизводимость

Сначала строится локальный payload, проверяются сохранность каждой исходной
записи, duplicate names/memberships, типы endpoints, namespace, отсутствие
наследования фактов и детерминизм при повторе/перестановке входов. При любой
ошибке запись в Neo4j не начинается.

При копировании меняются только key, namespace и fingerprint. Исторические
source_json, source_namespace, source_key, source_relationship_json, source
scores, mapping, labels, relation types и все остальные properties сохраняются.
Дополнительно записываются stage_2g_baseline_key/fingerprint. В namespace key
включён исходный стабильный ID; membership/fact endpoints получают тот же префикс.

Используются существующие Stage 2F.3 `import_once`, `assert_namespace`, `verify`,
`metrics`, Cypher counts/WCC и Stage 2E `outside_snapshot`, `DOCUMENT_QUERY`,
`EDGE_QUERY`, `connect_neo4j`. Новый графовый framework не создаётся.
Первый import атомарен; counts проверяются до commit. Заполненный target с
другим fingerprint либо неполным составом отвергается. Повторный запуск с
тем же fingerprint сверяет каждую запись и выполняет no-op.

До/после выполняются полный baseline hash, per-record snapshot hashes, снимок
всех чужих namespace с element IDs/properties, schema constraints/indexes,
проверка всех отношений, касающихся target, на cross-namespace endpoints/owner.
Read-back проверяет каждый label/type/property/ID/endpoint целевого payload.
После двух проходов target hashes должны совпасть, второй проход создаёт 0/0.
Фактическая WCC отдельно проверяется существующим Cypher без GDS/APOC.
Глобальные constraints и чужие namespace не изменяются.

Manifest фиксирует parent commit, code hashes, policy, версии библиотек,
входные hashes, hash корпуса, решения и результаты. Credentials читаются
существующим helper только из окружения/указанного env; в отчёты и Git не попадают.

## Метрики и ограничения интерпретации

KG для WCC состоит из Entity+Concept и RELEX_REL+SIMILAR_TO+HAS_SHARED_CONCEPT.
Document и RELEX_MENTIONS исключены. Направление игнорируется только при расчёте
слабых компонент. Петля считается ненулевой степенью; вершины с одними петлями
учитываются отдельно, как в Stage 2F.3.

Отдельно сравниваются прямые пары документов через Concept, новые пары
(concept, doc A, doc B), entity→concept←entity мосты и первая достижимость
пары документов в полном KG. Число произвольных walks не заявляется: циклы
сделали бы его бесконечным. Примеры используют реальные source facts с raw
triple IDs и внутридокументный BFS произвольной длины. Они являются
семантическими маршрутами, а не новыми доказанными фактическими зависимостями.

Больше рёбер/крупнее компонента не означает лучше KG. В concept_audit показаны
размер и распределение по документам каждого hub. Broad approved Patients/Genes
требуют осторожного использования в retrieval. Качество ACCEPT не равно
измеренной precision: отдельного независимого gold-набора для этих 30 текстов нет.

## Минимальный интерфейс будущего incremental resolution

```python
resolver = ConceptResolver(catalog, reference_entities, existing_memberships)
delta = resolver.resolve_batch(local_entities, similar_pairs)
# delta.decisions, delta.concepts, delta.memberships
```

Caller передаёт уже извлечённые локальные entities, проверенные aliases и
контекстные evidence. Resolver не загружает документы и не вызывает NER/RE.
Новый документ сначала ищет существующие концепты, затем принимает/отклоняет
memberships по той же политике. Новое понятие может связать новые entities
с затронутыми ранее несвязанными references, если есть минимум два документа.
Это покрыто отдельным synthetic test. Полная ingestion-система сейчас
не реализуется, никакой новый реальный документ в эксперимент не поступает.

Для следующих 30/100/1000 документов caller должен хранить индексы
normalized alias→concept/entity IDs, token→candidate IDs, scope abbreviations,
versioned catalog и отдельный журнал решений. Текущий batch API допускает
передачу только найденных references и affected concepts; весь factual graph
не пересчитывается. Простая реализация на текущих 811 entities сканирует
маленький каталог; для 1000 документов retrieval и evidence loading ограничивают
локальной окрестностью кандидата. Нельзя использовать транзитивное замыкание
SIMILAR_TO как доказательство одного concept.

Дубликаты предотвращаются lookup по aliases, стабильными IDs, проверкой
catalog collisions и namespace-scoped persistence. Глобальные Neo4j constraints
на этом этапе не добавляются. При реальной конкурентной ingestion потребуется
один сериализованный writer/version check каталога; нынешний single-writer
эксперимент не заявляет поддержку конкурентного обновления.

Исправление ошибочной membership должно добавлять новую версию решения с
ссылкой на прежнее evidence и supersedes/retraction, затем обновлять активный
concept view в новом namespace/catalog version. Нельзя менять source entity,
переносить факты или незаметно переписывать baseline. Текущий 2G — аддитивный
снимок: 118 прежних memberships не удаляются даже при более строгой новой
политике. Следующая версия resolver использует новый namespace и fingerprint;
перезапись существующего target с другой политикой блокируется.

## Запуск и проверка

```powershell
.venv/Scripts/python.exe -m pytest tests/test_dynamic_concepts.py -q
.venv/Scripts/python.exe scripts/build_dynamic_concept_graph.py --prepare-only --neo4j-env <existing-file>
.venv/Scripts/python.exe scripts/build_dynamic_concept_graph.py --neo4j-env <existing-file>
```

Для offline подготовки доступен `--prepare-only --baseline-cache <snapshot.json>`.
Он всё равно требует полного совпадения baseline SHA-256; cached payload нельзя
использовать для импорта. Перед реальной записью baseline читается и проверяется
заново. Длительные операции предваряются CAPS-сообщениями; tqdm обновляется
по реальным entities, snapshots и завершённым batches.
