# Shared Concepts V2 в графе RESEARCH

Подбор параметров завершён. Штатный вход в сборку графа RESEARCH — `scripts/build_corpus_graph.py`. Новый граф публикуется в изолированной локальной Neo4j под namespace `relex_corpus100_shared_concepts_v2`. [Исследование Corpus100](../../reports/retrieval/shared_concepts_v2_study/corpus100/REPORT.md)

## Утверждённая конфигурация

| Компонент | Значение |
|---|---|
| Векторизация | `intfloat/e5-small-v2`, модель и токенизатор `ffb93f3bd4047442299a41ebb6fa998a38507c52`, два канала по 384 |
| Контекст | S1, предложение исходного упоминания и по одному соседнему |
| EE Name / Context | 0,94 / 0,86, логическое AND и существующие EE-защиты |
| EC Name / Context | 0,91 / 0,87 |
| CC Name / Context | 0,93 / 0,90 |
| δ | 0,05, точная дробь 1/20 |

Штатный загрузчик принимает только эту конфигурацию. Код использует прямые окрестности, повторную EC-валидацию с накопительным δ и CC-дедупликацию с производным от состава tie-break при равном Жаккаре. При append EC проверяет всех участников заново, без исключения прежних участников самим append. Отдельные CC merge могут исключать memberships по утверждённому δ. [Конфигурационный файл](../../config/shared_concepts_v2_approved.json)

## Построение, состояние и восстановление

Первичная сборка использует ровно 100 документов исходной заморозки Stage 3A.5.11. Перед расчётом проверяются контрольные суммы исходных документов и checkpoint ER, семантических связей и двух каналов E5. Проверяется цепочка из 100 NER и 100 RE checkpoint с абсолютными offsets и идентичностью их входов. NER, RE, ER и `SIMILAR_TO` читаются из готовых результатов, E5 на Corpus100 также читается из сохранённых артефактов. Затем применяется полный batch V2. Связи `RELEX_MENTIONS`, `RELEX_REL` и `SIMILAR_TO` проецируются существующим `corpus_graph_payload.build_payload`. Для `HAS_SHARED_CONCEPT` источник — только V2, фактические флаги установлены в false. Исторический Hybrid C остаётся воспроизводимым в своей зафиксированной редакции. [Исторический процесс](STAGE_3A5_11_CORPUS100_GRAPH.md)

Файлы поколения находятся в `artifacts/relex_corpus100_shared_concepts_v2/generations/<request>/`. `source.json` и `entities.json` сохраняют проверенный источник и два вектора с provenance упоминаний и окон. `catalog/` содержит EE, чекпойнт очереди CC, версии, исключённые memberships, evidence, lineage и выгрузку принадлежностей. `payload.json` фиксирует свойства и концы рёбер перед первым обращением к Neo4j на запись. Узлы `RELEXConcept` несут ID, версию, оба центроида, обе суммы, ID участников и событий происхождения. Рёбра `HAS_SHARED_CONCEPT` несут два cosine score, версию концепта, evidence и ссылку на поколение. Опорный документ хранит ID поколения и SHA-256 файла каталога, поэтому дополнительных узлов состояния нет.

`PENDING.json` блокирует запуск другой команды поверх незаконченного поколения. Повтор первоначального импорта использует готовый файловый payload и добирает недостающие записи пакетами по 100. Повтор после атомарного append распознаёт уже записанное состояние по ID поколения. `CURRENT.json` обновляется только после контрольного чтения, сравнения свойств и концов всех рёбер, проверки уникальности ключей и серверных агрегатов. Завершённая команда при повторе проверяет состояние чтением. Нарушение ожидаемого состава или числа объектов останавливает публикацию и сохраняет причину в `last_error.json`. `tqdm` показывает проверку checkpoint, расчёт V2 и публикацию.

## Запуск из PowerShell

Из корня RESEARCH в Python 3.11 с установленным extra `relex`. Команда требует точного имени Docker-контейнера, имени пользовательской базы и прямого проброса его порта 7687 на `127.0.0.1`. Используйте значения изолированной базы RESEARCH в `RESEARCH_NEO4J_URI`, `RESEARCH_NEO4J_DATABASE` и `RESEARCH_NEO4J_CONTAINER`. Для аутентификации служат `NEO4J_USER` и `NEO4J_PASSWORD` из текущего процесса или `--neo4j-env <путь>`. Проверка цели исключает Bolt MAIN `127.0.0.1:17687`, соответствующие контейнер Compose и том данных, базу `system`, а также любые записи вне целевого namespace в пользовательской базе RESEARCH.

Первичная сборка и её resume используют одну и ту же команду.

```powershell
$env:PYTHONUTF8 = '1'
uv run --no-sync python scripts/build_corpus_graph.py --apply --neo4j-uri $env:RESEARCH_NEO4J_URI --database $env:RESEARCH_NEO4J_DATABASE --container $env:RESEARCH_NEO4J_CONTAINER
```

Проверка готовых входов отделена от batch и подключения к Neo4j.

```powershell
uv run --no-sync python scripts/build_corpus_graph.py --validate-frozen
```

### Пополнение

`--delta` принимает пакет результатов штатных upstream-стадий для новых документов. Пакет содержит следующие файлы.

| Файл | Содержание |
|---|---|
| `documents.jsonl` | Новые документы в контракте `ner-v2-source-1`. Старые ID запрещены. |
| `entity_resolution.json` | Успешный checkpoint `entity_resolution` с mapping, entities и relations новых документов, созданный форматом `build_corpus_graph.checkpoint`. |
| `semantic.json` | Успешный checkpoint `semantic` с рёбрами существующего `semantic_links`, которых касается хотя бы одна новая сущность. Рёбра новых и старых сущностей принимаются с исходными provenance. |
| `entities.json` | При наличии совместимые готовые пары векторов, профили и evidence S1 из файлового V2. |
| `manifest.json` | JSON с `base_generation` из `CURRENT.json`, `files_sha256` для перечисленных входов и `checksum` от канонического объекта. Для записи используется `shared_concepts_v2_pipeline.write_checked`. |

Для входов с подготовленными V2-векторами команда пополнения и её resume выглядят так.

```powershell
uv run --no-sync python scripts/build_corpus_graph.py --apply --neo4j-uri $env:RESEARCH_NEO4J_URI --database $env:RESEARCH_NEO4J_DATABASE --container $env:RESEARCH_NEO4J_CONTAINER --delta $env:RESEARCH_GRAPH_DELTA
```

Если пакет поставляет готовые ER и `SIMILAR_TO`, но пока не поставляет V2-векторы, добавляется `--e5-model <локальный каталог зафиксированной модели>`. В таком запуске только новые профили векторизуются через существующий проверяемый кэш. Новое поколение переиспользует старые векторы и EE-соседство. Изменения концептов и memberships, новые документы и связи публикуются одной транзакцией Neo4j. Старые фактологические узлы и рёбра сохраняют прежние свойства. Поколение Corpus100 и его пополнения принадлежат только указанному namespace RESEARCH.

## Neo4j Browser и серверные агрегаты

Выполняйте запрос полного графа в целевой пользовательской базе. Он включает узлы без рёбер. Ограничение визуализации больших графов в Neo4j Browser настраивается на стороне клиента.

```cypher
MATCH (n {namespace: 'relex_corpus100_shared_concepts_v2'})
OPTIONAL MATCH (n)-[r {namespace: 'relex_corpus100_shared_concepts_v2'}]->(m {namespace: 'relex_corpus100_shared_concepts_v2'})
RETURN n, r, m
```

Агрегаты всей пользовательской базы. Каждый запрос выполняется отдельно.

```cypher
MATCH (n) RETURN count(n) AS nodes
```

```cypher
MATCH (n) RETURN labels(n) AS labels, count(n) AS nodes ORDER BY nodes DESC
```

```cypher
MATCH ()-[r]->() RETURN count(r) AS relationships
```

```cypher
MATCH ()-[r]->() RETURN type(r) AS relationship_type, count(r) AS relationships ORDER BY relationships DESC
```

| Объект | Ожидается сразу после Corpus100 bootstrap |
|---|---:|
| `RELEXDocument` | 100 |
| `RELEXEntity` | 4 506 |
| `RELEXConcept` | 575 |
| `RELEX_MENTIONS` | 4 506 |
| `RELEX_REL` | 8 498 |
| `SIMILAR_TO` | 12 640 |
| `HAS_SHARED_CONCEPT` | 4 938 |

**PASS.** Проверены 100 checkpoint NER, 100 checkpoint RE и готовый локальный архив точки `91-87-93-90-05`. Совпали SHA-256 архива и канонического состава, 575 концептов, 4 938 memberships и 1 605 охваченных сущностей. Эти числа относятся к готовому исследовательскому каталогу.

**PASS.** Прошли 89 целевых тестов, включая сборку, повтор публикации и восстановление после серверного commit на синтетическом адаптере. Полная сборка Corpus100, inference и реальный импорт получили статус **NOT RUN**.

```powershell
uv run --no-sync python -m pytest tests/test_corpus_graph_v2.py tests/test_build_corpus_graph.py tests/test_shared_concepts_v2.py tests/test_shared_concepts_v2_study.py tests/test_contextual_shared_concepts.py tests/test_semantic_concepts.py -q
```

## Проверка локальной цели

Read-only сверка связала `bolt://127.0.0.1:17687` с контейнером `article-analysis-neo4j-1` и томом `article-analysis_neo4j_data`. На экземпляре онлайн две базы, пользовательская `neo4j` и системная `system`. В `neo4j` обнаружены 18 736 узлов и 55 381 отношение. Из них 1 450 узлов и 1 677 отношений относятся к MAIN `article-analysis-domain-v1` по namespace и концам рёбер. Полная очистка этой базы затронула бы MAIN. Очистка ждёт точного адреса отдельной пользовательской базы RESEARCH. Серверные числа 0/0 и агрегаты нового графа пока не подтверждены.
