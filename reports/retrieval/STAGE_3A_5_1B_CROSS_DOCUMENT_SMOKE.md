# Stage 3A.5.1b — Real Cross-Document Graph Retrieval Smoke Test

## 1. Версия графа и запуска

- Дата запуска: 19 ноября 2026, 17:35:54 UTC (20:35:54 MSK).
- HEAD на момент запуска: `e6e16bdbc703194954a8fe92f342459eac4c2acc`, ветка `design-v2-sequential-kg`. Runner и артефакты входят в отдельный коммит отчёта.
- Namespace: `relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v2_sem_name_t0830`.
- Graph fingerprint: `41901f76e83ee9ab65fedfba888790504c8f328066a319115fcb6f13f7c3525a`.
- [JSON запуска и независимой сверки](STAGE_3A_5_1B_CROSS_DOCUMENT_SMOKE.json) содержит все междокументные `path_id`, исходные ключи, направления, решения реранкера, Top-5 каждого пула и индекс provenance без полных документов.

## 2. Выбор примеров

Четыре компактных Cypher-запроса проверили строго заданные формы A, B и C с двумя возможными типами моста. Запросы ограничены frozen namespace и read-транзакцией с timeout. Они выбирают один маршрут на форму по минимальной степени стартового узла, затем по ключам. Для `SIMILAR_TO` требуются `RELATED` и два различных контекста. Для обеих memberships проверяются флаги trust, привязка к документу и список `SUSPICIOUS_CELL_PAIRS`. Для C оба фактических ребра должны сохранить направление от head к tail, каждое внутри своего документа.

В targeted-режиме Seed брался из найденного свидетеля. В E2E вопрос строился по имени найденной сущности, а Seed вычислялся из результата GLiNER и E5. Сопоставление последовательности ключей отличает точный Cypher-свидетель от других путей. `min_score=-1.0` — `smoke_only`. Порог GLiNER и рабочие настройки компонентов сохранены. Прогоны предназначены для проверки соединения компонентов.

## 3. A — доверенный `SIMILAR_TO`: PASS

`cooperation` (`W4285525942`) → `SIMILAR_TO` → `cooperation` (`W4309030044`). Это два различных узла, обратного traversal в данном примере нет. `resolution_decision_2f2=RELATED`, `similarity=0.9999998807907104`. Источник и оригинальные IDs сохранены в JSON. Переход помечен как нефактический. Evidence имеет статус `non_factual`, а реранкер вернул `no_factual_evidence`.

## 4. B — доверенный `HAS_SHARED_CONCEPT`: PASS

`medical complications` (`W2761455282`) → `HAS_SHARED_CONCEPT` → `Medical complications` ← `HAS_SHARED_CONCEPT` ← `complications` (`W3101922761`). GraphRetriever прошёл вторую membership в обратном направлении. У рёбер соответственно `document_ids=[W2761455282]` и `[W3101922761]`. Обе memberships прошли trust-фильтры и имеют `factual=false`, `identity=false`, `fact_inheritance_allowed=false`. Evidence имеет статус `non_factual`, реранкер вернул `no_factual_evidence`.

## 5. C — два факта из разных документов: PASS через `SIMILAR_TO`

GraphRetriever нашёл точный трёхрёберный путь после единственного повтора с расширенными лимитами. В этом прогоне вернулось пять междокументных маршрутов с фактами из обоих документов. Все пять прошли Evidence Extraction.

| Исходное направление в Neo4j | Документы | Роль |
|---|---|---|
| `RESULTS` → `RELEX_REL / HAS_RESULT` → `follow-up` | `W3101922761` | Фактическое ребро с исходной цитатой |
| `follow-up` → `SIMILAR_TO / RELATED` → `follow-up` | `W3101922761` → `W2761455282` | Нефактический переход |
| `follow-up` → `RELEX_REL / HAS_RESULT` → `improved clinical outcome` | `W2761455282` | Фактическое ребро с независимой цитатой |

На всех трёх рёбрах `traversed_reverse=false`. `GraphPath.document_ids` содержит `W2761455282` и `W3101922761`. Evidence подтвердил два factual-ребра, а центральное ребро сохранило статус `non_factual`. `path_id` — `path:3fc89f9d9ba18768c11259c74a6762596a62f33d8acdc9055d65464d6b589e70`.

| Документ и валидированное окно `[start:end)` | Сохранённый фрагмент `[start:end)` | Источник |
|---|---|---|
| `W3101922761` `[0:1730)` | `[1494:1623)` «y. Only 7 corneal grafts (17.5%) remained clear at last follow-up. CONCLUSIONS: A GDD can successfully control intractable glauco» | [W3101922761](https://openalex.org/W3101922761) |
| `W2761455282` `[0:1883)` | `[1586:1731)` «r careful preoperative evaluation. Most patients had an improved clinical outcome. There were no instances of prosthetic loosening or deep infect» | [W2761455282](https://openalex.org/W2761455282) |

Реранкер допустил этот путь в общем пуле и выдал `outside_top_k`. При отдельной подаче того же проверенного пути `score=2.0`: `relevance=1.5`, `validated_factual=1.0`, `seed_linking=0`, `extra_hops=−0.30`, `non_factual_transitions=−0.20`, `repeated_windows=0`, `diversity=0`. Этот score относится к отдельной проверке, где в пуле один путь.

**Вариант C через концепт — BLOCKED на проверенном бюджете.** Cypher нашёл физический маршрут `discharge` → `dialysis patients` → `Patients` ← `patients` → `anti-rHuEPO neutralizing antibodies` между `W2761455282` и `W3123056951`. В GraphRetriever точный путь отсутствовал при `max_candidates=200`, хотя девять других междокументных путей прошли проверку с фактами из одного документа. Независимый запрос подтвердил позиции нужных рёбер в отсортированных adjacency-списках: `1/1`, `7/9`, `4/16`, `1/16`. Лимит соседей здесь их не отсёк. Сработал `max_candidates`, а единственный разрешённый расширенный прогон уже был использован для C через семантику. Вариант C через концепт остался без подтверждения полного пути и его evidence.

## 6. E2E с реальными GLiNER и E5: PASS для retrieval

Frozen-кэш из 811 сущностей прошёл проверку hash и IDs. Модели выполнили CPU inference. Использовались два кандидата на mention. Во всех трёх вопросах найдены mentions, fallback по полному вопросу не понадобился.

| Вопрос | Mentions | Seed / уникальные узлы | Paths | Междокументные | Валидны / допущены | В общем Top-5 |
|---|---|---:|---:|---:|---:|---:|
| What is reported about follow-up? | `follow-up` (`METHOD`, `PROCESS`) | 4 / 2 | 200 | 13 | 9 / 9 | 0 |
| What is reported about dialysis patients? | `dialysis` (`OBJECT`), `dialysis patients` (`ACTOR`, `OBJECT`) | 6 / 2 | 200 | 17 | 3 / 3 | 0 |
| What is reported about RESULTS? | `RESULTS` (`ATTRIBUTE`, `CONCEPT`, `RESULT`) | 6 / 2 | 200 | 2 | 2 / 2 | 0 |

Например, `follow-up` распознан в вопросе с offsets `[23:32)` и точно связан с `follow-up` в `W3101922761` (`score=0.9999999999999126`). Оттуда BFS прошёл `SIMILAR_TO` к `follow-up` в `W2761455282`, затем прямое фактическое ребро `APPLIED_TO` к `dialysis patients` в `W2761455282`. Evidence взят из `[0:1883)` `W2761455282`, источник указан выше. `path_id` начинается с `3357beed`. Путь валиден, реранкер вернул `outside_top_k`.

Для вопроса про `dialysis patients` тот же фактический `APPLIED_TO` пройден в обратном направлении. В графе он по-прежнему направлен от `follow-up` к `dialysis patients`. Путь заканчивается обратным `SIMILAR_TO` в `W3101922761` (`path_id` начинается с `36681c30`). Для вопроса про `RESULTS` путь `932700de…` проходит обратный `HAS_RESULT` от `RESULTS` к `GDD implantation`, прямой `HAS_RESULT` от `GDD implantation` к `follow-up` и `SIMILAR_TO` в `W2761455282`. Оба factual-ребра последнего маршрута имеют evidence только в `W3101922761`.

Всего E2E вернул 32 уникальных междокументных пути. В 14 подтверждены factual-рёбра из одного документа и присутствует переход к другому. В оставшихся 18 только нефактические переходы. Цепочки с подтверждёнными фактами из обоих документов получены в targeted-режиме.

## 7. Evidence и provenance

Независимый read-only запрос сверил 52 уникальных ребра по типу, физическим head/tail и SHA-256 всех исходных свойств. Сверены 34 сохранённых evidence-записи из четырёх `RELEXDocument`: оригинальные `source_triple_id`, `window_id`, принадлежность документа, точный срез `text[start:end]`, revision, SHA-256 и URL. Из JSON можно восстановить ориентацию traversal независимо от направления фактического ребра. Для нефактических переходов массив свидетельств пуст.

По обоим режимам получено 54 уникальных междокументных `path_id`, из них 34 `valid` и 20 `non_factual`. Каждый из 34 валидных допущен к ранжированию хотя бы в одном прогоне. Счётчики прогонов могут пересекаться. Для междокументных путей не обнаружено `partial` или `invalid` evidence.

## 8. Реранкер и Top-5

Исключения в E2E: `no_factual_evidence` — 18, `duplicate_factual_sources` — 6, `outside_top_k` — 8. Последние две категории относятся к путям, допущенным к ранжированию. В каждой полной выборке реранкер вернул пять валидированных путей из одного документа. Состав Top-5 и score breakdown каждого участника сохранены в JSON.

| Пул E2E | Scores Top-5 в порядке результата |
|---|---|
| `follow-up` | `3.5`, `3.5`, `3.125`, `3.125`, `3.125` |
| `dialysis patients` | `3.5`, `3.125`, `3.125`, `3.125`, `3.125` |
| `RESULTS` | `3.0`, `3.0`, `2.625`, `2.625`, `2.625` |

Проверка обработки уже найденной двухдокументной цепочки C прошла: после повторной сверки исходных свойств, evidence и `path_id` существующий реранкер вернул этот путь с полным breakdown. Условие попадания междокументного пути в **общий** Top-5 осталось невыполненным (`Reranker=FAIL` в основном JSON-результате, изолированная проверка `PASS`). Полные промежуточные результаты и ответы реранкера сериализовались в строгий JSON (`allow_nan=False`).

## 9. Лимиты BFS

Обычный прогон: `max_hops=5`, `max_neighbors=50`, `max_candidates=200`, `max_relationships=1000`, `max_seeds=32`, transaction timeout 10 с. Единственный повтор для C-semantic: `5 / 100 / 1000 / 5000 / 32`, timeout 20 с. Максимум один `SIMILAR_TO` и один концептуальный мост задаёт существующий GraphRetriever.

Во всех восьми выборках `truncated=true`, единственный `limits_hit` — `max_candidates`. `max_neighbors` и `max_relationships` не сработали. Для того же Seed C-semantic обычная выборка просмотрела 335 relationship rows и дала два междокументных пути с фактами из одного документа. Расширенная просмотрела 1680 строк, дала 13 междокументных путей и восстановила точный свидетель. В повторе меняются три бюджета одновременно, но исчерпан был только бюджет кандидатов. Усечённые выборки не определяют полный набор достижимых путей.

## 10. Тайминги

Время в секундах, `perf_counter`. Для GLiNER и E5 первый E2E включает ленивую загрузку весов.

| Прогон | GLiNER | E5 + Linking | BFS | Evidence | Rerank |
|---|---:|---:|---:|---:|---:|
| A | — | — | 3.109 | 0.266 | 0.438 |
| B | — | — | 1.595 | 0.301 | 0.493 |
| C-semantic | — | — | 1.654 | 0.159 | 0.292 |
| C-semantic, единственный повтор | — | — | 8.311 | 1.418 | 2.466 |
| C-concept | — | — | 1.527 | 0.287 | 0.545 |
| E2E `follow-up` | 35.768 | 0.567 | 1.622 | 0.170 | 0.245 |
| E2E `dialysis patients` | 0.451 | 0.038 | 1.505 | 0.191 | 0.308 |
| E2E `RESULTS` | 0.471 | 0.035 | 1.758 | 0.138 | 0.589 |
| Изолированная дополнительная проверка C | — | — | — | 0.055 | 0.002 |

Весь основной запуск — **91.330 с**, включая preflight `2.855 с`, поиск свидетелей `4.920 с`, подготовку моделей и кэша `0.987 с`, postflight `2.849 с`, сериализацию и вспомогательные чтения.

## 11. Неизменность KG: PASS

Stage 3A.1 audit успешно выполнен до и после основного запуска, затем ещё раз после независимой read-only сверки. SHA-256 snapshot во всех случаях `ba06b63dfba69c4a930101426934c717b733257daf91309b6bb7ecb99ebcc1dc`. Fingerprint и counts также совпали. В namespace 877 узлов и 2748 рёбер, без связей с другими namespace. Все обращения к KG были запросами на чтение.

## 12. Итог: PARTIAL

| Критерий | Статус | Фактический результат |
|---|---|---|
| A, trusted `SIMILAR_TO` | PASS | Переход между двумя реальными документами |
| B, trusted concept bridge | PASS | Две membership, обход второй в обратном направлении |
| C, два фактических источника | PASS | Пять валидных путей через семантику, точный свидетель восстановлен |
| C через концепт | BLOCKED | Свидетель существует в KG, усечённая выдача его не вернула |
| E2E | PASS | 32 междокументных пути из вопросов, 14 допущены к реранкеру |
| Reranker | PARTIAL | Двухдокументный C получил score в отдельной проверке, общий Top-5 всех пулов внутридокументный |
| Integrity | PASS | Frozen snapshot и fingerprint совпали |

Focused tests Stage 3A.1–3A.4: **129 passed**. При самостоятельном ревью сопоставлены trust-условия Cypher и BFS, исходные направления, причины исключения, пределы поиска, сериализация и содержимое артефакта. Блокирующий дефект компонентов не выявлен. Статус относится к технической проверке маршрутов. Фактические связи сохраняют контекст исходных документов.

## Приложение. Все уникальные междокументные пути

Маршрут читается слева направо по порядку обхода. Стрелка показывает исходное направление ребра в Neo4j. F — factual, N — non-factual. Ссылки E ведут к фрагментам оригинальных окон и URL документов. Полные идентификаторы и решения по каждому прогону в JSON.

| Path ID, начало | Прогоны | Маршрут и evidence |
|---|---|---|
| `6d4ea405aba0` | A #1 | cooperation (W4285525942) / `SIMILAR_TO` N → cooperation (W4309030044) |
| `3802fdbee29f` | B #1 | medical complications (W2761455282) / `HAS_SHARED_CONCEPT` N → Medical complications / `HAS_SHARED_CONCEPT` N ← complications (W3101922761) |
| `b6d1254239d7` | C_semantic #1, C_semantic #2, E2E #3 | RESULTS (W3101922761) / `HAS_RESULT` F ([E1](#e1)) → follow-up (W3101922761) / `SIMILAR_TO` N → follow-up (W2761455282) |
| `932700de7073` | C_semantic #1, C_semantic #2, E2E #3 | RESULTS (W3101922761) / `HAS_RESULT` F ([E1](#e1)) ← GDD implantation (W3101922761) / `HAS_RESULT` F ([E1](#e1)) → follow-up (W3101922761) / `SIMILAR_TO` N → follow-up (W2761455282) |
| `26d9e084afa3` | C_semantic #2 | RESULTS (W3101922761) / `HAS_RESULT` F ([E1](#e1)) → follow-up (W3101922761) / `HAS_SHARED_CONCEPT` N → follow-up / `HAS_SHARED_CONCEPT` N ← follow-up (W2761455282) |
| `2794f964d0aa` | C_semantic #2 | RESULTS (W3101922761) / `HAS_RESULT` F ([E1](#e1)) → follow-up (W3101922761) / `SIMILAR_TO` N → follow-up (W2761455282) / `APPLIED_TO` F ([E2](#e2)) → dialysis patients (W2761455282) |
| `e15713e335be` | C_semantic #2 | RESULTS (W3101922761) / `HAS_RESULT` F ([E1](#e1)) → follow-up (W3101922761) / `SIMILAR_TO` N → follow-up (W2761455282) / `APPLIED_TO` F ([E2](#e2)) → dialysis patients (W2761455282) |
| `48670a4a61cb` | C_semantic #2 | RESULTS (W3101922761) / `HAS_RESULT` F ([E1](#e1)) → follow-up (W3101922761) / `SIMILAR_TO` N → follow-up (W2761455282) / `APPLIED_TO` F ([E2](#e2)) → total knee arthroplasty (W2761455282) |
| `3fc89f9d9ba1` | C_semantic #2 | RESULTS (W3101922761) / `HAS_RESULT` F ([E1](#e1)) → follow-up (W3101922761) / `SIMILAR_TO` N → follow-up (W2761455282) / `HAS_RESULT` F ([E2](#e2)) → improved clinical outcome (W2761455282) |
| `8dd5944001c5` | C_semantic #2 | RESULTS (W3101922761) / `HAS_RESULT` F ([E1](#e1)) → follow-up (W3101922761) / `SIMILAR_TO` N → follow-up (W2761455282) / `APPLIED_TO` F ([E2](#e2)) → dialysis patients (W2761455282) |
| `e7de520c744d` | C_semantic #2 | RESULTS (W3101922761) / `HAS_RESULT` F ([E1](#e1)) ← GDD (W3101922761) / `HAS_RESULT` F ([E1](#e1)) → follow-up (W3101922761) / `SIMILAR_TO` N → follow-up (W2761455282) |
| `ed8ddfbb1bf7` | C_semantic #2 | RESULTS (W3101922761) / `HAS_RESULT` F ([E1](#e1)) → glaucoma success (W3101922761) / `HAS_RESULT` F ([E1](#e1)) ← follow-up (W3101922761) / `SIMILAR_TO` N → follow-up (W2761455282) |
| `7134bf739ae4` | C_semantic #2 | RESULTS (W3101922761) / `HAS_RESULT` F ([E1](#e1)) ← GDD (W3101922761) / `COMPARED_WITH` F ([E1](#e1)) ← GDD (W3101922761) / `HAS_RESULT` F ([E1](#e1)) → follow-up (W3101922761) / `SIMILAR_TO` N → follow-up (W2761455282) |
| `3f60d67c3f31` | C_semantic #2 | RESULTS (W3101922761) / `HAS_RESULT` F ([E1](#e1)) ← GDD (W3101922761) / `HAS_RESULT` F ([E1](#e1)) → glaucoma success (W3101922761) / `HAS_RESULT` F ([E1](#e1)) ← follow-up (W3101922761) / `SIMILAR_TO` N → follow-up (W2761455282) |
| `a2ddfe33fc3d` | C_semantic #2 | RESULTS (W3101922761) / `HAS_RESULT` F ([E1](#e1)) ← GDD (W3101922761) / `COMPARED_WITH` F ([E1](#e1)) ← GDD implantation (W3101922761) / `HAS_RESULT` F ([E1](#e1)) → follow-up (W3101922761) / `SIMILAR_TO` N → follow-up (W2761455282) |
| `d6e38eaeb243` | C_concept #1 | discharge (W2761455282) / `APPLIED_TO` F ([E2](#e2)) → dialysis patients (W2761455282) / `APPLIED_TO` F ([E2](#e2)) ← follow-up (W2761455282) / `SIMILAR_TO` N ← follow-up (W3101922761) |
| `9579a3e63ec6` | C_concept #1 | discharge (W2761455282) / `APPLIED_TO` F ([E2](#e2)) → dialysis patients (W2761455282) / `HAS_SHARED_CONCEPT` N → Patients / `HAS_SHARED_CONCEPT` N ← patients (W3101922761) |
| `9d5ff2114c6b` | C_concept #1 | discharge (W2761455282) / `APPLIED_TO` F ([E2](#e2)) → dialysis patients (W2761455282) / `HAS_SHARED_CONCEPT` N → Patients / `HAS_SHARED_CONCEPT` N ← patients (W3123056951) |
| `92077c6b274e` | C_concept #1 | discharge (W2761455282) / `APPLIED_TO` F ([E2](#e2)) → dialysis patients (W2761455282) / `HAS_SHARED_CONCEPT` N → Patients / `HAS_SHARED_CONCEPT` N ← patients (W2791492120) |
| `6a2ae14e98ad` | C_concept #1 | discharge (W2761455282) / `APPLIED_TO` F ([E2](#e2)) → dialysis patients (W2761455282) / `HAS_SHARED_CONCEPT` N → Patients / `HAS_SHARED_CONCEPT` N ← patients (W4391351014) |
| `11cb86dbb442` | C_concept #1 | discharge (W2761455282) / `APPLIED_TO` F ([E2](#e2)) → dialysis patients (W2761455282) / `HAS_SHARED_CONCEPT` N → Patients / `HAS_SHARED_CONCEPT` N ← patients (W4391351014) |
| `a60ecddc47a2` | C_concept #1 | discharge (W2761455282) / `APPLIED_TO` F ([E2](#e2)) → dialysis patients (W2761455282) / `HAS_SHARED_CONCEPT` N → Patients / `HAS_SHARED_CONCEPT` N ← patients (W3123056951) |
| `8978f16a47b2` | C_concept #1 | discharge (W2761455282) / `APPLIED_TO` F ([E2](#e2)) → dialysis patients (W2761455282) / `HAS_SHARED_CONCEPT` N → Patients / `HAS_SHARED_CONCEPT` N ← cancer patients (W4391351014) |
| `7699372fb120` | C_concept #1 | discharge (W2761455282) / `APPLIED_TO` F ([E2](#e2)) → dialysis patients (W2761455282) / `APPLIED_TO` F ([E2](#e2)) ← improved clinical outcomes (W2761455282) / `HAS_RESULT` F ([E2](#e2)) ← surgery (W2761455282) / `SIMILAR_TO` N ← Surgical procedures (W3101922761) |
| `39dfad274a2a` | E2E #1 | follow-up (W3101922761) / `SIMILAR_TO` N → follow-up (W2761455282) |
| `e5eb51ab827b` | E2E #1 | follow-up (W2761455282) / `SIMILAR_TO` N ← follow-up (W3101922761) |
| `64cb6a4ed5f2` | E2E #1 | follow-up (W3101922761) / `HAS_SHARED_CONCEPT` N → follow-up / `HAS_SHARED_CONCEPT` N ← follow-up (W2761455282) |
| `3357beed4d02` | E2E #1 | follow-up (W3101922761) / `SIMILAR_TO` N → follow-up (W2761455282) / `APPLIED_TO` F ([E2](#e2)) → dialysis patients (W2761455282) |
| `6e05049d511a` | E2E #1 | follow-up (W3101922761) / `SIMILAR_TO` N → follow-up (W2761455282) / `APPLIED_TO` F ([E2](#e2)) → dialysis patients (W2761455282) |
| `3473e52b59de` | E2E #1 | follow-up (W3101922761) / `SIMILAR_TO` N → follow-up (W2761455282) / `APPLIED_TO` F ([E2](#e2)) → total knee arthroplasty (W2761455282) |
| `59f9aa039896` | E2E #1 | follow-up (W3101922761) / `SIMILAR_TO` N → follow-up (W2761455282) / `HAS_RESULT` F ([E2](#e2)) → improved clinical outcome (W2761455282) |
| `f62af19bd7fc` | E2E #1 | follow-up (W3101922761) / `SIMILAR_TO` N → follow-up (W2761455282) / `APPLIED_TO` F ([E2](#e2)) → dialysis patients (W2761455282) |
| `52742fd4e3c6` | E2E #1 | follow-up (W2761455282) / `HAS_SHARED_CONCEPT` N → follow-up / `HAS_SHARED_CONCEPT` N ← follow-up (W3101922761) |
| `c1ef9c0bf371` | E2E #1 | follow-up (W2761455282) / `SIMILAR_TO` N ← follow-up (W3101922761) / `HAS_RESULT` F ([E1](#e1)) ← GDD implantation (W3101922761) |
| `472a8ceb95d0` | E2E #1 | follow-up (W2761455282) / `SIMILAR_TO` N ← follow-up (W3101922761) / `HAS_RESULT` F ([E1](#e1)) ← RESULTS (W3101922761) |
| `d5b561f247be` | E2E #1 | follow-up (W2761455282) / `SIMILAR_TO` N ← follow-up (W3101922761) / `HAS_RESULT` F ([E1](#e1)) → glaucoma success (W3101922761) |
| `a1fa454d00c6` | E2E #1 | follow-up (W2761455282) / `SIMILAR_TO` N ← follow-up (W3101922761) / `HAS_RESULT` F ([E1](#e1)) ← GDD (W3101922761) |
| `36681c30f6fc` | E2E #2 | dialysis patients (W2761455282) / `APPLIED_TO` F ([E2](#e2)) ← follow-up (W2761455282) / `SIMILAR_TO` N ← follow-up (W3101922761) |
| `778008ba330e` | E2E #2 | dialysis patients (W2761455282) / `HAS_SHARED_CONCEPT` N → Patients / `HAS_SHARED_CONCEPT` N ← patients (W3101922761) |
| `a742f04de452` | E2E #2 | dialysis patients (W2761455282) / `HAS_SHARED_CONCEPT` N → Patients / `HAS_SHARED_CONCEPT` N ← patients (W3123056951) |
| `0e2578221c33` | E2E #2 | dialysis patients (W2761455282) / `HAS_SHARED_CONCEPT` N → Patients / `HAS_SHARED_CONCEPT` N ← patients (W2791492120) |
| `c0cbfdfa77bb` | E2E #2 | dialysis patients (W2761455282) / `HAS_SHARED_CONCEPT` N → Patients / `HAS_SHARED_CONCEPT` N ← patients (W4391351014) |
| `c6de5ee778a0` | E2E #2 | dialysis patients (W2761455282) / `HAS_SHARED_CONCEPT` N → Patients / `HAS_SHARED_CONCEPT` N ← patients (W4391351014) |
| `73ea2c3a17f9` | E2E #2 | dialysis patients (W2761455282) / `HAS_SHARED_CONCEPT` N → Patients / `HAS_SHARED_CONCEPT` N ← patients (W3123056951) |
| `0f05cb2f358e` | E2E #2 | dialysis patients (W2761455282) / `HAS_SHARED_CONCEPT` N → Patients / `HAS_SHARED_CONCEPT` N ← cancer patients (W4391351014) |
| `d9ea75dbac6c` | E2E #2 | dialysis patients (W2761455282) / `APPLIED_TO` F ([E2](#e2)) ← follow-up (W2761455282) / `SIMILAR_TO` N ← follow-up (W3101922761) |
| `89ea9a0a5a92` | E2E #2 | dialysis patients (W2761455282) / `HAS_SHARED_CONCEPT` N → Patients / `HAS_SHARED_CONCEPT` N ← patients (W3101922761) |
| `70c1ab9a7a6c` | E2E #2 | dialysis patients (W2761455282) / `HAS_SHARED_CONCEPT` N → Patients / `HAS_SHARED_CONCEPT` N ← patients (W3123056951) |
| `d6807f39ef2b` | E2E #2 | dialysis patients (W2761455282) / `HAS_SHARED_CONCEPT` N → Patients / `HAS_SHARED_CONCEPT` N ← patients (W2791492120) |
| `16284bceee0c` | E2E #2 | dialysis patients (W2761455282) / `HAS_SHARED_CONCEPT` N → Patients / `HAS_SHARED_CONCEPT` N ← patients (W4391351014) |
| `5aa54958728b` | E2E #2 | dialysis patients (W2761455282) / `HAS_SHARED_CONCEPT` N → Patients / `HAS_SHARED_CONCEPT` N ← patients (W4391351014) |
| `e6ad7651d280` | E2E #2 | dialysis patients (W2761455282) / `HAS_SHARED_CONCEPT` N → Patients / `HAS_SHARED_CONCEPT` N ← patients (W3123056951) |
| `985a9f4a26f5` | E2E #2 | dialysis patients (W2761455282) / `HAS_SHARED_CONCEPT` N → Patients / `HAS_SHARED_CONCEPT` N ← cancer patients (W4391351014) |
| `f6859a2020a4` | E2E #2 | dialysis patients (W2761455282) / `APPLIED_TO` F ([E2](#e2)) ← improved clinical outcomes (W2761455282) / `HAS_RESULT` F ([E2](#e2)) ← surgery (W2761455282) / `SIMILAR_TO` N ← Surgical procedures (W3101922761) |

### E1

`W3101922761`, окно `[0:1730)`, фрагмент `[0:220)`. [W3101922761](https://openalex.org/W3101922761)

> PRECIS: In eyes with intractable glaucoma, drainage devices provide long-term control of intraocular pressure also after penetrating keratoplasty (PK). There is a high incidence of corneal graft failure. PURPOSE: To comp

### E2

`W2761455282`, окно `[0:1883)`, фрагмент `[0:220)`. [W2761455282](https://openalex.org/W2761455282)

> BACKGROUND: Total hip arthroplasty in patients with end-stage renal disease (ESRD) may be associated with a high complication rate. However, the perioperative complication and mortality rates in these patients after tota
