# Этап 2D — Global Semantic Linking Preview

Все сценарии рассчитаны только в памяти: сохраняются 1122 канонические вершины, 30 документов и все 1760 направленных типизированных отношений 1C (включая 13 петель). Добавляется отдельный неориентированный тип SIMILAR_TO; вершины и исходные отношения не объединяются и не переназначаются. В Neo4j ничего не записано.

## Методика

Использован неизменённый worker 2C.1 с адаптером только в памяти отдельного процесса: text_name и расчёт добавляемых связей вместо merge. Четыре локальные модели загружены последовательно; BGE-small переиспользован для двух порогов. Векторы 2B/2C.1 не сохранялись, поэтому получены заново. Все четыре полных SHA-256 embeddings совпали с 2B и 2C.1. Протокол: CPU FP32, 4 потока, batch=8, max_length=512, CLS для BGE; masked mean для E5/GTE; E5 query: с обеих сторон; L2-нормализация. E5-small — PyTorch, остальные — ONNX Runtime. Скачиваний и новых зависимостей нет.

Для каждой конфигурации проверены все 628881 неупорядоченная пара (без самопар и top-5). Cosine вычисляется в float32; сравнение его точного числового значения с десятичным порогом — в float64 без округления, как в 2C.1. Отбираются все similarity >= threshold, включая внутри- и междокументные пары, без фильтров типов и решений 2C. Один SIMILAR_TO на неупорядоченную пару в рамках конфигурации; связь добавляется даже при наличии другого предметного отношения между её концами. Сценарии не суммируются.

Слабая связность: неориентированная проекция всех исходных отношений плюс SIMILAR_TO; типы и направления игнорируются только для обхода. Каждая из 1122 вершин остаётся в графе. Изоляция — отсутствие любых инцидентных отношений; исходная петля считается связью, но не соединяет разные вершины. Новых петель нет. Компоненты дополнительно проверены независимым Union-Find.

## Пять сценариев

Количество вершин во всех строках — **1122**, исходных отношений — **1760**. Новые рёбра считаются отдельно, без удвоения неориентированных связей.

| Сценарий | Новых связей | Внутри документов | Между документами | Всего рёбер | Изолированные | Доля I, % | Слабые компоненты | Крупнейшая, V |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Без SIMILAR_TO | 0 | 0 | 0 | 1760 | 324 | 28.88 | 391 | 61 |
| bge-small-en-v1.5 / name / 0.90 | 672 | 514 | 158 | 2432 | 201 | 17.91 | 256 | 557 |
| bge-small-en-v1.5 / name / 0.95 | 485 | 340 | 145 | 2245 | 234 | 20.86 | 290 | 456 |
| bge-base-en-v1.5 / name / 0.95 | 488 | 345 | 143 | 2248 | 235 | 20.94 | 293 | 455 |
| e5-small-v2 / name / 0.95 | 685 | 491 | 194 | 2445 | 200 | 17.83 | 258 | 580 |
| gte-small / name / 0.95 | 698 | 540 | 158 | 2458 | 195 | 17.38 | 251 | 588 |

## Междокументная связность

«Документов с прямой связью» — документы, инцидентные хотя бы одному междокументному SIMILAR_TO. Проекция документов имеет 30 вершин и одно ребро на связанную пару document IDs; включает документы без связей. Такая проекция может переоценивать сквозную связность: разные сущности внутри одного документа не обязательно связаны. Поэтому отдельно показаны реально достижимые пары документов (оба представлены в одной компоненте KG) и число документов в крупнейшей компоненте сущностей. Достижимость означает существование хотя бы одного пути между сущностями этих документов, не связность всех их сущностей.

| Сценарий | Документов с прямой связью / 30 | Прямых пар документов / 435 | Компонент проекции документов | Крупнейшая в проекции, docs | Достижимых пар документов в KG | Документов в крупнейшей компоненте KG |
|---|---:|---:|---:|---:|---:|---:|
| Без SIMILAR_TO | 0 | 0 | 30 | 1 | 0 | 1 |
| bge-small-en-v1.5 / name / 0.90 | 25 | 55 | 6 | 25 | 235 | 22 |
| bge-small-en-v1.5 / name / 0.95 | 22 | 49 | 9 | 22 | 159 | 18 |
| bge-base-en-v1.5 / name / 0.95 | 21 | 47 | 11 | 19 | 140 | 17 |
| e5-small-v2 / name / 0.95 | 24 | 56 | 7 | 24 | 254 | 23 |
| gte-small / name / 0.95 | 25 | 55 | 6 | 25 | 256 | 23 |

## Решения 2C и диагностические риски

ACCEPT/REJECT из 2C относятся к тождеству сущностей, а не к допустимости тематического SIMILAR_TO. REJECT — предупреждение против интерпретации перехода как эквивалентности, не доказанная ошибка новой связи. Типовые конфликты и повреждения также только сигналы риска. Решения не использованы как фильтры. Выборка 2C целевая и зависима от прежних моделей/top-5, не независимая gold-разметка; Precision/Recall/F1 и частота ошибок по этим числам не оцениваются. REVIEW и NOT_IN_2C не означают подтверждение.

| Сценарий | ACCEPT все / cross | REJECT все / cross | Cross REVIEW | Cross нет в 2C | Cross разные названия | Cross разные типы | Cross непересекающиеся типы | Cross повреждённое имя / partial_token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bge-small-en-v1.5 / name / 0.90 | 16 / 1 | 21 / 8 | 144 | 5 | 15 | 32 | 24 | 0 / 2 |
| bge-small-en-v1.5 / name / 0.95 | 5 / 1 | 19 / 7 | 132 | 5 | 2 | 24 | 17 | 0 / 2 |
| bge-base-en-v1.5 / name / 0.95 | 5 / 1 | 18 / 6 | 131 | 5 | 0 | 23 | 16 | 0 / 2 |
| e5-small-v2 / name / 0.95 | 15 / 1 | 24 / 7 | 180 | 6 | 51 | 30 | 22 | 0 / 2 |
| gte-small / name / 0.95 | 17 / 1 | 22 / 8 | 144 | 5 | 15 | 31 | 23 | 0 / 2 |

## 20 конкретных междокументных пар

По четыре примера на сценарий: высокий cosine; известный REJECT, если доступен, иначе высокий cosine; две пары с минимальным допущенным cosine. Повторы canonical-пары между сценариями исключены, поэтому минимальные выбираются среди ещё не показанных пар. Для высоких сходств предпочтены разные названия. boundary обозначает нижний край допущенных значений, а не обязательную близость к порогу: у BGE-base/0.95 все междокументные cosine >= 0.9999995, пограничных около 0.95 нет. У остальных сценариев показаны случаи около заданного порога. Отбор иллюстративный, не репрезентативный; полный состав и точные ID находятся в semantic_links.csv. Имена не исправлялись.

| Сценарий / отбор | Левая сущность | Документ | Правая сущность | Документ | Cosine | Решение 2C | Типы различаются | ID связи (префикс) |
|---|---|---|---|---|---:|---|---|---|
| bge-small-en-v1.5 / name / 0.90 / high | patients | W3123056951 | patients | W2761455282 | 1.00000000 | REVIEW | True | sl:003c3b69d263 |
| bge-small-en-v1.5 / name / 0.90 / known_REJECT_or_high | cancer | W4391351014 | cancer | W3123056951 | 0.99999982 | REJECT | False | sl:7dd226babe5b |
| bge-small-en-v1.5 / name / 0.90 / boundary | follow-up | W3101922761 | follow-up period | W2761455282 | 0.90308440 | REJECT | True | sl:e4c283595b57 |
| bge-small-en-v1.5 / name / 0.90 / boundary | experiments | W4389076493 | Experimental results | W4230116451 | 0.90366417 | REVIEW | True | sl:88dd9b9f1d08 |
| bge-small-en-v1.5 / name / 0.95 / high | RESULTS | W3101922761 | results | W2078771716 | 1.00000000 | REVIEW | False | sl:02c71f3720e3 |
| bge-small-en-v1.5 / name / 0.95 / known_REJECT_or_high | development | W3123056951 | development | W7215137876 | 0.99999982 | REJECT | False | sl:be1a02d13677 |
| bge-small-en-v1.5 / name / 0.95 / boundary | filtering | W2364973104 | filter | W2084709926 | 0.95198429 | REJECT | True | sl:29f8b38e5c98 |
| bge-small-en-v1.5 / name / 0.95 / boundary | grains | W2317432785 | grain | W2906067930 | 0.95940447 | REVIEW | False | sl:a0493f03aabd |
| bge-base-en-v1.5 / name / 0.95 / high | energy consumption | W3138328568 | energy consumption | W2991563224 | 1.00000000 | REVIEW | False | sl:0206c2483a13 |
| bge-base-en-v1.5 / name / 0.95 / known_REJECT_or_high | study | W2761455282 | study | W2791492120 | 0.99999994 | REJECT | False | sl:dd2d389f0f9a |
| bge-base-en-v1.5 / name / 0.95 / boundary | paper | W2991563224 | paper | W2906067930 | 0.99999952 | REVIEW | False | sl:a9ce8acaafe2 |
| bge-base-en-v1.5 / name / 0.95 / boundary | development | W3123056951 | development | W7215137876 | 0.99999976 | REVIEW | False | sl:47008067422f |
| e5-small-v2 / name / 0.95 / high | genes | W3138328568 | genes | W1995028339 | 1.00000000 | REVIEW | True | sl:06d5246b15cc |
| e5-small-v2 / name / 0.95 / known_REJECT_or_high | patients | W4391351014 | patients | W2761455282 | 1.00000000 | REJECT | False | sl:3ff06f7093dc |
| e5-small-v2 / name / 0.95 / boundary | supply line | W2906067930 | electric supply line | W2364973104 | 0.95053738 | REVIEW | True | sl:5019dc7be8a1 |
| e5-small-v2 / name / 0.95 / boundary | result | W4234628002 | outcome | W2761455282 | 0.95137519 | REVIEW | False | sl:7708b45e7d0e |
| gte-small / name / 0.95 / high | resolution | W2096307598 | resolution | W3027457624 | 1.00000000 | REVIEW | False | sl:76927d1964c4 |
| gte-small / name / 0.95 / known_REJECT_or_high | patients | W3101922761 | patients | W4391351014 | 0.99999988 | REJECT | False | sl:63f00fb48cec |
| gte-small / name / 0.95 / boundary | changes | W4234628002 | change | W2394633461 | 0.95410502 | REVIEW | False | sl:8351ad65b65a |
| gte-small / name / 0.95 / boundary | fluxes | W2096307598 | particle fluxes | W2084709926 | 0.95790476 | REVIEW | True | sl:9b7ddb653ff7 |

## Вывод и ограничения

Для следующей проверки глобальных переходов наиболее перспективен **E5-small / text_name / 0.95**: 194 междокументные связи, 23 документа в крупнейшей компоненте KG, 254 достижимые пары документов, 7 известных междокументных REJECT. По сравнению с BGE-small/0.95 число известных междокументных REJECT такое же, а охват крупнейшей компоненты выше (18 → 23 документов). Это условный выбор по связности и диагностическим рискам, не доказанный победитель по качеству: 186 из 194 междокументных пар остаются REVIEW или отсутствуют в 2C.

GTE-small/0.95 близок по глобальному эффекту: 23 документа в крупнейшей компоненте, 256 достижимых пар документов против 254 у E5; документов с прямыми связями 25 против 24. При этом известных междокументных REJECT 8 против 7 у E5. Во всём графе, наоборот, у E5 REJECT больше (24 против 22), поэтому рекомендация E5 ограничена целью междокументного linking и не означает минимального общего риска.

BGE-base/0.95 — наиболее консервативная контрольная точка по известным REJECT (18 всего, 6 междокументных), но крупнейшая компонента охватывает 17 документов. BGE-small/0.95 даёт близкий эффект (18 документов, 7 междокументных REJECT). Ослабление BGE-small до 0.90 повышает охват до 22 документов, но увеличивает междокументные REJECT до 8 и конфликты наборов типов до 32.

Во всех пяти сценариях присутствуют 143 междокументные пары с одинаковым нормализованным названием. Дополнительных пар с разными названиями: BGE-small/0.90 — 15, BGE-small/0.95 — 2, BGE-base/0.95 — 0, E5-small/0.95 — 51, GTE-small/0.95 — 15. Нормализация здесь только диагностическая (NFKC, casefold, пробелы), вход модели не менялся. Значительная часть общей связности опирается на совпадающие слова вроде patients, study, results, cancer; разные исследования, когорты и смыслы от этого не становятся тождественными. Даже cosine около 1.0 встречается у известных REJECT.

Структурные показатели сравниваются при постоянном числе вершин: уменьшение числа изолированных вершин и компонент здесь обусловлено добавлением путей, а не сжатием сущностей. Эти пути являются гипотезами similarity, а не извлечёнными фактами или доказанной эквивалентностью. Полезность маршрутов, семантическая корректность SIMILAR_TO и качество Graph-RAG не проверялись.

## Проверки и воспроизводимость

Проверены 1122 уникальных canonical ID, 1211 исходных entity ID, 1327 mention ID и 30 документов; все 1760 отношений и их конечные точки; неизменность исходных типов/направлений/полей через контрольные суммы. Для пяти сценариев числа всех, внутри- и междокументных пар точно совпали с 2C.1. Самопар и дубликатов внутри сценария нет; все конечные точки присутствуют в 2A. BFS и Union-Find согласованы. Исходные документы, результаты предыдущих этапов, код инференса и файлы моделей проверены SHA-256 до и после работы. Новые файлы — только этот отчёт и semantic_links.csv; код — scripts/preview_global_semantic_links.py.

Повторный запуск: `.venv/Scripts/python.exe scripts/preview_global_semantic_links.py`. Порядок сценариев и пар детерминирован; названия, контексты, исходные ID и флаги сохранены в CSV со ссылками на 2A/2C. Проверки не превращают similarity в семантическое доказательство. Neo4j, mapping, отношения и артефакты 1A–2C.1 не изменяются.

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
- `reports/entity_resolution/relex_base_t035/semantic_resolution/input/canonical_entities.jsonl`: `34fe3fd42e110b5b7a8b054360eee2a6182c67782e7afa54d2298e0491bf9151`
- `reports/entity_resolution/relex_base_t035/semantic_resolution/input/report.md`: `bd99b9d37b9461fcf932ff37428d1cbefb0d7fabb2b40977ae73430a212714dc`
- `reports/entity_resolution/relex_base_t035/semantic_resolution/model_comparison/candidate_pairs.csv`: `5b5101945ce513243d3a28969b77425c4e1549a96bb31bb616546727d815ae2f`
- `reports/entity_resolution/relex_base_t035/semantic_resolution/model_comparison/model_comparison.csv`: `e89fa1b0d133f33c0710b6bf595e1bbddc7fcc6c21d77bc2c9c222bf454aac6c`
- `reports/entity_resolution/relex_base_t035/semantic_resolution/model_comparison/report.md`: `40f5f0d7d5b0100a02c83b774430ec6538b99e3fde668c030f75c43c81a881bc`
- `reports/entity_resolution/relex_base_t035/semantic_resolution/threshold_sweep/report.md`: `68d301a45cdb24334fc189e4253a455995a01966eaf718d425a33386fbe9f017`
- `reports/entity_resolution/relex_base_t035/semantic_resolution/threshold_sweep/threshold_results.csv`: `f3a26801a3a505355043111cdab8bb9d99a544ea49ee59324dcbe5ccf0200f2a`
- `reports/ner_v2/research_run2/input_abstracts.jsonl`: `684ca3a758a128c6382ab924f5c131f0e27faa9f7625b431ae0d6db975e543b1`
- `reports/re_v2/relex_comparison/threshold035/e3_base/entities.jsonl`: `19039d87119130dcc37d2c472a7f2242c6a3ad5059e46a1ff431e6fa37e3b6ba`
- `reports/re_v2/relex_comparison/threshold035/e3_base/relations.jsonl`: `bbb4e602abccedcc776b59cfd526f8ffe7ffb46bdea7f7dc3c38df33174dd099`
- `scripts/benchmark_semantic_embeddings.py`: `c8acc0440e3d98f77d4903d54dee2ca0984662fe46b45dbc39c7dfba6853ee41`
- `scripts/sweep_semantic_thresholds.py`: `815a8deae004431db4e3b2db57a3c4d0036ba48131513d2faa9d50223ed93c38`
