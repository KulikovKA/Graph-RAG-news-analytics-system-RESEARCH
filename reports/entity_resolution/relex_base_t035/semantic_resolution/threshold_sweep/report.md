# Этап 2C.1 — Threshold Sweep

Выполнены все 90 сценариев: 9 моделей × 2 режима × пороги 0.75, 0.80, 0.85, 0.90, 0.95. Это экспериментальное замыкание similarity-пар через Union-Find в памяти, не семантически подтверждённое объединение. ACCEPT/REJECT из 2C и типы не применяются как фильтры.

Исходно: 1122 канонические сущности, 30 документов, 1760 существующих отношений 1C, 324 изолированные вершины. Каждое представление сравнивается по всем 628881 неупорядоченным парам без самопар, не по top-5. Внутри/между документами определяется по исходному document_id до экспериментального объединения.

## Протокол

В 2B векторы не сохранялись (нет пригодных npy/npz/pt файлов корпуса); top-5 CSV и квантили не позволяют восстановить все cosine. Поэтому пересчитаны 18 наборов векторов, каждый использован для пяти порогов. Повторно использованы helpers/константы 2B без изменения его скрипта: CPU FP32, 4 потока, batch=8, max_length=512, собственный tokenizer.json, CLS для BGE без префикса, masked mean для E5 с query: и для GTE без префикса, L2-нормализация. E5-small работает через PyTorch, остальные — через ONNX Runtime. Все 18 полных SHA-256 векторных матриц и текстов совпали с 2B; версии runtime и файлы моделей тоже проверены. Модели запускались последовательно в отдельных процессах, без сети и изменения весов.

Cosine вычислен в float32, как в 2B; далее значение каждой пары представлено в float64 и сравнивается с десятичным порогом без предварительного округления. Условие включения — similarity >= threshold. Пороги обрабатываются от строгого к мягкому, Union-Find дополняется всеми вновь допустимыми парами. В CSV: merge_groups — группы размера ≥2; entities_in_merge_groups — все их участники; removed_vertices = Σ(size−1); remaining_canonical_vertices включает также не объединённые singleton-узлы. При отсутствии объединений largest_merge_group=0.

Изоляция определяется **только существующими отношениями 1C**, чьи конечные точки в памяти заменяются корнями Union-Find. Similarity-пары не добавляются в отношения. Петля считается связью в isolated_vertices; isolated_vertices_without_self_loops считает изоляцию после исключения петель. Документный provenance сохранён в ключе (document_id, root_head, relation_type, root_tail) диагностических уникальных троек. Никаких новых mappings или отношений на диск не записано.

## Оставшиеся канонические вершины (гипотетически)

| Модель | Режим | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 |
|---|---|---:|---:|---:|---:|---:|
| BAAI/bge-small-en-v1.5 | text_name | 299 | 506 | 662 | 798 | 889 |
| BAAI/bge-small-en-v1.5 | text_name_context | 16 | 34 | 70 | 219 | 697 |
| BAAI/bge-base-en-v1.5 | text_name | 449 | 594 | 721 | 829 | 894 |
| BAAI/bge-base-en-v1.5 | text_name_context | 1 | 2 | 30 | 147 | 602 |
| BAAI/bge-large-en-v1.5 | text_name | 416 | 590 | 725 | 836 | 901 |
| BAAI/bge-large-en-v1.5 | text_name_context | 10 | 29 | 73 | 294 | 801 |
| intfloat/e5-small-v2 | text_name | 1 | 1 | 93 | 459 | 802 |
| intfloat/e5-small-v2 | text_name_context | 1 | 1 | 5 | 34 | 156 |
| intfloat/e5-base-v2 | text_name | 1 | 9 | 160 | 568 | 825 |
| intfloat/e5-base-v2 | text_name_context | 1 | 1 | 26 | 39 | 271 |
| intfloat/e5-large-v2 | text_name | 1 | 13 | 162 | 598 | 835 |
| intfloat/e5-large-v2 | text_name_context | 1 | 1 | 26 | 38 | 206 |
| thenlper/gte-small | text_name | 1 | 3 | 145 | 532 | 787 |
| thenlper/gte-small | text_name_context | 1 | 1 | 1 | 1 | 62 |
| thenlper/gte-base | text_name | 1 | 10 | 302 | 604 | 822 |
| thenlper/gte-base | text_name_context | 1 | 1 | 1 | 1 | 31 |
| thenlper/gte-large | text_name | 1 | 4 | 225 | 587 | 825 |
| thenlper/gte-large | text_name_context | 1 | 1 | 1 | 1 | 25 |

## Изолированные вершины с учётом петель

| Модель | Режим | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 |
|---|---|---:|---:|---:|---:|---:|
| BAAI/bge-small-en-v1.5 | text_name | 83 | 136 | 178 | 218 | 248 |
| BAAI/bge-small-en-v1.5 | text_name_context | 0 | 2 | 11 | 40 | 183 |
| BAAI/bge-base-en-v1.5 | text_name | 126 | 163 | 195 | 227 | 250 |
| BAAI/bge-base-en-v1.5 | text_name_context | 0 | 0 | 4 | 27 | 162 |
| BAAI/bge-large-en-v1.5 | text_name | 115 | 159 | 197 | 228 | 251 |
| BAAI/bge-large-en-v1.5 | text_name_context | 0 | 1 | 15 | 66 | 222 |
| intfloat/e5-small-v2 | text_name | 0 | 0 | 25 | 120 | 220 |
| intfloat/e5-small-v2 | text_name_context | 0 | 0 | 0 | 1 | 20 |
| intfloat/e5-base-v2 | text_name | 0 | 2 | 39 | 156 | 227 |
| intfloat/e5-base-v2 | text_name_context | 0 | 0 | 0 | 1 | 56 |
| intfloat/e5-large-v2 | text_name | 0 | 5 | 38 | 159 | 228 |
| intfloat/e5-large-v2 | text_name_context | 0 | 0 | 0 | 2 | 32 |
| thenlper/gte-small | text_name | 0 | 1 | 43 | 141 | 214 |
| thenlper/gte-small | text_name_context | 0 | 0 | 0 | 0 | 9 |
| thenlper/gte-base | text_name | 0 | 2 | 83 | 167 | 225 |
| thenlper/gte-base | text_name_context | 0 | 0 | 0 | 0 | 2 |
| thenlper/gte-large | text_name | 0 | 0 | 58 | 164 | 227 |
| thenlper/gte-large | text_name_context | 0 | 0 | 0 | 0 | 2 |

## Изолированные вершины без учёта петель

| Модель | Режим | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 |
|---|---|---:|---:|---:|---:|---:|
| BAAI/bge-small-en-v1.5 | text_name | 84 | 137 | 180 | 219 | 248 |
| BAAI/bge-small-en-v1.5 | text_name_context | 16 | 30 | 24 | 42 | 184 |
| BAAI/bge-base-en-v1.5 | text_name | 126 | 164 | 195 | 228 | 250 |
| BAAI/bge-base-en-v1.5 | text_name_context | 1 | 2 | 20 | 37 | 164 |
| BAAI/bge-large-en-v1.5 | text_name | 115 | 160 | 197 | 228 | 251 |
| BAAI/bge-large-en-v1.5 | text_name_context | 10 | 27 | 34 | 70 | 223 |
| intfloat/e5-small-v2 | text_name | 1 | 1 | 25 | 122 | 221 |
| intfloat/e5-small-v2 | text_name_context | 1 | 1 | 5 | 30 | 25 |
| intfloat/e5-base-v2 | text_name | 1 | 2 | 39 | 157 | 228 |
| intfloat/e5-base-v2 | text_name_context | 1 | 1 | 26 | 27 | 63 |
| intfloat/e5-large-v2 | text_name | 1 | 5 | 39 | 160 | 229 |
| intfloat/e5-large-v2 | text_name_context | 1 | 1 | 26 | 29 | 35 |
| thenlper/gte-small | text_name | 1 | 1 | 43 | 142 | 215 |
| thenlper/gte-small | text_name_context | 1 | 1 | 1 | 1 | 28 |
| thenlper/gte-base | text_name | 1 | 2 | 83 | 168 | 225 |
| thenlper/gte-base | text_name_context | 1 | 1 | 1 | 1 | 29 |
| thenlper/gte-large | text_name | 1 | 0 | 58 | 165 | 228 |
| thenlper/gte-large | text_name_context | 1 | 1 | 1 | 1 | 25 |

## Размер крупнейшей объединённой группы

| Модель | Режим | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 |
|---|---|---:|---:|---:|---:|---:|
| BAAI/bge-small-en-v1.5 | text_name | 329 | 49 | 19 | 16 | 16 |
| BAAI/bge-small-en-v1.5 | text_name_context | 217 | 125 | 70 | 54 | 19 |
| BAAI/bge-base-en-v1.5 | text_name | 38 | 22 | 16 | 16 | 16 |
| BAAI/bge-base-en-v1.5 | text_name_context | 1122 | 1082 | 239 | 60 | 25 |
| BAAI/bge-large-en-v1.5 | text_name | 98 | 22 | 17 | 16 | 16 |
| BAAI/bge-large-en-v1.5 | text_name_context | 600 | 161 | 70 | 34 | 13 |
| intfloat/e5-small-v2 | text_name | 1122 | 1122 | 982 | 109 | 17 |
| intfloat/e5-small-v2 | text_name_context | 1122 | 1122 | 1039 | 70 | 32 |
| intfloat/e5-base-v2 | text_name | 1122 | 1113 | 823 | 28 | 16 |
| intfloat/e5-base-v2 | text_name_context | 1122 | 1122 | 239 | 70 | 31 |
| intfloat/e5-large-v2 | text_name | 1122 | 1110 | 841 | 35 | 16 |
| intfloat/e5-large-v2 | text_name_context | 1122 | 1122 | 161 | 70 | 34 |
| thenlper/gte-small | text_name | 1122 | 1120 | 880 | 28 | 16 |
| thenlper/gte-small | text_name_context | 1122 | 1122 | 1122 | 1122 | 125 |
| thenlper/gte-base | text_name | 1122 | 1113 | 275 | 21 | 16 |
| thenlper/gte-base | text_name_context | 1122 | 1122 | 1122 | 1122 | 88 |
| thenlper/gte-large | text_name | 1122 | 1119 | 619 | 24 | 16 |
| thenlper/gte-large | text_name_context | 1122 | 1122 | 1122 | 1122 | 240 |

## Переходы к крупным транзитивным группам

Для каждой пары модель/режим показан соседний шаг сетки с максимальным абсолютным увеличением крупнейшей группы при снижении порога. Это числовая диагностика резкого роста, не граница семантической безопасности. Группа >50% корпуса отмечает структурное доминирование; порог 50% не используется для отбора пар.

| Модель | Режим | Наибольший шаг | Размер до → после | Уже >50% корпуса при |
|---|---|---|---:|---|
| BAAI/bge-small-en-v1.5 | text_name | 0.80 → 0.75 | 49 → 329 | не наблюдалось |
| BAAI/bge-small-en-v1.5 | text_name_context | 0.80 → 0.75 | 125 → 217 | не наблюдалось |
| BAAI/bge-base-en-v1.5 | text_name | 0.80 → 0.75 | 22 → 38 | не наблюдалось |
| BAAI/bge-base-en-v1.5 | text_name_context | 0.85 → 0.80 | 239 → 1082 | 0.8 |
| BAAI/bge-large-en-v1.5 | text_name | 0.80 → 0.75 | 22 → 98 | не наблюдалось |
| BAAI/bge-large-en-v1.5 | text_name_context | 0.80 → 0.75 | 161 → 600 | 0.75 |
| intfloat/e5-small-v2 | text_name | 0.90 → 0.85 | 109 → 982 | 0.85 |
| intfloat/e5-small-v2 | text_name_context | 0.90 → 0.85 | 70 → 1039 | 0.85 |
| intfloat/e5-base-v2 | text_name | 0.90 → 0.85 | 28 → 823 | 0.85 |
| intfloat/e5-base-v2 | text_name_context | 0.85 → 0.80 | 239 → 1122 | 0.8 |
| intfloat/e5-large-v2 | text_name | 0.90 → 0.85 | 35 → 841 | 0.85 |
| intfloat/e5-large-v2 | text_name_context | 0.85 → 0.80 | 161 → 1122 | 0.8 |
| thenlper/gte-small | text_name | 0.90 → 0.85 | 28 → 880 | 0.85 |
| thenlper/gte-small | text_name_context | 0.95 → 0.90 | 125 → 1122 | 0.9 |
| thenlper/gte-base | text_name | 0.85 → 0.80 | 275 → 1113 | 0.8 |
| thenlper/gte-base | text_name_context | 0.95 → 0.90 | 88 → 1122 | 0.9 |
| thenlper/gte-large | text_name | 0.90 → 0.85 | 24 → 619 | 0.85 |
| thenlper/gte-large | text_name_context | 0.95 → 0.90 | 240 → 1122 | 0.9 |

## Как читать агрессивность и перспективные пороги

Меньшее число оставшихся узлов, большая крупнейшая группа и большее число междокументных пар означают более агрессивное экспериментальное сжатие, а не лучшую модель. Для сравнения при самом строгом проверенном пороге:

| Модель | Режим, τ=0.95 | Осталось V | Max группа | Документов в ней | Смешанных типов групп | Косвенно связанных пар ниже τ | Min cosine в Max группе |
|---|---|---:|---:|---:|---:|---:|---:|
| BAAI/bge-small-en-v1.5 | text_name | 889 | 16 | 5 | 48 | 1 | 1.000000 |
| BAAI/bge-small-en-v1.5 | text_name_context | 697 | 19 | 1 | 119 | 465 | 0.800639 |
| BAAI/bge-base-en-v1.5 | text_name | 894 | 16 | 5 | 43 | 12 | 1.000000 |
| BAAI/bge-base-en-v1.5 | text_name_context | 602 | 25 | 1 | 133 | 817 | 0.786082 |
| BAAI/bge-large-en-v1.5 | text_name | 901 | 16 | 5 | 42 | 4 | 1.000000 |
| BAAI/bge-large-en-v1.5 | text_name_context | 801 | 13 | 1 | 99 | 204 | 0.856472 |
| intfloat/e5-small-v2 | text_name | 802 | 17 | 5 | 56 | 91 | 0.952161 |
| intfloat/e5-small-v2 | text_name_context | 156 | 32 | 1 | 121 | 4610 | 0.837397 |
| intfloat/e5-base-v2 | text_name | 825 | 16 | 5 | 54 | 110 | 1.000000 |
| intfloat/e5-base-v2 | text_name_context | 271 | 31 | 1 | 150 | 2794 | 0.829141 |
| intfloat/e5-large-v2 | text_name | 835 | 16 | 5 | 57 | 28 | 1.000000 |
| intfloat/e5-large-v2 | text_name_context | 206 | 34 | 1 | 144 | 3395 | 0.837007 |
| thenlper/gte-small | text_name | 787 | 16 | 5 | 68 | 108 | 1.000000 |
| thenlper/gte-small | text_name_context | 62 | 125 | 2 | 42 | 18648 | 0.776460 |
| thenlper/gte-base | text_name | 822 | 16 | 5 | 56 | 78 | 1.000000 |
| thenlper/gte-base | text_name_context | 31 | 88 | 3 | 29 | 13230 | 0.793602 |
| thenlper/gte-large | text_name | 825 | 16 | 5 | 55 | 91 | 1.000000 |
| thenlper/gte-large | text_name_context | 25 | 240 | 5 | 24 | 42730 | 0.820776 |

mixed_type_groups считает группы, где исходные наборы entity_types участников различаются; groups_with_multiple_type_labels дополнительно считает группы с несколькими метками в объединении всех типов, даже если одинаковая множественная типизация была у каждого участника. Ни один из этих признаков не блокирует экспериментальное объединение.

transitive_below_threshold_pairs = Σ size(size−1)/2 − число выбранных пар. Это точное число пар внутри полученных групп, которые сами **не прошли** порог, но оказались приравнены транзитивным замыканием. Низкий min cosine внутри крупнейшей группы показывает этот эффект непосредственно. Компонента может содержать все 30 документов и разные типы без высоких попарных сходств у всех её участников.

По названиям при τ=0.95 BGE сохраняют 889–901 узел и 248–251 изолированных вершин; E5/GTE — 787–835 узлов. При τ=0.80 у BGE крупнейшая группа содержит 22–49 сущностей, у E5/GTE — 1110–1122. Таким образом, BGE по названиям дают более умеренное структурное сжатие в этой сетке; это не сравнение семантического качества.

Контекст резко усиливает сжатие. При τ=0.95 остаётся: BGE 602–801 узел, E5 156–271, GTE 25–62. У GTE с контекстом уже при τ=0.90 все 1122 сущности образуют одну группу, а все 1760 отношений превращаются в петли. По названиям у E5 и GTE-small/large максимальный скачок крупнейшей группы виден при снижении 0.90 → 0.85, у GTE-base — 0.85 → 0.80 (крупнейшие группы всех E5/GTE при 0.85: 275–982); у GTE с контекстом — уже 0.95 → 0.90.

Для последующей проверки перспективны прежде всего τ=0.90 и 0.95 для BGE/text_name и τ=0.95 для E5/GTE/text_name: они предшествуют сильному росту групп, а крупнейшие группы остаются сравнительно небольшими. Для первой проверки BGE/text_name_context предпочтителен строгий край τ=0.95: уже там есть смешанные типы и транзитивные пары ниже порога. E5/text_name_context при 0.95 оставляет лишь 156–271 узел; GTE/text_name_context даже при 0.95 оставляет 25–62 узла, поэтому эти режимы в данной сетке не выглядят умеренным сценарием объединения. Это выбор кандидатов для проверки, не разрешение автоматического merge и не общий откалиброванный порог.

Нулевые изолированные вершины сами по себе не улучшение: при схлопывании в одну вершину исходные отношения превращаются в петли. Поэтому показаны оба определения изоляции. Резкое падение V, конфликтные типы и поглощение разных документов — сигналы для проверки, не подтверждение новой идентичности. Типы, повреждённые данные и решения 2C не фильтровались. Семантическая точность и качество Graph-RAG этим экспериментом не измерены.

## Проверки и воспроизводимость

Проверены 1122 ID, 30 документов, покрытие 1211 исходных entity ID и 1327 mention ID, все 1760 исходных отношений и полная сетка 90 конфигураций. Ровно 628881 проверяемая пара на представление, без самопар и двойного учёта направлений. Union-Find покрывает каждую вершину ровно один раз; число сокращений согласовано с составом групп. Число пар выше порога проверено прямым сравнением, а число пар, оставшихся вершин и размер максимальной группы проверены на монотонность. На 120 контрольных сценариях результат сопоставлен с независимым обходом графа (компоненты, изоляция, петли, типы, транзитивность и границы порогов); повторная симуляция дала идентичные результаты. Дополнительно проверены крайние сценарии без объединений и с полным схлопыванием на всех реальных 1122 сущностях и 1760 отношениях. Полные embedding checksums совпадают с 2B. SHA-256 старых артефактов и файлов моделей контролируются до и после работы.

Скрипт: `.venv/Scripts/python.exe scripts/sweep_semantic_thresholds.py`. Используются существующие библиотеки и код 2B; ничего не скачивается и не устанавливается. Создаются только threshold_results.csv и report.md. Canonical mapping, отношения и Neo4j не изменены, Semantic Merge/2D не выполнялся.

SHA-256 защищённых артефактов:

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
- `reports/ner_v2/research_run2/input_abstracts.jsonl`: `684ca3a758a128c6382ab924f5c131f0e27faa9f7625b431ae0d6db975e543b1`
- `reports/re_v2/relex_comparison/threshold035/e3_base/entities.jsonl`: `19039d87119130dcc37d2c472a7f2242c6a3ad5059e46a1ff431e6fa37e3b6ba`
- `reports/re_v2/relex_comparison/threshold035/e3_base/relations.jsonl`: `bbb4e602abccedcc776b59cfd526f8ffe7ffb46bdea7f7dc3c38df33174dd099`
- `scripts/benchmark_semantic_embeddings.py`: `c8acc0440e3d98f77d4903d54dee2ca0984662fe46b45dbc39c7dfba6853ee41`
