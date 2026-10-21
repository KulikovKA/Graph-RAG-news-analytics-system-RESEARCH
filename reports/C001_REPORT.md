# C-001 — отчёт о реальном CPU прототипе архитектуры C

Дата: 2026-10-21 (Europe/Moscow). Статус: **исполняемый baseline, недостаточное качество end-to-end RE**.

## 1. Результат и структура

Реализован независимый Python-проект без backend/frontend/API/БД: NER модель, contextual kinds, bounded directed pairs, обучаемый multi-label RE, scope/qualifier gates, conservative concepts, exact evidence/schema validation, JSON output, отдельный evaluator. Структура и команды полностью приведены в [README](../README.md). Код: `src/cgraph`, конфиг: `config/c001.json`, контракты: `contracts`, проверки: `tests`, результаты: `reports`.

Все 12 DEV-документов обработаны реальными моделями: 477 entities, 2907 candidate pairs, 566 raw proposals, 46 provisional assertions. Бюджет отбросил 594 дополнительных пар. Это не synthetic smoke experiment. Искусственные minimal-pair tests приведены отдельно.

## 2. Данные, split и защита от leakage

Upstream commit `169834dec45b527f078620bd429d7286eb2700a1` репозитория [Graph-based-RAG-for-article-analysis](https://github.com/KulikovKA/Graph-based-RAG-for-article-analysis). Dataset: `pilot24-owner-reviewed-gold-v1-20261021`. Оригинальная ontology v1.0 сохранена byte-for-byte, SHA-256 `c3c0186ce492560fb27be400194596f70fae66a89daf4b7844636264cd3e3692`. Её старый approval header не переписан: актуальный owner-reviewed release envelope хранится отдельно. Один owner review, annotation origin AI_GENERATED, independent secondary review отсутствует.

Seed 42. Train: **D004, D010, D017, D024, D026, D038, D060, D078**. Validation: **D001, D003, D035, D092**. Группы C02/C07 целиком оставлены для validation; тематические pilot24 clusters — только proxy для family grouping, независимой проверки семейства не было. Это descriptive DEV validation на четырёх документах, без подтверждающего holdout, confidence claims и настройки thresholds по validation.

Loader проверяет allowlist 12 IDs, split DEV, revision IDs и pinned revision-content hash, inventory chunk IDs/ordinals/sections/languages/Unicode boundaries, stored/source SHA-256 и byte hashes входов. Полный revision hash сверяется с release metadata: полный исходный revision вне frozen chunks не реконструируется. `fetch-dev` получает ровно 14 разрешённых файлов по pinned commit; все 14 SHA-256 проверены реальным скачиванием. CALIBRATION/HOLDOUT тексты, Gold и review packets не открывались.

Раздельные stores: `data/inputs`, `data/labels`, `artifacts/*predictions`, offline evaluation. Runtime ontology — нормативная проекция без Gold examples и document-specific boundaries. Реальный end-to-end run выполнен через audit guard, запрещающий open labels/gold/original ontology, импорт annotation-модуля и socket.connect. Gold entities использовались только в отдельном диагностическом run.

D024: `exhaustive_relations=false`; его positive annotations разрешены для обучения, все assumed negatives исключены. Остальные hard negatives — close same-sentence type-compatible pairs и reversed positives из exhaustive train-документов; unresolved evidence не становится negative Gold. Train precision на неполном D024 ограничена полнотой аннотаций и приведена только descriptively.

## 3. Реальные модели и обучение

NER: [urchade/gliner_small-v2.1](https://huggingface.co/urchade/gliner_small-v2.1), revision `4e091416cf7c3481db542c2a3d26156916f3a47f`, Apache-2.0. Реально загружено **152,648,704 параметра**, CPU float32, strict=True, класс `UniEncoderSpanGLiNER`. Окна 140 whitespace words, overlap 30, threshold 0.35, 12 зафиксированных label descriptions. Nested/overlapping spans сохраняются, dedup только exact span+kind. Exact codepoints проверяются сравнением с неизменным source string.

Tokenizer/backbone config: [microsoft/deberta-v3-small](https://huggingface.co/microsoft/deberta-v3-small), revision `a36c739020e01763fe789b4b85e2df55d6180012`, MIT по card, `DebertaV2Tokenizer`. Backbone config встраивается в derived runtime config; weights не меняются. File hashes, включая weights/tokenizer/config/cards, в [model_files.json](../contracts/model_files.json); фактическая runtime identity в [runs/end_to_end.json](runs/end_to_end.json). Никакой generative model не загружается.

RE: **TF-IDF marked-pair linear sigmoid heads**, one-vs-rest logistic regression (`liblinear`, C=1, balanced classes, seed 42). Case-sensitive Unicode word tokens, unigram+bigram, до 12 000 features. Head/tail boundary markers и direction marker различают направление; применяется immutable ontology domain/range. Это обученный лексический classifier, **без pretrained RE encoder**. SciBERT остаётся base encoder и здесь не называется готовой RE моделью.

Изучен [GLiNER-Relex base](https://huggingface.co/knowledgator/gliner-relex-base-v1.0), revision `e6a880049a19c5cc222a7a479c32e84b0d8cdd9a`, Apache-2.0: joint NER+RE относится к arm F, поэтому не подменяет раздельную архитектуру C. Его веса в этом эксперименте не запускались. Выбран минимальный полностью обучаемый CPU marked-pair baseline; перенос на encoder head оставлен C-002.

Обучение: 1289 примеров, 1073 hard negatives. Из 23 relation outputs 21 обучен; `INTEGRATED_IN` и `OPTIMIZES` имеют нулевой support и перечисляются в `unsupported_labels`; неопределённые пары получают ABSTAIN. Все 23 labels присутствуют в inventory; 24-й output — NO_RELATION. Multi-label positives допустимы; ABSTAIN — отдельная policy при низких scores, конфликте с NO_RELATION или unsupported labels. Thresholds 0.65 фиксированы заранее, не calibrated probabilities.

Точная identity local RE parameters: `cc9401bf53cb0a53b9d84aaddfa3412053016b95214d388514091af65a5c08da`; feature/tokenizer hash `5be5f97fb79b9d7c629f43d9853aad75dd108cdd73a627cddedc715408dd1068`. Train IDs, support каждого класса и точные settings в [training_metadata.json](training_metadata.json). Checkpoint из внешнего Hub для RE отсутствует; лицензия sklearn — BSD-3-Clause, отдельная лицензия новых fitted parameters не назначалась. Веса/NPZ/cache в Git не включены.

## 4. Поддержанные и неподдержанные возможности

Реализованы raw mention spans, contextual ANALYTE kind, directed/multi-label RE, NO_RELATION/ABSTAIN, domain/range, pair cap 400/chunk и gap 300 characters, local sentence candidates, parsing простых чисел/единиц, simple negation/pseudo-negation и modality cues, conservative NFC/whitespace name normalization с сохранением chemistry case/composition, exact local acronym rules, source-scoped concepts и evidence checks.

Нет полного contextual-role classifier (прочие roles UNKNOWN), coreference, cross-sentence RE, discontinuous mention model, надёжного range/comparator parsing, n-ary numeric/condition attachment, calibrated NLI или глобального identity classifier. Device instances не объединяются по имени; RELATED/hierarchy не превращаются в SAME.

Unresolved qualifiers/negation/comparison/distribution/overlapping endpoints получают ABSTAIN. Простые numeric spans сохраняются отдельно с attachment UNKNOWN; числа не назначаются всем соседним entities. Required qualifiers не выдумываются. Assertion evidence и qualifier rules проверяются до JSON Schema validation. Accepted assertions помечены **ACCEPTED_PROVISIONAL**, `qualifier_status=UNKNOWN`, support_decision=CLASSIFIER_AND_LEXICAL_SCREEN. Это кандидатный research graph, не independently established semantic truth.

Lexical predicate screen требует явный cue между endpoints, поэтому одна co-occurrence не является достаточной. Это только deterministic screen: ложные semantic relations и неверные NER types всё равно остаются. Grounded exact quote не доказывает смысл утверждения.

## 5. Качество на четырёх validation-документах

Основной end-to-end run: Gold не подаётся в NER/RE. Строгий one-to-one multiset matching exact chunk/codepoint spans+kind и directed predicate; concept IDs не участвуют. Нулевой precision denominator → null; Gold positives при пустом output → recall/F1=0.

| Метрика | TP / FP / FN | Precision | Recall | F1 |
|---|---|---|---|---|
| Entity span exact | 79 / 51 / 128 | 0.6077 | 0.3816 | 0.4688 |
| Typed entity exact | 70 / 60 / 137 | 0.5385 | 0.3382 | 0.4154 |
| Raw end-to-end relation | 5 / 86 / 127 | 0.0549 | 0.0379 | 0.0448 |
| Gated end-to-end relation | 0 / 21 / 132 | 0.0000 | 0.0000 | 0.0000 |
| RE given candidate endpoints | 0 / 0 / 7 | null | 0.0000 | 0.0000 |
| Qualified relation | 0 / 21 / 132 | 0.0000 | 0.0000 | 0.0000 |

Candidate recall: **7/132 = 0.0530**. Abstention: **508/623 = 0.8154**. Unknown qualifier completeness не получает qualified TP, а Gold qualifiers остаются в знаменателе; нулевой qualified F1 честно отражает отсутствие полной поддержки.

Отдельный Gold-entity RE diagnostic, только validation, без NER inference:

| Метрика | TP / FP / FN | Precision | Recall | F1 |
|---|---|---|---|---|
| Raw RE, Gold entities | 50 / 88 / 82 | 0.3623 | 0.3788 | 0.3704 |
| Gated RE, Gold entities | 8 / 2 / 124 | 0.8000 | 0.0606 | 0.1127 |
| RE given bounded candidate endpoints | 8 / 2 / 117 | 0.8000 | 0.0640 | 0.1185 |
| Qualified RE | 0 / 10 / 132 | 0.0000 | 0.0000 | 0.0000 |

Диагностический candidate recall **125/132 = 0.9470**. Entity quality для этой диагностики не публикуется: Gold spans дали бы тавтологическую оценку. Данные не смешиваются с end-to-end.

На train end-to-end gated RE F1 = 0.0353 (TP=5); это **не независимый результат**. Полные micro/macro-document и per-document counts: [validation_e2e.json](validation_e2e.json), [validation_gold_entities_re.json](validation_gold_entities_re.json), [train_descriptive_e2e.json](train_descriptive_e2e.json). Human semantic-support rate, confidence intervals, false-merge metrics и downstream retrieval quality — NOT_MEASURED.

## 6. Производительность

Нативная Windows, Python 3.11.9, AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD, 16 logical CPUs, 4 PyTorch threads, CPU float32; WSL2 execution не проверена. Полный lock: [uv.lock](../uv.lock), Linux torch закреплён с CPU index без CUDA packages. Runtime package versions сохранены в run manifests.

Реальный fresh-process run **всех 12 DEV** с локальными model files: wall **47.524 s**, CPU **90.266 s**, peak working set **1.624 GiB** (1743646720 bytes). Method: Windows GetProcessMemoryInfo PeakWorkingSetSize, process lifetime incl. imports/model load. Wall включает dependency imports после CLI dispatch и model loading; download/training не включены. NER model setup/load **11.877 s**, отдельно от per-document extraction. Это одна описательная CPU measurement, не доказательство speedup относительно A/B/F.

Model invocations: **27 NER calls**, instrumented **27 encoder forwards**, **19 RE batch calls** на 2907 pairs; всего 46 model API calls. **LLM calls = 0**; сетевые подключения во время extraction запрещены audit hook. Maximum observed encoder sequence — 295 tokens. Нет генеративного fallback.

Сумма стадий по всем 12 DEV (ms):

| Стадия | ms |
|---|---|
| candidates_ms | 40.43 |
| ner_ms | 14245.93 |
| relation_classifier_ms | 691.36 |
| scope_qualifiers_ms | 37.86 |
| validation_ms | 10.61 |

Linking учитывается отдельно в per-document execution. Сумма stage time не равна process wall: imports, loading, schema checks и serialization имеют отдельную стоимость.

| Validation doc | Typed entity F1 | Gated RE F1 | extraction wall ms | CPU ms |
|---|---|---|---|---|
| D001 | 0.3400 | 0.0000 | 1319.7 | 5156.2 |
| D003 | 0.3564 | 0.0000 | 1483.1 | 5750.0 |
| D035 | 0.5263 | 0.0000 | 620.0 | 2437.5 |
| D092 | 0.5063 | 0.0000 | 1175.6 | 4484.4 |

Отдельный Gold RE process: wall 4.556 s, CPU 3.578 s, peak 0.108 GiB, 6 RE calls, 0 encoder/LLM calls. Полная performance scope указана в JSON: validation quality не выдаётся за отдельный 12-document latency run.

## 7. Примеры реального output и основные ошибки

Ниже неизменённые model outputs. Успешные train-примеры иллюстрируют функционирование, а не независимое качество. Validation false positive включён явно. Exact offsets и evidence SHA-256 всех примеров: [examples.json](examples.json).

- **D001 / validation / validation_false_positive**: `E-textiles` (MATERIAL, [0,10)) → `HAS_PROPERTY` → `light weight` (PROPERTY, [68,80)). Score 0.9631; `ACCEPTED_PROVISIONAL`; reason `None`. Exact typed Gold match: `False`.

  Evidence chunk `6319af2d-0fe1-4daa-87c6-131c379a75c9`, [0,168):

  > E-textiles are gaining growing popularity recently due to low cost, light weight, and conformable compatibility with clothes in wearable and portable smart electronics.

- **D003 / validation / validation_correct_raw_abstained**: `wearable devices` (DEVICE, [393,409)) → `DETECTS` → `biomarkers` (ANALYTE, [420,430)). Score 0.7279; `ABSTAIN`; reason `UNSUPPORTED_CAPABILITY:NEGATION_SCOPE`. Exact typed Gold match: `True`.

  Evidence chunk `45f36916-7688-4edb-8dd1-2ac50ea13339`, [315,498):

  > Moisture-resistant, stretchable NO x gas sensors can expand the capability of wearable devices to detect biomarkers from humans and exposed environments for early disease diagnostics.

- **D017 / train / correct_train**: `Graphene` (MATERIAL, [302,310)) → `HAS_PROPERTY` → `strong surface activities` (PROPERTY, [457,482)). Score 0.9516; `ACCEPTED_PROVISIONAL`; reason `None`. Exact typed Gold match: `True`.

  Evidence chunk `52c7bac2-feff-4316-934f-2726c4842540`, [302,483):

  > Graphene and related layered inorganic analogues have shown great potential for gas-sensing applications because of their large specific surface areas and strong surface activities.

- **D026 / train / correct_train**: `2D metal carbide MXenes` (MATERIAL, [465,488)) → `HAS_PROPERTY` → `high metallic conductivity` (PROPERTY, [504,530)). Score 0.8901; `ACCEPTED_PROVISIONAL`; reason `None`. Exact typed Gold match: `True`.

  Evidence chunk `5a1139f9-9bf1-4665-96e1-c15a6477ded6`, [439,684):

  > Here, we demonstrate that 2D metal carbide MXenes, which possess high metallic conductivity for low noise and a fully functionalized surface for a strong signal, greatly outperform the sensitivity of conventional semiconductor channel materials.

Главные наблюдения: NER boundary/type errors оставили только 7 из 132 Gold relations среди end-to-end candidates, хотя с Gold entities проходят 125/132. Лексические heads плохо переносятся на другие topical clusters. Из 91 raw validation relations только 5 совпали с Gold; gates удалили и эти TP, а 21 оставшийся provisional assertion не совпал с typed Gold. Следовательно, deterministic screen одновременно теряет recall и не гарантирует precision. Увеличение strictness само по себе недостаточно.

Обнаружен конкретный дефект cue rules: uppercase chemical `NO x` попадает под case-insensitive `no` и вызывает ложный ABSTAIN по negation scope (пример D003 выше). Он сохранён в measured C-001 baseline; следующая версия должна отдельно различать химическое NO/NOx и языковое отрицание. После просмотра validation правила и thresholds ради улучшения scores не подгонялись.

Reasons по raw proposals всех 12 DEV:

| Reason | Count |
|---|---|
| NO_EXPLICIT_PREDICATE_BETWEEN_ENDPOINTS | 365 |
| UNSUPPORTED_CAPABILITY:OVERLAPPING_RELATION_ENDPOINTS | 89 |
| UNSUPPORTED_CAPABILITY:QUALIFIER_ATTACHMENT | 35 |
| UNSUPPORTED_CAPABILITY:NEGATION_SCOPE | 18 |
| UNSUPPORTED_CAPABILITY:COMPARISON_OR_DISTRIBUTION | 7 |
| UNSUPPORTED_CAPABILITY:NEGATION_ATTACHMENT | 6 |

Эти counts относятся к proposals, abstention denominator — к candidate pairs; их нельзя складывать как одну метрику.

Диагностика установки: первый Xet download завис, заменён на pinned HTTPS с timeout и checksum. Strict GLiNER load прошла. Transformers 5.1.0 вывел generic Mistral regex warning для DeBERTa tokenizer; архитектура/веса проверены, чужой tokenizer fix не применён. Это зарегистрированное ограничение runtime, а не скрытая подмена модели.

## 8. Проверки и C-002

22 теста: source identity/hash tamper, split/cluster isolation, immutable ontology, label-free projection, Unicode/repeated offsets, overlapping mentions, directed markers, pair budgets, simple/complex negation и not-only, co-occurrence rejection, numeric spans, chemical/device cannot-merge, multiset/empty metrics, strict qualifier penalty, schema и zero-LLM constraint, real classifier fitting/serialization, train labels boundary и provenance без чтения annotation-модуля. Полный model run — отдельная реальная CPU интеграционная проверка с denied label/network access. См. [tests.json](tests.json).

Приоритеты C-002: (1) scientific NER и contextual typing на train-only domain data; (2) замена TF-IDF головы на marked-pair encoder head либо shared span classifier с сохранением arm identity; (3) расширение train coverage редких relations, exhaustive negative judgments; (4) sentence/condition attachment, negation/coreference scope и n-ary qualifiers; (5) независимый semantic audit и calibration внутри DEV, без подгонки на текущих четырёх validation docs; (6) family grouping, macro/cluster uncertainty и независимый будущий protocol. Текущий DEV validation уже просмотрен и не должен называться untouched confirmatory test в C-002.

Репозиторий содержит только код, контракты, measured metrics, hashes и разрешённые DEV excerpts. Model weights/cache, DEV labels и полные predictions остаются локально под gitignore. HOLDOUT/private data/secrets не публикуются. Основной проект использован только для контрактов, DEV-данных и документации, его рабочее дерево не изменено.
