# Stage 3A.0 — Architecture Audit for Hybrid Graph Retrieval

Дата: 2026-11-16. База аудита: commit `5b177c8`, ветка `design-v2-sequential-kg`.

## Область проверки и ограничения доказательств

Проведён статический аудит кода и локальных артефактов, чтение NumPy-кэша, проверка его hashes, размерностей, норм и соответствия ID; проверены hashes семи файлов E5. Модели, inference, обучение, LLM, импортеры и субагенты не запускались. `.env` не читался. Единственный создаваемый файл — этот отчёт, согласно заключительной просьбе пользователя; commit/push запрещены для этого аудита.

Целевой namespace: `relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v2_sem_name_t0830`.

В процессе нет переменных NEO4J_URI/USER/PASSWORD/DATABASE. Live Neo4j не подключался: фактическое текущее содержимое сервера, индексы, доступность read-only пользователя и latency НЕ проверены. Числа 30 документов, 811 сущностей, 36 концептов, 1307 RELEX_REL, 492 SIMILAR_TO, 138 HAS_SHARED_CONCEPT подтверждены ожидаемым payload импортера и контекстом пользователя, но не новым live read-back. Локальный полный snapshot относится к предшественнику `_concept_v1`: 34 концепта и 130 memberships. Импортер 2G.3 копирует его, изменяет namespace/fingerprint и добавляет 8 memberships и 2 концепта; сущности и факты сохраняет. Не выдавать проверки snapshot за проверку live 2G.3.

## 1. Уже реализовано и пригодно для повторного использования

### 1.1 GLiNER: точная линия происхождения

Источники: `reports/ner_v2/research_run2/E3/config.json`, `src/cgraph/ner_v2/quality_audit.py`, `scripts/build_relex_graphs.py`, `scripts/reconstruct_global_kg_2f3.py`.

- Checkpoint: `fastino/gliner2.5-base-v1`, revision `54785d51df8d86a0d8f2eb212fec15871aad2c4d`.
- API: `gliner2.AutoExtractor`, локальная директория `artifacts/ner_v2/models/gliner25`; config, tokenizer и weights физически существуют. Hash весов из сохранённого E3 config: `7274094de2e0c2a37a386f55fc4e23061a954da5bd7a335e7dfe56f2743c277a`. Веса GLiNER повторно не хешировались и не загружались.
- CPU FP32, seed 42, четыре потока, `attention_backend='standard'`.
- `extract_entities(..., include_confidence=True, include_spans=True, overlap_policy='allow', max_len=None)`; schema создаётся через `.entities(SHORT)`.
- NER descriptors — именно короткие E3 labels, а не длинные descriptions из ontology и не baseline GLiNER-small:

| Код | Descriptor |
|---|---|
| OBJECT | physical, biological or digital object |
| ACTOR | person or organization |
| CONCEPT | scientific concept or theory |
| METHOD | algorithm or research method |
| PROCESS | event or scientific process |
| DATA_RESOURCE | dataset or research corpus |
| ATTRIBUTE | property, parameter or metric |
| RESULT | measured or observed result |
| CONDITION | experimental or contextual condition |
| TASK | research task or application |
| MECHANISM | mechanism explaining a phenomenon |

Пороги относятся к разным операциям: исходный inference E3 выполнялся с 0.15; для ReLEx использованы `E3/mentions_035.jsonl` (0.35). В актуальный KG попали contextual entities с `max_member_confidence >= 0.46`, после canonicalization и согласованных merges. 0.46 — фильтр сущности по максимуму исходных members, а не повторный NER inference. Порог ReLEx/семантического linking нельзя смешивать с NER confidence.

Preprocessing: исходный текст без перезаписи; окна по `\S+`, 110 слов, overlap 25. Prompt-aware token budget 512; слишком длинное окно делится по предложениям/границам слов с перекрытием. `detection.postprocess` проверяет точный текст span, переводит локальные offsets в абсолютные и дедуплицирует `(start,end,label)` по максимуму score, сохраняя вложения и альтернативные labels. Индексы — Python Unicode `[start,end)`.

N3 — последующая нормализация через `ScientificNormalizer`: spaCy `en_core_web_sm` без NER/parser, осторожное приведение конечного существительного во множественном числе, обработка possessive, NFC и whitespace для поисковой формы, abbreviation candidates. Не меняет исходные evidence offsets. Имена для E5 берутся из сохранённого representative surface/name, не из произвольно нормализованного вопроса.

Для короткого английского вопроса extractor и postprocess пригодны, обычно достаточно одного окна. Research runner целиком непригоден как online API: привязан к frozen corpus, checkpoint/report записи и метрикам. Нужен тонкий адаптер с ленивой однократной загрузкой. Качество на вопросах, confidence threshold, аббревиатуры без определения и русский язык не проверены. Для пустого NER предусмотреть явно помеченный fallback vector search по полному вопросу; совпадение имён не объединяет контекстные сущности разных документов.

### 1.2 Embeddings и локальный поиск

Физический кэш: `.cache/semantic_concept_sweep/`:

- `text_name_vectors.npy`: `(888,384)`, float32; SHA-256 `0b986f44c8a83a4e4ee3fc78f99405f150257c8026c3e5e9504f924e2e9de327`.
- `text_name_context_vectors.npy`: `(888,384)`, float32; SHA-256 `947a07bb58c6a13bc93da831a44898f07b06c8b140089689813f015e9eb084c9`.
- `representations.json`: упорядоченные `entities` (811) и `concepts` (77 reference variants).
- `embedding_manifest.json`: fingerprint, hashes, модель, рецепт; `baseline.json`: полный snapshot `_concept_v1`.
- `text_name_ee.npy`, `text_name_ec.npy` и соответствующие context-матрицы: вспомогательные cosine matrices, НЕ embeddings вопросов.

Первые 811 строк каждой vector matrix соответствуют entities в порядке JSON; последние 77 — concept reference variants. Это не 77 уникальных концептов и не готовые векторы всех 36 актуальных концептов. Все шесть `.npy` совпали с manifest hashes; оба vector массива конечны и имеют нормы примерно 1. Для 811 entity IDs проверены равенство множеств, уникальное соответствие snapshot nodes, document IDs и exact names. Для целевого KG key строится как `namespace + ':' + entity_id`; импортер 2G.3 сохраняет entity IDs. Live совпадение ещё требуется проверить read-only.

Модель E5: `intfloat/e5-small-v2`, 384 dimensions; физическая директория `C:/Users/Kirill/Desktop/embedding-models/intfloat__e5-small-v2`. Семь model/config/tokenizer файлов присутствуют, их hashes совпали с manifest. Hash `model.safetensors`: `45bfa60070649aae2244fbc9d508537779b93b6f353c17b0f95ceccb1c5116c1`. Git revision E5 из изученного manifest не установлен; воспроизводимость здесь фиксируется hashes файлов.

Exact recipe (`scripts/sweep_semantic_concepts.py`, `scripts/benchmark_semantic_embeddings.py`, `src/cgraph/semantic_concepts.py`):

1. `text_name` = сохранённое имя без дополнительного casefold/лемматизации. Context mode: `Entity: <name>\nContext: <representative_context.text>`.
2. Prefix **`query: ` для обеих сторон**; кэш не использует `passage: `.
3. Собственный `tokenizer.json`, special tokens, truncation 512, padding до длины batch; CPU FP32, batch 8.
4. Attention-masked mean `last_hidden_state`, затем L2 normalization; сохранение float32. Cosine в semantic sweep — float64, clipping [-1,1], отдельная обработка byte-identical vectors. Нельзя округлять scores перед threshold.

Для retrieval оставить name-кэш и тот же рецепт для query mentions. Переключение corpus на `passage: ` потребовало бы новых embeddings и сейчас не нужно. Query vector в будущем требует online E5 inference; наличие corpus cache не устраняет эту операцию. Сейчас inference не запускался.

NumPy cosine Top-K достаточно: матрица 811×384 float32 занимает 1,245,696 байт (~1.19 MiB). Использовать стабильную сортировку `(-score, entity_id)`, проверку ID/order/hashes и явный abstain при слабом linking. FAISS, vector DB и новый Neo4j vector index для этого размера не нужны. Пороги 0.95 (SIMILAR_TO) и 0.830 (concept membership) не переносить автоматически на вопрос→entity.

### 1.3 Реальные отношения и evidence

| Тип | Физическое направление | Свойства и использование |
|---|---|---|
| RELEX_REL | RELEXEntity → RELEXEntity | `relation_type`, `document_id`, `source_triple_ids`, `source_confidence_values`, `source_relations_json`, `source_json`, original endpoints. Направленный извлечённый факт, не Gold-истина. |
| SIMILAR_TO | одна физическая стрелка на пару | `directed=false`, `undirected=true`, `similarity`, `model`, `mode`, `threshold`, original/current IDs, context/provenance. Обход в обе стороны; не factual evidence. |
| HAS_SHARED_CONCEPT | RELEXEntity → RELEXConcept | `factual=false`, `identity=false`, `fact_inheritance_allowed=false`; conceptual bridge entity→concept←entity. |
| RELEX_MENTIONS | RELEXDocument → RELEXEntity | `document_id`, `source_entity_ids`, `source_mention_ids`, `source_offsets_json`, `source_memberships_json`. Принадлежность и evidence lookup; не свободный тематический переход через весь документ. |

Источник: `reconstruct_global_kg_2f3.py`, `build_dynamic_concept_graph.py`, `import_hybrid_concept_2g3.py` и локальный snapshot. `RELEX_REL` сохраняет параллельные рёбра и 39 фактических self-loops в snapshot. При обходе не расширять петли; не удалять их из графа.

В текущей линии данных **нет плоских `window_text` и `evidence_json` на 1307 фактических рёбрах**. Их наличие в раннем `build_relex_graphs.edge_row` не означает наличие после canonical relation remapping. Рабочий источник: `json.loads(source_relations_json)[*].evidence[*]` с полями `window_id`, `document_id`, `start`, `end`, `text`, `entity_ids`. Проверены все 1490 исходных окон: `document.text[start:end] == text`, ошибок 0. В этих окнах нет revision_id: брать его у RELEXDocument вместе с `text_sha256`, `source_url`, DOI/title и source_reference.

У сущностей есть `representative_context_json`, aliases, `source_offsets_json`, `source_confidences_json`, `source_json`; плоские start/end относятся к representative, а confidence — max member. Для слитых nodes нельзя считать их одним mention. У новых concept memberships есть `evidence_json` с decision evidence; это доказательство основания membership, не факта RE.

Новый chunking pipeline не требуется: при подтверждении live свойств достаточно чтения Neo4j, JSON decoding и exact slice validation. Каждое окно привязывать к исходному triple/document/revision; semantic/context snippets маркировать отдельно. Окно может содержать несколько предложений — это не гарантированно минимальная цитата и не доказательство любого вывода LLM.

## 2. Чего не хватает для Retrieval

В `src/` и `tests/` не обнаружены реализации QueryAnalyzer, EntityLinker, GraphRetriever, PathReranker, EvidenceExtractor. Есть исследовательский cosine candidate search и проверка graph connectivity, но нет готового online retrieval orchestration.

Нужны адаптер вопроса, проверяемый loader existing vector cache, linking/abstention, read-only bounded traversal, path policy, ranking/diversity и evidence decoder. Отдельно нужны согласованные query fixtures и оценка entity-linking/Recall@5/evidence correctness. LLM aggregation пока только последующий потребитель результата, не часть данного аудита и не основание для нового framework.

## 3. Проблемы и политика доверия

### Статусы SIMILAR_TO нельзя интерпретировать как один approval flag

Все 492 связи сохранены намеренно, включая REJECT/REVIEW:

| Поле | Распределение в snapshot |
|---|---|
| decision_2c | REVIEW 449; REJECT 20; NOT_IN_2C 15; ACCEPT 8 |
| resolution_decision_2f2 | SHARED_CONCEPT 233; RELATED 80; REJECT 37; REVIEW 70; MERGE 72 |
| preview_status_2f2 | replaceable_by_concept_path 233; related_retained 80; rejected_candidate 37; review_pending 70; identity_redundant 72 |

2C оценивал identity; REJECT там не доказывает отсутствие смысловой связи, ACCEPT не доказывает факт. Поэтому простая фильтрация всех status fields без знания семантики некорректна. Консервативный запрет REJECT/REVIEW одновременно по обоим decision fields оставляет всего 19 связей; это вычислено на snapshot, не измерение retrieval recall.

Предложение для согласования: в trusted режиме сначала разрешать только явное `resolution_decision_2f2 == RELATED`, а старый 2C статус сохранять как provenance; SHARED_CONCEPT проходить через проверенные memberships, REVIEW/REJECT исключить, MERGE не использовать как semantic bridge. Это максимум 80 физических semantic links до удаления петель и иных фильтров, НЕ итоговая оценка доступных маршрутов. Такой приоритет позднего решения требует согласования. Альтернатива — строгие 19 связей с существенно меньшей связностью. Непонятные/отсутствующие статусы не включать молча. Ничего не удалять и не менять в Neo4j.

### Две SUSPICIOUS memberships Cell

Импортер 2G.3 устанавливает им `requires_review=true`, `review_status='SUSPICIOUS'`, хотя исходный автоматический decision был ACCEPT. Исключать такие memberships до расширения маршрута в обе стороны. Концепт: `concept:2g1:7ae66a4459e9d0bec5010222`.

- `e:231c4582bed80098121ab6d8da48133468e9eb8717b057ffa9bfd461baa3b66d`, документ W2039836112, имя cell.
- `e:b97ece67b0a4af2e49316cdea76c641bd0ac8d808822305d4e7abcc4bef49e24`, документ W2791492120, peripheral blood mononuclear cells.

Фильтр: `requires_review != true` и `review_status` не SUSPICIOUS/REVIEW/REJECT; дополнительно точный запрет этих двух пар защищает от отсутствующих flags. Не запрещать сами entity nodes или все их фактические отношения. Из 138 memberships останется максимум 136 trusted memberships до иных фильтров.

### Другие риски

- Live target и его fingerprint пока не проверены; импортер хранит итоговую verification в stdout, не создаёт постоянный отчёт 2G.3 своим main.
- Кэш исключён из Git и модель E5 лежит вне репозитория: клонирование само по себе не обеспечивает локальный поиск.
- Query язык, порог abstain и точность NER на вопросах не определены. Английские labels/нормализатор не дают гарантии для русского ввода.
- Семантические переходы и concept hubs расширяют поиск, но не переносят фактические утверждения между исследованиями.
- RE confidence, cosine и query similarity не являются одинаково калиброванными вероятностями.

## 4. Предлагаемый алгоритм и интерфейсы

### Ограниченный multi-hop

Предлагаемые начальные параметры, НЕ установленные настройки проекта: до 5 кандидатов на mention, до 10 seed entities суммарно; 3 физических KG hops; максимум 1 SIMILAR_TO и 1 concept bridge (две HAS_SHARED_CONCEPT); 20 соседей на узел, beam 50 на глубину, максимум 200 конечных кандидатов. Ввести отдельные limits для числа прочитанных записей, server timeout и явный `truncated`/причины отсечения.

Обход BFS/beam в Python с параметризованными one-hop read-only Cypher. Проверять namespace у каждого узла И ребра, label/endpoints и policy до расширения. Не использовать безграничное `[*]` с LIMIT после перечисления всех путей. Для этого небольшого графа допустимо однократное ограниченное чтение adjacency в память с fingerprint; это не новый индекс и не изменение графа.

RELEX_REL можно искать по входящим и исходящим соседям, сохраняя исходную стрелку и `traversed_reverse`; никогда не инвертировать смысл предиката. Concept bridge считать двумя физическими hops, семантическую связь обходить неориентированно. Не повторять узлы/edge keys, подавлять петли при расширении. RELEX_MENTIONS использовать отдельно для document/evidence lookup. Возвращать factual paths только при наличии хотя бы одного RELEX_REL с валидируемым evidence; pure semantic/concept paths возвращать отдельно как related context, не factual answer. Путь не доказывает транзитивность его предикатов.

### Reranking

Минимальный вариант — детерминированный scorer. После hard safety/evidence фильтров вычислять query relevance из linking scores и явного overlap вопроса с именами/предикатами/evidence. Пример стартовой, ещё не проверенной формулы:

`S = .45*R + .25*E + .20*F + .10*C - .12*Nsemantic - .05*(L-1)`.

Здесь R — relevance [0,1]; E — доля factual edges с exact evidence; F — среднее сохранённых max source confidence по каждому factual edge; C — ограниченный бонус за coverage нескольких query mentions. Число facts учитывать ограниченным бонусом внутри F, не поощрять бесконечно длинные цепи. Weights и преобразования подлежат проверке на согласованных вопросах; не перемножать confidence как вероятность истинности пути. Хранить score breakdown.

Top-5: дедупликация по factual edge IDs и документным окнам; greedy diversity с наказанием за повтор document/edge sets, без принудительного добора нерелевантных результатов. Возвращать менее пяти при нехватке evidence.

Отдельный cross-encoder reranker потенциально лучше сопоставляет формулировку вопроса и текст evidence, но добавляет модель, задержку, зависимости и необходимость выбора checkpoint/лицензии/оценки. Найденного готового reranker нет. Он не заменяет hard filters и не определяет истинность semantic edges. Рекомендация для первого этапа — scorer; модель только после baseline и отдельного разрешения.

### Минимальные Python-контракты (предложение, не существующий API)

Достаточно функций с указанными логическими именами и `TypedDict`/обычных dict; пять иерархий классов не нужны. Ниже перечисленные структуры — новые контракты, не свойства Neo4j.

| Модуль | Сигнатура | Обязательные поля результата |
|---|---|---|
| QueryAnalyzer | `analyze(question: str) -> QueryAnalysis` | `question`, `mentions`, `model_id`, `revision`, `threshold`, `warnings`; mention: `mention_id`, `surface_text`, `start`, `end`, `label`, `score`; пустой NER допустим. |
| EntityLinker | `link(query: QueryAnalysis, top_k: int) -> LinkResult` | `namespace`, `cache_fingerprint`, `candidates`, `unresolved_mentions`; candidate: `mention_id` (null для full-question fallback), `entity_id`, `node_key`, `document_id`, `score`, `rank`, `match_method`; не объединять разные IDs по имени. |
| GraphRetriever | `retrieve(seeds: list[Seed], limits: dict) -> RetrievalResult` | `namespace`, `graph_fingerprint`, `paths`, `truncated`, `diagnostics`; Seed: `node_key`, `score`, `mention_id`; Path: `path_id`, `seed_key`, ordered node keys, ordered edges, document IDs. Edge: `key`, `type`, `relation_type` (null допустим), `head_key`, `tail_key`, `traversed_reverse`, source properties/provenance и safety flags. |
| PathReranker | `rank(query, paths, top_k=5) -> list[RankedPath]` | исходный Path, `score`, `score_breakdown`, `diversity_penalty`, evidence validation summary; stable tie-break по path_id. |
| EvidenceExtractor | `extract(paths) -> EvidenceBundle` | `items`, `errors`; item: `path_id`, `edge_key`, `source_triple_id`, `window_id`, `document_id`, `revision_id`, `text_sha256`, `start`, `end`, `text`, `source_url`, `provenance`, `kind`, `validation_status`. Невосстановимое обязательное поле — ошибка/отказ, не выдуманное значение. |

Проверку доступности и точности evidence проводить до финального ранжирования; после Top-5 извлекать полный bundle. Это совместимо с целевой цепочкой: до rank нужен лёгкий evidence summary, после него — сами цитаты. LLM получает отдельно фактические assertions, semantic/concept context и provenance, с запретом делать фактический вывод только по similarity.

## 5. Независимая работа Agent A и Agent B

Это план, агенты не запускались. Все перечисленные ниже файлы — предлагаемые новые файлы, сейчас не созданы.

- Agent A: `src/cgraph/query_analysis.py`, `src/cgraph/entity_linking.py`, `tests/test_query_analysis.py`, `tests/test_entity_linking.py`. Реализует lazy E3 adapter, cache validation, deterministic Top-K и abstention. Тесты: fake extractor/encoder, маленькие synthetic arrays, пустой/Unicode вопрос, одинаковые имена в разных документах, повреждённый cache/order. Без Neo4j и реального model inference.
- Agent B: `src/cgraph/graph_retrieval.py`, `tests/test_graph_retrieval.py`. Реализует read-only adjacency reader и bounded traversal. Тесты: synthetic graph/fake driver, все edge types/directions, namespace leakage, циклы, параллельные facts, budgets, REJECT/REVIEW, SUSPICIOUS Cell. Вход seeds фиксирован, Agent A не требуется.
- Общий контракт: Seed, Edge, Path и namespace/policy из предыдущего раздела. Один интегратор заранее создаёт `src/cgraph/retrieval_types.py`; A/B его читают, но не редактируют. Общий `__init__.py`, dependencies и AGENTS.md не менять в параллельных задачах.
- Разные Git worktrees и ветки `stage3a-query-linking` / `stage3a-graph-traversal`; push только своей назначенной ветки после отдельного разрешения на начало разработки. Общую интеграцию и изменения контракта выполняет интегратор после обеих задач.
- Минимальный integration test: фиксированные Seed records → ожидаемые paths и запрещённые отсутствующие edges. Реальный NER/E5 и live read-only Neo4j — отдельные согласованные smoke checks.

## 6. План Stage 3A.1–3A.5

1. **3A.1 — контракты и baseline.** Согласовать язык, trust policy и budgets. Подготовить небольшую утверждённую подборку вопросов/ожидаемых сущностей и evidence. Read-only проверить live counts, fingerprint, ID coverage, status distributions и свойства target namespace без `.env` чтения. Создать общий контракт одним интегратором. Критерий: явная версия snapshot и отсутствие неизвестных mappings.
2. **3A.2 — Query Analyzer + Entity Linking (A).** Существующий E3 extractor через адаптер, существующие 811 vectors, NumPy Top-K, confidence/abstention. Сначала synthetic tests; model smoke check только после разрешения. Критерий: source-exact query spans, stable IDs, no forced linking.
3. **3A.3 — bounded traversal (B).** Реализация независимо от A по общему Seed контракту. Сначала fake adjacency, затем read-only live smoke при согласованном доступе. Критерий: строгие бюджеты и namespace, запрет unsafe memberships, разделение fact/semantic/concept.
4. **3A.4 — Evidence + deterministic ranking.** Предлагаемые новые `src/cgraph/evidence_extraction.py`, `src/cgraph/path_reranking.py` и соответствующие focused tests. JSON decoding, exact slices, confidence provenance, diversity Top-5. Критерий: каждая factual citation проверяема; отсутствие fabricated evidence.
5. **3A.5 — интеграция и оценка.** Тонкая orchestration функция без нового framework. Измерить seed Recall@K, factual path/evidence correctness, Top-5 relevance/diversity, no-answer cases и latency. Выдать evidence bundle для будущего LLM aggregation. Запуск самого LLM/reranker и расширение экспериментов требуют отдельного согласования. Критерий: воспроизводимый retrieval baseline при неизменном KG.

## 7. Решения для согласования

1. Разрешить реализацию Stage 3A.1–3A.5 и только затем запуск A/B; сейчас разрешён только аудит.
2. Язык вопросов: английский baseline либо отдельная стратегия для русского. Автоперевод/LLM не подразумеваются.
3. Trust policy SIMILAR_TO: приоритет позднего RELATED (до 80) или консервативное исключение REJECT/REVIEW по обоим этапам (19). REVIEW не включать в trusted paths по умолчанию.
4. Query NER threshold и abstention/linking cutoff определить отдельно от frozen KG thresholds. Предпочтительный старт — извлечение кандидатов 0.35 с последующей проверкой, без перестройки корпуса; это предложение, не измеренный оптимум.
5. Глубина 3 физических hops, ограничение semantic/concept transitions, объём кандидатов и separate related-only output.
6. Начать с deterministic scorer и name embeddings; model reranker/context retrieval добавлять только после оценки baseline.
7. Способ read-only live подключения через уже установленное безопасное окружение и разрешение на ограниченный model smoke test на следующем этапе. Секреты не выводить и не сохранять в отчёт.

До утверждения этих решений реализация, model inference и изменение графа не начинаются.
