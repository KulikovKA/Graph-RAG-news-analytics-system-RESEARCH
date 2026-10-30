# B-001 — Improved Source-Grounded Hybrid Graph Extraction

Реальный end-to-end запуск завершён на 12 DEV, 78 LLM-вызовов. **Текущий B-001 не пригоден для автоматической публикации научного графа**: final precision 25%, recall 4.10%, всего 16 TP. Relation F1 выше C control, но typed entity F1, candidate recall и число найденных правильных связей ниже; wall time выше примерно в 22.8 раза. Это не доказательство общего превосходства архитектуры B.

Exploratory DEV: те же 12 уже исследованных документов, что в финальном C-DIAG-002. Это не независимый финальный тест. D024 имеет неполную relation-разметку: unmatched prediction не обязательно семантически ложно. Gold использован только offline evaluator; CALIBRATION/HOLDOUT не открывались, обучения и fine-tuning нет.

Источник, настройки, диагностические технические перезапуски и ограничения: [протокол](B001_PROTOCOL.md). Модель `qwen3.5:4b-q4_K_M`, digest `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`; Ollama 0.35.0, CPU, seed 42, temperature 0, top-k 1, repeat penalty 1.1/256, context 8192, max output 4096. Исходная ontology v1.0 и её 12 kinds/23 relations неизменны; NLI не используется.

## Качество по общему C evaluator

Точное совпадение Unicode-span и типа, направленное отношение, one-to-one multiset matching; P/R/F1 и TP/FP/FN. Raw по общему evaluator включает предложения с привязанными endpoints и evidence. Все результаты без независимой проверки научного смысла.

| Метрика | B P/R/F1 | B TP/FP/FN | C P/R/F1 | C TP/FP/FN |
|---|---|---|---|---|
| Typed entities | 0.3069/0.2632/0.2834 | 155/350/434 | 0.5126/0.4160/0.4592 | 245/233/344 |
| Raw relations | 0.0874/0.0410/0.0558 | 16/167/374 | 0.0090/0.1282/0.0168 | 50/5508/340 |
| Final relations | 0.2500/0.0410/0.0705 | 16/48/374 | 0.0107/0.1231/0.0196 | 48/4455/342 |
| Qualified relations | 0.0000/0.0000/0.0000 | 0/64/390 | 0.0000/0.0000/0.0000 | 0/4503/390 |

Candidate recall при одинаковом C single-sentence generator: B 50/390 = 0.1282; C 71/390 = 0.1821. Это diagnostic endpoint coverage, а не качество отношений; candidate pairs не ограничивают B extraction.

Raw-all, где неразрешимые relation proposals дополнительно считаются FP: 0.0693/0.0410/0.0515, TP/FP/FN 16/215/374; таких proposals 48. Необработанные из-за усечённого JSON окна отдельно учитываются в coverage; их количество предложений неизвестно.

Явный model ABSTAIN: 36/183 = 0.1967 от структурно разобранных relation records. До mapping: 46/231 явных model ABSTAIN. Pair abstention по диагностике C: B 4006/4068 = 0.9848; C 1425/3942 = 0.3615. Эти знаменатели различаются: отсутствие принятого отношения на паре не обязательно является явным решением LLM.

Qualifier completeness всегда UNKNOWN: без подтверждения полноты строгий qualified scorer не присваивает TP. Числа и условия, не привязанные к конкретной связи, сохраняются как UNKNOWN/UNSUPPORTED. В данном запуске принято **0 qualifiers**, сохранено 16 unresolved records. Полное qualified extraction не реализовано как доказанная capability. Точная цитата подтверждает provenance, но не семантическую поддержку.

## Стадии и потери recall

Raw parsed LLM: 503 entity records и 231 relation records. Exact mapped entities: 505 (включая новые endpoints из relations); структурно адресуемые relations: 183; final accepted: 64. Отклонено 128 entity records на mapping, 48 relation records на mapping и 119 relations на ontology/scope. Raw ответы, структурные решения, accepted, rejected и unresolved хранятся отдельно.

Точные Gold edges, остающиеся доступными: **390 → 72** untyped endpoints → **51** typed endpoints → **50** candidate pairs → **16** raw → **16** final. Последовательные FN: 318 boundary/mapping/failed-window losses + 21 kind losses + 1 candidate/context loss + 34 model losses + 0 final-gate losses = 374. Final validation удалил 119 unmatched-to-Gold proposals и не удалил ни один из 16 raw TP; это accounting по annotation, а не доказательство семантической ложности каждого удалённого proposal. Рост precision не компенсирует низкий upstream recall.

Final FP по первой применимой категории: {'boundary': 42, 'relation_or_unannotated': 4, 'kind': 2}. Ошибки границ включают усечённые химические названия и слишком короткие носители свойств. Одна потеря candidate context — межпредложенческий HAS_PERFORMANCE в D026.

## Ошибки по всем отношениям

Final TP/FP/FN относительно annotations; C — тот же GLiNER control из C-DIAG-002. Нулевой Gold support не доказывает capability.

| Relation | B TP/FP/FN | C TP/FP/FN |
|---|---|---|
| HAS_PROPERTY | 0/8/52 | 7/170/45 |
| HAS_PERFORMANCE | 0/8/52 | 4/64/48 |
| HAS_MORPHOLOGY | 0/1/17 | 1/3/16 |
| HAS_COMPONENT | 0/0/57 | 4/1325/53 |
| HAS_MEMBER | 0/12/44 | 0/2069/44 |
| USES_MATERIAL | 3/1/20 | 5/24/18 |
| FUNCTIONS_AS | 2/1/1 | 0/36/3 |
| DETECTS | 2/10/11 | 3/12/10 |
| USED_FOR | 7/3/40 | 10/191/37 |
| FABRICATED_BY | 1/1/13 | 7/75/7 |
| IMPROVES | 0/1/7 | 1/5/6 |
| ACTS_ON | 0/0/13 | 2/86/11 |
| INTEGRATED_IN | 0/0/1 | 0/11/1 |
| HAS_MECHANISM | 0/0/6 | 0/9/6 |
| MECHANISM_INVOLVES | 0/0/9 | 0/1/9 |
| LOCATED_IN | 0/1/2 | 0/2/2 |
| OPTIMIZES | 0/0/1 | 0/14/1 |
| FACILITATES | 1/0/2 | 0/97/3 |
| CAUSES | 0/1/3 | 0/48/3 |
| EXPLAINS | 0/0/11 | 4/103/7 |
| CHANGES_PARAMETER | 0/0/8 | 0/0/8 |
| DEPENDS_ON | 0/0/2 | 0/110/2 |
| MEETS_REQUIREMENT | 0/0/2 | 0/0/2 |

## Стоимость и покрытие

B: wall 8187.7 s; CPU 32700.4 s; peak sampled aggregate RSS 4.776 GiB; LLM calls 78, input tokens 212272, output tokens 57059, retries 0, JSON/truncation errors 1, request errors 0. Успешные окна 77/78; failed 1.

C control: wall 358.6 s; CPU 1322.2 s; peak process RSS 1.684 GiB; LLM calls 0. B включает Python client, Ollama server и его потомков, суммируя RSS с шагом 0.25 s; C — peak одного процесса. Общий сервер и совместно отображённые страницы ограничивают прямую сравнимость RAM и CPU. Первичные измерения включают локальную загрузку модели, но исключают предварительные пробы, скачивание weights, тесты и evaluator; они не являются полным временем исследования.

Оборудование: AMD Ryzen 7 8845HS, 8 физических/16 логических CPU, 32 GB RAM; Python 3.11.9, четыре inference threads, GPU offload 0. Основной inference выполнен 30 октября 2026, около 12:30–14:46 Europe/Moscow. Input tokens включают кешируемый системный prompt; это не число заново вычисленных prompt tokens. Одно failed окно D078 (sentence 4 первого chunk) достигло 4096 output tokens, его partial JSON не принят, все Gold relations оставлены в знаменателе. PowerShell-обёртка сообщила exit 1 при progress в native stderr; traceback в логе отсутствует, все 12 файлов и финальный manifest записаны и затем успешно проверены offline evaluator. Чистый exit оболочки не заявляется как свидетельство завершения.

Модель использовалась локально; внешней API-тарификации не было. Энергопотребление и денежная стоимость локальной машины не измерялись. Peak RSS — sampled aggregate, не точный общий объём уникальных физических страниц.

Причины rejection: `{'ONTOLOGY_DOMAIN_RANGE': 67, 'ONTOLOGY_DOMAIN_RANGE;MODEL_ABSTAIN;UNKNOWN_OR_UNSUPPORTED_POLARITY_MODALITY': 17, 'NEGATION_NOT_IN_EVIDENCE': 2, 'SURFACE_NOT_AT_OCCURRENCE': 95, 'ONTOLOGY_DOMAIN_RANGE;NEGATION_NOT_IN_EVIDENCE': 6, 'INVALID_OCCURRENCE_ANCHOR': 66, 'MODEL_ABSTAIN;UNKNOWN_OR_UNSUPPORTED_POLARITY_MODALITY': 14, 'ONTOLOGY_DOMAIN_RANGE;MODEL_ABSTAIN;NEGATION_NOT_IN_EVIDENCE': 3, 'INVALID_EVIDENCE_SENTENCE': 15, 'REQUIRED_QUALIFIER_UNSUPPORTED': 1, 'ONTOLOGY_DOMAIN_RANGE;NEGATION_NOT_IN_EVIDENCE;REQUIRED_QUALIFIER_UNSUPPORTED': 1, 'ONTOLOGY_DOMAIN_RANGE;MODEL_ABSTAIN': 2, 'ONTOLOGY_DOMAIN_RANGE;REQUIRED_QUALIFIER_UNSUPPORTED': 2, 'ONTOLOGY_DOMAIN_RANGE;UNKNOWN_OR_UNSUPPORTED_POLARITY_MODALITY': 4}`.

## Примеры

Ниже exact-сопоставление с owner-reviewed DEV annotations; несовпадение с Gold не является независимым доказательством семантической ошибки.

**D001: `gas sensor` → USES_MATERIAL → `cotton/elastic threads`**

Корректная базовая связь: текст прямо описывает материал устройства; exact typed directed Gold TP. Полнота qualifiers и modality этим совпадением не подтверждены.

> The proposed gas sensor is built using commercially available cotton/elastic threads as flexible/stretchable templates and reduced graphene oxide/mesoporous zinc oxide nanosheets as sensing layers to form conducting fibers.

**D026: `Ti3C2Tx MXene gas sensors` → DETECTS → `VOC gases`**

Корректная базовая detection-связь и Gold TP, но указанные в тексте 50–100 ppb и room temperature не сохранены как accepted qualifiers. Это пример неполного qualified assertion.

> Ti3C2Tx MXene gas sensors exhibited a very low limit of detection of 50–100 ppb for VOC gases at room temperature.

**D010: `Ti` → HAS_MEMBER → `C`**

Семантически неверное утверждение при точной цитате: Ti и C вырваны из химической записи, не заданы как analyte collection и member. Ошибка chemical span + contextual kind + predicate.

> The Ti 3 C 2 Tx MXene/GO/CuO/ZnO nanocomposite gas sensor exhibited the best NH 3 gas sensor.

**D010: `GO` → HAS_MEMBER → `CuO`**

Семантически неподдержанная group-связь: формула перечисляет компоненты nanocomposite, но не утверждает, что GO является коллекцией с member CuO. Domain/range и exact evidence сами этого не обнаружили.

> The Ti 3 C 2 Tx MXene/GO/CuO/ZnO nanocomposite gas sensor exhibited the best NH 3 gas sensor.

**D092: `CO` → HAS_MEMBER → `gas molecules`**

Неверное направление: source перечисляет CO среди gas molecules; модель направила HAS_MEMBER от отдельного газа к группе. Типы формально допустимы, смысл — нет.

> The interactions between four different graphenes (including pristine, B- or N-doped and defective graphenes) and small gas molecules (CO, NO, NO(2) and NH(3)) were investigated by using density functional computations to exploit their potential applications as gas sensors.

**Реальные FN**

- kind: D001 `E-textiles` (DEVICE) → HAS_PROPERTY → `low cost` (PROPERTY).
- boundary: D001 `E-textiles` (DEVICE) → HAS_PROPERTY → `conformable compatibility with clothes` (PROPERTY).
- model: D001 `fiber gas sensor` (DEVICE) → HAS_PROPERTY → `conductive` (PROPERTY).
- candidate: D026 `Ti3C2Tx MXene gas sensors` (DEVICE) → HAS_PERFORMANCE → `signal-to-noise ratio` (PERFORMANCE).

Например, `E-textiles` → HAS_PROPERTY → `low cost` потерян из-за kind, а `fiber gas sensor` → HAS_PROPERTY → `conductive` — на model stage при доступных typed endpoints. Эти примеры выбраны после scoring для объяснения ошибок и не использованы для изменения prompt или правил.

## Ограничения и проверка

Один владелец проверял AI-origin разметку; независимой второй разметки нет. Окна ограничены одним предложением, поэтому межпредложенческие связи и coreference не оцениваются как поддержанные. Repeat penalty выбран до Gold по техническому повторению на одном DEV-окне, и DEV уже исследован; статистическая независимость отсутствует. Потери boundary/type и unsupported qualifiers сохраняются в итоговых метриках. Никакие результаты B не подменяются модельными или синтетическими примерами.

`python -m pytest -q`: **46 passed, 1 deselected**, 3.75 s. Исключён сохранённый historical_training test; обучение не запускалось. Проверены repeated/nested Unicode spans, chemical NO и отрицание, направление/ontology, unsupported qualifiers, новые exact endpoints, Gold/network access guard, model digest mismatch и явное сохранение усечённой генерации как failure.

Нормативные contracts, весь `src/cgraph/`, main и C-DIAG-002 сохранены без изменений. CLI: `python -m bgraph.cli extract --output <new-directory>` и отдельный `python -m bgraph.cli evaluate --predictions <directory> --output <metrics.json>`. Аудит hashes проверил исходные predictions и raw traces; source-only access guard запрещал Gold/closed paths и разрешал только loopback Ollama. Синтетические тесты не входят в DEV metrics. Файл metrics содержит старую общую формулировку про wider windows; фактический primary config — одно предложение, что уточнено в error_analysis.json и данном отчёте.

Primary decoding/config закреплены до Gold в `144d63e`, exact-surface endpoint schema/rules — в `5a90136`; base worktree — `b368a63`, ветка `b-001-source-grounded-hybrid`. После scoring менялись только отчёты, offline анализ и дополнительные runtime-тесты; frozen extraction, prompt и predictions сохранены.

Воспроизведение и границы доступа приведены в [протоколе](B001_PROTOCOL.md); полные агрегаты и per-relation counts — [metrics.json](b001/metrics.json), фиксированный C baseline — [c_diag_002_control_baseline.json](b001/c_diag_002_control_baseline.json). Run manifest с hashes — [run.json](b001/run.json); разложение потерь и примеры — [error_analysis.json](b001/error_analysis.json). Локальные raw traces, predictions и источники не коммитятся.

Документация runtime: [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-4B), [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs), [Ollama generate API](https://docs.ollama.com/api/generate). Реальные параметры и blob hashes из локального запуска имеют приоритет над advertised maximum context.
