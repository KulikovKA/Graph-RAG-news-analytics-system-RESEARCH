# C-RESTART-001 — pretrained-only архитектура C

Дата: 2026-10-29, Europe/Moscow. Реальный CPU inference завершён на всех 12 DEV.
Обучение, fine-tuning, fit, использование Gold в inference, генеративные LLM и
downstream reranker отсутствуют. **Исполнимость и exact evidence не означают
правильность графа**. Качество ниже пригодного для автоматической публикации фактов.

## Что реализовано

Независимые компоненты: scientific text → готовый GLiNER-small → 12 типов spans →
направленные candidate pairs → pretrained entailment как zero-shot RE → partial
scope/qualifiers → проверка evidence → conservative canonicalization → assertions.
Новый CLI `cgraph-zero` не содержит команды обучения. Исторические C-001 код,
конфиг и отчёты сохранены. Рабочее дерево C-002 осталось в исходном каталоге;
новая реализация находится в отдельном worktree/ветке.

Повторно использованы DEV allowlist/loader, frozen ontology v1.0, source hashes,
нормативная проекция без Gold examples, JSON Schema, exact multiset evaluator,
GLiNER adapter, evidence и conservative linking. Ontology не менялась.

## Модели и выбор RE

NER: `urchade/gliner_small-v2.1`, revision
`4e091416cf7c3481db542c2a3d26156916f3a47f`, Apache-2.0,
152,648,704 параметров. Сохранены 12 исходных label
descriptions, threshold 0.35, окна 140 words/overlap 30. Никаких Gold-based
boundary repairs или нового supervised entity typing.

RE: `cross-encoder/nli-deberta-v3-xsmall`, revision
`a150876415327c80daeff35ca6f68f5ed8cf5c24`, Apache-2.0,
70,831,107 параметров. Это готовая модель NLI с тремя
реальными классами contradiction/entailment/neutral, обученная её авторами на
SNLI/MultiNLI. На наших текстах и метках ничего не обучалось. Новые 23 heads
не создавались. Применение entailment к формулировкам отношений следует
[Sainz et al., EMNLP 2021](https://aclanthology.org/2021.emnlp-main.92/).
Результаты той статьи не являются оценкой нашей научной онтологии.

Выбор обусловлен отдельным RE-интерфейсом, доступным полным checkpoint,
CPU-стоимостью и лицензией. Это компонент RE, а не downstream reranker.
Данные модели: [model card](https://huggingface.co/cross-encoder/nli-deberta-v3-xsmall).
Контроль из model card воспроизведён: entailment 0.9397 и contradiction 0.9996;
он исключён из DEV метрик и времени основного запуска.

Предварительно исследован GLiREL-large-v0 (466,577,920 параметров). Он не
воспроизвёл опубликованный пример: около 0.001–0.002 для `characters` вместо
около 0.99. SHA256 и strict load успешны; официальный loader даёт тот же результат.
Проверены старые/новые библиотеки, fast/slow tokenizer и обычная загрузка весов.
Причина **не установлена**. Первые 12-DEV predictions с нулём raw сохранены
локально; они исключены из валидированного сравнения качества, а не скрыты.
Диагностика: [glirel_diagnostic.json](c_restart_001/glirel_diagnostic.json).
GLiREL card/repository указывают CC BY-NC-SA 4.0 при Apache-2.0 в package metadata.
Joint GLiNER-Relex относится к F и не использовался.

Отдельно устранена несовместимость локальной DeBERTa tokenization в
Transformers 5.1.0: служебные CLS/SEP отсутствовали. Финальный стек закрепляет
Transformers 4.51.3/tokenizers 0.21.4 и проверяет их наличие. Это также меняет
NER runtime относительно C-001; сравнение не изолирует только замену RE.
Исходные pretrained веса сохранены без изменений.

## Зафиксированная политика и capabilities

До оценки записаны [аудит](C_RESTART_001_PROTOCOL.md) и
[окончательный протокол](C_RESTART_001_PROTOCOL_ADDENDUM.md).
Конфиг SHA256: `80be728c4614bdb92c60670d949f38a64fc6903d0f96429e0bea38c8af6ea3ff`.
Порог NLI entailment >0.5, batch 32, single-sentence context, без поиска thresholds
и verbalizations по Gold. Negative hypothesis дополнительно проверяется только
при linguistic negation в source. Полные три scores каждой проверки сохранены
локально в `re_inference_trace`. Ни один score не объявляется калиброванной
вероятностью правильности научного факта.

Все 23 relation capabilities явно перечислены в конфиге. 22 имеют статус
EXPERIMENTAL_ZERO_SHOT; это возможность выполнить проверку гипотезы, а не
доказательство поддержки научной семантики. `MEETS_REQUIREMENT` UNSUPPORTED:
совместный requirement bundle и collective scope требуют n-ary attachment.
`CHANGES_PARAMETER` требует явный direction cue; неизменность параметра и
сложные causal/explanatory claims поддерживаются неполно. Per-type фактические
TP/FP/FN находятся в `loss_audit.per_relation` отчётов JSON, включая нулевые TP.

Кандидаты перечисляются до domain/range pruning в обоих направлениях, без
прежнего cap 400/chunk и gap 300 characters. NLI оценивает только совместимые
с онтологией hypotheses. Cross-sentence/coreference в основном run не поддержаны.
Неоднозначные типы одного span сохраняются и могут увеличивать FP.
RE score относится к текстовым spans, поэтому raw proposals проецируются на
альтернативные NER kinds тех же spans до final domain/range gate; 889 таких
proposals нарушили domain/range и были удалены. Число raw proposals отличается
от числа NLI queries: порог прошли 4701
queries, затем альтернативные polarity/type записи объединяются/проецируются.
Все 5558 raw scores проверены по реальным локальным NLI traces.
Это также ограничивает прямую сопоставимость raw output с C-001, который заранее
ограничивал classifier labels по domain/range; final matching policy едина.

Удалён обязательный узкий predicate cue между endpoints. Uppercase NO/NOx/NO₂
отделены от English negation; simple polarity/modality и scope проверяются
детерминированно. Неоднозначное отрицание/cross-clause scope дают ABSTAIN.
Числа/units сохраняются с exact evidence; произвольного присоединения к соседним
relations нет. Реализованы явные trailing temperature и physical direction
qualifiers, но completeness остаётся UNKNOWN: qualified TP за них не начисляется.
Опциональные unresolved qualifiers не удаляют все binary assertions.
На реальном DEV присоединённых qualifiers оказалось ноль; attachment подтверждён
только минимальными тестами, а не успешными qualified predictions этого run.

Каждый final assertion имеет exact source hash/quote/codepoint offsets и
ACCEPTED_PROVISIONAL. Evidence verification подтверждает происхождение текста,
**не entailment и не научную истинность**. Каноникализация сохраняет chemical
case/composition и не объединяет device instances по одному имени; глобальная
entity resolution не заявляется.

## Фактические метрики: все 12 DEV

| Метрика | TP / FP / FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Entity span | 277 / 201 / 312 | 0.5795 | 0.4703 | 0.5192 |
| Entity typed | 245 / 233 / 344 | 0.5126 | 0.4160 | 0.4592 |
| Raw directed relation | 50 / 5508 / 340 | 0.0090 | 0.1282 | 0.0168 |
| Final directed relation | 48 / 4455 / 342 | 0.0107 | 0.1231 | 0.0196 |
| Qualified relation | 0 / 4503 / 390 | 0.0000 | 0.0000 | 0.0000 |

Candidate recall: **71/390 = 0.1821**.
Abstention по candidate pairs: **1425/3942 = 0.3615**.
478 entities, 3942 candidates,
6841 реальных NLI queries, 5558 raw proposals,
4503 final assertions. Извлечено 16
числовых spans; в final assertions сохранено 0 partial qualifiers.
Контексты/queries сверх лимита: 0.

Exact Unicode spans+kind, directed predicate, one-to-one multiset matching;
relaxed spans/coreference credit отсутствуют. UNKNOWN qualifier completeness
никогда не получает qualified TP. При пустом prediction denominator precision
обозначается null/«—», а не искусственным нулём/единицей.

Gold используется исключительно offline evaluator после сохранения predictions.
D024 имеет incomplete relation annotation: precision всех DEV ограничена её
полнотой. DEV уже исследовался в C-001 и не является untouched confirmatory set.
CALIBRATION/HOLDOUT не открывались. Один owner review, AI-origin annotations,
независимого второго annotator нет. Human semantic-support rate не измерен.

## Где теряются правильные утверждения

| Стадия | Правильных edges осталось | Потеря на стадии |
|---|---:|---:|
| gold | 390 | 0 |
| untyped_endpoints | 86 | 304 |
| typed_endpoints | 76 | 10 |
| candidate_pairs | 71 | 5 |
| model_served | 71 | 0 |
| re_capability | 71 | 0 |
| raw_re | 50 | 21 |
| domain_range | 50 | 0 |
| scope | 48 | 2 |
| evidence | 48 | 0 |
| final | 48 | 0 |

Это exact typed directed Gold edges, не автоматически подтверждённые научные
факты. Candidate recall ограничен уже наличием точных NER endpoints. Потери
raw→scope и причины отбрасывания отдельно записаны в `loss_audit`; там же
per-document funnel и per-relation counts. Не смешиваются pair abstention и
число rejected proposals. Canonicalization не меняет scoring keys, поэтому
может дать нулевую потерю на final даже без проверки глобальной identity.

## Сравнение с историческим C-001

Одни и те же четыре legacy validation DEV: D001, D003, D035, D092.

| Вариант | Typed entity F1 | Candidate coverage | Raw relation F1 | Final relation F1 | Qualified F1 |
|---|---:|---:|---:|---:|---:|
| Исторический C-001 | 0.4154 | 7/132 | 0.0448 | 0.0000 | 0.0000 |
| Pretrained-only C | 0.4154 | 7/132 | 0.0141 | 0.0190 | 0.0000 |

Полная таблица нового варианта на этих четырёх документах:

| Метрика | TP / FP / FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Entity span | 79 / 51 / 128 | 0.6077 | 0.3816 | 0.4688 |
| Entity typed | 70 / 60 / 137 | 0.5385 | 0.3382 | 0.4154 |
| Raw directed relation | 7 / 851 / 125 | 0.0082 | 0.0530 | 0.0141 |
| Final directed relation | 7 / 596 / 125 | 0.0116 | 0.0530 | 0.0190 |
| Qualified relation | 0 / 603 / 132 | 0.0000 | 0.0000 | 0.0000 |

C-001 обучался на восьми остальных DEV, новый C не обучался ни на одном.
Сравнение всех 12 для C-001 включает training documents и потому не является
независимым результатом: historical final F1 0.0229,
новый final F1 0.0196.
Все aggregate counts доступны в [comparison.json](c_restart_001/comparison.json).
Сравнение descriptive: маленький DEV, разные RE и runtime, без доверительных
интервалов, настройки на Gold и утверждения превосходства по общей архитектуре.
На legacy validation candidate recall остался 7/132: снятие pair caps не исправило
отсутствие точных NER endpoints. Новый scope сохраняет все 7 raw TP, но оставляет
596 FP. Raw F1 снизился с 0.0448 до 0.0141; рост final F1 с нуля до 0.0190
не означает пригодность графа. На всех DEV final precision составляет всего
1.07% (48 TP, 4455 FP). Неразборчивость NLI здесь столь же существенна, как NER.

## Измеренная стоимость

Windows CPU, Python 3.11.9, 16 logical CPUs,
4 PyTorch threads, float32. Один fresh process; включает
imports, загрузку/hash verification моделей, inference и сериализацию документов;
исключает downloads, probes, невалидированные попытки и evaluator.

| Показатель | Измерено |
|---|---:|
| Wall time | 389.959 s |
| CPU process time | 1443.359 s |
| Peak working set | 1.683 GiB (1807134720 bytes) |
| NER API / encoder calls | 27 / 27 |
| RE batch API / encoder calls | 223 / 223 |
| Model API calls всего | 250 |
| LLM calls | 0 |
| Максимум NER / RE subtokens | 295 / 128 |

Peak RAM: Windows GetProcessMemoryInfo PeakWorkingSetSize; process lifetime including imports/load. RE scoring/queries и model API calls имеют
разные denominators: один batch обрабатывает до 32 premise/hypothesis pairs.
Одна measurement на общей машине; это не контролируемый speed benchmark.
Исторический C-001: 47.524 s wall, 90.266 s CPU, 1.624 GiB peak, 46 API calls
на 12 DEV; его обученный lexical RE существенно дешевле pretrained entailment.
Per-document/stage timing — в JSON. Промежуточные контрольные прогоны не
суммируются с основным временем и отдельно раскрыты в диагностике.

## Проверки, воспроизведение и вывод

`uv run --no-sync python -m pytest -q`: **37 passed, 1 deselected**.
Исключён только сохранённый C-001 supervised fitting test; он не запускался.
Проверены source/ontology hashes, DEV allowlist, Unicode/nested spans,
направленность/неограниченные пары, chemical vs linguistic negation, scope,
required direction, partial numeric evidence, raw/final domain filtering,
strict qualified penalty, loss funnel и отсутствие training imports/calls в
новых inference modules. Искусственные fixtures тестируют код и не входят в
экспериментальные метрики. Реальная интеграция — отдельные NLI control и 12 DEV.

Guard завершил реальный inference с запретом Gold/closed splits/network и
старых fitted RE. [access_audit.json](c_restart_001/access_audit.json) связан
hash с [run.json](c_restart_001/run.json). Source/config/lock/model/prediction
hashes сохранены; run выполнен до финального commit (`code_dirty=true`), точные
inference modules проверены по hashes. Полный lock закреплён. В Git публикуются
только код, contracts, aggregate metrics и provenance; полные predictions,
per-query traces, Gold labels и все веса остаются локально.

Воспроизведение — команды в [README](../README.md); для нового run нужен новый
output directory. Качество: [12 DEV](c_restart_001/dev.json),
[legacy validation](c_restart_001/legacy_validation.json),
[counts](c_restart_001/counts.json).

**Что работает:** реальные отдельные pretrained NER/RE, запрет обучения и
leakage, воспроизводимые predictions, exact provenance, детальная диагностика
потерь. **Что не доказано:** корректность scientific graph, полнота qualifiers,
каузальная/механистическая семантика, глобальная identity и downstream retrieval.
Низкий relation precision/recall нельзя скрыть structural PASS.

Продолжать C имеет смысл как ограниченное исследование bottlenecks, прежде всего
готового scientific NER/typing и семантики verbalizations, с заранее
зафиксированными Gold-free решениями. Текущий вариант не готов к автономному
построению надёжного научного графа. Ввиду стоимости NLI и потерь endpoints
не стоит масштабировать его без улучшения этих компонентов. Обучение на Gold,
reranker и joint F остаются вне текущего эксперимента.
