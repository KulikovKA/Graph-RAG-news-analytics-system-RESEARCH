# Architecture C — pretrained-only zero-shot pipeline (C-RESTART-001)

Текущий протокол **запрещает обучение и дообучение на наших документах и Gold**.
Рабочий inference: GLiNER-small → типизированные spans → направленные пары →
готовый NLI checkpoint как zero-shot RE → scope/qualifiers → exact evidence →
консервативная каноникализация → provisional graph assertions. Генеративных LLM
и downstream reranker нет. NLI использует свои три готовые головы
contradiction/entailment/neutral; новые relation heads не создаются.

Основной конфиг: [config/c_restart_001.json](config/c_restart_001.json).
Реальный DEV-эксперимент и ограничения:
[C_RESTART_001_REPORT.md](reports/C_RESTART_001_REPORT.md).
Предварительный аудит: [protocol](reports/C_RESTART_001_PROTOCOL.md),
[runtime audit и окончательный протокол](reports/C_RESTART_001_PROTOCOL_ADDENDUM.md).

```console
uv sync --frozen --extra test
uv run --no-sync cgraph fetch-dev
uv run --no-sync cgraph-zero download-models
uv run --no-sync python scripts/probe_nli_checkpoint.py artifacts/nli_checkpoint_probe.json
uv run --no-sync python scripts/assert_zero_shot_inference.py --subset all --output artifacts/c_restart_001_nli_dev
uv run --no-sync cgraph-zero evaluate --subset all --predictions artifacts/c_restart_001_nli_dev --output reports/c_restart_001/dev.json
uv run --no-sync cgraph-zero evaluate --subset validation --predictions artifacts/c_restart_001_nli_dev --output reports/c_restart_001/legacy_validation.json
uv run --no-sync python -m pytest -q
uv run --no-sync python scripts/build_c_restart_001_report.py --predictions artifacts/c_restart_001_nli_dev
```

Запускать из корня репозитория, Python 3.11. Для повторного inference нужен новый
output-каталог. DEV downloader ограничен allowlist и hashes; CALIBRATION/HOLDOUT
не получает. Extraction и scoring выполняются отдельными процессами. Guard
запрещает доступ к Gold, старым обученным RE-параметрам и сети и сохраняет
`access_audit.json`. Структурный PASS не подтверждает правильность отношений.
Gold/weights/cache/полные predictions и per-query traces остаются в исключённых
из Git `data/` и `artifacts/`. Подтверждающей HOLDOUT-оценки нет.

22 отношения имеют только статус **EXPERIMENTAL_ZERO_SHOT**: формулировка
гипотезы не доказывает поддержку научной семантики. `MEETS_REQUIREMENT`
не поддержан. Qualifier completeness остаётся UNKNOWN и не получает strict
qualified credit. Cross-sentence RE/coreference в основном NLI-run не поддержаны.

GLiREL отдельно исследован, но не воспроизвёл опубликованный контрольный пример
даже через официальный loader. Его конфиг сохранён в
`config/c_restart_001_glirel.json`, диагностика описана в отчёте; основная модель
RE — `cross-encoder/nli-deberta-v3-xsmall`, Apache-2.0. GLiREL card указывает
CC BY-NC-SA 4.0 при конфликтующем Apache-2.0 в package metadata.

`pytest` сохраняет и по умолчанию исключает один помеченный
`historical_training` тест, обучающий C-001 на DEV. Он не выполняется в текущем
протоколе; остальные старые контрактные проверки и новые тесты выполняются.

## Исторический C-001: supervised baseline

Ниже сохранена документация первоначального эксперимента. Его `train` и
Gold-entity diagnostic команды и планы fine-tuning относятся только к истории и
**не входят в текущий разрешённый протокол**. Обученные параметры C-001 новый
pipeline не загружает. Исторический отчёт и метрики не переписаны.

Работающий исследовательский CPU pipeline: GLiNER-small → contextual kinds → bounded directed pairs → обученный marked-pair classifier → правила scope/qualifiers → консервативные concepts → проверка exact evidence → JSON. Генеративные модели не используются.

На независимых от обучения **четырёх DEV validation-документах** exact typed entity F1 = **0,4154**, raw relation F1 = **0,0448**, relation F1 после gates = **0**. При Gold entities в отдельной RE-диагностике F1 после gates = **0,1127**. Это исполняемый начальный baseline с низким качеством отношений. Полные результаты и ограничения: [C001_REPORT.md](reports/C001_REPORT.md).

## Структура

```text
config/c001.json              единый конфиг, seed, модели, thresholds, budgets
contracts/                    неизменная ontology v1.0, проекция без Gold-примеров,
                              DEV allowlist/hashes, split, model hashes, JSON Schema
src/cgraph/
  inputs.py, prepare.py        проверка и получение только DEV inputs
  labels.py                   отдельный доступ к train/evaluation labels
  ner.py, candidates.py       GLiNER spans, направленные ограниченные пары
  relations.py                обучаемые sigmoid linear heads, NO_RELATION/ABSTAIN
  rules.py, validation.py      scope, числа, linking, exact evidence, schema
  pipeline.py, cli.py          extraction и отдельные train/evaluation команды
scripts/                      audit inference, формирование отчёта
tests/                        контрактные и обучающие тесты
reports/                      реальные метрики, execution manifests, примеры
data/inputs/, data/labels/     создаются отдельно; исключены из Git
artifacts/                    локальные weights и predictions; исключены из Git
uv.lock                       полные закреплённые зависимости с хешами
```

## Запуск

Нужны Python **3.11**, [uv](https://docs.astral.sh/uv/), CPU и интернет для первоначального получения разрешённых данных и моделей. Команды одинаковы для PowerShell и WSL2 Ubuntu; выполнять из корня нового репозитория. Нативная Windows проверена реальным запуском. WSL2 не запускалась; для Linux в lock выбран CPU-only PyTorch без CUDA-зависимостей.

```console
uv sync --frozen --extra test
uv run --no-sync cgraph fetch-dev
uv run --no-sync cgraph download-models
uv run --no-sync cgraph train
uv run --no-sync python scripts/assert_label_free_inference.py --subset all --output artifacts/dev_e2e
uv run --no-sync cgraph evaluate --subset validation --predictions artifacts/dev_e2e --output reports/validation_e2e.json
uv run --no-sync cgraph evaluate --subset train --predictions artifacts/dev_e2e --output reports/train_descriptive_e2e.json
uv run --no-sync cgraph diagnose-re --subset validation --output artifacts/dev_gold_re
uv run --no-sync cgraph evaluate --subset validation --predictions artifacts/dev_gold_re --output reports/validation_gold_entities_re.json
uv run --no-sync python -m pytest -q
uv run --no-sync python scripts/build_c001_report.py --e2e artifacts/dev_e2e --diagnostic artifacts/dev_gold_re
```

Для обычного extraction без audit wrapper:

```console
uv run --no-sync cgraph extract --subset validation --output artifacts/another_run
```

Каталог output должен быть новым/пустым: существующие runs не перезаписываются. `evaluate` читает только перечисленные документы выбранного subset и отдельно проверяет их IDs, chunk hashes, JSON Schema, exact spans и отсутствие пересечения validation с train модели.

Вместо `fetch-dev` можно импортировать DEV из уже имеющегося исходного проекта:

```console
uv run --no-sync cgraph import-dev --source-root "C:/Users/Kirill/Desktop/Graph-based RAG for article analysis"
```

`fetch-dev` получает только DEV source snapshot, release manifest и 12 явно перечисленных DEV Gold-файлов с pinned upstream commit. CALIBRATION/HOLDOUT texts, annotations и review packets не загружаются. Ни source clone, ни старая инфраструктура для исполнения не нужны.

## Модели и контракт

- NER: [urchade/gliner_small-v2.1](https://huggingface.co/urchade/gliner_small-v2.1), revision `4e091416cf7c3481db542c2a3d26156916f3a47f`, Apache-2.0. CPU float32, strict загрузка весов, nested mentions сохраняются. Измерено 152 648 704 параметра; это отличается от округлённой таблицы model card.
- Tokenizer/backbone config: [microsoft/deberta-v3-small](https://huggingface.co/microsoft/deberta-v3-small), revision `a36c739020e01763fe789b4b85e2df55d6180012`, MIT по model card. Backbone config встраивается локально, обращения к mutable main отсутствуют. Все downloaded file SHA-256 в `contracts/model_files.json`.
- RE: **обучаемые case-sensitive TF-IDF marked-pair linear sigmoid heads**, one-vs-rest logistic regression. Это компактный лексический classifier, без pretrained RE encoder. 23 relation outputs + NO_RELATION; несколько labels на паре допустимы. ABSTAIN — отдельное решение по confidence/conflict/capabilities. Параметры сериализуются в NPZ, features — JSON; pickle не используется. Точная identity обученной модели в `reports/training_metadata.json`. Fine-tuning SciBERT и encoder RE в C-001 не выполнялись.
- `INTEGRATED_IN` и `OPTIMIZES` присутствуют в интерфейсе, но не обучены из-за нулевого support в train; `unsupported_labels` сохраняются на парах. На неопределённых таких парах не выдаётся полный NO_RELATION.

Ontology original сохранена без изменения байтов. Inference читает только отдельную проекцию нормативных definitions/domain/range/qualifier rules без source examples и document-specific boundaries. Gold читает только обучение, диагностика и evaluator; audit wrapper запрещает открывать labels и оригинальную ontology, импорт annotation-модуля и все сетевые соединения во время extraction.

Seed 42; cluster-disjoint train: D004, D010, D017, D024, D026, D038, D060, D078. Validation: D001, D003, D035, D092. Кластеры — доступный консервативный proxy, не независимо подтверждённые семейства публикаций. D024 имеет `exhaustive_relations=false`, поэтому даёт только positive training examples, без assumed negatives.

## Ограничения

Scope и qualifiers поддержаны частично. Числа/единицы извлекаются, но не назначаются соседним отношениям автоматически. Unresolved conditions, сложное отрицание, comparison/distribution, overlapping relation endpoints и required qualifiers дают ABSTAIN. Accepted assertions имеют **ACCEPTED_PROVISIONAL**, `qualifier_status=UNKNOWN` и не получают credit в строгой qualified relation метрике. Lexical predicate screen уменьшает co-occurrence false positives, но не является независимым semantic verifier.

Полноценные contextual roles, coreference, cross-sentence RE, n-ary condition attachment, calibrated NLI, расширенный scientific NER fine-tuning и междокументное identity resolution оставлены для C-002. Пороги не настраивались по validation. Тесты на искусственных минимальных примерах не смешиваются с метриками реального DEV.

При первом скачивании Xet завис в этом окружении; `download-models` использует HTTPS с timeout и проверяет закреплённые хеши. Transformers 5.1.0 выводит generic Mistral-regex warning при загрузке этого локального DeBERTa tokenizer; он зафиксирован в отчёте, tokenizer fix для другой архитектуры не применён.

Weights, cache, labels и полные локальные predictions в Git не добавляются. В отчёт включены только разрешённые DEV-примеры, метрики и provenance. Лицензии внешних моделей не распространяются автоматически на исходный dataset и ontology; их provenance сохранён отдельно.
