# ENTITY-TYPING-V2-001 — локальный стенд Entity Typing

Стенд сравнивает T1 GLiNER-only, T2.1 Qwen без thinking, T2.2 Qwen с thinking и T3 hybrid на одних замороженных mentions. Подготовка и разработка не требуют генерации Qwen. Запуск `qwen` без `--dry-run` и launcher без `-DryRun` запускают настоящий эксперимент; это делает пользователь.

## Установка на Windows

Из корня репозитория можно использовать существующее окружение:

```powershell
$py = '.\.venv-ner-v2\Scripts\python.exe'
& $py -m pytest tests/test_entity_typing_v2.py -q
```

Для отдельного лёгкого окружения (Python 3.11, без GLiNER/torch/transformers):

```powershell
py -3.11 -m venv .venv-entity-typing
$py = '.\.venv-entity-typing\Scripts\python.exe'
& $py -m pip install -r config/entity_typing_v2_requirements.txt
```

Ollama должен быть запущен локально. Модель никогда не скачивается скриптом. Если нужного checkpoint нет, установите его самостоятельно:

```powershell
ollama pull qwen3.5:4b-q4_K_M
```

Discovery читает `/api/version`, `/api/tags`, `/api/show`, `/api/ps`, выбирает только теги Qwen3.5-4B, предпочитает Q4_K_M и проверяет архитектуру/число параметров. Он не подменяет модель на Qwen3, Tev1 или другую размерность. Для явного выбора используйте `--model qwen3.5:4b-q4_K_M`. Digest, quantization, native context и версия Ollama сохраняются; обновление тега между ON/OFF не допускается.

При подготовке 05.11.2026 обнаружены Ollama 0.35.0 и `qwen3.5:4b-q4_K_M`, Q4_K_M, digest `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`, native context 262144. Это наблюдение API метаданных, не результат inference. Фактическое состояние следующего запуска проверяется заново.

Для одного server slot задайте переменную **серверу Ollama**. Завершите Ollama через значок в трее, затем в отдельном PowerShell:

```powershell
$env:OLLAMA_NUM_PARALLEL = '1'
ollama serve
```

Клиент всегда отправляет один запрос за раз. OS lock предотвращает одновременный запуск режимов этим стендом даже для разных каталогов/datasets в одном checkout. Если доступный локальный процесс Ollama содержит `OLLAMA_NUM_PARALLEL` со значением, отличным от `1`, запуск отклоняется. Отсутствующая/недоступная переменная явно отмечается как непроверенная: клиентская настройка не может изменить уже запущенный сервер. Во время эксперимента не посылайте сторонние запросы в тот же Ollama.

## Подготовка и dry-run без загрузки модели

```powershell
& $py scripts/run_entity_typing_v2.py prepare --dataset pilot
& $py scripts/run_entity_typing_v2.py qwen --dataset pilot --dry-run
```

Первый шаг читает только `reports/ner_v2/research_run2/input_abstracts.jsonl`, `E3/mentions_035.jsonl`, `E3/config.json` и Ontology v2. Он проверяет исходные SHA-256, provenance каждого hint, N3, точные Unicode codepoint offsets и score ≥0.35. Имеются 30 документов, 1327 hints и 1211 уникальных spans. Несколько hints одного `(document_id,start,end)` объединяются, для каждого типа берётся максимальный score. Нормализованная N3 form переиспользуется; NLP и модели не загружаются. Нижепороговые hints не добавляются: отсутствие top-2 означает `null` margin, а не нулевую альтернативу.

Pilot содержит 240 candidates с seed 4401. Страты — discipline × top-1 type; минимум один кандидат каждой непустой страты, оставшиеся места распределяются пропорционально её ёмкости методом наибольших остатков. Внутри страт выборка циклически охватывает score bands и неоднозначность (`margin<0.15`), с seeded shuffle. Это диагностическая выборка, а не оценка распространённости типов в корпусе без поправки на sampling. Предварительные assisted labels вообще не читаются.

Стабильные непрозрачные candidate IDs зависят от документа, редакции, хеша текста и offsets. `manifest.json`, `candidate_dataset.jsonl` и копия `ontology.json` замораживаются до inference. Повторный prepare идемпотентен; изменение входов требует нового `--output`.

Dry-run создаёт `dry_run.json` и `dry_run_batches.jsonl`, проверяет доступность модели, настройки, пути, RAM и prompt sizes. Он **не вызывает `/api/chat` или `/api/generate`**, не загружает и не выгружает модель. Код возврата: 0 — модель доступна и план построен, 2 — метаданные/ресурсы недоступны, 1 — ошибка конфигурации или данных. Недоступность модели не выдаётся за успешную проверку. В `model_settings.json` предложенные настройки отмечены `DRY_RUN_PROPOSED`.

## Последовательный запуск четырёх тестов

```powershell
& $py scripts/run_entity_typing_v2.py gliner --dataset pilot
& $py scripts/run_entity_typing_v2.py qwen --mode no-thinking --dataset pilot --resume
& $py scripts/run_entity_typing_v2.py qwen --mode thinking --dataset pilot --resume
& $py scripts/run_entity_typing_v2.py hybrid --dataset pilot
& $py scripts/run_entity_typing_v2.py summarize --dataset pilot
```

Либо launcher, который сначала выполняет prepare и dry-run, затем последовательно T1 → T2.1 → T2.2 → T3 → summary:

```powershell
.\scripts\run_entity_typing_v2.ps1 -Python $py -Dataset pilot -NumThread 4
```

Только безопасная подготовка и dry-run через launcher:

```powershell
.\scripts\run_entity_typing_v2.ps1 -Python $py -Dataset pilot -DryRun
```

Для всех 1211 mentions замените `--dataset pilot` на `--dataset full` во всех командах или используйте `-Dataset full`. Pilot сохраняется в `reports/entity_typing_v2/run1/`, full — в `reports/entity_typing_v2/run1/full/`; manifests не смешиваются. `--output reports/entity_typing_v2/run2` выбирает новый корень (для full также добавится `full/`). Если ExecutionPolicy запрещает запуск `.ps1`, используйте Python-команды выше.

## Максимальная ёмкость контекста

`max_model_context=262144`. Native context читается из `model_info`. Кандидаты `num_ctx` проверяются в порядке 262144 → 131072 → 65536 → 32768 → 16384 → 8192, без RoPE/YaRN. Ограничение runtime можно дополнительно указать `--runtime-context-limit`.

RAM определяется через psutil. Оценка включает полный размер модели, f16 K/V cache из per-layer head counts и key/value dimensions, множитель KV 1.2 и резерв runtime/SSM/scratch 2 GiB. Используется максимум 85% доступной RAM с учётом памяти уже загруженного **того же digest**, которую можно переиспользовать. При неполных метаданных оценка консервативна. GPU по умолчанию отключён (`num_gpu=0`), модель сравнивается на CPU. Неизвестные детали allocator/SSM/workspace не позволяют по одному `/api/show` доказать абсолютный физический максимум; этот расчёт задаёт безопасный оценочный потолок, а не измерение throughput.

Только при настоящей команде `qwen` выполняется load-only `/api/generate` **без prompt**, `num_predict=0`, с выбранным `num_ctx`. Это выделение модели/контекста, не тестовая генерация. `/api/ps` обязан подтвердить тот же digest и запрошенный context length. При ошибке памяти/runtime скрипт явно печатает причину и пробует следующий безопасный вариант. При иных ошибках он останавливается. Выбранное значение выводится до typing и закрепляется в manifest. Для следующего режима/resume оно не уменьшается: при недостатке RAM нужно освободить память или начать отдельный эксперимент. Полностью завершённый resume не загружает модель заново.

```powershell
& $py scripts/run_entity_typing_v2.py qwen --mode no-thinking --resume --strict-max-context
```

`--strict-max-context` запрещает запуск, если **полное native окно** недоступно по оценке RAM/runtime или не подтверждено allocator. Dry-run проверяет только оценочную часть и явно сохраняет `runtime_verified=false`.

Ёмкость не заполняется искусственно. Batch содержит полный исходный abstract и до 8 mentions этого документа. Для каждого запроса проверяются верхняя оценка input tokens + output budget. Оценка — UTF-8 byte bound сериализованных messages и schema + 1024 токена резерва renderer; она намеренно завышена. Старый tokenizer Qwen3-4B из NER-V2-003 не используется как tokenizer Qwen3.5. Реальные `prompt_eval_count`/`eval_count` сохраняются отдельно. Abstract не обрезается: если даже один mention с контекстом не помещается, preflight останавливается до inference. Очень малый `num_ctx` может быть отклонён этой консервативной проверкой.

## Настройки и идентичность ON/OFF

Общие настройки находятся в `config/entity_typing_v2.json`: temperature 0, seed 42, top_p 1, top_k 0, min_p 0, repeat_penalty 1, presence/frequency_penalty 0, CPU threads 4, num_parallel 1, stream false, keep_alive 30m. Penalties заданы явно, чтобы defaults checkpoint не меняли сравнение. Оба режима используют `/api/chat` с реальным boolean `think=false`/`true` и JSON Schema в `format`. Prompt и исходные batches идентичны, включая все 11 нормативных русских определений и английские descriptions онтологии. В них нет NER score/type, normalized form или оценочных меток.

По умолчанию оба output budget равны 4096; ON резервирует 2048 для thinking при планировании общего batch. Настройки можно менять **до первого inference**, например:

```powershell
& $py scripts/run_entity_typing_v2.py qwen --dry-run --num-thread 4 --batch-size 6 --num-predict 4096 --num-predict-thinking 8192
& $py scripts/run_entity_typing_v2.py qwen --mode no-thinking --resume --num-thread 4 --batch-size 6 --num-predict 4096 --num-predict-thinking 8192
& $py scripts/run_entity_typing_v2.py qwen --mode thinking --resume --num-thread 4 --batch-size 6 --num-predict 4096 --num-predict-thinking 8192
```

Те же overrides нужны в обоих режимах. Удобнее заранее изменить JSON. После первого запуска замораживаются digest, версия runtime, context, sampling, budgets, batching, prompt version и hash кода стенда. Изменение отклоняется, чтобы результаты двух конфигураций не смешивались. Из-за разбиения ошибочных batches retry prompts могут различаться между режимами; каждый prompt hash и точный payload сохраняются для аудита. Основной план остаётся единым.

Ollama валидирует options при загрузке/запросе; `/api/ps` подтверждает фактический context. Public API не возвращает effective sampling settings или effective num_thread — скрипт сохраняет точный request и явно обозначает эту границу проверки. Sampling temperature 0 повышает воспроизводимость, но не гарантирует битовую идентичность между версиями runtime/оборудованием. Описание API: [chat](https://docs.ollama.com/api/chat), [running models](https://docs.ollama.com/api/ps), [runtime settings](https://docs.ollama.com/faq), [model](https://ollama.com/library/qwen3.5:4b).

## Остановка, recovery и resume

Ctrl+C завершает процесс с кодом 130. Повторите соответствующую команду с `--resume` либо весь launcher. Каждый полученный raw response сначала записывается атомарно с fsync в `attempts/<mode>/<attempt_id>.json`; затем сохраняются parsed records и полный JSONL snapshot. Если процесс завершился между этими шагами, resume восстанавливает корректные решения из durable journal. Успешные ID больше не отправляются модели. Если прервать HTTP-вызов **до получения ответа**, результат такого незавершённого вызова недоступен и его придётся повторить.

JSON Schema проверяет поля/типы/диапазоны, parser дополнительно проверяет IDs, дубликаты и согласованность status/type. Для повреждённого JSON извлекаются только полностью сформированные объекты: тип, ID, confidence и justification никогда не придумываются. Сохраняются корректные записи, остаток помечается NOT_PROCESSED и делится на меньшие batches. По умолчанию максимум 2 дополнительных попытки на candidate за один запуск; повторный явный `--resume` даёт незавершённым ID новый ограниченный retry budget. Timeout 900 секунд на запрос, на метаданные 10 секунд. Ошибки с attempt ID и исходным контекстом попадают в `failure_log.jsonl`; raw content и thinking сохраняются раздельно внутри Ollama message. Если runtime не даёт отдельного thinking token counter, сохраняется null с пояснением; разделение не оценивается выдуманным числом.

tqdm показывает candidates, batches, elapsed/ETA, текущий batch, errors, накопленные input/output tokens и candidates/min; после batch печатается краткая строка validation/timing. Длинное thinking в терминал не выводится. Batch ETA меняется при добавлении retry batches.

## T3 и метрики

T3 читает только замороженный набор и сохранённые решения T1/T2.1/T2.2; сетевого клиента у него нет. Detection остаётся 0.35. Для routing thresholds 0.50/0.55/0.60/0.65 score ≥ threshold принимает GLiNER; ниже берётся сохранённое решение Qwen. Для каждого режима дополнительно считается risk-aware policy: при имеющемся top-2 и margin <0.15 берётся Qwen независимо от высокого top-1. Получаются 16 комбинаций. Отсутствующий top-2 не создаёт margin. Необработанные Qwen ID остаются NOT_PROCESSED, а не заменяются выдуманным решением.

Для каждой policy сохраняются исходные решения, итог и причина маршрута. Плановые calls и input tokens пересчитываются после реальной **офлайн-перепаковки** маршрутизированных mentions по документам с теми же batch/output/context ограничениями. Доля mentions, принятых GLiNER, не равна доле сэкономленных batch calls. Input savings используют консервативные byte token bounds и помечаются оценочными; гипотетическая hybrid inference latency не объявляется измеренной.

Автоматический summary после успешных этапов и отдельная `summarize` работают также с частичными результатами: agreements считаются по пересечению валидных IDs, missing IDs перечисляются явно. Сохраняются distributions 11 типов/UNKNOWN/NON_ENTITY, конфликтующие пары, JSON errors, изменения типов, planned hybrid savings, реальные prompt/decode/load timings и counts из Ollama, latency каждого attempt, retries, sampled peak RSS клиента и суммы Ollama processes. RSS может дважды учитывать shared pages, sampled peak может пропустить короткий пик. Повторные попытки включены в расход токенов и время.

Обычные отчёты не содержат истинных Precision/Recall/F1. Для последующей **независимой ручной** проверки подготовьте JSONL:

```json
{"candidate_id":"c:<ID из candidate_dataset.jsonl>","entity_type":"METHOD","review_status":"HUMAN_REVIEWED"}
```

```powershell
& $py scripts/run_entity_typing_v2.py evaluate --dataset pilot --gold .\human_gold.jsonl
```

Evaluator проверит IDs, допустимые типы и статус review, рассчитает per-type и micro Precision/Recall/F1 на проверенных candidates; missing/abstention дают FN для соответствующего положительного типа. Результат — `human_evaluation.json`. Эти метрики оценивают typing существующих candidates, а не полноту обнаружения mentions во всём корпусе. Предварительные assisted reference не используются как Gold.

## Артефакты и ZIP

Общий каталог pilot — `reports/entity_typing_v2/run1/`:

- `manifest.json`, `candidate_dataset.jsonl`, `ontology.json` — frozen входы и runtime identity.
- `T1_gliner.jsonl`, `T21_qwen_no_thinking.jsonl`, `T22_qwen_thinking.jsonl`, `T3_hybrid.jsonl` — решения.
- `metrics.json`, `metrics.csv`, `disagreements.csv`, `experiment_summary.md` — результаты сравнения.
- `hardware.json`, `model_settings.json`, `failure_log.jsonl` — ресурсы и диагностика.
- `batch_plan.jsonl`, `T3_plan_estimates.json`, `attempts/`, `sessions/`, `command_sessions/` — аудит запросов и timings.
- `dry_run.json`, `dry_run_batches.jsonl` — проверка без генерации.

Файлы этапов создаются только после соответствующего запуска; T2 outputs до inference отсутствуют. Все записи UTF-8, offsets — codepoints, end exclusive. Исторические отчёты, Ontology v2, NER-V2 и N3 не изменяются. Relation Extraction, verification и resolution не выполняются.

```powershell
& $py scripts/run_entity_typing_v2.py export-results --dataset pilot
& $py scripts/run_entity_typing_v2.py export-results --dataset full --zip .\reports\entity_typing_v2\full_results.zip
```

Pilot ZIP по умолчанию: `reports/entity_typing_v2/run1_pilot_results.zip`. Он содержит результаты, raw responses/thinking, manifest, метрики, логи, frozen ontology, исходники стенда/конфиг/README для воспроизведения и `archive_sha256_manifest.json`. Веса и caches не включаются; full подкаталог не попадает в pilot ZIP. Можно экспортировать незавершённый эксперимент: summary явно укажет пропущенные результаты. Передайте ZIP для анализа после своего запуска.
