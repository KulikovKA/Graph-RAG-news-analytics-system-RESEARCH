# ENTITY-TYPING-V2-001 — локальный стенд Entity Typing

Стенд сравнивает T1 GLiNER-only, T2.1 Qwen без thinking, T2.2 Qwen с thinking и T3 hybrid на одних замороженных mentions. **Основной режим по умолчанию — full, все 1211 candidates, два независимых сервера Qwen.** Pilot (240) оставлен для быстрой проверки. Подготовка и разработка не требуют генерации Qwen. Запуск `qwen`/`night` без `--dry-run` и launcher без `-DryRun` запускают настоящий эксперимент; это делает пользователь.

## Два отдельных сервера

По запросу пользователя 05.11.2026 используются `http://localhost:11434` и `http://127.0.0.1:11435`. В [Ollama 0.35.0](https://github.com/ollama/ollama/blob/v0.35.0/server/sched.go#L481-L485) для архитектуры qwen35 принудительно разрешён только один вычислительный слот на сервер. Поэтому два асинхронных запроса здесь адресуются **двум процессам Ollama**, у каждого своя модель и своё окно контекста. Это не два запроса в очередь одного сервера. Общие файлы весов на диске используются из существующего Ollama model store, без повторного скачивания.

Внутри T2.1 одновременно выполняются до двух OFF batches; после завершения всех OFF запросов так же выполняются ON batches. Обе копии обязаны иметь одинаковые tag, digest, quantization, native context и версию Ollama. Каждый candidate отправляется одному серверу на попытку; основной поток единолично обновляет общие JSONL и failure log. Worker сохраняет уникальный attempt journal до отправки и сразу после ответа. Journal содержит server URL, worker index, model digest и точные timings.

При настоящем запуске существующий сервер переиспользуется. Если endpoint отсутствует, Python запускает `ollama serve` скрыто с `OLLAMA_HOST` соответствующего порта, `OLLAMA_NUM_PARALLEL=1` и `OLLAMA_MAX_LOADED_MODELS=1`, ждёт readiness и затем работает с ним. При завершении/ошибке останавливаются только процессы, созданные этим launcher; уже работавший сервер не завершается. Логи серверов — `full/server_logs/`, ownership/PID — `full/server_processes.json`. Дополнительные запросы Qwen для readiness не выполняются. При Ctrl+C клиент закрывает активные HTTP sockets; уже полученные raw ответы остаются в durable journal для resume.

Общий RAM budget учитывает **две модели, два независимых KV cache и резерв runtime для каждой**. Одинаковый `num_ctx` выбирается максимальным из native ladder для обеих копий и затем проверяется на обоих `/api/ps`. Например, два окна по 32768 — два отдельных окна, а не общее окно 65536. Если даже минимальное окно не помещается по RAM или не вмещает полный prompt, эксперимент останавливается; освободите память и повторите dry-run. Тексты не обрезаются. По умолчанию `num_thread=4` на модель, то есть до 8 compute threads для двух серверов; `-NumThread 2` задаёт по два. Ускорение зависит от CPU и памяти, оно не гарантировано.

Новая конфигурация пишет в **`reports/entity_typing_v2/run3_three_servers/`**, оставляя прежнюю попытку `run1` без изменений. Frozen runtime старого прогона не мигрируется и результаты разных настроек не смешиваются. Для полноценного одиночного режима можно создать отдельный config с `server_urls: []` и отдельным `output`.

Dry-run не запускает серверы и не загружает модели. Если 11435 ещё не запущен, он использует метаданные checkpoint с работающего `base_url` (11434), явно отмечает второй сервер как `ready=false` и считает RAM для **двух** копий. Это проверка плана, а не подтверждение фактической аллокации второго экземпляра.

## Ночной запуск

Из корня репозитория:

```powershell
# Только проверка CLI, метаданных и полного набора, без загрузки/генерации модели
.\run_all_night.ps1 -DryRun

# Настоящий ночной запуск: full=1211, два сервера, по четыре CPU-потока, resume включён
.\run_all_night.ps1

# Быстрая проверка на pilot без inference
.\run_all_night.ps1 -Dataset pilot -DryRun
```

Порядок: prepare → T1 → T2.1 OFF → T2.2 ON → T3 offline → summarize → completeness → export-results. Команда `python scripts/run_entity_typing_v2.py night` выполняет тот же pipeline. Старый `scripts/run_entity_typing_v2.ps1` перенаправляет в новый launcher и тоже выбирает full по умолчанию. CLI берёт default dataset из `config/entity_typing_v2.json` (`default_dataset=full`); явный `--dataset pilot` переключает набор.

Launcher работает в текущем foreground PowerShell, ждёт Python до завершения, не создаёт detached jobs и не запрашивает подтверждений. На время реального запуска он запрещает автоматический сон Windows через [SetThreadExecutionState](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate), затем восстанавливает предыдущее состояние; постоянный план питания не меняется, экран может выключаться. Не закрывайте PowerShell и не отправляйте компьютер в сон вручную: такой пользовательский сон API не блокирует. `-DryRun` состояние питания не меняет.

Полный каталог: `reports/entity_typing_v2/run3_three_servers/full/`; итоговый ZIP: `reports/entity_typing_v2/run3_three_servers/full_results.zip`. Технические логи — `full/night_runs/`: PowerShell transcript и Python log (stdout/stderr, tqdm и traceback), отдельные файлы на каждый запуск. `manifest.json` содержит requested config, checkpoint/digest/quantization, фактический context, thinking flags, версии Python/пакетов/Ollama, timings и историю этапов каждого запуска. При model preflight failure фактический контекст остаётся неизвестным — он не подменяется оценочным.

После исчерпания retries неудачного batch сохраняются NOT_PROCESSED, обработка остальных batches продолжается. Неполный T2.1 не блокирует T2.2 и офлайн-отчёты. Ошибки этапов сохраняются в `stage_errors.jsonl` и общем `failure_log.jsonl`. Повторите `.\run_all_night.ps1`, чтобы продолжить: полностью завершённые T2.1/T2.2 проверяются по durable journal и пропускаются **без обращения к Ollama**, незавершённые отправляют только оставшиеся IDs. Между этапами удерживается единый OS lock, поэтому второй launcher не может вмешаться в тот же сервер/checkout.

Итоговые `completeness.json` и `completeness.csv` проверяют каждый из 1211 ID для T1, T2.1, T2.2 и всех 16 T3 policies, включая отсутствующие/дублированные/посторонние ID, согласованность решений и актуальность T3 относительно входов. Строка NOT_PROCESSED обеспечивает учёт ID, но не считается успешным предсказанием. Даже при недоступном Ollama создаются явные строки для всех ID и ZIP с доступными результатами. Status COMPLETE и код 0 выдаются только при полной успешной классификации и успешном экспорте. Код 3 — частичный прогон/ошибка этапа/экспорта; 130 — Ctrl+C с попыткой сохранить офлайн-отчёты и ZIP. Ошибки конфигурации/исходных данных до начала pipeline дают код 1. При неисправности диска/повреждении данных создание полного набора артефактов может оказаться невозможно; подробности останутся в доступном техническом логе.

Параметры launcher: `-Python`, `-Config`, `-Output`, `-NumThread`, `-ServerUrls`, `-StrictMaxContext`, `-Dataset`, `-DryRun`. CLI поддерживает `--server-urls URL1 URL2`. Никаких вызовов Qwen для отчётов, проверки полноты или ZIP нет. Чтобы проверить полноту отдельно:

```powershell
.\.venv-ner-v2\Scripts\python.exe scripts/run_entity_typing_v2.py check-completeness
```

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

Клиент отправляет по одному запросу на каждый независимый сервер, до двух одновременно. OS locks удерживаются для обоих нормализованных endpoint addresses и каталога результатов на весь pipeline, включая переходы между этапами. Если доступный локальный процесс Ollama содержит `OLLAMA_NUM_PARALLEL` со значением, отличным от `1`, запуск отклоняется. Отсутствующая/недоступная переменная явно отмечается как непроверенная: клиентская настройка не может изменить уже запущенный сервер. Во время эксперимента не посылайте сторонние запросы в те же серверы Ollama.

## Подготовка и dry-run без загрузки модели

```powershell
& $py scripts/run_entity_typing_v2.py prepare --dataset full
& $py scripts/run_entity_typing_v2.py qwen --dataset full --dry-run
```

Первый шаг читает только `reports/ner_v2/research_run2/input_abstracts.jsonl`, `E3/mentions_035.jsonl`, `E3/config.json` и Ontology v2. Он проверяет исходные SHA-256, provenance каждого hint, N3, точные Unicode codepoint offsets и score ≥0.35. Имеются 30 документов, 1327 hints и 1211 уникальных spans. Несколько hints одного `(document_id,start,end)` объединяются, для каждого типа берётся максимальный score. Нормализованная N3 form переиспользуется; NLP и модели не загружаются. Нижепороговые hints не добавляются: отсутствие top-2 означает `null` margin, а не нулевую альтернативу.

Pilot содержит 240 candidates с seed 4401. Страты — discipline × top-1 type; минимум один кандидат каждой непустой страты, оставшиеся места распределяются пропорционально её ёмкости методом наибольших остатков. Внутри страт выборка циклически охватывает score bands и неоднозначность (`margin<0.15`), с seeded shuffle. Это диагностическая выборка, а не оценка распространённости типов в корпусе без поправки на sampling. Предварительные assisted labels вообще не читаются.

Стабильные непрозрачные candidate IDs зависят от документа, редакции, хеша текста и offsets. `manifest.json`, `candidate_dataset.jsonl` и копия `ontology.json` замораживаются до inference. Повторный prepare идемпотентен; изменение входов требует нового `--output`.

Dry-run создаёт `dry_run.json` и `dry_run_batches.jsonl`, проверяет доступность модели, настройки, пути, RAM и prompt sizes. Он **не вызывает `/api/chat` или `/api/generate`**, не загружает и не выгружает модель. Код возврата: 0 — модель доступна и план построен, 2 — метаданные/ресурсы недоступны, 1 — ошибка конфигурации или данных. Недоступность модели не выдаётся за успешную проверку. В `model_settings.json` предложенные настройки отмечены `DRY_RUN_PROPOSED`.

## Последовательный запуск четырёх тестов

```powershell
& $py scripts/run_entity_typing_v2.py gliner --dataset full
& $py scripts/run_entity_typing_v2.py qwen --mode no-thinking --dataset full --resume
& $py scripts/run_entity_typing_v2.py qwen --mode thinking --dataset full --resume
& $py scripts/run_entity_typing_v2.py hybrid --dataset full
& $py scripts/run_entity_typing_v2.py summarize --dataset full
```

Либо ночной launcher с prepare, последовательными этапами, проверкой полноты и ZIP:

```powershell
.\run_all_night.ps1 -Python $py -NumThread 4
```

Только безопасная подготовка и dry-run через launcher:

```powershell
.\run_all_night.ps1 -Python $py -DryRun
```

Без явного выбора dataset используются все 1211 mentions. Для быстрой проверки задайте `--dataset pilot`/`-Dataset pilot`. Pilot сохраняется в `reports/entity_typing_v2/run3_three_servers/`, full — в `reports/entity_typing_v2/run3_three_servers/full/`; manifests не смешиваются. `--output reports/entity_typing_v2/run2` выбирает новый корень (для full также добавится `full/`). Если ExecutionPolicy запрещает запуск `.ps1`, используйте Python-команды выше; временный запрет сна реализован именно PowerShell launcher.

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

По умолчанию оба output budget равны 4096; ON резервирует 2048 для thinking при планировании общего batch. Оба сервера используют одинаковые настройки. Настройки можно менять **до первого inference**, например:

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

tqdm показывает candidates, batches, elapsed/ETA, текущий batch, количество активных workers, errors, накопленные input/output tokens и candidates/min; после batch печатается worker и краткая строка validation/timing. Длинное thinking в терминал не выводится. Batch ETA меняется при добавлении retry batches. Сумма batch/decode durations при параллельной работе может превышать elapsed wall time: обе величины записываются отдельно, throughput по candidates считается по wall time сессий.

## T3 и метрики

T3 читает только замороженный набор и сохранённые решения T1/T2.1/T2.2; сетевого клиента у него нет. Detection остаётся 0.35. Для routing thresholds 0.50/0.55/0.60/0.65 score ≥ threshold принимает GLiNER; ниже берётся сохранённое решение Qwen. Для каждого режима дополнительно считается risk-aware policy: при имеющемся top-2 и margin <0.15 берётся Qwen независимо от высокого top-1. Получаются 16 комбинаций. Отсутствующий top-2 не создаёт margin. Необработанные Qwen ID остаются NOT_PROCESSED, а не заменяются выдуманным решением.

Для каждой policy сохраняются исходные решения, итог и причина маршрута. Плановые calls и input tokens пересчитываются после реальной **офлайн-перепаковки** маршрутизированных mentions по документам с теми же batch/output/context ограничениями. Доля mentions, принятых GLiNER, не равна доле сэкономленных batch calls. Input savings используют консервативные byte token bounds и помечаются оценочными; гипотетическая hybrid inference latency не объявляется измеренной.

Автоматический summary после успешных этапов и отдельная `summarize` работают также с частичными результатами: agreements считаются по пересечению валидных IDs, missing IDs перечисляются явно. Сохраняются distributions 11 типов/UNKNOWN/NON_ENTITY, конфликтующие пары, JSON errors, изменения типов, planned hybrid savings, реальные prompt/decode/load timings и counts из Ollama, latency каждого attempt, retries, sampled peak RSS клиента и суммы Ollama processes. RSS может дважды учитывать shared pages, sampled peak может пропустить короткий пик. Повторные попытки включены в расход токенов и время.

Обычные отчёты не содержат истинных Precision/Recall/F1. Для последующей **независимой ручной** проверки подготовьте JSONL:

```json
{"candidate_id":"c:<ID из candidate_dataset.jsonl>","entity_type":"METHOD","review_status":"HUMAN_REVIEWED"}
```

```powershell
& $py scripts/run_entity_typing_v2.py evaluate --dataset full --gold .\human_gold.jsonl
```

Evaluator проверит IDs, допустимые типы и статус review, рассчитает per-type и micro Precision/Recall/F1 на проверенных candidates; missing/abstention дают FN для соответствующего положительного типа. Результат — `human_evaluation.json`. Эти метрики оценивают typing существующих candidates, а не полноту обнаружения mentions во всём корпусе. Предварительные assisted reference не используются как Gold.

## Артефакты и ZIP

Общий каталог pilot — `reports/entity_typing_v2/run3_three_servers/`:

- `manifest.json`, `candidate_dataset.jsonl`, `ontology.json` — frozen входы и runtime identity.
- `T1_gliner.jsonl`, `T21_qwen_no_thinking.jsonl`, `T22_qwen_thinking.jsonl`, `T3_hybrid.jsonl` — решения.
- `metrics.json`, `metrics.csv`, `disagreements.csv`, `experiment_summary.md` — результаты сравнения.
- `hardware.json`, `model_settings.json`, `failure_log.jsonl` — ресурсы и диагностика.
- `batch_plan.jsonl`, `T3_plan_estimates.json`, `attempts/`, `sessions/`, `command_sessions/` — аудит запросов и timings.
- `dry_run.json`, `dry_run_batches.jsonl` — проверка без генерации.

Файлы этапов создаются только после соответствующего запуска; T2 outputs до inference отсутствуют. Все записи UTF-8, offsets — codepoints, end exclusive. Исторические отчёты, Ontology v2, NER-V2 и N3 не изменяются. Relation Extraction, verification и resolution не выполняются.

```powershell
& $py scripts/run_entity_typing_v2.py export-results --dataset full
& $py scripts/run_entity_typing_v2.py export-results --dataset full --zip .\reports\entity_typing_v2\full_results.zip
```

Pilot ZIP по умолчанию: `reports/entity_typing_v2/run3_three_servers_pilot_results.zip`. Он содержит результаты, raw responses/thinking, manifest, метрики, логи, frozen ontology, исходники стенда/конфиг/README для воспроизведения и `archive_sha256_manifest.json`. Веса и caches не включаются; full подкаталог не попадает в pilot ZIP. Можно экспортировать незавершённый эксперимент: summary явно укажет пропущенные результаты. Передайте ZIP для анализа после своего запуска.
