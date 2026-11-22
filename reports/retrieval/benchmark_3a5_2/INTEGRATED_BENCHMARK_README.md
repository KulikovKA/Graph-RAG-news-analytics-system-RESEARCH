# Stage 3A.5.7 — Unified End-to-End Retrieval Benchmark

Runner выполняет 140 оценок для 35 замороженных вопросов. В эксперимент входят 8 вопросов A, 7 вопросов B, 10 утверждённых вопросов C и 10 исторических вопросов C. В отчётах используется ключ `группа/ID`, например `C/3A52-C08` и `HOLDOUT/3A52-C08`.

| Вариант | Reranking | Linking cutoff | Linking top_k | Fallback | BFS max_candidates |
| --- | --- | ---: | ---: | --- | ---: |
| A | A | 0.85 | 5 | baseline | 200 |
| B | C | 0.85 | 5 | baseline | 200 |
| C | C | 0.85 | 10 | baseline | 200 |
| D | C | 0.85 | 10 | baseline | 100 |

Общие параметры включают порог GLiNER 0.15, FIFO BFS, 5 переходов, 50 соседей, 1 000 прочитанных связей, максимум 32 входных seed, таймаут транзакции 10 секунд и Top-5 путей после reranking. Fallback `baseline` использует текст всего вопроса, когда GLiNER не нашёл mentions. Gold применяется при расчёте метрик после retrieval и reranking.

## 1. Подготовьте PowerShell и Python

Запускайте команды из корня репозитория на ветке `design-v2-sequential-kg` с объединёнными изменениями Stage 3A.5.5 и 3A.5.6.

```powershell
Set-Location "$env:USERPROFILE\Desktop\Graph-RAG-news-analytics-system-RESEARCH"
git status --short --branch
. .\.venv\Scripts\Activate.ps1
python --version
$Runner = '.\reports\retrieval\benchmark_3a5_2\run_integrated_benchmark.py'
python $Runner --help
```

Если PowerShell запрещает выполнение `Activate.ps1`, используйте `& .\.venv\Scripts\python.exe $Runner ...` вместо `python $Runner ...`. Нужен Python 3.11. Если venv ссылается на удалённый интерпретатор или в нём не хватает зависимостей, синхронизируйте окружение по `uv.lock`.

```powershell
uv python install 3.11
uv sync --frozen --python 3.11 --extra benchmark --extra relex --extra test
. .\.venv\Scripts\Activate.ps1
```

Основные зависимости для запуска включают PyTorch, `gliner2[local]`, Transformers, Tokenizers, Hugging Face Hub, Safetensors, NumPy, Neo4j Python Driver и tqdm. Extra `test` добавляет pytest. Preflight проверяет импорт API и записывает версии зависимостей. Для сопоставимых результатов используйте окружение из `uv.lock`.

## 2. Проверьте локальные артефакты

- Замороженные документы и lock находятся в `reports\ner_v2\research_run2\input_abstracts.jsonl` и `reports\entity_resolution\relex_base_t035\dynamic_concept_graph\document_lock.json`.
- Gold и историческая выборка находятся в `questions_draft.jsonl` и `validate_promote_c\holdout_original_c.jsonl` рядом со скриптом.
- GLiNER E3 находится в `artifacts\ner_v2\models\gliner25`.
- Замороженные векторы, snapshot и манифест E5 находятся в `.cache\semantic_concept_sweep`. Пути E5 берутся из `embedding_manifest.json`. Если те же файлы E5 лежат в другом каталоге, передайте `--e5-model-dir 'D:\models\intfloat__e5-small-v2'` во все запуски, включая resume. Хеши файлов должны совпадать с манифестом.

## 3. Подключите Neo4j

Запустите существующий экземпляр Neo4j с замороженным namespace Stage 2G.3. Runner использует общий helper проекта `connect_neo4j`. В PowerShell задайте путь к существующему `.env` с переменными `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` и `NEO4J_DATABASE`.

```powershell
$Neo4jEnv = 'C:\path\to\existing\neo4j\.env'
Test-Path -LiteralPath $Neo4jEnv
# Для стандартного локального Bolt из helper проекта
Test-NetConnection 127.0.0.1 -Port 17687
```

Замените пример пути и при необходимости адрес и порт. Вместо `--neo4j-env $Neo4jEnv` можно использовать уже заданные переменные процесса. Переменные процесса имеют приоритет над файлом, поэтому при ошибке авторизации проверьте их актуальность. При ошибке namespace проверьте `NEO4J_DATABASE` и наличие существующего снимка Stage 2G.3 в выбранной базе. Не помещайте пароль в аргументы команды или журналы.

## 4. Запустите preflight

```powershell
python $Runner --preflight --neo4j-env $Neo4jEnv
$LASTEXITCODE
```

`Preflight PASS` и код `0` означают, что проверены Gold, корпус, векторы, хеши весов, Neo4j namespace, graph fingerprint и SHA-256 снимка. Проверка читает данные и хеширует локальные файлы моделей. Inference и каталог запуска появляются только при запуске benchmark.

## 5. Проверьте один вопрос

```powershell
$Runs = '.\reports\retrieval\benchmark_3a5_2\integrated_runs'
$SmokeDir = Join-Path $Runs ('smoke-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
python $Runner --neo4j-env $Neo4jEnv --groups C --question-id C08 --variants D --output-dir $SmokeDir
Get-Content -Encoding utf8 (Join-Path $SmokeDir 'comparison.md')
```

Это один вопрос и один вариант. Чтобы проверить совместное использование данных четырьмя вариантами, передайте `--variants A B C D` и выберите новый каталог. Фильтр `--question-id C08` без `--groups` выберет обе версии C08.

## 6. Запустите полный benchmark

```powershell
$Runs = '.\reports\retrieval\benchmark_3a5_2\integrated_runs'
$RunDir = Join-Path $Runs ('full-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
python $Runner --neo4j-env $Neo4jEnv --output-dir $RunDir
```

Без `--output-dir` runner создаст каталог с UTC ID и выведет его путь. В интерактивном PowerShell или Windows Terminal видны три полосы tqdm. **Overall benchmark** считает оценки, **Questions** считает вопросы, закрытые по всем выбранным вариантам, **Variant A/B/C/D** показывает текущий вариант. Строка вопроса показывает группу, ID, стадию, прошедшее время и ETA по мере накопления измерений. После каждого результата выводятся попадания Gold и factual Gold, число кандидатов и время, после каждой группы — промежуточный micro recall.

Для вывода в файл используйте `--no-progress`.

```powershell
python $Runner --neo4j-env $Neo4jEnv --output-dir $RunDir --no-progress 2>&1 |
    Tee-Object -FilePath '.\integrated-benchmark-console.log'
```

Выберите новый `$RunDir` для отдельного запуска. Внутренний `execution.log` создаётся и при выключенных полосах.

## 7. Продолжите прерванный запуск

```powershell
# Укажите каталог именно того запуска, который требуется продолжить.
$RunDir = '.\reports\retrieval\benchmark_3a5_2\integrated_runs\full-20261122-180000'
python $Runner --neo4j-env $Neo4jEnv --output-dir $RunDir --resume
```

Фильтры берутся из `config.json`, если их не передали явно. Для resume нужны тот же Git commit, тот же код, набор вопросов, версии зависимостей, модели, корпус, namespace и graph fingerprint. Runner после каждого варианта атомарно сохраняет checkpoint. При возобновлении он проверяет хеши checkpoint и артефактов, учитывает восстановленные оценки в прогрессе и повторяет незавершённые варианты. Каталог завершённого эксперимента защищён от повторной записи.

Коды выхода описывают состояние запуска. `0` означает, что завершены все запрошенные оценки и проверка графа прошла. `2` означает проблему preflight или конфигурации, `3` — ошибку или частичный результат, `130` — прерывание. Исходные ошибки при возобновлении остаются в `errors.jsonl`.

## 8. Откройте отчёты

| Файл | Содержимое |
| --- | --- |
| `config.json` | Политики, фильтры, Git SHA, environment, модели и fingerprints |
| `manifest.json` | Статус, количество завершённых оценок и попытки запуска |
| `results.json` | Сводки, парные эффекты и изменения документов и рангов по вопросам |
| `per_question.jsonl` | Одна строка на оценку с метриками, seeds, Top-5 и provenance |
| `comparison.csv` | Численные метрики по всем вопросам и по группам для pandas или Excel |
| `comparison.md` | Итоговые таблицы, сравнения A → B, B → C, C → D, A → D и разбор C03, C05, C07, C09 |
| `errors.jsonl`, `execution.log` | Диагностика и журнал выполнения |
| `checkpoints\`, `cache\` | Checkpoints, пулы путей, evidence с полным текстом и результаты отдельных стадий |

```powershell
Get-Content -Encoding utf8 (Join-Path $RunDir 'comparison.md')
Import-Csv (Join-Path $RunDir 'comparison.csv') |
    Where-Object group -eq 'ALL' |
    Format-Table variant, completed, gold_recall_micro, gold_recall_macro, factual_recall_micro,
                 all_required_rate, window_overlap_micro, mean_unique_factual_docs, mean_standalone_seconds
```

Micro recall делит сумму найденных Gold-документов на сумму обязательных документов. Macro recall усредняет долю по вопросам. Factual recall учитывает документы с проверенной привязкой фактических рёбер к источникам. Окна сравниваются по ID документа, ревизии и пересечению offsets. `Semantic Evidence Coverage = NOT_EVALUATED`. Для ручной проверки используйте фактические рёбра и исходные окна из `cache\rerank\`; путь к файлу указан у каждой оценки.

Колонка `mean_standalone_seconds` суммирует измеренные затраты стадий с полной общей стоимостью для каждого сравниваемого варианта. `mean_computed_stage_seconds` показывает выполненные заново вычисления, а `mean_wall_seconds` — наблюдаемое время оценки с cache reuse, сериализацией артефактов стадий и расчётом метрик. Первая загрузка моделей учитывается в стадии первого использования. Чтение кэша, запись checkpoint и подготовка отчётов отражаются по-разному в этих измерениях. Суммарное время попыток и уникальных вычислений указано в `results.json`. Парные сравнения используют одинаковый набор успешно выполненных вопросов. Документы вне Gold могут быть релевантными.
