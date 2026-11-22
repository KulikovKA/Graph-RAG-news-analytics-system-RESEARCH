# Stage 3A.5.8 — Neural & Hybrid Graph-RAG Reranking

Runner оценивает нейросетевые и гибридные политики на сырых путях конфигурации B из Stage 3A.5.7. Он работает офлайн, последовательно загружает модели с диска и сохраняет отдельный эксперимент в `runs/`. Production C остаётся действующим реранкером. [Методология](METHODOLOGY.md)

## 1. Окружение Windows CPU

Команды выполняются из корня репозитория в PowerShell. Lock рассчитан на Windows x64 и Python 3.11. Для запуска достаточно доступа к локальным артефактам Stage 3A.5.7 и весам реранкеров.

```powershell
$Stage = '.\reports\retrieval\benchmark_3a5_8'
$Runner = Join-Path $Stage 'run_neural_reranker_benchmark.py'
uv venv .cache/venv-3a5-8 --python 3.11
$Python = '.\.cache\venv-3a5-8\Scripts\python.exe'
uv pip sync --python $Python --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match (Join-Path $Stage 'requirements-win-cpu.txt')
& $Python $Runner --help
```

Окружение Stage 3A.5.7 остаётся в `.venv`. Для CUDA потребуется отдельная совместимая сборка PyTorch. Укажите устройство `--device cuda:0` и новый каталог запуска. Windows CPU lock проверен на PyTorch 2.11, Transformers 5.7 и Sentence Transformers 5.4.1. Ошибки минимальных версий и доступности устройства выдаёт preflight.

## 2. Источники и preflight

По умолчанию runner читает `benchmark_3a5_2/integrated_runs/20261122T191609-112996Z/`, включая `cache/bfs`, `cache/evidence`, `cache/rerank`, `config.json`, `manifest.json`, `results.json` и `per_question.jsonl`. Требуются исходный корпус, замороженные наборы вопросов и `.cache/semantic_concept_sweep/representations.json` с оригинальными именами сущностей. Для перемещённых файлов используйте `--source-run` и `--entity-names`.

Основной каталог весов — `$env:USERPROFILE\Desktop\embedding-models\rerankers`. Для отсутствующей папки runner ищет точный pinned snapshot в локальном кэше Hugging Face. Можно направить все модели в другой корень через `--model-root` или задать отдельные папки повторяемым `--model-path ID=PATH`. Передавайте путь к папке, содержащей `config.json`. SHA-256 весов проверяется по реестру, а параметры токенизации и конфигурации входят в fingerprint. Ошибки отдельно показывают отсутствующий файл, неподходящую архитектуру или изменённые веса. [Реестр моделей и ревизий](MODEL_REGISTRY.json)

```powershell
& $Python $Runner --preflight
# Если Qwen лежит вне основного каталога и локального HF cache
# & $Python $Runner --preflight --models qwen3-0.6b --model-path 'qwen3-0.6b=D:\models\qwen3-reranker-0.6b'
```

Preflight проверяет все 35 пулов и выбранные веса, импортирует API и проверяет устройство. Чтение и хеширование весов занимает время. Модели не инстанцируются, а каталог результатов не создаётся. `--preflight --smoke` ограничивает проверку данных двумя вопросами.

### Однократное восстановление отсутствующего кэша

Сначала попробуйте восстановить оригинальные `cache/bfs`, `cache/evidence`, `cache/rerank` рядом с завершённым исходным прогоном. Если резервной копии нет, явно запустите Stage 3A.5.7 только для B в новом каталоге. Нужны существующий замороженный Neo4j snapshot, разрешённый доступ на чтение, прежние E5/GLiNER и зависимости `.venv` из Stage 3A.5.7. Команда может занять несколько минут. [Инструкции Stage 3A.5.7](../benchmark_3a5_2/INTEGRATED_BENCHMARK_README.md)

```powershell
$Neo4jEnv = 'C:\path\to\existing\neo4j\.env'
$PoolDir = Join-Path $Stage ('runs\pool-B-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
& .\.venv\Scripts\python.exe .\reports\retrieval\benchmark_3a5_2\run_integrated_benchmark.py --variants B --output-dir $PoolDir --neo4j-env $Neo4jEnv
& $Python $Runner --preflight --source-run $PoolDir
```

Если блок B повторно получен с другими seeds, порядком путей, evidence или baseline, проверка `FROZEN_B.json` отклонит его. Для основного сравнения потребуется исходный кэш. Сам runner не подключается к Neo4j и не восстанавливает пул во время прогона.

## 3. Smoke и полный запуск

```powershell
$Smoke = Join-Path $Stage ('runs\smoke-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
& $Python $Runner --smoke --models minilm-l6 ettin-68m --candidate-budget 20 --representation both --device cpu --batch-size 4 --threads 4 --output-dir $Smoke

$Run = Join-Path $Stage ('runs\full-B20-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
$Models = 'ettin-68m','ettin-400m','bge-m3','mxbai-base','minilm-l6','qwen3-0.6b'
& $Python $Runner --models $Models --candidate-budget 20 --representation both --device cpu --batch-size 4 --threads 4 --output-dir $Run
```

Smoke использует A01 и approved C03. `--representation both` последовательно оценивает `fact_focused` и `document_aware` в рамках одной загрузки модели. Для первого короткого прогона используйте один вариант представления. Бюджет 50 запускайте отдельно, например с `--candidate-budget 50` и новым `--output-dir`. Бюджеты 100 и 200 требуют явного выбора. При budget 20 и двух представлениях верхняя граница полного прогона — 8 400 пар. `--max-length` задаёт суммарную длину вопроса, текста и шаблона в токенах, по умолчанию 512. `--warmup-batches` по умолчанию равен 1. Для вывода без полос используйте `--no-progress`.

В проверенном пуле бюджет 20 даёт такой же потолок factual Gold recall, как результат C. Бюджет 50 добавляет один доступный Gold-документ для approved C05. [Результаты проверки](VALIDATION.md)

## 4. Resume

```powershell
& $Python $Runner --resume --output-dir $Run
```

Настройки, порядок моделей и 35 ключей восстанавливаются из `config.json`. Resume повторно сверяет код, источники, runtime, веса и checkpoint. Завершённые блоки `модель × представление × вопрос` сохраняются при Ctrl+C и ошибке модели. Текущий незаписанный блок считается заново. Если модель была недоступна при фиксации fingerprint и появилась позже, для новой комбинации файлов используйте новый каталог. Коды выхода: `0` завершён, `2` ошибка preflight или fingerprint, `3` частичный результат, `130` прерывание.

## 5. Отчёты

```powershell
Get-Content -Encoding utf8 (Join-Path $Run 'comparison.md')
Import-Csv (Join-Path $Run 'comparison.csv') | Where-Object group -eq 'ALL' |
    Format-Table model, tier, policy, representation, completed, 'FactualGoldDocRecall@5_micro', latency_p95_seconds, pairs_per_inference_second, peak_process_rss_mib
```

| Файл | Содержимое |
| --- | --- |
| `config.json`, `model_fingerprints.json` | Настройки, данные, код, runtime, версии и SHA-256 |
| `manifest.json` | Завершённые блоки, попытки, загрузка, warmup и peak RSS по моделям |
| `comparison.md`, `comparison.csv` | Качество, время, парные дельты и наблюдения по вопросам |
| `results.json` | Сводки, парные дельты, candidate oracle, изменения путей, Gold, распределения scores |
| `per_question.jsonl` | C и обе нейросетевые политики, ранги, evidence и метрики |
| `baseline/`, `scores/` | Контрольные суммы checkpoint, C preselection, scores и ранги каждой пары, статистика truncation |
| `errors.jsonl`, `execution.log` | События, ошибки, трассировки |

Весовые файлы и локальные результаты исключены из Git. Точные фрагменты и offsets сохраняются в выбранных путях. Ссылки на полные входные BFS и evidence caches записаны для каждого вопроса.
