# Четыре эксперимента, по две копии модели в каждом

Все четыре скрипта подготовлены для самостоятельного запуска пользователем.
После изменения новые эксперименты не запускались. Общая реализация находится в
`scripts/relex_experiment.py`, функции E3 и Neo4j переиспользуются из
`scripts/build_relex_graphs.py`.

| Скрипт | Источник сущностей | Модель отношений | Namespace | Каталог результата |
|---|---|---|---|---|
| `build_e3_relex_base.py` | Исходный E3 GLiNER | RelEx Base | `relex_base` | `e3_base/` |
| `build_e3_relex_large.py` | Исходный E3 GLiNER | RelEx Large | `relex_large` | `e3_large/` |
| `build_joint_relex_base.py` | Сама RelEx Base | Та же RelEx Base | `relex_joint_base` | `joint_base/` |
| `build_joint_relex_large.py` | Сама RelEx Large | Та же RelEx Large | `relex_joint_large` | `joint_large/` |

Каталоги находятся внутри `reports/re_v2/relex_comparison/`.

В **каждом скрипте по умолчанию два Python worker**. Каждый загружает одну копию
нужной модели один раз. На 30 документах worker 1 обрабатывает 15 документов,
worker 2 — остальные 15. После загрузки обеих копий барьер одновременно разрешает
им инференс. Каждый документ обрабатывается один раз в рамках эксперимента;
результаты собираются в один подграф, а не в два дублирующих графа.

У каждого worker собственные progress bar, `worker1.log`/`worker2.log`, API audit
и timing. `--resume` пропускает готовые документы; если всё уже обработано,
модели не загружаются. Смена числа workers не меняет идентификаторы и совместима
с сохранёнными checkpoints.

По умолчанию `--workers 2 --threads 2`: два CPU-потока на одну копию модели.
Если запустить все четыре скрипта одновременно, получится **8 копий моделей и
16 вычислительных CPU-потоков**, плюс родительские процессы. Ускорение и RAM
новой конфигурации пока не измерялись.

## Запуск

Используются уже скачанные модели `artifacts/models/gliner-relex-base-v1.0` и
`artifacts/models/gliner-relex-large-v1.0`. Скачивания запрещены offline-режимом.
Зависимости включены в extra `relex` существующего `pyproject.toml` и `uv.lock`.

Учётные данные существующего Neo4j задаются через окружение PowerShell:

```powershell
$neo4jCredential = Get-Credential -UserName 'neo4j' -Message 'Existing Neo4j credentials'
$env:NEO4J_USER = $neo4jCredential.UserName
$env:NEO4J_PASSWORD = $neo4jCredential.GetNetworkCredential().Password
$env:NEO4J_URI = 'bolt://127.0.0.1:17687'
```

Пароли не сохраняются в код, отчёты или Git. Текущий `.env` старого проекта при
предыдущей проверке не совпадал с действующим паролем базы; применялся исходный
пароль из старого `compose.yaml`. Ни пароль, ни старые данные не менялись.

`Test-Path` подтверждает только наличие файла. При `Unauthorized` нужно задать
действующий пароль, а не повторять чтение того же `.env`. Уже заданные переменные
PowerShell имеют приоритет над `--neo4j-env`, поэтому обновите `NEO4J_PASSWORD`
в текущем терминале либо откройте новый терминал и задайте корректное окружение.

При `AuthenticationRateLimit` прекратите повторные попытки и дождитесь окончания
временной блокировки. В существующем контейнере healthcheck также использует
устаревший пароль из `NEO4J_AUTH` и периодически выполняет неудачный вход.
Фактический пароль базы, конфигурация контейнера и его healthcheck скриптами не
изменяются. Ошибки авторизации теперь выводятся коротким сообщением до загрузки
моделей; автоматического перебора паролей или повторных попыток нет.

Из корня проекта, каждый эксперимент отдельно:

```powershell
.\.venv\Scripts\python.exe scripts\build_e3_relex_base.py
```

```powershell
.\.venv\Scripts\python.exe scripts\build_e3_relex_large.py
```

```powershell
.\.venv\Scripts\python.exe scripts\build_joint_relex_base.py
```

```powershell
.\.venv\Scripts\python.exe scripts\build_joint_relex_large.py
```

Для одновременного запуска всех четырёх из одного PowerShell с уже заданными
переменными окружения можно выполнить следующий блок самостоятельно. Он
создаст скрытые процессы, выведет PID и запишет stdout/stderr каждого запуска
в отдельные файлы:

```powershell
$relexPython = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$relexConsole = Join-Path (Get-Location) 'reports\re_v2\relex_comparison\console'
New-Item -ItemType Directory -Path $relexConsole -Force | Out-Null
$relexScripts = 'build_e3_relex_base', 'build_e3_relex_large', 'build_joint_relex_base', 'build_joint_relex_large'
foreach ($relexScript in $relexScripts) {
    Start-Process -WindowStyle Hidden -FilePath $relexPython `
        -ArgumentList @('-u', "scripts\$relexScript.py", '--resume') `
        -WorkingDirectory (Get-Location).Path `
        -RedirectStandardOutput (Join-Path $relexConsole "$relexScript.stdout.log") `
        -RedirectStandardError (Join-Path $relexConsole "$relexScript.stderr.log") `
        -PassThru | Select-Object Id, ProcessName
}
```

Если запускать вручную в четырёх терминалах, переменные подключения нужно задать
в каждом терминале. Последовательное выполнение четырёх обычных команд в одном
терминале запускает эксперименты по очереди; **внутри каждого всё равно работают
две копии модели параллельно**.

Общие аргументы всех четырёх скриптов:

```powershell
# Две копии одной модели, по два потока; продолжить готовые документы.
.\.venv\Scripts\python.exe scripts\build_joint_relex_base.py --workers 2 --threads 2 --resume

# Smoke test, отдельный каталог. Пользователь запускает его самостоятельно.
.\.venv\Scripts\python.exe scripts\build_joint_relex_base.py --limit 2 --output reports/re_v2/relex_comparison/smoke_joint_base
```

Есть `--inference-only` (без Neo4j), `--neo4j-env`, `--relation-threshold`
(по умолчанию 0.7), `--entity-threshold` (только совместный режим, по умолчанию 0.3).
Повторный запуск с готовыми checkpoints требует `--resume`.

## Различия режимов и результаты

Первые два скрипта используют оригинальные `research_run2/E3/mentions_035.jsonl`:
1327 исходных записей, 1211 уникальных spans. Повторного GLiNER NER нет.
Все исходные типы, offsets, normalized_form, confidence и provenance сохраняются.
Relation head получает только эти сущности, без добавления новых.

В совместных режимах вызывается родной `model.inference(..., return_relations=True)`
без `input_spans`. Файл E3 **не читается**. Узлы создаются из собственных NER
предсказаний RelEx, связи — из её RE-предсказаний. Границы, текст, confidence,
типы, направление и принадлежность endpoints одному документу проверяются.
Сохраняются также сущности без отношений. Нормализация joint-упоминаний явно
обозначена как `joint-nfc-casefold-v1` и не подменяет исходную E3-нормализацию.

В обоих режимах используются те же 30 аннотаций и 20 типов отношений Ontology v2.
Base и Large имеют общий план окон внутри режима, учитывающий оба токенизатора.
E3-окна защищают известные spans от разрезания; joint-окна строятся только по
тексту, поскольку готовых сущностей в этом режиме нет. Планы между режимами
могут отличаться. Gold-разметка не используется.

В каждом каталоге: `experiment_config.json`, `prepared_inputs.json`, `windows.jsonl`,
атомарные checkpoints по документам, `entities.jsonl`, `predictions.jsonl`
(включая полные raw-предсказания по окнам), `relations.jsonl`, `report.md`,
`summary.json`, `metrics.csv`, `relation_distribution.csv` (все 20 типов),
`confidence_distribution.csv`, `per_document.csv`, логи workers и `graph.cypher`.
RAM измеряется как суммарный RSS родителя и его workers; другие эксперименты и
Docker не входят в этот показатель.

Для каждого графа используются детерминированные ключи с собственным namespace.
Совпадающие названия из разных статей не объединяются. Импорт — только MERGE
выделенных RELEX labels/relationships, без удаления старых данных.
Одновременные четыре запуска имеют разные каталоги и namespaces.

Browser: <http://127.0.0.1:17474>, подключение `bolt://127.0.0.1:17687`.
В каждом `graph.cypher` есть запросы на сущности, directed relations и документы.
Пример для совместного Base:

```cypher
MATCH (e:RELEXEntity {namespace: 'relex_joint_base'})
OPTIONAL MATCH (e)-[r:RELEX_REL {namespace: 'relex_joint_base'}]->(t:RELEXEntity {namespace: 'relex_joint_base'})
RETURN e,r,t;
```

Автоматические проверки не загружают pretrained модели и не подключаются к Neo4j:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_build_relex_graphs.py tests/test_relex_four_experiments.py -q
```
