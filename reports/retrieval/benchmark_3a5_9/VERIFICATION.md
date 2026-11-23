# Stage 3A.5.9 · воспроизведение и отдельная проверка

Первый проход записан в results/. Второй запуск в новом процессе для каждой фазы использовал те же frozen score rows. Независимый от evaluator код verify.py заново прочитал рёбра графа, source_relations_json, исходные документы и Gold, восстановил направление обхода и проверил опубликованную выдачу. Работу проверял основной исполнитель отдельным проходом. Внешний разметчик не участвовал.

| Фаза | Выдач | Уникальных путей внутри фазы | Проверенных окон в этих путях | Пар профиль–группа |
|---|---:|---:|---:|---:|
| Диагностика контролей | 50 | 140 | 145 | 8 |
| Однофакторная матрица | 425 | 182 | 188 | 68 |
| Дополнительный query-only linking | 25 | 115 | 120 | 4 |
| Совместные профили | 50 | 126 | 127 | 8 |
| Historical и переформулировки | 32 | 80 | 80 | 4 |

Повторение путей между фазами допустимо. Источники и окна хранятся один раз на path ID внутри каждой фазы.

| Проверка | Статус | Результат |
|---|---|---|
| Снимок и контракты | PASS | Восстановленный Stage 2G.3 совпал со snapshot SHA ba06b63d… и fingerprint 41901f76…. Сверены документы, ревизии, digest интегрированного manifest, наборы вопросов и хеши векторов. [Полные значения](PROFILE.json) |
| Паритет frozen B и текущего production | PASS | 35/35 подписей B включают текст, seeds, все сырые пути, evidence и C с компонентами score. Для текущих настроек сверены порядок Top-5 D и количество прочитанных строк с 3A.5.7. |
| Утечка и split | PASS | Код profiles.py принимает текст, mentions, cosine и структурный индекс. В нём отсутствуют Gold, question/group IDs и выбор параметров по документу. В run.py ссылки передаются в measure/diagnose после выдачи. Выбор компонентов читает только approved в matrix/lexical/combined. В confirmation фиксируется один финалист до вычисления historical и paraphrase metrics. Конфиг и хеши всего набора доступны валидатору входов. |
| Негативные примеры, пороги, совместимость | PASS | 151 профильный тест. Контракты query/linker, cycles, cap, доверие к navigation, invalid windows, source hashes, пустой поиск, round-robin, локальная замена seed и повтор источника проверены на fixtures. Корпусные числа получены отдельно. |
| Provenance | PASS | Для каждого выбранного пути проверены физические head/tail, признак обратного обхода, непрерывность, отсутствие повторной вершины, длина и path ID. Фактические источники сопоставлены с физическим RELEX_REL, типом и source triple. У каждого окна проверены document ID, revision, hash и точное равенство текста срезу. |
| Метрики | PASS | Из сохранённых рангов пересчитаны factual/doc recall, AllRequired, overlap, nDCG/MRR, разнообразие и повторы. Повторно проверены числители и знаменатели групп, макросредние и p50/p95 из поэтапных измерений. |
| Детерминизм | PASS | Совпали порядок всех 582 Top-5, seeds, сырые path ID digests, промежуточные диагностические данные и качественные метрики. Второй полный аудит потерь дал тот же SHA-256 af7c3706c7475913bc2eec8999a44d973abfcd8086b35a7ff7aed4816f25a6ca. |
| Предметная проверка | FAIL для promotion | Две новые явно посторонние статьи, в восстановленной выдаче C05 проверены случаи ошибочной атрибуции и необоснованной зависимости. [Разбор на уровне рангов и source triples](SOURCE_AUDIT.md) |
| Измерение production latency на Bolt, независимая оценка Precision, генерация | NOT RUN | Измерения snapshot/cache и целевой аудит имеют отдельные ограничения. Внешнее подтверждение на новом независимо размеченном наборе BLOCKED до появления разметки. |

### Входы и команды

Запуск из корня RESEARCH на Windows с локальным Python 3.11.9 проекта. Нужны tracked документы, исходные результаты 3A.5.7, их оригинальные локальные кэши ner/ и e5/ с digest из per_question.jsonl, а также каталог .cache/semantic_concept_sweep с закреплёнными файлами предшественника. Для подготовки дополнительных scores нужны обе локальные модели по путям существующих manifests. Доступ к Neo4j, сеть, seed-списки из Gold и GPU не требуются. Разрешение на работу распространяется на локальные материалы предыдущих этапов. Отдельная новая лицензия на распространение исходных abstract в этом эксперименте не устанавливалась. В evidence опубликованы ссылки через IDs и offsets.

Точные версии NER, E5, зависимостей и production-кода указаны в PROFILE.json. За исходные данные отвечают [config.json](../benchmark_3a5_2/integrated_runs/20261122T191609-112996Z/config.json) и [FROZEN_B.json](../benchmark_3a5_8/FROZEN_B.json). Хеши outputs и research-кода каждого прохода записаны в соответствующем manifest. Диагностика и первая матрица записаны перед документированными дополнениями к исследованию. Полный повтор проверяет их семантическое тождество на итоговой версии кода.

```powershell
$Py = '.\.venv\Scripts\python.exe'
$Stage = 'reports/retrieval/benchmark_3a5_9'
$Original = "$Stage/results"
$Replay = '.cache/stage_3a59_repeat_v1'

# Однократно для новых фрагментов и шести зафиксированных переформулировок.
if (!(Test-Path .cache/stage_3a59_query_scores_v1.json)) {
    & $Py -X utf8 "$Stage/inputs.py"
    if ($LASTEXITCODE -ne 0) { throw 'Query preparation failed' }
}
# Первый запуск использовал $Original, повтор — свежий $Replay.
foreach ($Phase in 'diagnosis','matrix','lexical','combined','confirmation') {
    & $Py -X utf8 "$Stage/run.py" $Phase $Replay
    if ($LASTEXITCODE -ne 0) { throw "Failed phase $Phase" }
}
& $Py -X utf8 "$Stage/audit_losses.py" "$Replay/final_losses.json"
Get-FileHash -Algorithm SHA256 "$Replay/final_losses.json"
& $Py -X utf8 "$Stage/stage_coverage.py" $Original "$Replay/stage_coverage.json"
& $Py -X utf8 "$Stage/verify.py" $Original $Replay
& $Py -X utf8 -m pytest -q tests/test_entity_linking.py tests/test_query_analysis.py tests/test_graph_retrieval.py tests/test_evidence_extraction.py tests/test_path_reranking.py tests/test_fact_aware_reranking.py tests/test_retrieval_types.py "$Stage/test_profiles.py"
```

Код завершения независимой проверки 0. Вывод pytest первого прохода — **151 passed in 6.50s**. Для очередного повтора нужен новый каталог результатов. Полный прогон моделей GLiNER/E5 на оригинальных вопросах не повторялся. Оба прохода проверяют воспроизводимость retrieval при фиксированных выходах этих моделей. В отличие от digest данных, измеренные секунды, RSS и digest конверта кэша с таймингами могут меняться между подготовками.
