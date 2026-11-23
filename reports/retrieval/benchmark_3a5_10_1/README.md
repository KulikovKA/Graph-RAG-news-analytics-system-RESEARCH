# Stage 3A.5.10.1 · persistent runtime

Два warm-запроса на live Neo4j подготовили prompt за **8,36 и 8,35 с** после **31,40 и 31,37 с** в исходном pipeline. На каждом из трёх входов совпали упорядоченные seeds после cap, Top-5 paths с provenance и байты prompt. Результаты относятся к CPU-процессу с замороженным профилем C. [Спецификация](SPEC.md) · [Машиночитаемая сводка](RESULTS.json)

## Запуск

Подготовьте окружение, файлы моделей и настройки Neo4j из предыдущего этапа. Команды ниже выполняются из корня checkout. [Конфигурация Stage 3A.5.10](../benchmark_3a5_10/LLM_INTEGRATION.md)

```powershell
$Py = '.\.venv\Scripts\python.exe'
& $Py -X utf8 -m cgraph.llm_qa --interactive
```

Вводите по одному вопросу на строку. Первый вопрос загружает GLiNER и E5. Последующие вопросы используют модели, их tokenizer, готовые QueryAnalyzer, EntityLinker, entity cache, Harmony tokenizer и тот же driver Neo4j. Для сохранения JSON используйте `--output` с существующим каталогом. CLI запишет `question-0001.json` и следующие файлы без перезаписи. В интерактивных результатах видны `session_query`, суммарный `peak_rss_bytes` и `runtime_setup` у первого вопроса. Ошибка отдельного вопроса возвращается в его JSON, сессия принимает следующий ввод.

Команда `exit`, EOF или Ctrl+C закрывают Neo4j driver и освобождают ресурсы сессии. При следующем запуске процесса модели снова загружаются. Одиночный `--question` создаёт сессию на один запрос. Архивный `--question … --replay` использует прежний маршрут.

## Замеры

Обе ветви получили три вопроса в одной и той же последовательности. В исходном процессе каждый вопрос заново создавал компоненты, а импорт Python-библиотек после первой строки уже был прогрет. В новом процессе первый вопрос был cold, следующие два warm. End-to-end в таблице измерен до готового prompt, в секундах. Столбцы со стадиями показывают **исходный → persistent**.

| Вопрос | Runtime | Query analysis | Entity linking | Retrieval целиком | End-to-end без API | Peak RSS, GiB |
|---|---|---:|---:|---:|---:|---:|
| A01 | cold | 29,676 → 29,970 | 0,520 → 0,487 | 44,052 → 43,410 | 45,705 → 45,135 | 2,043 → 2,044 |
| B01 | warm | 19,535 → 0,504 | 0,575 → 0,045 | 29,259 → 8,344 | 31,403 → 8,363 | 2,145 → 2,044 |
| C09 | warm | 20,373 → 0,458 | 0,518 → 0,055 | 29,474 → 8,333 | 31,372 → 8,354 | 2,162 → 2,044 |

| Наблюдение | Исходный | Persistent |
|---|---:|---:|
| Первый импорт GLiNER, с | 9,412 | 9,353 |
| Первый загрузчик GLiNER `AutoExtractor.from_pretrained`, с | 18,380 | 18,934 |
| Первая загрузка E5, включая tokenizer и SHA-проверки, с | 0,450 | 0,407 |
| Вызовы загрузчиков GLiNER / E5 | 3 / 3 | 1 / 1 |

Первый запрос новой сессии занял 45,14 с, включая 5,01 с на инициализацию runtime и 29,97 с на анализ вопроса. В исходном процессе повторные загрузки GLiNER заняли 17,88 и 18,48 с. Это основная измеренная причина выигрыша. Сохранялись проверки снимка Neo4j до и после каждого запроса. Вместе они заняли около 5 с в каждой warm-строке, отдельно от BFS. Повторное использование Harmony tokenizer сократило построение контекста с 1,90–2,14 с до 0,019–0,021 с. Значения 67,94 и 87,47 с из исходного наблюдения относятся к другому замеру. Три пары показывают эффект на этих запросах. Долгосрочный профиль памяти, latency Groq и распределение по корпусу имеют статус **NOT RUN**. Вызовов Groq — **0**.

## Проверка и воспроизведение

| Критерий | Статус | Доказательство |
|---|---|---|
| R1 · Модели и cache живут в сессии | PASS | Один реальный вызов каждого загрузчика, unit-проверка переиспользования объектов после ошибки. |
| R2 · Очистка ресурсов | PASS | Unit-проверки `exit`, EOF, Ctrl+C, ошибки и дрейфа графа. Live CLI закончил работу после двух вопросов и `exit`. |
| R3 · Паритет | PASS | Все три пары совпали по seeds, paths, triple IDs, source offsets и сообщениям. Отдельная проверка сверила их с frozen архивом и исходным корпусом. |
| R4 · Метрики | PASS | Обе ветви измерены с реальными моделями и Bolt, полные timings, loader spans и peak RSS зафиксированы. |
| R5 · CLI | PASS | Реальный интерактивный запуск и отдельный one-shot replay воспроизвели контекст. Ключ был удалён из окружения этих проверок, оба маршрута остановились перед отправкой в Groq. Одиночный live route проверен тестом lifecycle. |

Прошли **206 pytest-тестов**. Отдельное ревью сверило границы закрытия driver, lazy-init компонентов, сохранение проверок Neo4j, отсутствие изменений pinned retrieval и точность парной проверки. Main benchmark измерял тот же runtime, который использует интерактивная команда. Интерактивный smoke отдельно сверил два результата с benchmark. Фиксация cold/warm использует наблюдения отдельного процесса, Windows peak working set отражает максимум процесса с начала измерения.

```powershell
$Py = '.\.venv\Scripts\python.exe'
$Stage = 'reports/retrieval/benchmark_3a5_10_1'
& $Py -X utf8 "$Stage/measure.py" baseline artifacts/stage_3a5_10_1_baseline.json
& $Py -X utf8 "$Stage/measure.py" persistent artifacts/stage_3a5_10_1_persistent.json
& $Py -X utf8 -m pytest -q tests/test_llm_runtime.py tests/test_llm_qa.py tests/test_entity_linking.py tests/test_query_analysis.py tests/test_graph_retrieval.py tests/test_evidence_extraction.py tests/test_path_reranking.py tests/test_fact_aware_reranking.py tests/test_retrieval_types.py reports/retrieval/benchmark_3a5_9/test_profiles.py --junitxml=.cache/stage_3a5_10_1_final_tests.xml
& $Py -X utf8 "$Stage/verify.py" artifacts/stage_3a5_10_1_baseline.json artifacts/stage_3a5_10_1_persistent.json artifacts/stage_3a5_10_1_verified.json --junit .cache/stage_3a5_10_1_final_tests.xml
```

Измерение использует существующие `NEO4J_*` или `--neo4j-env` с утверждённым файлом и настройки `LLM_*` для построения контекста. Для проверки интерактивного smoke передайте в `verify.py` также `--cli-output` с каталогом из двух JSON (A01 и B01 в этом порядке). Полные локальные артефакты содержат исходные окна и prompts. Сводка фиксирует SHA этих файлов, хеши вывода и версии, не дублируя исходные тексты.
