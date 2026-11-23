# Верификация Stage 3A.5.10

| Критерий | Статус | Проверка |
|---|---|---|
| C1 · Идентичность retrieval C | PASS | Совпали source hashes, версии, fingerprint, SHA снимка Neo4j до/после. У live A01 и архива совпали пять путей в порядке, все физические рёбра, triple IDs, окна и prompt. Отдельно сверены 25 строк replay-индекса. |
| C2 · Provenance, offsets, дедупликация | PASS | Unit cases для ошибок offsets, revision, hash, навигации и независимых документов. Отдельная сверка всех отправленных окон с замороженными abstracts. |
| C3 · Токены и роли | PASS | Unit cases для порядка, исключений без обрезки и управляющих маркеров. Реальный tokenizer превышал usage на 183 токена во всех семи ответах API с метаданными. |
| C4 · Статусы и ссылки | PASS | Unit cases для неизвестных и исключённых ID, повторных ссылок, malformed/empty JSON, статусов. Live и replay дали четыре принятых содержательных ответа/частичных ответа со структурно действительными цитатами. В двух отрицательных live-случаях модель вернула отказ без цитат. |
| C5 · Groq, обработка ошибок и ключ | PASS | Реальный запрос и usage. HTTP mocks охватывают 401/403, 429, 5xx, timeout, malformed API payload и budget. CLI без ключа вернул код 2 с инструкцией. Первый transport, недоступная модель и один таймаут задокументированы. |
| C6 · CLI, replay и live | PASS | Live и replay через финальный CLI без ключа воспроизвели Top-5 и отправленный prompt. Реальный smoke использовал ту же точку `run_question` через GLiNER/E5, Neo4j и Groq. Сохранены полный prompt, контекст, provenance, времена и usage. |
| C7 · Ручная оценка примеров | PASS | Проверены атрибуция, неподтверждённые расширения, ограничения и отказы. Полный ответ с двумя подтверждёнными вкладами на выбранных cross-document вопросах **FAIL**. Полная factual accuracy и сравнительная эффективность **NOT RUN**. |
| C8 · План следующего этапа | PASS | Зафиксированы предложенные правила корпуса, построения вопросов, изоляции, метрик и согласования лимитов. Массовый запуск **NOT RUN**. |

## Команды и вид доказательств

```powershell
$Py = '.\.venv\Scripts\python.exe'
& $Py -X utf8 -m pytest -q tests/test_llm_qa.py tests/test_entity_linking.py tests/test_query_analysis.py tests/test_graph_retrieval.py tests/test_evidence_extraction.py tests/test_path_reranking.py tests/test_fact_aware_reranking.py tests/test_retrieval_types.py reports/retrieval/benchmark_3a5_9/test_profiles.py --junitxml=.cache/stage_3a5_10_final_tests.xml
uv lock --check --offline
& $Py -X utf8 scripts/smoke_retrieval_3a1.py
& $Py -X utf8 reports/retrieval/benchmark_3a5_10/smoke.py artifacts/stage_3a5_10_smoke_v3
& $Py -X utf8 reports/retrieval/benchmark_3a5_10/verify.py artifacts/stage_3a5_10_smoke_v3 artifacts/stage_3a5_10_verification.json --earlier-minimal artifacts/stage_3a5_10_smoke_v1/minimal.json --earlier-minimal artifacts/stage_3a5_10_smoke_v2/minimal.json --final-cli artifacts/3a5_10_cli_missing_key.json --final-cli artifacts/3a5_10_cli_live_no_key.json --junit .cache/stage_3a5_10_final_tests.xml
```

Команды live используют заранее заданные NEO4J_* и Groq-настройки. В runs v1/v2 был исходный Llama preset, подробности доступны в [smoke-отчёте](SMOKE_RESULTS.md). У v3 ненулевой код завершения из-за одиночного таймаута, при этом остальные результаты записаны. `verify.py` проверяет артефакты отдельным проходом. [RESULTS.json](RESULTS.json) содержит сводку JUnit, usage и структурные результаты. Тесты pytest используют mocks и frozen файлы. Реальные данные и сетевые вызовы представлены отдельными smoke-артефактами.

## Отдельное кодовое ревью

Проверены границы API-клиента, точный текст, восстановление ссылок, вывод CLI, доступ к графу и места записи. После review replay переведён на отдельный индекс без Gold. Валидация отклоняет ложные inline ID и некорректный Unicode. Матрица использована отдельной проверкой для сравнения всех 25 строк индекса. Live A01 и архивный A01 сформировали одинаковый prompt до этой правки. Финальный CLI в обоих режимах воспроизвёл эти пути и prompt без обращения к API. Сетевые артефакты сохранены из этапа smoke. Лимит 10 попыток исчерпан.

Проверены staged path list, diff и отсутствие Groq token, Authorization value, локальных `.env`, весов и исходных временных артефактов. Рабочий `query_analysis.py` соответствует хешу профиля в CRLF, blob Git хранит тот же текст в LF. Проверка исходников допускает только эквивалентные окончания строк Python. Хеши корпуса, индекса и артефактов проверяются по исходным байтам. Пользовательские изменения и Stage 3A.5.9 исключены из commit.
