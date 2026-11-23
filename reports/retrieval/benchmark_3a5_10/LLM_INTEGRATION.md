# Запуск Stage 3A.5.10

Из корня репозитория используйте его Python 3.11. Для live нужны зависимости frozen PROFILE.json, локальные GLiNER 2.5 Base и E5-small-v2 cache и доступ к замороженному Neo4j 2G.3 через `NEO4J_*` или `--neo4j-env`. Для replay нужны файлы Stage 3A.5.9, корпус, document lock и пакет `tokenizers`. Адаптер вызывает OpenAI-compatible API через зафиксированный HTTPX 0.28.1.

Live smoke выполнен в действующем исследовательском venv с Python 3.11.9 и версиями из [PROFILE.json](../benchmark_3a5_9/PROFILE.json). Общий lock проекта закрепляет `huggingface-hub==1.28.0`, а профиль live — `1.33.0`. Для live используйте окружение Stage 3A.5.9, затем запускайте его Python напрямую или через `uv run --no-sync`. Replay работает без моделей и Neo4j.

Проверенный preset в [example config](../../../.env.example) задаёт `openai/gpt-oss-120b`, JSON Object Mode, temperature 0, reasoning effort low, лимиты 6000/1000 и timeout 30 секунд. Параметры описаны в [Groq model card](https://console.groq.com/docs/model/openai/gpt-oss-120b) и [Groq API reference](https://console.groq.com/docs/api-reference). Максимум контекста модели — 131072, максимум completion — 65536. При смене GPT-OSS модели задайте совместимый tokenizer и параметры. Текущий счётчик использует [Harmony](https://developers.openai.com/cookbook/articles/openai-harmony), официальный tokenizer из закреплённой ревизии и резерв 256 токенов для возможных служебных добавлений провайдера. Точный серверный расход берётся из API usage. CLI не выбирает резервную модель автоматически.

```powershell
$Py = '.\.venv\Scripts\python.exe'
New-Item -ItemType Directory -Force .cache,artifacts | Out-Null
curl.exe -fsSL 'https://huggingface.co/openai/gpt-oss-120b/resolve/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/tokenizer.json' -o .cache/stage_3a5_10_gpt_oss_tokenizer.json
# Загрузить проверенный preset без учётных данных, затем ввести ключ без эха.
Get-Content .env.example | Where-Object { $_ -match '^LLM_' } | ForEach-Object {
    $Name, $Value = $_ -split '=', 2
    [Environment]::SetEnvironmentVariable($Name, $Value, 'Process')
}
$Secret = Read-Host 'GROQ_API_KEY' -AsSecureString
$env:GROQ_API_KEY = [System.Net.NetworkCredential]::new('', $Secret).Password
$Question = 'What limits microbial production of limonene?'
& $Py -X utf8 -m cgraph.llm_qa --question $Question --replay --output artifacts/3a5_10_replay.json
& $Py -X utf8 -m cgraph.llm_qa --question $Question --neo4j-env <existing-env> --output artifacts/3a5_10_live.json
& $Py -X utf8 reports/retrieval/benchmark_3a5_10/smoke.py artifacts/stage_3a5_10_new_smoke --neo4j-env <existing-env>
Remove-Item Env:GROQ_API_KEY
```

Подставьте путь к существующему `--neo4j-env` в кавычках либо удалите аргумент при уже заданных `NEO4J_*`. Локальный файл `.env` исключён из Git. Настройки CLI читаются из окружения процесса. `LLM_BASE_URL` — доверенный адрес Groq или заданный оператором доверенный HTTPS-прокси, которому разрешено получить ключ и документы. HTTP-редиректы запрещены. CLI возвращает код 0 для ANSWERED, PARTIAL и обоснованного INSUFFICIENT_EVIDENCE, код 2 при ошибке. Если нет ключа при непустом контексте, результат содержит `missing_GROQ_API_KEY`.

Таймаут действует на сетевые операции. 429, 5xx и временные сетевые сбои допускают не более `LLM_MAX_RETRIES` (0–3) дополнительных отправок. Экспоненциальная задержка с jitter ограничена. Числовой Retry-After до 10 секунд учитывается. Если сервер просит более долгую паузу, управление возвращается вызывающей стороне. Авторизация, невалидный JSON, неверная модель и исчерпание бюджета завершают запрос. Даже после сетевого таймаута попытка могла быть учтена провайдером. Значения usage относятся к полученному ответу, по попыткам без usage расход неизвестен.

Поля `context.sources` описывают все переданные окна, `citations` — процитированные. `context.messages` сохраняет точный prompt для повторения. `llm.attempts` показывает число отправок, `request_id` берётся из API и может отсутствовать при сетевом сбое. `timings_seconds` разделяет retrieval, его подстадии, context building, LLM и общее время. Полнота и научная обоснованность требуют чтения источников по `source_url`, offsets и hash. JSON хранит контекст исходных текстов и пользовательский вопрос. Публиковать такой файл следует только в пределах разрешённых условий использования этих данных.
