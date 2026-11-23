# Общий MLflow для RESEARCH

Tracking Server развёртывается в MAIN. Сюда передаются только агрегированные результаты из доверенных отчётов. За качество данных, split и provenance отвечает исходный эксперимент. Личная информация, текст вопросов, документы, prompts, reasoning и ключи в tracking запрещены. Клиент проверяет допустимые поля и формат, но не может распознать секрет, помещённый в разрешённое текстовое поле.

Схема версии 1 и сервер описаны в MAIN в `docs/specs/RES-EVAL-001.md`, `docs/specs/tracking-schema.md` и `docs/TRACKING.md`. Для `retrieval_quality` метрики требуют commit, config hash, snapshot hash, профиль, split и источник оценки. Gold-метрики допускаются при подтверждённом качестве разметки и хеше корпуса. Для качественных метрик `llm_as_judge` нужны версии judge/generator, rubric и prompt, dataset hash, split и источник оценки.

## Подключение

Установите Python 3.11 и пакет клиента из `requirements.txt`, закреплённый за коммитом MAIN. Укажите переменные из `env.example` в окружении текущего процесса. На Windows при локальном сервере URI равен `http://127.0.0.1:5000`. Для контейнера URI зависит от сети развертывания. Файл env является образцом и не загружается автоматически. Пароли и токены туда не записываются.

```powershell
python -m pip install -r tracking/requirements.txt
$env:MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
$env:TRACKING_ENABLED = 'true'
$env:TRACKING_ALLOW_CREATE = 'false'
$env:TRACKING_OUTBOX = 'data/tracking-outbox/events.sqlite3'
python tracking/send_report.py retrieval_quality путь-к-доверенному-отчёту.json
python -m project_tracking.client replay
```

Отчёт — JSON-объект с `external_id`, `params`, `metrics` и `tags`. Опционально разрешены `status` и `parent_run_id`. Словари используют имена полей общей схемы. Возвращаются статусы `delivered`, `queued`, `disabled` и `error`. Повтор использует тот же external ID и те же данные. Для одинаковой идемпотентной привязки процессы на одном хосте должны использовать общий файл очереди. Очередь следует сохранять и регулярно повторять отправку при восстановлении сервера. Статус `error` означает, что событие могло не сохраниться. Отправку можно вызвать через `send_report.send(experiment, report)`; функция возвращает статус и не прерывает исследование при ошибке.

## Контрольный исторический импорт

```powershell
python tracking/import_stage_3a59_tracking.py .
```

Скрипт читает файлы из Git-объекта `bf541d34e34c557bd24846648a3c78f3303f78fe`, проверяет хеш сводки и согласованность манифеста с профилем. Измеренный commit — `320a85612daa414645154c64b6e340b974b0d727`. Отправляется запись `imported=true` для production C, historical HOLDOUT. Данные этого среза могут пересекаться с approved C и дают описательное сравнение. Метрики p50/p95 относятся к snapshot replay с кэшированными данными. Память является максимумом процесса. Gold-метрики опубликованы в отчёте, но не импортируются, так как наличие независимо подтверждённой разметки не установлено.

Агрегаты Stage 3A.5.10 импортируются отдельным run:

```powershell
python tracking/import_stage_3a510_tracking.py .
```

Это смешанный smoke минимального API, retrieval и генерации. Зафиксированы 10 попыток Chat Completions и usage семи ответов, 9607 входных и 762 выходных токена. Оценка 0.00189825 USD использует опубликованный тариф на 23 ноября 2026 года и только известный usage. Расход таймаута и фактический биллинг неизвестны. `estimated_groq_cost_usd` содержит оценку, `groq_cost_usd` отмечен как NOT_AVAILABLE. Профиль и fingerprint snapshot относятся к retrieval-кейсам, а не ко всем API-вызовам. В `config_hash` указан SHA-256 байтов опубликованного SPEC smoke. Скрипт сверяет хеши SPEC, RESULTS.json, отчёта и профиля в Git-коммите `85b69302f6d3c0b4a5d3630b46194f7646ddafda`. Повторного запуска моделей этот клиент не выполняет.

## Проверка подключения 24.11.2026

В отдельном окружении Python 3.11 установлен клиент MAIN f976e6e5cc9f5acb93a972d085fa001828c81827 из tracking/requirements.txt. Команда `python -m pytest tracking -q` прошла 7 тестов. Обе команды исторического импорта выше выполнены дважды. REST подтвердил два FINISHED run с imported=true и JSON artifacts, совпадающими с записанными metrics. Очередь после `python -m project_tracking.client replay` пуста. Stage 3A.5.9 — run 3a9c361e5b6c424f94fce4da0cda08d4; Stage 3A.5.10 — run 4010d0bf40454e97a25bc80b4189f78f. Массовая оценка Judge и новые измерения не выполнялись.
