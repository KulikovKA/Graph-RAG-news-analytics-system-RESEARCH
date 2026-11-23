# Stage 3A.5.10 · Graph-RAG C + Groq

Команда `python -m cgraph.llm_qa --question "…"` выполняет live retrieval в замороженном Neo4j, упаковывает оригинальные окна и вызывает `openai/gpt-oss-120b` в Groq. Флаг `--replay` использует одну из 25 проверенных выдач Stage 3A.5.9. Цитаты разрешаются до URL оригинала, document ID, ревизии и абсолютных offsets.

[Запуск и конфигурация](LLM_INTEGRATION.md) · [Спецификация](SPEC.md) · [Архитектура](ARCHITECTURE.md) · [Шаблоны](PROMPTS.md)

[Локальные тесты и приёмка](TEST_REPORT.md) · [Live smoke и ручная проверка ответов](SMOKE_RESULTS.md) · [Машиночитаемый отчёт](RESULTS.json) · [План на 100 документов](STAGE_3A5_11_PLAN.md)
