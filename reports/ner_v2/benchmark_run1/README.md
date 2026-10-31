# NER v2: benchmark_run1

Результаты benchmark, выполненного пользователем 31 октября 2026 года.
Исходный локальный каталог: `artifacts/ner_v2/benchmark_run1/`.
При архивировании inference повторно не запускался.

| Модель | Загрузка, с | Detection + normalization, с | Записей mentions | Уникальных spans | Документов с mentions |
|---|---:|---:|---:|---:|---:|
| large | 14.866 | 97.648 | 99 | 95 | 12/12 |
| medium | 5.899 | 31.789 | 54 | 47 | 10/12 |
| small | 3.918 | 15.895 | 25 | 25 | 9/12 |

Один span может иметь несколько предварительных hints. Их записи считаются
отдельно; они не означают разные canonical entities. Попарный Jaccard по spans:
large/medium — 0.2909, large/small — 0.1765, medium/small — 0.3846.
Это agreement и наблюдаемая скорость одного запуска. Gold v2 отсутствует,
поэтому precision/recall/F1 и вывод о лучшей по качеству модели не устанавливаются.

Архив содержит:

- `manifest.json` — побайтовая копия исходного manifest с настройками, revisions,
  версиями библиотек, timings и agreement.
- `large.jsonl`, `medium.jsonl`, `small.jsonl` — побайтовые копии предсказаний.
- `input_abstracts.jsonl` — точная копия использованных 12 входных аннотаций.
- `validation.json` — проверка hash входа/онтологии/model lock, всех 178 source spans,
  контрактов, уникальности mention IDs, model identifiers и пересчёта agreement.
- `article_links.md`, `article_links.json` — ссылки на все публикации, найденные
  открытые версии и metadata OpenAlex на момент проверки ссылок.

Пути внутри исходного manifest сохранены как были при запуске. Для просмотра
архива входные тексты доступны в `input_abstracts.jsonl`; SHA-256 этого файла
совпадает с `input_sha256` исходного manifest. Commit исполняемого кода не был
записан benchmark, поэтому `validation.json` отдельно фиксирует HEAD при
архивировании и не выдаёт его за подтверждённый execution commit.

Новые результаты находятся отдельно от исторических C/B-001. Веса, окружения,
PDF и большие локальные артефакты в этот архив не включены.
