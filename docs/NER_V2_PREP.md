# NER-V2-PREP

Подготовка выполнена 31 октября 2026 года в `design-v2-sequential-kg`. Полная
архитектура остаётся проектом; Ontology v2.0 утверждена: **11 типов / 20 отношений**.
GLiNER forward pass, NER benchmark/inference, Qwen, RE, verification, resolution,
embedding inference, обучение и создание Gold на этом этапе не выполнялись.

## Код и контракты

Изолированный пакет `src/cgraph/ner_v2/`:

| Модуль | Назначение |
|---|---|
| `model_setup.py` | Скачать и проверить pinned файлы, собрать offline runtime; отдельная lazy-функция загрузки для будущего запуска |
| `detection.py` | Будущее mention detection, контроль budget, exact offsets, сохранение пересечений и предварительных hints |
| `normalization.py` | Контекстная tokenization/POS/morphology/lemmatization и локальные определения сокращений |
| `io.py` | Source-only JSONL, проверка hash/offsets/provenance и атомарная запись результатов |
| `openalex.py` | Поиск Field IDs, seeded выборка, проверка inverted index, воспроизводимый offline replay |
| `benchmark.py` | Будущий последовательный CPU-запуск трёх моделей с общим протоколом |
| `cli.py` | Подготовка и валидация; отдельный явный opt-in для будущего inference |

Старые модули `cgraph.ner`, `cgraph.model_setup`, ontology v1.0, Gold, C и B-001
не изменены. Повторно используется чистая функция `cgraph.ner.windows`.
Из старого downloader перенесён подход: pinned backbone config и tokenizer
в локальном runtime, хеши файлов и hardlink весов; новые файлы находятся отдельно.
Версии всех исторических зависимостей в `uv.lock` сохранены; optional extra
`benchmark` согласован с уже существовавшей записью старого lock. Новое окружение
использует только extras `ner-v2` и `test`, прежняя `.venv` не изменяется.

Нормативная схема: `contracts/ner_v2/ontology.v2.json`. Русские определения
сохранены дословно из задания, английские описания служат общим label input GLiNER.
Схемы JSON: `source.schema.json` и `mention.schema.json` в том же каталоге.
Исполняемые проверки `io.py` дополнительно проверяют семантические инварианты:
hash текста, соответствие source reference и `text[start:end] == surface_text`.
Все offsets — полуинтервалы Unicode codepoints относительно неизменённого `text`.

Выход mention: `document_id`, nullable `revision_id`, `mention_id`, `surface_text`,
`start`, `end`, `normalized_form`, `lemma`, `ner_label_hint`, `score`,
`model_identifier`, `source_reference`; дополнительно — tokens с POS/morphology,
кандидаты сокращений и версия normalizer. Это предварительные кандидаты;
окончательного типа и canonical entity ID здесь нет.

Одинаковый span с разными hints сохраняется как несколько кандидатных записей.
Повтор одного span/hint в перекрывающихся окнах оставляет максимальный score.
Совпадение normalized form, lemma или сокращения не объединяет объекты.
Mention ID детерминирован по документу, текстовой редакции, span, hint и модели.

## NLP и общий протокол

Выбраны **spaCy 3.8.18 + en_core_web_sm 3.8.0**. Используются tokenizer,
`tok2vec`, `tagger`, `attribute_ruler`, `lemmatizer`; `ner` и `parser` исключены.
Это компактный английский pipeline для CPU с контекстным POS и морфологией,
без отдельной transformer-модели нормализации. Наличие компонентов описано в
[официальной документации spaCy](https://spacy.io/models/en), POS/morphology и
лемматизация — в [Linguistic Features](https://spacy.io/usage/linguistic-features).
Сравнение скорости/качества со Stanza или scispaCy не проводилось; выбор не
является экспериментальным утверждением о превосходстве на научных текстах.

Лемматизируются обычные lowercase ASCII слова; proper names, формулы,
аббревиатуры, mixed-case, символы и частичные токены сохраняются консервативно.
Специализированный термин может остаться неизменённым либо получить ошибочный POS;
это ограничение требует последующей оценки, а не скрытого исправления по Gold.
Притяжательные суффиксы удаляются только при `tag_ == POS`, с защитой парных
кавычек. Полная морфология и оригинальные token spans сохраняются отдельно.
Нормализация NFC и whitespace затрагивает только производную форму; NFKC,
casefold и переписывание исходного текста не применяются. `NH₃ != NH3`, `pH != ph`.

Abbreviation detection — ограниченное локальное выравнивание символов по принципу
Schwartz–Hearst, включая `long form (SHORT)` и `SHORT (long form)`. Сохраняются
точные spans и все неоднозначные определения. Это candidate scope на документ,
не coreference/resolution и не автоматическая замена сокращения длинной формой.

Общие настройки всех моделей находятся в `config/ner_v2.json`:

- Все 11 `ner_description_en` из Ontology v2.0, в фиксированном порядке.
- Threshold `0.35`; окна `60` whitespace-слов, overlap `20`, без изменения source.
- `flat_ner=False`, `multi_label=True`; одинаковая сортировка и дедупликация.
- Budget `512` prompt+text subword tokens; перед будущим forward проверяется
  фактическая GLiNER tokenization. Превышение останавливает запуск. Протокол
  корректируется для всех трёх моделей совместно, не под отдельный checkpoint.
- CPU, 4 потока, seed 42, deterministic torch algorithms, без fine-tuning.

Архитектура остаётся последовательной:
Mention Detection → Linguistic Normalization → будущие Qwen Entity Typing →
будущие Qwen Relation Extraction. Окончательная типизация планируется Qwen 4B.

## Модели и offline-файлы

До загрузки проверено примерно 235 ГБ свободного места. Downloader также
проверяет запас для partial-файлов и copy fallback перед каждым полным запуском.
Использованы официальные репозитории автора:
[large](https://huggingface.co/urchade/gliner_large-v2.1),
[medium](https://huggingface.co/urchade/gliner_medium-v2.1),
[small](https://huggingface.co/urchade/gliner_small-v2.1).

| Checkpoint | Commit SHA | Файл весов | Размер весов, bytes |
|---|---|---|---:|
| `urchade/gliner_large-v2.1` | `abd49a1f1ebc12af1be84d06f6848221cf96dcad` | `pytorch_model.bin` | 1782044554 |
| `urchade/gliner_medium-v2.1` | `40ec419335d09393f298636f471328b722c6da9e` | `model.safetensors` | 780735696 |
| `urchade/gliner_small-v2.1` | `4e091416cf7c3481db542c2a3d26156916f3a47f` | `pytorch_model.bin` | 610652234 |

Локальный корень — `artifacts/ner_v2/models/`:
`gliner_large`, `gliner_medium`, `gliner_small`; tokenizer/backbone metadata —
`backbone_large`, `backbone_medium`, `backbone_small`; готовые offline-каталоги —
`runtime_large`, `runtime_medium`, `runtime_small`.

Pinned backbone revisions:

- `microsoft/deberta-v3-large`: `64a8c8eab3e352a784c658aef62be1662607476f`.
- `microsoft/deberta-v3-base`: `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`.
- `microsoft/deberta-v3-small`: `a36c739020e01763fe789b4b85e2df55d6180012`.

Скачаны `gliner_config.json`, реальные веса, README, `config.json`,
`tokenizer_config.json` и `spm.model` для каждого backbone. Отдельные pretrained
backbone weights не нужны: encoder weights уже входят в GLiNER checkpoint.
Для medium скачан один формат весов, без дублирующего `.bin` и ONNX.
Всего pinned исходных файлов — **3180854257 bytes**; runtime использует hardlinks
для весов, где это поддерживается. Исторический cache переиспользовался только
при совпадении размера и upstream digest, без изменения старых каталогов.

`contracts/ner_v2/model_lock.json` фиксирует repo IDs, commit SHA, имена,
размеры и upstream LFS SHA-256 либо Git blob SHA. Полные локальные SHA-256 и
версии основных библиотек — `reports/ner_v2/model_inventory.json`.
Runtime config включает pinned `encoder_config`; tokenizer находится рядом.
Будущая загрузка задаёт `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`local_files_only=True`, `map_location=cpu`, `strict=True`.
Веса не десериализовались и GLiNER instance не создавался: проверены файлы,
конфигурации, совместимость зависимостей и offline tokenizers всех трёх моделей
на синтетическом тексте (`reports/ner_v2/offline_tokenizer_check.json`).
Проверочный prompt с окном 60 слов даёт 214 tokens при общем лимите 512.
Transformers 5.1.0 может выдавать Mistral regex warning при чтении этого локального
DeBERTa tokenizer: его проверка использует широкий vocabulary heuristic. Здесь
backend — Unigram/Metaspace; Mistral regex patch не применяется, pinned tokenizer
сохраняется. Передача `fix_mistral_regex` в этой версии вызывает duplicate-keyword
ошибку библиотеки, поэтому соответствующий флаг не добавлен.
Полная runtime-проверка весов отложена до разрешения
inference. `artifacts/`, `data/`, `.venv-ner-v2/` и веса исключены из Git.

## OpenAlex

Изучен исходный репозиторий на commit
[`169834dec45b527f078620bd429d7286eb2700a1`](https://github.com/KulikovKA/Graph-based-RAG-for-article-analysis/tree/169834dec45b527f078620bd429d7286eb2700a1):

- [`integrations/openalex.py`](https://github.com/KulikovKA/Graph-based-RAG-for-article-analysis/blob/169834dec45b527f078620bd429d7286eb2700a1/src/app/integrations/openalex.py)
  канонизирует `W…`, читает metadata и восстанавливает abstract с запретом gaps,
  duplicate positions и malformed offsets.
- [`workers/ingest.py`](https://github.com/KulikovKA/Graph-based-RAG-for-article-analysis/blob/169834dec45b527f078620bd429d7286eb2700a1/src/app/workers/ingest.py)
  преобразует работу в источник `openalex` с разделом `abstract`, DOI и landing URL.
- [`services/ingestion.py`](https://github.com/KulikovKA/Graph-based-RAG-for-article-analysis/blob/169834dec45b527f078620bd429d7286eb2700a1/src/app/services/ingestion.py)
  разделяет документ и редакцию, считает content hashes и сохраняет точные offsets.

Новый корпус получен из официального `https://api.openalex.org`, отдельно от
старых 100 работ и Pilot24. Механика запросов — в
[официальной документации OpenAlex](https://help.openalex.org/api/).
Field IDs не угадывались: выполнен поиск `/fields?search=…`. History отсутствует
как field; найдено подполе `/subfields?search=History`, затем его родительский field.

| Область | Найденный Field | Два OpenAlex work ID |
|---|---|---|
| Computer Science / Machine Learning | `17`, Computer Science | `W3195791401`, `W4312890372` |
| Medicine | `27`, Medicine | `W4387756919`, `W2092571965` |
| Physics | `31`, Physics and Astronomy | `W7130738450`, `W2020268674` |
| Biology | `13`, Biochemistry, Genetics and Molecular Biology | `W3182220660`, `W2355493570` |
| Economics | `20`, Economics, Econometrics and Finance | `W7220014427`, `W2130076420` |
| History | `12`, Arts and Humanities; subfield `1202`, History | `W2057261224`, `W2073881503` |

Отбор: один seeded sample 30 на область (`seed=42`, `has_abstract`, `language:en`,
`type:article`); локально source type journal/conference, корректный abstract
80–600 слов, сортировка ID, seeded shuffle, первые две разные primary topics.
Глобальная дедупликация по OpenAlex ID. Field/subfield проверены по primary topic.
Это небольшой междисциплинарный pilot; metadata OpenAlex не гарантирует чистоту
дисциплинарной классификации или экспертную оценку каждой публикации.

Выполнено 14 успешных запросов при формировании окончательной выборки:
6 поисков fields, поиск History subfield, получение родительского field и 6
выборок works. Отдельные предварительные проверки API не включены в acquisition log.
Timeouts 10/30 секунд, максимум 3 попытки, небольшая пауза, ограниченный Retry-After.
Необязательный `OPENALEX_API_KEY` читается только из environment; ключ не включается
в manifest, исключения или логи. Недоступность API вызывает явную ошибку, без fake text.
PDF не скачивались.

JSONL: **`data/ner_v2/openalex_12.jsonl`**. `text` содержит только восстановленную
аннотацию, без добавления title; metadata хранится отдельно. `revision_id` — SHA-256
этого текста, а дата обновления metadata хранится отдельно в source reference.
Исходные выбранные metadata/inverted indexes, field lookup results, параметры,
candidate IDs и fingerprints сохранены в `contracts/ner_v2/openalex_snapshot.json`.
Небольшой snapshot хранится в Git, JSONL восстанавливается из него без сети.
Полные ответы candidate pools лежат локально в `artifacts/ner_v2/openalex/`.
Seed гарантирует порядок на замороженном наборе; повторное обращение к живому
OpenAlex со временем может дать другой набор. Для нового отбора нужен `--refresh`.
Названия, DOI, topics и hashes всех 12 работ — `reports/ner_v2/corpus_manifest.json`.

## Воспроизведение и проверки

Прогресс отображается через `tqdm` 4.70.1: файлы моделей и объём загрузки в bytes,
проверка offline-каталогов/tokenizers, получение и подготовка аннотаций. В будущем
benchmark будут вложенные индикаторы моделей, документов и окон с percentage,
скоростью и ETA; во время загрузки модели показывается статус `loading`.
Индикаторы пишутся в stderr и автоматически отключаются при перенаправлении вывода
без терминала. JSONL, параметры извлечения и контракты данных не меняются.

PowerShell, из корня репозитория:

```powershell
$env:UV_PROJECT_ENVIRONMENT = '.venv-ner-v2'
uv sync --locked --extra ner-v2 --extra test
$py = '.\.venv-ner-v2\Scripts\python.exe'
& $py -m cgraph.ner_v2.cli prepare-models
& $py -m cgraph.ner_v2.cli prepare-corpus
& $py -m cgraph.ner_v2.cli validate-input
& $py -m cgraph.ner_v2.cli verify-models
& $py -m cgraph.ner_v2.cli verify-tokenizers
& $py -m pytest tests/test_ner_v2.py tests/test_contracts.py -q
```

Технические проверки: **39 новых + 19 существующих = 58 passed**.
Тесты используют синтетические строки, статические detector-output fixtures и
замороженные OpenAlex metadata; GLiNER inference не вызывается. Проверены NFC,
формулы/регистр, plural/possessives/quotes, POS/morphology, abbreviations и ambiguity,
overlaps, повторные Unicode spans, JSONL, schemas, hashes, seeded selection,
offline replay, ограниченные API retries, отсутствие ключей в логах и CLI guard.
`tests/test_training.py` не запускался: он обучает classifier и выполняет predictions,
что не входит в разрешённый подготовительный этап.

Будущий benchmark, **не запускался**:

```powershell
.\.venv-ner-v2\Scripts\python.exe -m cgraph.ner_v2.cli benchmark --allow-inference --output artifacts/ner_v2/benchmark_run1
```

Он последовательно загрузит large/medium/small локально, запишет mentions JSONL,
общий manifest, load time, detection+normalization time, throughput и попарное
span agreement. Без v2 Gold agreement не является quality score; precision/recall/F1
в этом benchmark не вычисляются. Существующий output directory не перезаписывается.
После NER-V2-PREP работа остановлена до следующего задания на извлечение.
