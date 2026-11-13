# Stage 2F.4 — Local GLiREL Evaluation

Дата клиента: 2026-11-13 (Europe/Moscow). Source HEAD `8175e8e`, ветка `design-v2-sequential-kg`. До эксперимента отслеживаемые файлы не изменены; скачанная пользователем `model/` оставалась untracked. Это последний ограниченный модельный тест перед отдельным решением о расширении корпуса; расширение не выполнялось.

## Результат

GLiREL: exact sanity hits при >=0.35 — **0/6**; междокументных outputs при >=0.35 — **0**, P-пар с outputs — **0/6**. Содержательно принятых cross outputs — **0**. Все три заданных порога показаны ниже. Положительного улучшения относительно ReLEx S2 не продемонстрировано.

Результаты относятся только к установленной GLiREL 1.2.1, локальному `jackboyla/glirel-large-v0`, фиксированным девяти prompts онтологии D и сохранённым научным примерам. Пороги диагностические; scores разных моделей не считаются одинаково калиброванными. P01–P06 не являются полноценным gold benchmark, F1 не вычислялся.

| Набор | Порог | Outputs | Пары с output | Вердикты |
|---|---:|---:|---:|---|
| sanity | 0.35 | 0 | 0 | нет положительных outputs |
| sanity | 0.50 | 0 | 0 | нет положительных outputs |
| sanity | 0.70 | 0 | 0 | нет положительных outputs |
| cross | 0.35 | 0 | 0 | нет положительных outputs |
| cross | 0.50 | 0 | 0 | нет положительных outputs |
| cross | 0.70 | 0 | 0 | нет положительных outputs |

## Локальная модель, окружение и диагностика загрузки

Использована папка `C:\Users\Kirill\Desktop\Graph-RAG-news-analytics-system-RESEARCH\model` с glirel_config.json и pytorch_model.bin. Веса прочитаны на месте; не перемещались, не перезаписывались и повторно не скачивались. SHA-256 обоих файлов до/после сохранены в manifest.json. Размер checkpoint: 1866504702 bytes. Предусмотренное альтернативное расположение `%USERPROFILE%/glirel-test/model` не потребовалось; окружение `%USERPROFILE%/glirel-test/.venv` при диагностике не найдено.

Python: `C:\Users\Kirill\Desktop\Graph-RAG-news-analytics-system-RESEARCH\.venv\Scripts\python.exe`. Версии: {'glirel': '1.2.1', 'torch': '2.11.0', 'transformers': '5.1.0', 'huggingface-hub': '1.33.0', 'tqdm': '4.70.1', 'loguru': '0.8.0'}. В основном .venv уже был glirel, но отсутствовал импортируемый им loguru (пакет не объявлен в glirel Requires-Dist). loguru и его небольшие зависимости установлены только в `.cache/glirel_runtime` через pip --target; .venv, pyproject.toml и uv.lock не менялись. Flair этой установленной версии не нужен: её TokenRepLayer использует собственный TransformerWordEmbeddings.

Обнаружены две проблемы загрузки, устранённые только в процессе:

- GLiREL._from_pretrained требует устаревшие proxies/resume_download, которые установленный Hub mixin уже не передаёт. Scoped wrapper передаёт None/False внутреннему loader; файлы пакетов не изменены.
- Локально есть DeBERTa-v3-large config и SentencePiece tokenizer, но отдельных базовых весов нет. Encoder создаётся из локального config, после чего публичный GLiREL.from_pretrained загружает **весь** GLiREL state_dict с strict=True и assign=True. Это не случайно инициализированная модель для inference: все encoder/head параметры заменены checkpoint-весами; missing/unexpected keys нет. Дополнительный backbone checkpoint не скачивался.

Загружены 420 state_dict entries, 466577920 параметров; tokenizer vocabulary 128005. CPU, 2 threads, seed=0, deterministic algorithms, eval/no_grad. Две ранние проверки public loader остановились до создания модели из-за Hub signature; успешная проверка загрузки до эксперимента не выполняла inference. Итоговый исследовательский inference содержит ровно 34 encoder forwards.

Loader также печатает config.json not found: у checkpoint конфигурация называется glirel_config.json, она успешно прочитана. Transformers 5.1.0 вывел generic warning о Mistral regex при чтении DeBERTa tokenizer. Дополнительная проверка фактически показала DebertaV2Tokenizer, Unigram SentencePiece и Metaspace pre-tokenizer, а не Mistral tokenizer. Предупреждение зафиксировано; автоматическая правка regex и замена версии пакета не проводились. Совместимость tokenization с исходным обучающим стеком не доказана только успешной загрузкой, поэтому отрицательный результат нельзя переносить на все версии GLiREL. Все actual token/mention и subword/word alignments в этом запуске прошли проверки.

## Адаптер и отсутствие leakage

Реальная сигнатура: `(self, text, labels, flat_ner=True, threshold=0.5, ner=None, ground_truth_relations=None, top_k=-1)`. В этой версии аргумент называется text, но принимает list[str]. Текст разбит ровно regex установленного GLiREL API: `\w+(?:[-_]\w+)*|\S`. Использованы **включительные** ner end indices, по две известные сущности ENTITY, без NER. На выходе public API end уже **исключительный**, и преобразование обратно проверяется по точным tokens и IDs. ENTITY type не ограничивает relation labels.

Перед forward проверены source/joint character spans, token boundaries, mention tokens, 2 entity spans, 2 направленные пары и ordered mapping девяти prompts. Проверяется вся subword→word карта; фактическая длина encoder input сверяется forward hook с preflight. Параметры public API: threshold=0.0, top_k=-1. Вендорский API выбирает scores строго >0; поэтому отсутствующие outputs сохранялись бы как unreturned score slots с null, а не нули. Диагностические outputs выделяются фильтром >=0.35/0.50/0.70 из сохранённых scores, новых forwards для порогов нет.

Prompts D, без изменения порядка: `is a type of`, `is a stage of`, `is part of`, `has participant`, `has property`, `is used for`, `is applied to`, `may be applied to`, `is derived from`. Определения, expected labels, оценки аудитора, прежние outputs и внешние научные знания модели не передавались. В cross использован только исходный S2 joint text, с теми же DOCUMENT/END DOCUMENT boundaries. Никакого искусственного E2→REL→E3 утверждения.

| Набор | Пары | Побайтно исходный текст/targets | Доступных scores | Не возвращённых slots | Max words / subwords с prompts |
|---|---:|---:|---:|---:|---|
| sanity | 6 | 6 | 108 | 0 | 148 / 250 |
| cross | 28 | 28 | 504 | 0 | 421 / 486 |

Все 28 S2 inputs сохранены без сокращения: прямая сопоставимость исходного текста и targets с ReLEx S2 сохранена. Токенизация и архитектура моделей различаются.

Использован консервативный предел 512 transformer subwords с prompts, дополнительно к GLiREL config.max_len=512 исходных words. Если исходный вход не помещается, код уменьшает обе source области на одинаковую долю вокруг полных targets; ни один mention не отрезается. Фактические source substrings и provenance сохраняются в каждой JSONL записи. Результаты fsync после каждой пары; resume проверяет signature и не повторяет завершённые forwards.

## Шесть внутридокументных sanity-пар

| Пара | Expected / direction | GLiREL expected score | Максимальный output score / label / direction | Exact hit 0.35 / 0.50 / 0.70 |
|---|---|---:|---|---|
| S01 | SUBTYPE_OF E2_TO_E3 | 0.00406825 | 0.00856989 USED_FOR E2_TO_E3 | 0 / 0 / 0 |
| S02 | HAS_PROPERTY E2_TO_E3 | 0.00252183 | 0.00252183 HAS_PROPERTY E2_TO_E3 | 0 / 0 / 0 |
| S03 | HAS_PROPERTY E2_TO_E3 | 0.14442462 | 0.14442462 HAS_PROPERTY E2_TO_E3 | 0 / 0 / 0 |
| S04 | DERIVED_FROM E2_TO_E3 | 0.00013655 | 0.01428191 HAS_PROPERTY E3_TO_E2 | 0 / 0 / 0 |
| S05 | APPLIED_TO E2_TO_E3 | 0.00063880 | 0.00984926 HAS_PROPERTY E2_TO_E3 | 0 / 0 / 0 |
| S06 | USED_FOR E2_TO_E3 | 0.00042601 | 0.00046989 HAS_PROPERTY E2_TO_E3 | 0 / 0 / 0 |

Для каждой пары сохранены исходный фрагмент, head/tail, токенные и символьные spans, все фактические scores, expected relation и source-based review. Expected relations подтверждены исходным текстом; их вердикт SUPPORTED_CONCEPT_RELATION относится к проверяемому эталонному смыслу, а не к отсутствующему положительному output модели. В сохранённом прежнем sanity онтологии D GLiNER ReLEx распознала все 6/6 при >=0.35; здесь GLiREL не распознала ни одного при заданных cutoffs. Это отдельный sanity comparison, а не cross-document результат.

Диагностика не обнаружила некорректных API arguments, смешения inclusive/exclusive spans, потерянных labels, дополнительных targets, отсутствующего направления, неполной загрузки параметров или truncation. Содержательный нулевой sanity сохранён и не использован для остановки cross. Причина низких scores (формулировки D, предметный домен, режим токенизации/версия runtime или checkpoint) причинно не отделена. Переписывание labels, подбор порога и внешние положительные демонстрации после просмотра результатов не выполнялись.

Ниже порога есть различимый отклик: S02 и S03 ставят ожидаемое HAS_PROPERTY в верном направлении на первое место, но scores всего около 0.00252 и 0.14442. У S01 максимум относится к USED_FOR вместо subtype, у S04/S05 — к HAS_PROPERTY вместо derivation/application. У S06 обратный USED_FOR немного выше правильного. Это ошибки ранжирования/воздержание на известных смыслах, а не положительные ложные edges: выдаваемых при выбранных порогах edges нет. Нулевой threshold-yield не означает, что API возвращал пустые сырые векторы.

## Междокументные пары и сравнение с ReLEx S2

| P-пара | Expected / direction | GLiREL expected | ReLEx S2 expected | GLiREL max / label / direction |
|---|---|---:|---:|---|
| P01 | STAGE_OF E2_TO_E3 | 0.00000604 | 0.00063823 | 0.00001515 HAS_PROPERTY E2_TO_E3 |
| P02 | POTENTIALLY_APPLICABLE_TO E2_TO_E3 | 0.00036225 | 0.00744994 | 0.00107269 HAS_PROPERTY E3_TO_E2 |
| P03 | HAS_PARTICIPANT E2_TO_E3 | 0.00000894 | 0.00001159 | 0.00085273 HAS_PROPERTY E2_TO_E3 |
| P04 | SUBTYPE_OF E2_TO_E3 | 0.00000849 | 0.00052577 | 0.00055222 HAS_PROPERTY E2_TO_E3 |
| P05 | PART_OF E3_TO_E2 | 0.00078390 | 0.00106168 | 0.00106012 HAS_PROPERTY E2_TO_E3 |
| P06 | POTENTIALLY_APPLICABLE_TO E2_TO_E3 | 0.00019170 | 0.00025848 | 0.00237111 HAS_PROPERTY E2_TO_E3 |

Более высокий raw score одной модели не считается улучшением научной корректности: шкалы не откалиброваны. P01 требует определения ESRD/CKD, которого нет в двух абстрактах; P04/P05 консервативно зависят от определения overexpression/bundle, а не устанавливают общий эксперимент или идентичность оборудования. P02/P06 — условные гипотезы переносимости, не observed application. P03 допускает узкое activity→cell-class отношение по лексическому смыслу, но не подтверждает Breg→rHuEPO механизм. Нетривиальных бесспорных TEXT-SUPPORTED SPLIT positives в этом наборе нет; это ограничение прежнего source audit сохранено.

| Сравнение при >=0.35 | GLiREL | GLiNER ReLEx S2 (для sanity — прежний D-run) |
|---|---:|---:|
| Expected sanity hits / 6 | 0 | 6 |
| Cross outputs / 28 пар | 0 | 0 |
| Принятых cross relations | 0 | 0 |
| Ошибочных положительных cross outputs | 0 | 0 |
| P-пар с любым положительным output / 6 | 0 | 0 |

## Содержательный аудит outputs

При >=0.35 нет положительных предсказаний ни в sanity, ни в cross; следовательно, нет outputs для распределения по SUPPORTED_CONCEPT_RELATION / PLAUSIBLE_HYPOTHESIS / UNCERTAIN / REJECTED. В quality_review.csv для каждой пары явно сохранён NO_POSITIVE_PREDICTION, поля correctness помечены not_asserted. Это обозначение abstention, не пятый вердикт для фактически предсказанного отношения. Нулевой accepted output не означает precision=100%.

Наибольший cross score: (0.09467815607786179, 'W09', 'SUBTYPE_OF', 'E3_TO_E2'). Все значения ниже диагностического cutoff сохраняются как raw scores, а не установленный научный claim. Они позволяют видеть ранжирование/ошибочный относительный head-tail порядок, но не являются основанием опускать порог постфактум. Шесть positive candidates и N/W оцениваются по источникам; префикс N сам по себе не основание отклонить любой возможный concept relation.

Для максимума W09 смысл E3(papal politics)→SUBTYPE_OF→E2(synod on synodality) источниками не поддерживается: историческая папская репрезентация и политика 1605–1700 не определяются как подтип современного синода. При искусственном повышении такого score до статуса edge этот claim был бы REJECTED по type/scope; в текущем эксперименте он остаётся ниже порога и не включён в число положительных ошибок. Другие слабые ответы на P также не заменяют нужный predicate: например, HAS_PROPERTY не выражает применение MRI-denoising к маммограммам.

## Ответы на вопросы

1. **Локальная работа:** checkpoint успешно загружен и public API работает после описанных локальных compatibility bridges. Все веса загружены строго; никакие модели не скачивались дополнительно.
2. **Sanity:** exact expected hits и scores приведены выше для всех шести пар и всех трёх cutoffs; нулевой sanity является ограничением оценки модели в выбранной конфигурации, а не скрытым успешным тестом.
3. **Число cross outputs:** приведено для каждого порога в основной таблице; raw-score inventory хранит все доступные labels/directions, включая ниже cutoffs.
4. **Содержательная корректность:** положительные outputs, если они есть, перечислены в quality_review.csv; отсутствие outputs не позволяет оценить precision извлечения.
5. **P01–P06:** expected и maximum scores показаны в таблице; отсутствие positive outputs не превращает эти кандидаты в gold negatives.
6. **Ошибки:** различаются нераспознавание явно поддержанных sanity-смыслов и ложные cross claims. Возможные версии/tokenization/label/domain факторы не выдаются за доказанную архитектурную причину.
7. **Сравнение с ReLEx S2:** оцениваются accepted/rejected outputs и конкретные expected смыслы, а не общий raw-score максимум. ReLEx S2 уже имела нулевой cross yield на этих 28 парах; GLiREL не должна объявляться лучше только за низкую уверенность/воздержание.
8. **Корпус из 30 разнородных статей:** позволяет установить поведение двух локальных конфигураций на выбранных примерах и диагностировать недостаточность benchmark. Он не позволяет оценить общий cross-document recall или доказать превосходство/непригодность всей модельной семьи. При расширении нужен независимый source-supported набор с двумя необходимыми текстовыми посылками и совместимыми негативными контролями; такое расширение сейчас не выполнялось.

## Файлы и воспроизведение

Компактный набор: sanity_results.jsonl, cross_document_predictions.jsonl, quality_review.csv, report.md, manifest.json. Каждый JSONL checkpoint содержит фактический вход и provenance, токены/offsets, ner, raw API predictions и преобразованные scores. manifest хранит frozen sources, модельные/package hashes, runtime, подпись inference, проверки и threshold summaries. Hash собственного manifest не включён из-за самоссылки.

Команды: `.venv/Scripts/python.exe scripts/evaluate_glirel.py --stage sanity`, затем `scripts/report_glirel.py --stage sanity`, затем `scripts/evaluate_glirel.py --stage cross`, затем `scripts/report_glirel.py --stage finalize` с тем же интерпретатором. Четыре новых offset/endpoint/crop теста прошли. Повторный запуск стадий использует signature-checked checkpoints; report не вызывает inference. Веса, cache и окружения в commit не включаются. Neo4j, первый проход GLiNER ReLEx, ER, SIMILAR_TO, HAS_SHARED_CONCEPT и исходные документы не менялись; другие модели и обучение не запускались. Push не выполнялся.
