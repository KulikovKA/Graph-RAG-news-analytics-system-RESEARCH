# Stage 3A.5.2 · Evidence-grounded benchmark

В benchmark сохранены 30 исходных идентификаторов вопросов и охват всех 30 замороженных аннотаций. Человек утвердил формулировки вопросов и ожидаемых ответов для 8 записей блока A и 7 записей блока B. Эти 15 записей имеют статус `HUMAN_REVIEWED` и флаг `requires_human_review: false`. Остальные 15 записей сохраняют статус `DRAFT_UNREVIEWED` и флаг `requires_human_review: true`. Итоговое утверждение benchmark и присвоение статуса Gold остаются отдельными этапами.

| Category | Questions | Answerability | Review status |
|---|---:|---|---|
| A · Single-document, 1-hop | 8 | 8 answerable | `HUMAN_REVIEWED` |
| B · Single-document, multi-hop | 7 | 7 answerable | `HUMAN_REVIEWED` |
| C · Cross-document | 10 | 10 answerable | `DRAFT_UNREVIEWED` |
| D · Negative / ambiguous | 5 | 4 insufficient evidence, 1 ambiguous | `DRAFT_UNREVIEWED` |

The benchmark records **25 SUPPORTED, 1 PARTIAL and 4 UNSUPPORTED**. Support describes the inference requested by the question. Every D record cites evidence for the reason to abstain or qualify the answer. D04 is partial because both directions of the Kerr effect are supported conditionally and the question omits sublayer thickness. For D, the document count refers to the sources needed to explain the limitation.

## Source and method

The starting revision was `1e8d154a86f8420b724029b1b93dfac925ca5ddc` on `design-v2-sequential-kg`. The Stage 2G.3 importer specifies a 30-document graph. The architecture audit and Stage 3A.5.1b report distinguish within-document factual evidence from similarity and concept membership. Stage 3A.5.1 has a tracked smoke runner which emits its report to stdout. A separate saved 3A.5.1 run report was unavailable in the checkout. [2G.3 importer](../../../scripts/import_hybrid_concept_2g3.py) · [Architecture audit](../../../docs/architecture/STAGE_3A_0_RETRIEVAL_AUDIT.md) · [3A.5.1 runner](../../../scripts/smoke_retrieval_3a5.py) · [3A.5.1b report](../STAGE_3A_5_1B_CROSS_DOCUMENT_SMOKE.md)

I read the archived OpenAlex abstract text for every locked ID and matched titles, source URLs, revisions and SHA-256 digests against the document lock. Hashes use UTF-8 bytes of the decoded text. Each evidence range uses Python Unicode indices with an exclusive end, including original punctuation and spacing. The review links to the DOI where available and otherwise to the frozen OpenAlex source URL. [Input abstracts](../../ner_v2/research_run2/input_abstracts.jsonl) · [Document lock](../../entity_resolution/relex_base_t035/dynamic_concept_graph/document_lock.json)

Questions and answers were drafted from these texts without external scientific sources, retrieval outputs, model predictions or graph paths as evidence. No retrieval run or paraphrase generation was performed. The executor was the active Codex session. The requested model and effort were GPT-6.2 Astra High. The exact runtime variant and effort could not be independently established.

Исходный benchmark зафиксирован коммитом `808c3d14c49547f4252067f2675a90f2b5aa6cb5`, ревизия блока A — коммитом `e39c5ecbc7240b9c70dca7a923cc69ebfb34fc6b`. Поля `generation_method: EVIDENCE_GROUNDED_DRAFT` и `source_of_truth: ORIGINAL_DOCUMENT` сохраняют происхождение записей. При ручной ревизии блоков A и B формулировки взяты из утверждённых человеком текстов, а опорные фрагменты повторно сверены с замороженными аннотациями. Ссылки, идентификаторы и хеши документов сохранены.

## Ручная ревизия блока A

- A01. Источник описывает основное препятствие для получения высокого титра в инженерных микроорганизмах.
- A02. Новый фрагмент подтверждает преобладание memory B cells среди B-клеток в полипах. Исследование охватывает 10 пациентов с хроническим риносинуситом с назальными полипами.
- A03. Два фрагмента одного документа подтверждают диапазон наблюдений и положение максимума. Источник описывает положение словами “to the west” без точной долготы.
- A04. Обоснование выбора tanh ansatz относится к приведённому в статье редуцированному уравнению.
- A05. Источник называет расширение инвертированного повтора основной причиной различия. Утверждённая формулировка вопроса шире этой оговорки.
- A06. Ответ воспроизводит перечень основных инструментов из статьи. Его полнота ограничена охватом источника.
- A07. Причина ухода Легия приведена в изложении титульного листа The Book of Mormon.
- A08. Новый фрагмент описывает предложение об установке SPD на трёх типах линий для экранирования и фильтрации. Источник рассматривает косвенное грозовое перенапряжение в исследованной системе электроснабжения.

При ревизии блока A независимое смысловое ревью подтвердило поддержку всех восьми ответов замороженными текстами и точное совпадение формулировок с утверждённой версией. Структурное ревью подтвердило согласованность JSONL и Markdown, метаданных, offsets и SHA-256. Ревью Git diff подтвердило неизменность записей B–D на этапе A, исходного корпуса и document lock. Блокирующих дефектов не выявлено.

## Ручная ревизия блока B

Цепочки описывают операции над одним источником. Число приведённых цитат отражает покрытие утверждений. Количество извлекаемых чанков и маршруты в KG здесь не проверялись.

| ID | Смысловая цепочка и характер вывода | Методологическое ограничение |
|---|---|---|
| B01 | Аннотация прямо связывает фиксированное число proposals с ограниченной адаптивностью к распределению людей. Затем описывает mini-density map, основанную на идее crowd counting, для адаптивного выбора числа proposals и сообщает о снижении избыточности при сохранении качества proposal boxes. | Утверждение о качестве относится к region proposals. Качество итоговой детекции по всем показателям этой фразой не установлено. |
| B02 | Из двух рядов извлекаются показатели за одинаковый десятилетний срок, 70% для контроля глаукомы и 26% для прозрачности трансплантата. Из них вычисляется разница в 44 процентных пункта и делается сравнение частот. | Ретроспективное сравнение двух маргинальных исходов не устанавливает причинность или наличие обоих исходов у одних и тех же глаз. Показатели последнего визита относятся к отдельной конечной точке. |
| B03 | Фактор риска и трудоёмкое, зависящее от специалиста измерение объясняют интерес к автоматизации. Внешняя проверка дала корреляцию с Cumulus `r = 0.884`. Авторы описывают потенциальную применимость в масштабных исследованиях. | Корреляция сама по себе не устанавливает взаимозаменяемость методов или клиническую диагностическую точность. |
| B04 | CNV обнаружены при сравнении крайних фенотипов мышиных линий. Авторы сравнили дозу генов с экспрессией, не нашли простой корреляции во всех случаях и указали на необходимость дальнейшей работы по экспрессии. | Недостаточность одного обнаружения CNV для объяснения поведения — интерпретация ограниченных функциональных результатов. |
| B05 | В объяснении авторов неценовые факторы сдвинули спрос сильнее предложения, цены выросли при увеличении производства, совокупный доход фермеров не снизился. | Период 2003–2014 опущен в вопросе, но сохранён в evidence. Результат относится к Китаю в этот период и к совокупному доходу. Данных о доходах отдельных фермеров нет. |
| B06 | Автор предлагает IFS inner work для проработки травм отношений, связывает это с развитием эмоциональных навыков и описывает зрелость и проработку травм привязанности как предпосылки открытости и сотрудничества на равных. | Это концептуальная гипотеза автора. Экспериментального подтверждения терапевтического эффекта в аннотации нет. |
| B07 | Ежегодные медали отмечали достижения предыдущего года и продвигали образы строителя, воина и импресарио. Их периодичность позволяет сопоставлять саморепрезентацию и прослеживать политический курс. | Вопрос говорит о папских медалях шире. Ответ ограничен ежегодными выпусками 1605–1700 годов, изученными в статье. Источник описывает задуманный образ и воспринимаемый успех. |

Независимое смысловое ревью подтвердило дословное совпадение утверждённых формулировок и поддержку существенных звеньев каждой цепочки. Структурное ревью подтвердило согласованность записей и артефактов, включая метаданные, SHA-256 и offsets. Ревью Git diff подтвердило, что изменены ровно семь записей B, а блоки A, C и D, корпус и document lock сохранены. Блокирующих дефектов не выявлено.

## Coverage and evidence

| Frozen discipline label | Unique documents cited | Questions citing the discipline |
|---|---:|---:|
| Biology | 5 | 5 |
| Computer Science / Machine Learning | 5 | 4 |
| Economics | 5 | 5 |
| History | 5 | 6 |
| Medicine | 5 | 7 |
| Physics | 5 | 5 |

Question counts within a discipline count each question once. A question spanning disciplines contributes to both rows. Source labels are retained, including for religious studies under History and composite shape-coding under Computer Science / Machine Learning.

All **10 cross-document questions** have separate passages supporting the contribution of each of two documents in the drafting review. For example, C04 needs the convergence paper for the Kalman filter and the fare paper for which solver belongs to which level. C09 needs both the grain article's total-income finding and the comment's attribution of the voluntary-notice finding to Nord and Ting. Neither comparison requires a factual link between articles.

## Review findings and limits

- C05 uses the writer's explicit comparison of the 1930s and 1960s; the archived abstract ends mid-thought and only its earlier complete statements are used.
- D01 cannot yield a clinical benefit rate from a count of variants tested in vitro. D02 cannot establish a universal device-dose threshold from one cohort. D03 cannot quantify a legislative effect from preceding voluntary arrangements. D05 lacks identifiers linking individuals across cohorts. These limitations apply to the requested conclusions in this corpus.
- Coverage is deliberately broad in a small corpus. Some papers support several questions and some contribute only to a comparison or a negative control. Difficulty labels are drafting judgments based on the operations needed. The named study contexts in category C may make locating sources easier.
- Для дальнейшей ревизии C сохраняется проверка достаточности одного источника и естественности сравнения, особенно междисциплинарного сравнения в C02. Для D сохраняется проверка критериев зачёта ограниченного или условного ответа.

## Reproducing the checks

From the repository root, run:

```sh
python reports/retrieval/benchmark_3a5_2/validate_draft.py
```

Исходный валидатор зафиксировал 30 уникальных ID, 65 вхождений evidence и охват 30 из 30 документов. После ревизии A было 66 вхождений, включая дополнительное подтверждение диапазона наблюдений в A03. После ревизии B валидатор подтвердил **30 исходных уникальных ID, 15 записей после ручной ревизии, 15 черновых записей, 74 вхождения evidence и охват 30 из 30 документов**. Он проверяет байты блока A по контрольной сумме коммита ревизии A, а байты блоков C и D — по контрольной сумме исходного benchmark. В сумме это 23 побайтно неизменные записи при 7 обновлённых. Проверки включают соответствие метаданных и SHA-256 замороженному корпусу и его lock, границы и точный текст цитат, согласованность вкладов документов и число различных источников. Проверка пригодности benchmark для retrieval и присвоение статуса Gold остаются отдельными задачами. [Question records](questions_draft.jsonl) · [Review copy](questions_review.md)
