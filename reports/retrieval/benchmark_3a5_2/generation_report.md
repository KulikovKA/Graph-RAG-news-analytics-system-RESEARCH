# Stage 3A.5.2 · Evidence-grounded benchmark

В benchmark сохранены 30 исходных идентификаторов вопросов и охват всех 30 замороженных аннотаций. У восьми записей блока A человек утвердил формулировки вопросов и ожидаемых ответов. Эти записи имеют статус `HUMAN_REVIEWED` и флаг `requires_human_review: false`. Остальные 22 записи сохраняют статус `DRAFT_UNREVIEWED` и флаг `requires_human_review: true`. Итоговое утверждение benchmark и присвоение статуса Gold остаются отдельными этапами.

| Category | Questions | Answerability | Review status |
|---|---:|---|---|
| A · Single-document, 1-hop | 8 | 8 answerable | `HUMAN_REVIEWED` |
| B · Single-document, multi-hop | 7 | 7 answerable | `DRAFT_UNREVIEWED` |
| C · Cross-document | 10 | 10 answerable | `DRAFT_UNREVIEWED` |
| D · Negative / ambiguous | 5 | 4 insufficient evidence, 1 ambiguous | `DRAFT_UNREVIEWED` |

The benchmark records **25 SUPPORTED, 1 PARTIAL and 4 UNSUPPORTED**. Support describes the inference requested by the question. Every D record cites evidence for the reason to abstain or qualify the answer. D04 is partial because both directions of the Kerr effect are supported conditionally and the question omits sublayer thickness. For D, the document count refers to the sources needed to explain the limitation.

## Source and method

The starting revision was `1e8d154a86f8420b724029b1b93dfac925ca5ddc` on `design-v2-sequential-kg`. The Stage 2G.3 importer specifies a 30-document graph. The architecture audit and Stage 3A.5.1b report distinguish within-document factual evidence from similarity and concept membership. Stage 3A.5.1 has a tracked smoke runner which emits its report to stdout. A separate saved 3A.5.1 run report was unavailable in the checkout. [2G.3 importer](../../../scripts/import_hybrid_concept_2g3.py) · [Architecture audit](../../../docs/architecture/STAGE_3A_0_RETRIEVAL_AUDIT.md) · [3A.5.1 runner](../../../scripts/smoke_retrieval_3a5.py) · [3A.5.1b report](../STAGE_3A_5_1B_CROSS_DOCUMENT_SMOKE.md)

I read the archived OpenAlex abstract text for every locked ID and matched titles, source URLs, revisions and SHA-256 digests against the document lock. Hashes use UTF-8 bytes of the decoded text. Each evidence range uses Python Unicode indices with an exclusive end, including original punctuation and spacing. The review links to the DOI where available and otherwise to the frozen OpenAlex source URL. [Input abstracts](../../ner_v2/research_run2/input_abstracts.jsonl) · [Document lock](../../entity_resolution/relex_base_t035/dynamic_concept_graph/document_lock.json)

Questions and answers were drafted from these texts without external scientific sources, retrieval outputs, model predictions or graph paths as evidence. No retrieval run or paraphrase generation was performed. The executor was the active Codex session. The requested model and effort were GPT-6.2 Astra High. The exact runtime variant and effort could not be independently established.

Исходный benchmark зафиксирован коммитом `808c3d14c49547f4252067f2675a90f2b5aa6cb5`. Поля `generation_method: EVIDENCE_GROUNDED_DRAFT` и `source_of_truth: ORIGINAL_DOCUMENT` сохраняют происхождение записей. При ручной ревизии авторитетным источником формулировок стал утверждённый текст блока A, а опорные фрагменты повторно сверены с замороженными аннотациями. Ссылки, идентификаторы и хеши документов сохранены.

## Ручная ревизия блока A

- A01. Источник описывает основное препятствие для получения высокого титра в инженерных микроорганизмах.
- A02. Новый фрагмент подтверждает преобладание memory B cells среди B-клеток в полипах. Исследование охватывает 10 пациентов с хроническим риносинуситом с назальными полипами.
- A03. Два фрагмента одного документа подтверждают диапазон наблюдений и положение максимума. Источник описывает положение словами “to the west” без точной долготы.
- A04. Обоснование выбора tanh ansatz относится к приведённому в статье редуцированному уравнению.
- A05. Источник называет расширение инвертированного повтора основной причиной различия. Утверждённая формулировка вопроса шире этой оговорки.
- A06. Ответ воспроизводит перечень основных инструментов из статьи. Его полнота ограничена охватом источника.
- A07. Причина ухода Легия приведена в изложении титульного листа The Book of Mormon.
- A08. Новый фрагмент описывает предложение об установке SPD на трёх типах линий для экранирования и фильтрации. Источник рассматривает косвенное грозовое перенапряжение в исследованной системе электроснабжения.

Независимое смысловое ревью подтвердило поддержку всех восьми ответов замороженными текстами и точное совпадение формулировок с утверждённой версией. Структурное ревью подтвердило согласованность JSONL и Markdown, метаданных, offsets и SHA-256. Ревью Git diff подтвердило неизменность записей B–D, исходного корпуса и document lock. Блокирующих дефектов не выявлено.

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

- B02 pairs the ten-year entries of two outcome series. The last visit is a different endpoint and the two marginal rates cannot identify which eyes experienced both outcomes.
- B06 rests on an author's proposed link between relational healing and a synodal mindset. Its expected answer preserves that qualification. C05 uses the writer's explicit comparison of the 1930s and 1960s; the archived abstract ends mid-thought and only its earlier complete statements are used.
- D01 cannot yield a clinical benefit rate from a count of variants tested in vitro. D02 cannot establish a universal device-dose threshold from one cohort. D03 cannot quantify a legislative effect from preceding voluntary arrangements. D05 lacks identifiers linking individuals across cohorts. These limitations apply to the requested conclusions in this corpus.
- Coverage is deliberately broad in a small corpus. Some papers support several questions and some contribute only to a comparison or a negative control. Difficulty labels are drafting judgments based on the operations needed. The named study contexts in category C may make locating sources easier.
- Для дальнейшей ревизии блоков B–D сохраняются исходные задачи. В B нужно проверить необходимость каждого шага. В C нужно проверить, можно ли ответить с одним источником и естественно ли сравнение, особенно междисциплинарное сравнение в C02. В D нужно проверить критерии зачёта ограниченного или условного ответа.

## Reproducing the checks

From the repository root, run:

```sh
python reports/retrieval/benchmark_3a5_2/validate_draft.py
```

Исходный валидатор зафиксировал 30 уникальных ID, 65 вхождений evidence и охват 30 из 30 документов. Актуализированный валидатор подтвердил **30 уникальных ID, 8 записей после ручной ревизии, 22 побайтно идентичные исходным записи B–D, 66 вхождений evidence и охват 30 из 30 документов**. Дополнительное вхождение подтверждает диапазон наблюдений в A03. Проверки включают соответствие метаданных и SHA-256 замороженному корпусу и его lock, границы и точный текст цитат, согласованность вкладов документов и число различных источников. Проверка пригодности benchmark для retrieval и присвоение статуса Gold остаются отдельными задачами. [Question records](questions_draft.jsonl) · [Review copy](questions_review.md)
