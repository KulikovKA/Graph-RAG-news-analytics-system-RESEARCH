# C-DIAG-002: frozen exploratory DEV protocol

Дата: 2026-10-29 (Europe/Moscow). Base: `b368a63`, отдельный worktree/ветка
`c-diag-002-pretrained-only`. Этот файл и config фиксируются до predictions/evaluation.
Ни один новый Gold F1 не использован для выбора моделей или правил.

## Ограничения и выбор

Только 12 DEV из неизменного manifest; CALIBRATION/HOLDOUT запрещены.
Ontology v1.0 (12 kinds, 23 relations) и все gates C-RESTART-001 неизменны.
CPU float32, четыре torch threads, seed 42, deterministic algorithms, <32 GiB RAM.
Обучение, fit, новые classifier heads, генеративные LLM запрещены.
External checkpoints, fine-tuned их авторами, допустимы; локального обучения нет.

Контроль: `urchade/gliner_small-v2.1@4e091416cf7c3481db542c2a3d26156916f3a47f`,
threshold 0.35, прежние 12 prompts, окна 140/30 words.

Два выбранных готовых scientific NER:

| Alias | Checkpoint revision | Source class → ontology kind |
|---|---|---|
| chem | `teman67/matscibert-chem-ner@e8bd203aec2ec59c8705f199c108b976d170d953` | CHEM → MATERIAL; O → no entity |
| battery | `Dmjdxb/batterybert-electrocatalyst-ner-v2@13e458cef185382b7a54ad2352f51946aa18b0f3` | MATERIAL → MATERIAL; CONDITION → OPERATING_CONDITION; DURATION/CURRENT/FAILURE_MODE → unsupported; O → no entity |

Mapping фиксирован по исходным классам и определениям ontology. CHEM описывает
вещество, но не contextual target role, поэтому ANALYTE не назначается. CONDITION
интерпретируется как контекст среды/эксплуатации. DURATION/CURRENT обычно могут быть
qualifiers; FAILURE_MODE не имеет однозначного kind. Их spans считаются отдельно и
не превращаются в OTHER. Остальные kinds этими checkpoint не поддерживаются.
Каждый BIO-класс из config явно покрыт mapping после удаления B-/I-.

Выбор обусловлен доступностью полных token-classifier checkpoints, понятными
source labels, materials-domain backbone и работоспособностью стандартного CPU
loader. Это ограниченная проверка, не исчерпывающий обзор всех scientific NER.
MatSciBERT/MatBERT/SciBERT base encoders сами по себе NER-моделями не считаются.
Проверка `output_loading_info` обязана дать ноль missing/unexpected/mismatched keys;
форма сохранённой classifier.weight и file hashes сохраняются. Нельзя инициализировать
новую head или менять num_labels/id2label при загрузке.

Scientific decoding: pretrained argmax BIO + Transformers `simple` aggregation,
без дополнительного score threshold. Окна chem 45/10 words (лимит 128 subtokens из
model card), battery 100/20 (512). При overflow — явная ошибка, без truncation.
Spans сохраняют исходные Unicode offsets; boundary repair/lexicons не добавляются.

## Прогоны и RE

Пять заранее заданных вариантов: control, chem, battery, union, gold.
`union` — точное объединение трёх NER по (start,end,kind), duplicate=max score;
конфликтующие kinds и nested spans сохраняются. Никакого выбора победителя по Gold.
Все четыре predicted-entity варианта проходят тот же NLI/gates pipeline.
NER time отдельно суммируется из per-chunk timing; полная стоимость каждого прогона
измеряется отдельным fresh process с imports/load/hashing/serialization.

RE control: `cross-encoder/nli-deberta-v3-xsmall@a150876415327c80daeff35ca6f68f5ed8cf5c24`,
entailment >0.5, batch 32, 512 subtokens, single sentence; 22 исходные
verbalizations, без изменений по Gold. `MEETS_REQUIREMENT` unsupported; остальные
EXPERIMENTAL_ZERO_SHOT, scientific support не доказан. Domain/range до NLI queries
и последующие projection/scope/evidence/canonicalization — как в baseline.
Ни альтернативные thresholds, ни verbalizations после evaluation не исследуются.

Gold entities — отдельная явно помеченная диагностика. Exporter читает только DEV
Gold и экспортирует исключительно spans/kinds/IDs + source/provenance hashes. RE не
получает Gold relations, qualifiers, polarity, contextual roles или answers.
Gold-entity метрики не называются end-to-end результатом архитектуры и не служат
разрешением применять Gold в обычном inference. Gold используется evaluator только
после записи immutable predictions. Все пять прогонов завершаются до общего scoring.

GLiREL: одна повторная CPU-проверка того же официального checkpoint
`jackboyla/glirel-large-v0@40a523e12a8432d6da364cf2a195a28755ff04d3` через официальный
`GLiREL.from_pretrained`, strict loading. Опубликованный Doctor Who control,
те же tokens/NER/labels, threshold 0, top_k=1; оба `characters` score должны быть
>0.5 (в README около .99). Budget: один процесс, максимум 10 минут.
При повторном FAIL — остановка GLiREL, точный blocker и исключение из scientific
quality comparison. Не считать нулевой output неработающего runtime качеством RE.
При PASS: тот же adapter/threshold 0.5/22 prompts/contexts и те же сохранённые
control/Gold entities; сравнить с NLI, не запуская повторно NER.
Генеративный REBEL и joint GLiNER-Relex вне C; TACRED/SciERC fixed-label heads без
наших 23 labels не выдаются за zero-shot поддержку ontology.

## Evaluation и публикация

Exact Unicode span и typed entities, directed multiset relation matching из
неизменного evaluator; полный denominator: все 12 DEV. Per-kind typed recall и
untyped boundary recall считаются отдельно. Candidate recall сохраняет denominator
Gold relations, даже если часть типов не поддержана. Raw/final P/R/F1, FP,
strict Qualified F1 (UNKNOWN completeness не получает TP), pair abstentions,
unsupported labels/Gold opportunities, stage-loss funnel и per-relation counts.
Gold-mode отдельно также conditional metrics на доступных кандидатах.
Разложение потерь: boundaries → kind → candidate context → served/capability →
RE → domain/range → scope → evidence → final; это ordered accounting, не causal
эксперимент для каждого фильтра. Детерминированные примеры TP/FP/FN выбираются после
predictions без исправления модели. Все 23 отношения выводятся, включая нули.

Wall time, process CPU time, Windows PeakWorkingSetSize; downloads, evaluation и
control probes не входят в основные времена. NER-only cost берётся из stage timers;
peak RAM — весь fresh process, а не отдельный NER encoder. RAM не суммируется между
последовательными прогонами. Runs/predictions/input/model/config/code hashes и audit
сохраняются. Network/Gold/closed-split/fitted-model access запрещён audit hook при
обычном inference; Gold mode разрешает только выделенный entity-only export.

DEV повторно исследован: результаты exploratory, без confirmatory significance.
D024 annotation incomplete; отсутствует independent semantic review. Ни высокий
F1, ни конкретный итог рекомендации заранее не гарантируются. В Git публикуются
код, frozen protocol/config, aggregate metrics/provenance и короткие DEV examples;
полные Gold/predictions/query traces/weights остаются локальными.
