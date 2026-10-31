"""Russian thesis-oriented report generated from measured NER-V2-003 artifacts."""
import csv
from collections import Counter
from pathlib import Path

from cgraph.common import read_json
from cgraph.ner_v2.research import jsonl,SHORT
from cgraph.ner_v2.quality_audit import ROOT,THRESHOLDS


def table(headers,rows):
    def value(x):return str(x).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(value(x) for x in r)+' |' for r in rows])


def csv_rows(name):return list(csv.DictReader((ROOT/name).open(encoding='utf8')))


def main():
    metrics=read_json(ROOT/'threshold_metrics.json');costs=read_json(ROOT/'qwen_cost_estimates.json')
    reference=csv_rows('reference_metrics.csv');quality=csv_rows('audit_quality_metrics.csv')
    by_type=csv_rows('reference_by_type.csv');type_quality=csv_rows('type_hint_audit_metrics.csv')
    labels=csv_rows('label_metrics.csv');discipline=csv_rows('discipline_metrics.csv')
    docs=jsonl(ROOT/'input_abstracts.jsonl');audit=jsonl(ROOT/'audit_candidates_UNREVIEWED.jsonl')
    sample=[a for a in audit if a['audit_partition']=='stratified_120']
    sampling=read_json(ROOT/'audit_sampling_manifest.json');manifest=read_json(ROOT/'corpus_manifest.json')
    rec=read_json(ROOT/'recommendation.json');sensitivity=read_json(ROOT/'recommendation_sensitivity.json')
    env=read_json(ROOT/'environment.json');tokenizer=read_json(ROOT/'qwen_tokenizer_lock.json')
    performances={n:read_json(ROOT/n/'performance.json') for n in ['E1','E2','E3']}
    def metric(n,t):return next(x for x in metrics if x['configuration']==n and x['threshold']==t)
    def ref(n,t):return next(x for x in reference if x['configuration']==n and float(x['threshold'])==t)
    def cost(n,t):return next(x for x in costs if x['configuration']==n and x['threshold']==t)
    def q(n,t,p):return next(x for x in quality if x['configuration']==n and float(x['threshold'])==t and x['partition']==p)
    text='''# NER-V2-003: Candidate Quality Audit + Threshold Optimization

Период исследования: **31 октября — 1 ноября 2026 года, Europe/Moscow**. Ветка: `design-v2-sequential-kg`.
Исходный commit: `0366724d620d076adc8035114424cb06a57ff5d1`.
Статус: фактические CPU-измерения; semantic audit имеет статус **UNREVIEWED**, independent Human Gold отсутствует.

Отдельный [готовый фрагмент для дипломной работы](NER_V2_003_THESIS_FRAGMENT.md).

## 1. Постановка задачи

Исследование продолжает NER-V2-002 и NORM-V2-001 и рассматривает генерацию source-exact кандидатов
для будущей Qwen Entity Typing 4B. Требуется сопоставить модели и пороги по выходу, потерям предполагаемо
значимых терминов, предварительно выявленному шуму, времени CPU и объёму планового входа LLM.
Число spans не трактуется как Recall; совпадение с assisted reference не является истинным Recall или F1.
Научная цель — обоснованный рабочий вариант для следующего этапа, а не доказательство оптимальности
на всей совокупности научных публикаций.

Исходные отчёты, raw results NER-V2-002/NORM-V2-001, normalizer N3 и Ontology v2 сохранены.
Набор типов содержит 11 категорий, отношений — 20; изменения схемы не проводились.
Qwen inference, RE, Semantic Verification, embeddings, Entity Resolution, fine-tuning и построение графа
не выполнялись. Использование tokenizer для подсчёта входа не требует загрузки весов Qwen.

## 2. Научная мотивация выбора confidence thresholds

Score threshold управляет решением decoder о сохранении кандидата. Повышение порога уменьшает выход,
но может одновременно удалить шум, термины с редкой формой, короткие сокращения и полезные альтернативные
подсказки типов. Высокий score не доказывает корректность: модель может уверенно присвоить неподходящий hint.
Scores GLiNER v2.1 и GLiNER2.5 не считаются одинаково калиброванными вероятностями правильного извлечения.
Порог следует оценивать отдельно для каждой архитектуры и относительно явных требований к потерям.

До сравнения зафиксирована прозрачная условная политика: рассмотреть пороги не ниже 0,35, сохранить
не менее 90% exact совпадений с assisted reference относительно 0,35 и не менее 80% в каждой дисциплине,
не уменьшать число совпадений в редких draft-type группах с 1–4 исходными совпадениями, не увеличивать
число draft-нерелевантных spans в полном наборе шести документов. Из допустимых точек выбирается
наибольший threshold. Это исследовательские требования к рабочей точке, а не статистически установленные
законы качества. Чувствительность к защите редких групп анализируется отдельно.

## 3. Описание экспериментального корпуса

Получено 30 новых аннотаций через официальный OpenAlex API: по пять публикаций на каждую из шести
дисциплин, 100–400 whitespace-слов на документ, пять разных primary topics в каждой дисциплине.
Корпус содержит **5744 слова**. Seed отбора — **4303**. Исключены совпадения с прежними 12 работами
по OpenAlex ID, DOI и SHA-256 текста; дубликатов внутри новой выборки также нет.

Использован существующий `OpenAlexClient`, строгая реконструкция inverted index и контракт source_record.
На дисциплину запрошен seeded sample 150 работ с `language:en`, `type:article`, `has_abstract:true`,
`is_retracted:false`; далее выполнены локальные проверки journal/conference, длины, тематики и уникальности.
Кандидаты сортируются по ID, перемешиваются `Random(4303)`; выбираются первые пять разных тем.
History отбирается по subfield History, остальные области — по соответствующим fields.

Проверка до NER выявила, что metadata `language=en` не гарантирует английский abstract, а article-type
не исключает книжную рецензию. Добавлены проверки доли Latin letters и английских служебных слов,
а также title/abstract markers рецензий и редакционных материалов. Первоначальный предварительный
отбор пересчитан **до любых новых предсказаний** с тем же seed и теми же сохранёнными candidate pools.
Причины и SHA исходного предварительного набора сохранены в `preflight_correction.json`.
После окончательной заморозки ни один документ не заменялся.
Фактически выполнены шесть успешных works API requests. Второй preflight использовал те же сохранённые
ответы, поэтому requests в финальном snapshot пуст; исходный acquisition log и совпадение response hashes
сохранены отдельно в [acquisition_provenance.json](research_run2/acquisition_provenance.json).

Metadata и критерии не являются независимой экспертизой жанра. Например, History включает историко-
религиоведческий и устно-исторический материал; принадлежность к области взята из OpenAlex.
В одной физической аннотации присутствует U+FFFD в строках формул; исходный текст не исправлялся.
Все offsets отсчитываются по Unicode codepoints неизменённого abstract, без добавления title.

**Таблица 1. Замороженные публикации.**

'''
    text+=table(['OpenAlex ID','Дисциплина','Год','Слов','Название / DOI'],[
        [f"[{d['document_id']}]({d['source_url']})",d['discipline'],d['year'],len(d['text'].split()),
         d['title']+' / '+str(d.get('doi') or 'DOI отсутствует')] for d in docs])
    text+='''

Исходные abstracts: [input_abstracts.jsonl](research_run2/input_abstracts.jsonl).
Полные выбранные OpenAlex records/inverted indexes: [openalex_snapshot.json](research_run2/openalex_snapshot.json).
Manifest включает title, year, DOI, scientific field, topic, source URL и SHA-256.
Полные ответные candidate pools сохранены локально в `artifacts/ner_v2/openalex_run2/`; параметры,
candidate IDs и response hashes включены в snapshot. Для воспроизведения выводов достаточно замороженного
snapshot и JSONL; новый запрос к живому API со временем может дать иной набор даже при том же seed.

Для подробного аудита выбран один документ на дисциплину: `Random(4303).choice` на отсортированных
пяти IDs каждой области, независимо от outputs моделей.
Генератор отдельно переинициализируется seed 4303 для каждой области, поэтому выбран одинаковый
порядковый индекс в отсортированных пятёрках; статистическая независимость выборов между областями не заявляется.

'''
    reference_lock=read_json(ROOT/'reference_lock.json')
    text+=table(['Дисциплина','Документ аудита','Reference spans'],[
        [d['discipline'],d['document_id'],reference_lock['by_discipline'][d['discipline']]]
        for d in docs if d['document_id'] in manifest['audit_document_ids']])
    text+='''

## 4. Сравниваемые NER-модели

'''
    text+=table(['Конфигурация','Checkpoint','Окно / overlap','Decoder'],[
        ['E1','GLiNER Large v2.1','60 / 20','flat_ner=False, multi_label=True'],
        ['E2','GLiNER Large v2.1','110 / 25','flat_ner=False, multi_label=True'],
        ['E3','GLiNER2.5 Base','110 / 25','AutoExtractor → BoundaryExtractor, overlap_policy=allow']])
    gliner_lock=read_json('contracts/ner_v2/gliner25_research_lock.json')
    text+=f'''

Pinned Large revision: `abd49a1f1ebc12af1be84d06f6848221cf96dcad`.
Pinned GLiNER2.5 revision: `{gliner_lock['revision']}`.
Модели обрабатывались последовательно, каждая в отдельном процессе; одновременно два NER-checkpoint
в память не загружались. CPU float32, четыре torch threads, seed 42, deterministic torch algorithms.
Все варианты используют неизменённую N3 и одинаковые короткие descriptions:

'''
    text+=table(['Hint','Short description'],SHORT.items())
    text+='''

N3 сохраняет исходные surface и offsets, меняет только допустимое конечное plural noun, сохраняет
модификаторы и определения сокращений на уровне документа. Совпадение нормализованных строк не сливает
сущности. Для GLiNER2.5 сохранён официальный schema API: ключи labels и descriptions передаются отдельно;
для v2.1 descriptions являются label input. Это ограничивает изоляцию архитектурного фактора.

Nested mentions и альтернативные hints одного span сохраняются. Decoder v2.1 при `flat_ner=False`
разрешает вложенность, но может подавлять перекрёстные overlaps внутри окна; после объединения окон
перекрёстные пересечения возможны. В E3 явно разрешены overlaps. Эта разница не скрывается.
Дедупликация NER удаляет только одинаковые `(document, start, end, hint)` с сохранением максимального score.
Для Qwen выполняется безопасная группировка одинаковых source spans с сохранением всех альтернативных hints.

## 5. Методика оценки кандидатов

Assisted reference составлен непосредственно по шести текстам до запуска моделей. Зафиксировано
158 source spans: выбранные ассистентом научно значимые выражения и все их точные вхождения, включая
вложенные термины. Файл [assisted_reference_frozen.jsonl](research_run2/assisted_reference_frozen.jsonl)
имеет статус **ASSISTED_REFERENCE_DRAFT**, review_status **UNREVIEWED**; SHA подтверждается в
конфигурации каждой модели. Это неполный список диагностических anchors, а не исчерпывающая разметка.
Отсутствие кандидата в этом списке не означает его нерелевантность.

Полный union предсказаний при 0,15 на шести документах содержит **433 уникальных source spans**.
Для каждого сохранены весь abstract, контекст, границы, scores, hints всех моделей и survival при восьми
thresholds. На этих 433 spans предложены осторожные правила: exact reference → VALID_SPAN,
ограниченный discourse/header watchlist → IRRELEVANT, очевидные нарушения границ → соответствующая
категория; остальные остаются AMBIGUOUS. Это предварительная разметка для последующей ручной проверки.

Из остальных 24 документов отобраны **120 уникальных кандидатов, по 20 на область**.
Воспроизводимое многомерное балансирование учитывает focal model, hint, confidence bin, соглашение моделей
и документ, с seeded tie order. Страты не образуют полный декартов набор: 120 случаев недостаточно для
всех сочетаний шести областей, трёх моделей, 11 типов и восьми confidence ranges. Реальная coverage:

'''
    text+=table(['Измерение','Распределение'],[[field,'; '.join(f'{k}: {v}' for k,v in values.items())]
                                              for field,values in sampling['coverage'].items()])
    text+='''

Ассистент дополнительно прочитал контекст всех 120 случаев и записал содержательные поправки в
[assistant_audit_amendments.json](research_run2/assistant_audit_amendments.json). Все эти оценки также
остаются UNREVIEWED. Поля human reviewer, category и notes оставлены пустыми; человек ничего не подтверждал.
Исходный независимый от NER reference после этого не изменялся.

Категории границ: VALID_SPAN, PARTIAL_SPAN, WRONG_BOUNDARY, IRRELEVANT, AMBIGUOUS.
MISSED_ENTITY применяется к отдельному списку reference spans без соответствующих outputs.
Качество типа оценивается отдельно: TYPE_HINT_CORRECT, TYPE_HINT_INCORRECT, TYPE_HINT_AMBIGUOUS.
Если у draft-интерпретации несколько допустимых типов, совпавшая альтернатива остаётся типологически
неоднозначной. Типовые counts в таблицах ниже условны на VALID_SPAN, чтобы не смешивать границы и labels.

'''
    sample_counts=Counter(a['draft_assessment']['boundary_category'] for a in sample)
    text+=table(['Draft-категория','Из 120'],sorted(sample_counts.items()))
    text+='''

Файлы: [все 553 audit records](research_run2/audit_candidates_UNREVIEWED.jsonl),
[стратифицированные 120](research_run2/audit_stratified_120.jsonl),
[шесть полных документов с outputs](research_run2/audit_full_documents.jsonl).
Полный census и дополнительные 120 анализируются раздельно. Отбор 120 не является простой случайной
выборкой всех predicted spans; доля ошибок на нём не экстраполируется на весь корпус.

## 6. Методика threshold optimization

Исследованы **0,15; 0,25; 0,35; 0,45; 0,50; 0,55; 0,60; 0,65**.
Основной inference каждой модели выполнен один раз при 0,15. На первом окне каждого из шести
аудитных документов проведены native checks при 0,35, 0,50 и 0,65: всего 18 на модель, 54 в серии.
Сравнивались exact spans, labels и scores с допуском 10⁻⁶; все проверки совпали.

Постфильтрация также проверена по decoder: у v2.1 применяется строгое `score > threshold`,
у E3 — включительное сравнение. При greedy декодировании по убывающему score низкопороговые
кандидаты не меняют решение для более высоких. У boundary checkpoint adaptive threshold=false,
abstention не зависит от span threshold, overlap_policy=allow. Код фильтра сохраняет различие
строгой/нестрогой границы. Если equivalence check не проходит, pipeline останавливается и требует
нового output root с native inference всех thresholds; дополнительные вызовы в таком режиме учитываются.
Native checks не являются доказательством для произвольной будущей конфигурации decoder.

Для модели m и порога t обозначим S_m(t) множество source spans, M_m(t) — records с hints.
N_spans=|S_m(t)|; spans/document=N_spans/30; spans/100 words=100·N_spans/5744.
Потери относительно 0,15: |S_m(0,15) \ S_m(t)|.
Доля сохранения baseline 0,35: |S_m(t) ∩ S_m(0,35)| / |S_m(0,35)|.
При t<0,35 эта доля равна единице; дополнительный count_ratio отдельно показывает рост числа кандидатов.
Model-specific spans отсутствуют у двух других моделей при том же численном t; это описание множеств,
а не сравнение вероятностей ошибки. Nested/crossing pairs считаются по уникальным spans; альтернативные
типы — дополнительные hint records относительно числа уникальных spans.

Exact assisted matching: число reference spans с идентичными document/start/end, делённое на 158.
Overlap matching требует только непустого пересечения в том же документе и поэтому является более
слабым диагностическим признаком. Длинный неверный span может пересекать несколько reference anchors;
overlap matching нельзя трактовать как полноту корректных границ.

## 7. Результаты E1/E2/E3

**Таблица 2. Все 24 комбинации model + threshold.** Время NER общее для восьми постфильтров одной модели.

'''
    text+=table(['Модель','t','Mentions','Spans','Spans/doc','Spans/100 слов','Пустых документов','Exact draft /158','Qwen tokens','Qwen calls'],[
        [m['configuration'],f"{m['threshold']:.2f}",m['mentions'],m['unique_spans'],f"{m['spans_per_document']:.2f}",
         f"{m['spans_per_100_words']:.2f}",m['documents_without_spans'],ref(m['configuration'],m['threshold'])['exact_matches'],
         cost(m['configuration'],m['threshold'])['input_tokens'],cost(m['configuration'],m['threshold'])['batch_calls']] for m in metrics])
    text+='''

E3 имеет больше exact совпадений с выбранным assisted reference на всех исследованных порогах,
но это не доказанное преимущество по истинному Recall. Reference мал, создан ассистентом и используется
при выборе рабочей точки. Истинный Precision не оценён: существенная часть census остаётся AMBIGUOUS.
Исторические 435 spans E3 при 0,35 относятся к прежним 12 документам; 1211 spans в новой серии
не являются улучшением или деградацией относительно того числа без учёта другого корпуса.

## 8. Влияние threshold на количество spans

**Рисунок 1. Количество уникальных spans.**

![Threshold и число кандидатов](research_run2/figures/01_candidate_counts.png)

**Рисунок 2. Сохранение baseline 0,35.**

![Сохранение кандидатов](research_run2/figures/02_candidate_retention.png)

'''
    text+=table(['Модель','t','Потеряно от 0,15','Сохранено от 0,35, %','Nested pairs','Crossing pairs','Альтернативные hints','Model-specific'],[
        [m['configuration'],f"{m['threshold']:.2f}",m['spans_lost_vs_015'],f"{100*m['retained_fraction_vs_035']:.1f}",
         m['nested_pairs'],m['crossing_pairs'],m['alternative_hints'],m['model_specific_spans']] for m in metrics])
    text+='''

E1/E2 теряют значительную часть spans уже при 0,45. E2 оставляет один физический документ
`W2317432785` без единого кандидата начиная с 0,45; при 0,65 пустыми становятся ещё
`W2084709926` и `W3138328568`. У E1/E3 ни один документ полностью не опустел на исследованных порогах.
Это не означает достаточную полноту: документ с одним оставшимся кандидатом может потерять почти
все содержательные термины. Все 720 document/model/threshold наблюдений доступны в
[document_metrics.csv](research_run2/document_metrics.csv).

## 9. Анализ потери значимых научных сущностей

**Рисунок 3. Exact matching с assisted reference.** Это диагностическая доля совпадений, не кривая Recall.

![Совпадения с assisted reference](research_run2/figures/04_assisted_reference.png)

'''
    text+=table(['Модель','t','Exact','Overlap','Нет exact','Нет overlap','Exact retained vs 0,35, %'],[
        [r['configuration'],r['threshold'],r['exact_matches'],r['overlap_matches'],r['missing_exact'],r['missing_overlap'],
         f"{100*float(r['exact_retention_vs_035']):.1f}"] for r in reference])
    text+='''

Гипотеза для E3 проверена непосредственно на последовательных шагах. Таблица показывает новые exact
потери относительно предыдущего исследованного threshold; разные вхождения одного surface считаются отдельно.
При 0,55 новых exact потерь на этих 158 anchors не было, но overlap уменьшился на единицу и полезные
кандидаты вне reference всё равно могли исчезнуть.

'''
    events=jsonl(ROOT/'threshold_examples.jsonl')
    text+=table(['Переход к t','Новые exact потери E3','Примеры (draft)'],[
        [f"{t:.2f}",len(next(e for e in events if e['configuration']=='E3' and e['threshold']==t and e['event']=='newly_lost_assisted_terms')['examples']),
         '; '.join(f"{x['surface']} [{x['suggested_type']}, {x['document_id']}]" for x in
            next(e for e in events if e['configuration']=='E3' and e['threshold']==t and e['event']=='newly_lost_assisted_terms')['examples']) or 'Новых exact потерь не наблюдалось'] for t in THRESHOLDS])
    text+='''

Особенно существенны `optimised blockwise non-local means` и `OB-NLM`, удалённые при 0,45;
`magnetic resonance imaging`, `inbreeding protocol` и `SNP-based invariant genomic probes`, удалённые при 0,50.
Повышение до 0,60 дополнительно теряет `graphics processing units`, `grain formation` и отдельные B-cell terms;
при 0,65 исчезают `denoising`, `magnetic properties` и `grain price` в соответствующих вхождениях.
В дополнительной выборке есть важный низкопороговый артефакт `papal annual medals` со score E3 около 0,19:
его отсутствие уже при 0,35 подчёркивает, что baseline сам не обеспечивает полноту.

Для каждой комбинации сохранены пять категорий событий: полезный термин сохранился; draft-шум удалён;
значимый anchor потерян; неверная альтернативная метка удалена при сохранении подходящей; документ опустел.
Если событие не наблюдалось, оно явно отмечено `NO_OBSERVED_EVENT`, пример не выдумывался:
[threshold_examples.jsonl](research_run2/threshold_examples.jsonl).
Списки предполагаемых пропусков и пересекающихся кандидатов:
[audit_missing_reference_UNREVIEWED.jsonl](research_run2/audit_missing_reference_UNREVIEWED.jsonl).

**Таблица 3. E3: exact matching по предполагаемым типам reference.** Это группировка по draft-типам anchors,
не качество окончательной классификации NER.

'''
    text+=table(['Draft-тип','Всего reference']+[f'{t:.2f}' for t in THRESHOLDS],[
        [label,next(r['total'] for r in by_type if r['configuration']=='E3' and r['reference_type']==label)]+
        [next(r['exact_matches'] for r in by_type if r['configuration']=='E3' and r['reference_type']==label and float(r['threshold'])==t) for t in THRESHOLDS]
        for label in SHORT])
    text+='''

При 0,45 теряется единственный exact совпавший RESULT anchor из baseline E3. CONDITION и MECHANISM
не имели exact совпадений уже при 0,35; порог не может исправить upstream отсутствие такого span.
Защита редких групп чувствительна к единичным ассистентским решениям о границах и типах;
на этой таблице нельзя устанавливать популяционную полноту или надёжное ранжирование классов.

## 10. Анализ нерелевантных кандидатов

**Рисунок 4. Предположительно нерелевантные кандидаты в двух отдельных audit partitions.**

![Предположительный шум](research_run2/figures/03_draft_irrelevant.png)

'''
    text+=table(['Модель','t','Census present','Census IRRELEVANT','120-sample present','120-sample IRRELEVANT','Sample AMBIGUOUS'],[
        [n,f'{t:.2f}',q(n,t,'six_document_census')['present'],q(n,t,'six_document_census')['IRRELEVANT'],
         q(n,t,'stratified_120')['present'],q(n,t,'stratified_120')['IRRELEVANT'],q(n,t,'stratified_120')['AMBIGUOUS']]
        for n in ['E1','E2','E3'] for t in THRESHOLDS])
    text+='''

В полном census E3 при 0,35/0,45/0,50/0,55 сохраняются три однозначных по правилам discourse-кандидата;
их число снижается до двух только при 0,60. Например, `paper` как DATA_RESOURCE имеет score около 0,69,
а одно `study` как TASK — около 0,74, поэтому они сохраняются при 0,65. В отобранных 120 E3 действительно
снижает число draft IRRELEVANT с 6 при 0,35 до 4 при 0,45 и до 1 при 0,65, но одновременно
теряет проверяемые полезные spans. Улучшение доли на этой стратифицированной выборке не является
измерением истинного Precision корпуса.

Большинство не включённых в reference census-кандидатов осталось AMBIGUOUS. Поэтому низкое число
IRRELEVANT нельзя интерпретировать как высокую доказанную точность. Лексический watchlist также
не универсален: `study` может быть заголовком/самоссылкой, а может быть глаголом с неполной границей
исследовательской задачи; в contextual amendments такие случаи разведены.

## 11. Анализ ошибок границ и типов

Две оси оценки сохраняются отдельно. Ниже приведены реальные source spans и предварительные
интерпретации ассистента, не подтверждённые независимым человеком.

'''
    selected_surfaces=['photon energy','total knee arthroplasty','medical records','stochastic nature','Yarrowia lipolytica',
        'military power supply','McMaster Universities Arthritis Index scores','PSCAD/EMTDC','UEDGE and BOUT',
        'papal annual medals','coding methodology','product titer']
    text+=table(['Surface / document','Draft boundaries','Предполагаемые типы','Hints и scores'],[
        [a['surface']+' / '+a['document_id'],a['draft_assessment']['boundary_category'],
         ', '.join(a['draft_assessment']['suggested_types']) or 'нет',
         '; '.join(n+': '+', '.join(p['label']+' '+f"{p['score']:.3f}" for p in a['predictions'][n]) for n in ['E1','E2','E3'] if a['predictions'][n])]
         for a in sample if a['surface'] in selected_surfaces])
    text+='''

Фрагменты контекста:

> Strict selection of **photon energy** and alignment with manufacturer-recommended dose limits

Источник W4391351014. По контексту это параметр (ATTRIBUTE), тогда как E3 предлагает CONCEPT со score ≈0,774.

> We retrospectively reviewed **medical records** of all patients

Источник W3101922761. При t=0,50 E1 теряет DATA_RESOURCE (≈0,368), сохраняя OBJECT (≈0,713).
Граница остаётся, но предварительная типизация становится менее подходящей по draft-интерпретации.

> Western Ontario and **McMaster Universities Arthritis Index scores**

Источник W2761455282. Кандидат не включает начало составного названия; высокая уверенность в DATA_RESOURCE
не устраняет предположительную ошибку границы и роли показателя.

Есть и полезное удаление альтернативы: у E3 `magnetic properties` при 0,50 сохраняет ATTRIBUTE
(≈0,604), теряя CONCEPT (≈0,491). У `blood samples` при 0,65 остаётся OBJECT, а DATA_RESOURCE исчезает.
Таким образом, влияние порога на типизацию не однонаправленно. Решение следует проверять на уровне
конкретного span/hint, а не считать исчезновение альтернативных типов автоматическим улучшением.

'''
    text+=table(['Модель','t=0,35, partition','TYPE_HINT_CORRECT','TYPE_HINT_INCORRECT','TYPE_HINT_AMBIGUOUS'],[
        [r['configuration'],r['partition'],r['TYPE_HINT_CORRECT'],r['TYPE_HINT_INCORRECT'],r['TYPE_HINT_AMBIGUOUS']]
        for r in type_quality if float(r['threshold'])==.35])
    text+='''

Counts относятся к hint records на spans с draft-категорией VALID_SPAN; это не проценты истинной точности.
Все thresholds и partitions представлены в [type_hint_audit_metrics.csv](research_run2/type_hint_audit_metrics.csv).

## 12. Междисциплинарное сравнение

**Таблица 4. Уникальные spans при 0,35 / 0,65 (по пять документов на область).**

'''
    fields=sorted({d['discipline'] for d in docs})
    text+=table(['Дисциплина','E1','E2','E3'],[
        [field]+[' / '.join(next(r['unique_spans'] for r in discipline if r['configuration']==n and float(r['threshold'])==t and r['discipline']==field) for t in [.35,.65])
                 for n in ['E1','E2','E3']] for field in fields])
    text+='''

История, биомедицина и физика существенно различаются по жанру, словарю, длине терминов и доступности
локальных сокращений. Пять работ на область и один reference-документ не позволяют установить устойчивый
discipline-specific threshold. Общие показатели могут скрыть почти полную потерю содержательных терминов
в отдельном тексте, как в физическом audit document E2. Counts всех типов/дисциплин/порогов сохранены
в [label_metrics.csv](research_run2/label_metrics.csv) и [discipline_metrics.csv](research_run2/discipline_metrics.csv).

**Таблица 5. Распределение hints E3 по всем порогам.** Оно не подтверждает корректность этих типов.

'''
    text+=table(['Hint']+[f'{t:.2f}' for t in THRESHOLDS],[
        [label]+[next(r['mentions'] for r in labels if r['configuration']=='E3' and float(r['threshold'])==t and r['label']==label) for t in THRESHOLDS]
        for label in SHORT])
    text+='''

## 13. Производительность CPU и RAM

Аппаратная платформа: AMD Ryzen 7 8845HS, 8 физических / 16 логических ядер; модели используют 4 CPU threads.
GPU не использовался. NER-конфигурации запускались последовательно отдельными процессами.
Во всех вариантах одинаковая политика: warm-up отсутствует, первый forward включён в inference time,
load time измеряется отдельно. На модель получен один успешный основной проход; три независимых timing
повтора не проводились. Следовательно, разброс и статистическая значимость времени неизвестны.

'''
    text+=f"Python {env['python']}; ОС `{env['platform']}`; RAM {env['ram_bytes']/2**30:.2f} GiB.\n\n"
    text+=table(['Библиотека','Версия'],env['versions'].items())
    text+='\n\n**Таблица 6. Фактические затраты при inference 0,15.**\n\n'
    text+=table(['Модель','Load, с','NLP load, с','NER, с','NORM, с','Processing, с','Total phases, с','Peak RSS, GiB','Doc/s','Words/s','Окон / encoder calls','Native checks, с'],[
        [n,f"{p['model_load_seconds']:.3f}",f"{p['normalizer_load_seconds']:.3f}",f"{p['inference_seconds']:.3f}",
         f"{p['normalization_seconds']:.3f}",f"{p['processing_seconds']:.3f}",f"{p['total_phase_seconds']:.3f}",
         f"{p['peak_rss_bytes']/2**30:.3f}",f"{p['documents_per_second']:.3f}",f"{p['words_per_second']:.3f}",
         f"{p['windows']} / {p['encoder_calls']}",f"{p['native_check_seconds']:.3f}"] for n,p in performances.items()])
    text+='''

Processing включает подготовку реального token budget, NER, нормализацию и внутри-документную сборку;
исключает отдельные native checks, checkpoint I/O и финальный экспорт. Total phases = model load + NLP load
+ processing; это не end-to-end время с сетью, tokenization Qwen и построением отчёта. Native checks
добавляют 18 encoder calls на модель и оплачиваются отдельно. Все восемь постфильтров разделяют одно
NER-наблюдение; восемь независимых замеров скорости не заявляются.

Peak RAM — максимум абсолютного process RSS с шагом сэмплирования 10 мс, включая библиотеки и модель.
Короткие пики могут быть пропущены; это working set процесса, не размер одних весов.
До каждого forward проверен фактический prompt+text budget; максимумы encoder inputs E1/E2/E3 —
185/265/333 токена при лимите 512. Покрыты все 5744 whitespace-слова, скрытого усечения не было.

**Рисунок 5. Кандидаты после фильтра и один фактический NER timing на модель.**

![NER cost и кандидаты](research_run2/figures/06_candidates_ner_time.png)

Реализованы атомарные per-document checkpoints с hash конфигурации и source text. После ошибки
повторно обрабатывается только незавершённый документ; несовпадение frozen inputs/config останавливает resume.
Завершённые 90 model/document checkpoints повторно прочитаны с запрещёнными конструкторами моделей:
получено ноль новых model loads/forward calls, содержимое checkpoints не изменилось.
Тест имитирует ошибку записи и подтверждает сохранность предыдущего завершённого checkpoint.

## 14. Влияние порога на нагрузку Qwen Entity Typing

Qwen не запускалась. Для конкретного planning checkpoint **Qwen/Qwen3-4B** загружены только tokenizer,
vocabulary, merges и конфигурации. Revision:
'''+f"`{tokenizer['revision']}`.\n\n"+'''
Токены посчитаны фактически этим tokenizer с chat template, `enable_thinking=False`,
`add_generation_prompt=True`, `return_dict=False`. Последний параметр существенен для установленной
версии transformers: по умолчанию результат является BatchEncoding, длина которого не равна числу tokens.
В финальном аудите повторно токенизирован по одному сохранённому batch для каждой из 24 комбинаций.

Единый prompt содержит правила решений ENTITY/NON_ENTITY/UNKNOWN и определения 11 типов Ontology v2.
User JSON включает document_id, полный неизменённый abstract и кандидатов с ID, start/end, surface,
N3 form и всеми альтернативными NER hints. Объединяются только идентичные source spans; semantic identity
не устанавливается. Batching выполняется отдельно внутри документа: не более 32 уникальных кандидатов
и 4096 input tokens на запрос. Пустые документы не требуют запросов. Контекст и system prompt повторяются
в каждом batch, поэтому token savings не пропорциональны сокращению числа spans.

JSON bytes измерены как длина UTF-8 compact JSON. Input tokens включают system, user payload и chat framing,
но не будущие output tokens. Это точные counts конкретного планового шаблона, а не измеренная latency Qwen;
другой prompt, checkpoint или batch policy изменит объём.

**Рисунок 6. Tokenizer-only оценка планового входа Qwen.**

![Вход Qwen](research_run2/figures/05_qwen_load.png)

'''
    text+=table(['Модель','t','Spans/doc','JSON без batching, bytes','JSON с batching, bytes','Input tokens','Batch calls','Экономия tokens vs 0,35, %'],[
        [c['configuration'],f"{c['threshold']:.2f}",f"{c['candidates_per_document']:.2f}",c['unbatched_json_bytes'],
         c['batched_json_bytes'],c['input_tokens'],c['batch_calls'],f"{100*c['token_reduction_vs_035']:.1f}"] for c in costs])
    text+='''

Для E3 повышение 0,35 → 0,45 снижает spans на 17,2%, input tokens на 14,3%, calls 53 → 48;
переход к 0,65 снижает spans на 45,7%, tokens на 41,1%, calls 53 → 35.
При этом exact reference совпадения падают 106 → 98 → 76. Экономия вычислений следующего этапа
достигается ценой предположительно значимых терминов. Без запуска Qwen нельзя утверждать, сколько
секунд будет сэкономлено и окажется ли вся архитектура быстрее при выборе E3 вместо E2.
E3 при 0,35 требует 151572 input tokens против 65314 у E2: преимущество времени NER само по себе
не означает преимущество полной будущей цепочки.

Полный шаблон: [qwen_prompt_template.json](research_run2/qwen_prompt_template.json).
Все реальные planning payloads и counts: [qwen_planned_batches.jsonl](research_run2/qwen_planned_batches.jsonl).

## 15. Обсуждение ограничений

Корпус мал, содержит только аннотации и не является репрезентативной выборкой всех научных областей.
Metadata и автоматические language/genre preflight checks не заменяют экспертизу. Часть формул повреждена
в исходном OpenAlex тексте; эти дефекты не исправлялись. Разные языковые и жанровые нормы гуманитарных
работ могут существенно влиять на NER и допустимые сущности.

Reference создан ассистентом до NER, однако не проверен человеком, неполон и включает субъективные
решения о nested spans и типах. Он снижает непосредственную зависимость от outputs сравниваемых моделей,
но не обеспечивает независимость от общих знаний/предубеждений ассистента. Например, статус народов,
исторических повествовательных акторов, измерений и mathematical results требует согласования.
Рекомендация выбрана на том же audit material; независимого holdout для threshold selection нет.

120 кандидатов отобраны с многомерной стратификацией, а не равновероятно из всего output. Доли шума
на этой выборке не являются population estimates. Полный census имеет много AMBIGUOUS cases, поэтому
его малое число IRRELEVANT также не доказывает высокой точности. Contextual amendments — работа ассистента,
не Human Gold. Интервалы неопределённости, статистическая значимость и истинные Precision/Recall/F1
не вычислялись. Boundary quality и type hints анализируются отдельно.

Scores архитектур не калиброваны; input API и overlap policies различаются. При E1/E2 меняются одновременно
окно и overlap. Один timing-проход без warm-up не оценивает дисперсию; background activity и page cache
могут влиять на load и скорость. Qwen cost — только токенизация planning prompt, без генерации,
KV-cache, output lengths, runtime scheduling или latency. Реальную пригодность рабочей точки нужно проверить
независимым человеческим аудитом, затем на отдельном корпусе и в отдельном задании на Qwen.

## 16. Выбор рекомендуемой модели и порога

'''
    text+=table(['Модель','Рабочий threshold по заданной политике','Spans','Exact draft /158','Qwen tokens','Qwen calls'],[
        [r['configuration'],r['recommended_threshold'],r['unique_spans'],r['reference_exact_matches'],r['qwen_input_tokens'],r['qwen_batch_calls']] for r in rec['per_model']])
    text+='''

Основной предварительный вариант — **E3 + threshold 0,35 + N3**. Он даёт 106/158 exact совпадений
с draft reference, 1211 source spans, 53 плановых Qwen batches и наименьшее измеренное время NER
среди сравниваемых конфигураций. Выбор обусловлен сочетанием наблюдаемого покрытия выбранных anchors,
ошибок и CPU-стоимости; он не означает доказанно лучшего качества или минимальной стоимости всей цепочки.
Контроль для экономного candidate generation — **E2 + 0,35 + N3**; E1 + 0,35 следует сохранить как
контроль с большим числом draft exact matches, чем E2, но более дорогим NER.

Гипотеза безопасного глобального повышения E3 до 0,45–0,65 **не подтверждена в строгом варианте**.
При 0,45 общие ограничения 90% сохранения exact matches и 80% по дисциплинам ещё выполняются,
но исчезает редкий RESULT anchor и полезные METHOD spans. В census шум при этом не уменьшается.
В диапазоне 0,50–0,65 потери scientific anchors заметно выше. Высокий threshold не устраняет уверенные
ошибки подсказок, в частности MECHANISM для surgical procedure и CONCEPT для photon energy.

При исключении защиты редких draft-групп рабочая точка E3 сдвигается к **0,45**; E1/E2 остаются на 0,35.
Дополнительная проверка, разрешающая выбирать из всех восьми порогов, включая 0,15 и 0,25,
с сохранением остальных ограничений также возвращает 0,35 для всех моделей. Низкие пороги не скрыты:
для E3 0,25 добавляет пять exact anchors к baseline при росте планового token input на 15,6%,
а 0,15 — одиннадцать anchors при росте input на 37,5%. Эти альтернативы требуют отдельного решения
о допустимой нагрузке; преимущество baseline не является универсально установленным оптимумом.
Следовательно, **0,35–0,45 — диапазон для последующего независимого аудита**, а 0,35 — более осторожный
рабочий вариант этой серии. Если допускается потеря восьми текущих exact anchors ради 14,3% меньшего
планового token input, 0,45 является явной альтернативой, а не установленным оптимумом.
Пороги 0,55/0,65 также могут быть полезны при жёстком бюджете, но в данной reference-диагностике
не удовлетворяют выбранной политике сохранения значимых терминов.
Сравнение при близком плановом бюджете показывает эту зависимость от приоритетов: E1/0,35 требует
85155 tokens и даёт 49 exact anchors, тогда как E3/0,65 требует 89207 tokens и даёт 76 anchors,
при существенно меньшем измеренном времени NER. Поэтому 0,65 нельзя считать заведомо непригодным:
он перспективен для отдельного budget-constrained сценария. Его недостаток здесь — потеря 30 exact
anchors относительно собственного E3 baseline и чувствительных научных терминов. Единую оптимальную
точку без явной цены такой потери и независимого аудита установить нельзя.

Рекомендация не подбирает параметры индивидуально под отдельные документы или discipline outputs;
корпус не менялся. Выбранные trade-off правила и их чувствительность доступны в
[recommendation.json](research_run2/recommendation.json) и
[recommendation_sensitivity.json](research_run2/recommendation_sensitivity.json).

## 17. Выводы

Проведены реальные inference E1/E2/E3 на 30 новых замороженных OpenAlex abstracts, проверены восемь
thresholds и эквивалентность постфильтрации native decoding в 54 контрольных вызовах. Зафиксированы
158 независимых от новых предсказаний draft-reference spans, 433 census-кандидата и 120 дополнительно
стратифицированных кандидатов с contextual assistant review. Высокие scores не гарантируют корректной
границы или hint; повышение порога может как исправить, так и ухудшить набор альтернативных типов.

У E3 снижение числа кандидатов и Qwen tokens при более строгом threshold сопровождается потерями
именованных методов, сокращений, объектов, процессов и результатов. Поэтому рабочий выбор сделан
в пользу E3/0,35/N3, с исследовательской альтернативой 0,45 при явном принятии выявленных потерь.
Влияние порога на время самого NER при постфильтрации не измерялось как восемь отдельных inference.
Следующая необходимая проверка — human adjudication boundary/type axes и независимый threshold holdout.
Ни один последующий архитектурный этап в этой серии не запускался.

## 18. Источники и воспроизводимость

1. Zaratiana U. et al. GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer.
   [arXiv:2311.08526](https://arxiv.org/abs/2311.08526).
2. Zaratiana U. et al. GLiNER2: An Efficient Multi-Task Information Extraction System with Schema-Driven Interface.
   [arXiv:2507.18546](https://arxiv.org/abs/2507.18546). Эта работа описывает семейство, а не измерения текущей серии.
3. [Официальный GLiNER2 repository](https://github.com/fastino-ai/GLiNER2),
   [GLiNER2.5 Base model card](https://huggingface.co/fastino/gliner2.5-base-v1).
4. [OpenAlex API reference](https://help.openalex.org/api/) и
   [API recipes с sample/seed](https://help.openalex.org/how-to/api-recipes/).
5. [Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B). В исследовании использован только tokenizer.
6. [Предыдущее исследование NER-V2-002/NORM-V2-001](NER_V2_002_NORM_V2_001_RESEARCH_REPORT.md),
   [архитектура v2](../../docs/architecture/SEQUENTIAL_HYBRID_KG_V2_DRAFT.md),
   [Ontology v2](../../contracts/ner_v2/ontology.v2.json).

Внешние числовые результаты не переносятся на текущий корпус. Все таблицы получены из сохранённых
outputs, проверок или фактической tokenization. Официальные API и model card использованы для интерфейсов
и provenance; выводы о качестве ограничены собственными UNREVIEWED материалами.

Карта артефактов:

- [Corpus manifest](research_run2/corpus_manifest.json), [reference lock](research_run2/reference_lock.json),
  [протокол](research_run2/protocol.json), [checkpoint E1](research_run2/E1/config.json),
  [E2](research_run2/E2/config.json), [E3](research_run2/E3/config.json).
- Raw window predictions: `research_run2/E*/raw_windows.jsonl`; восемь exports:
  `E*/mentions_015.jsonl` … `E*/mentions_065.jsonl`; atomic checkpoints: `E*/documents/*.json`.
- [Все threshold metrics](research_run2/threshold_metrics.csv),
  [reference matching](research_run2/reference_metrics.csv),
  [draft quality](research_run2/audit_quality_metrics.csv),
  [Qwen cost](research_run2/qwen_cost_estimates.csv); machine-readable graph data находятся в этих CSV.
- [Environment](research_run2/environment.json), [пакеты](research_run2/requirements.txt),
  [Qwen tokenizer lock](research_run2/qwen_tokenizer_lock.json), [resume check](research_run2/resume_verification.json).
- [Технические tests](research_run2/technical_tests.xml): **79 passed**.
  [Integrity audit](research_run2/integrity_audit.json), [SHA-256 manifest](research_run2/sha256_manifest.json).
- [Runner](../../src/cgraph/ner_v2/quality_audit.py), [анализ](../../scripts/analyze_ner_v2_003.py),
  [reference builder без доступа к predictions](../../scripts/freeze_ner_v2_003_reference.py),
  [contextual draft annotations](../../scripts/annotate_ner_v2_003_sample.py),
  [генератор отчёта](../../scripts/write_ner_v2_003_report.py).

Воспроизведение из checkout, PowerShell (старые outputs не перезаписывать):

```powershell
$env:UV_PROJECT_ENVIRONMENT = '.venv-ner-v2'
uv sync --locked --extra ner-v2 --extra test --extra research
$py = '.\\.venv-ner-v2\\Scripts\\python.exe'
# При отсутствии локальных NER-asset: prepare-models и prepare_gliner25_research.py из предыдущего протокола.
& $py scripts/prepare_ner_v2_003_tokenizer.py
$replay = 'reports/ner_v2/research_run2_replay'
New-Item -ItemType Directory -Path $replay
$files = @('input_abstracts.jsonl','corpus_manifest.json','assisted_reference_frozen.jsonl',
           'reference_lock.json','protocol.json','qwen_tokenizer_lock.json','assistant_audit_amendments.json')
foreach ($file in $files) { Copy-Item -LiteralPath "reports/ner_v2/research_run2/$file" -Destination $replay }
foreach ($name in @('E1','E2','E3')) {
    & $py -m cgraph.ner_v2.quality_audit infer --config $name --output-root $replay
}
# Та же infer-команда после прерывания пропускает завершённые документы.
& $py scripts/analyze_ner_v2_003.py --output-root $replay
& $py -m pytest tests/test_ner_v2_003.py tests/test_ner_v2_research.py tests/test_ner_v2.py tests/test_contracts.py -q
```

Для новой выборки требуется отдельное исследование: нельзя заменять документы текущего frozen corpus.
Файлы NER/Qwen weights не добавляются в Git. Все source hashes и checkpoints закреплены; изменения
исторических результатов и ontology проверяются final audit относительно исходного commit.
'''
    Path('reports/ner_v2/NER_V2_003_CANDIDATE_QUALITY_AUDIT.md').write_text(text,encoding='utf8',newline='\n')
    archive=text.replace('](research_run2/','](').replace('](../../','](../../../').replace('](NER_V2_002_','](../NER_V2_002_')
    (ROOT/'NER_V2_003_CANDIDATE_QUALITY_AUDIT.md').write_text(archive,encoding='utf8',newline='\n')
    write_thesis_fragment()
    print('report words',len(text.split()))


def write_thesis_fragment():
    text='''# Исследование качества NER-кандидатов и выбор порога уверенности

Готовый фрагмент для дипломной работы. Таблицы и рисунки относятся к
[полному отчёту NER-V2-003](NER_V2_003_CANDIDATE_QUALITY_AUDIT.md).

На этапе построения графа знаний из научных публикаций обнаружение упоминаний должно обеспечивать
не только достаточно широкий набор терминов, но и приемлемую нагрузку на последующую контекстную
типизацию. Избыточные кандидаты увеличивают объём входа языковой модели, тогда как чрезмерная
фильтрация создаёт необратимые пропуски. Поэтому в эксперименте NER-V2-003 исследовано влияние
порога уверенности на состав научных упоминаний, предварительные ошибки границ и типов,
вычислительную стоимость NER и плановый объём Qwen Entity Typing 4B.

Для сравнения сформирован новый междисциплинарный корпус из тридцати англоязычных аннотаций
OpenAlex, по пять работ из информатики, медицины, физики, биологии, экономики и истории. Отбор
использовал seed 4303, разные primary topics внутри области, проверку длины 100–400 слов и
исключение прежних двенадцати публикаций по идентификатору, DOI и хешу текста. Дополнительная
проверка исходных аннотаций до NER позволила исключить ошибочно помеченный английским русский
текст и книжную рецензию. После окончательной заморозки документы не заменялись. Корпус содержит
5744 whitespace-слова; его состав и provenance приведены в таблице 1 полного отчёта.

Сопоставлены GLiNER Large v2.1 с окном 60 слов и overlap 20 (E1), тот же checkpoint с окном
110 слов и overlap 25 (E2), а также GLiNER2.5 Base с boundary architecture при окне 110/25 (E3).
Применялись одинаковые короткие описания одиннадцати категорий Ontology v2 и утверждённая
нормализация N3. Веса закреплены точными revisions, обучение не проводилось. Каждый вариант
выполнялся отдельно на CPU с четырьмя потоками и seed 42. Точные Unicode spans и исходные
поверхностные формы сохранялись независимо от нормализованных строк и подсказок типов.

Для всех моделей исследованы пороги 0,15, 0,25, 0,35, 0,45, 0,50, 0,55, 0,60 и 0,65.
Основной inference выполнялся при минимальном пороге, после чего outputs фильтровались по scores.
Допустимость переиспользования проверена по decoder и отдельными native вызовами при 0,35,
0,50 и 0,65 на первом окне каждого из шести аудитных документов. Все 54 проверки дали
одинаковые spans, hints и scores в установленном допуске. При этом сохранено различие строгой
границы score > threshold у v2.1 и включительной границы у E3. Восемь постфильтров не
рассматриваются как восемь независимых измерений скорости.

Поскольку независимая человеческая разметка отсутствует, основной акцент сделан на диагностике
конкретных ошибок. До запуска NER для одного случайно выбранного документа каждой дисциплины
составлены списки предполагаемо значимых научных выражений непосредственно по источнику.
Зафиксировано 158 точных reference spans, включая повторные и вложенные упоминания. Эти списки
имеют статус ASSISTED_REFERENCE_DRAFT: они неполны, созданы ассистентом и не являются Gold.
Измеряются exact и overlap совпадения с ними, но не истинные Recall или F1.

Полный union низкопороговых outputs на шести документах содержит 433 уникальных spans.
Из остальных двадцати четырёх статей дополнительно выбраны 120 кандидатов, по двадцать на
область, с балансированием по модели, hint, confidence range и наличию точного совпадения у
нескольких моделей. Контекст всех 120 кандидатов дополнительно прочитан ассистентом. Предложены
отдельные категории качества границы и качества типа; все оценки сохраняют статус UNREVIEWED,
а поля для будущей человеческой проверки остаются пустыми. Доли ошибок в этой стратифицированной
выборке не переносятся автоматически на весь корпус.

Результаты всех двадцати четырёх комбинаций приведены в таблице 2, изменение числа spans
показано на рисунках 1 и 2. При пороге 0,35 E1 даёт 635 уникальных кандидатов, E2 — 413,
E3 — 1211. При 0,65 эти значения снижаются соответственно до 202, 124 и 658. Однако
уменьшение выходного набора само по себе не означает улучшения качества. В E2 один физический
документ остаётся без единого кандидата уже при 0,45; при 0,65 полностью опустошаются три
документа. У E1 и E3 таких случаев не обнаружено, но отсутствие пустых outputs не гарантирует
сохранения необходимых терминов внутри каждого текста.

Exact совпадения с assisted reference при 0,35 составляют 49 из 158 для E1, 35 для E2
и 106 для E3. Диагностические кривые представлены на рисунке 3. Для E3 повышение до 0,45
снижает число совпадений до 98, до 0,50 — до 92, а до 0,65 — до 76. Среди потерянных
при 0,45 выражений находятся optimised blockwise non-local means и сокращение OB-NLM.
При 0,50 дополнительно исчезают magnetic resonance imaging, inbreeding protocol и отдельные
генетические термины. Таким образом, повышение confidence затрагивает научные методы и
объекты, а не только слабые или неинформативные кандидаты.

Анализ по предполагаемым типам reference в таблице 3 показывает неоднородность этих потерь.
У E3 единственный exact совпавший RESULT anchor baseline исчезает уже при 0,45.
CONDITION и MECHANISM не имели exact совпадений при 0,35, поэтому повышение порога не может
исправить их исходное отсутствие. Эти observations требуют осторожной интерпретации:
draft-типы и выбранные границы могут быть пересмотрены человеком, а единичный anchor
не характеризует устойчивую полноту модели по целому классу.

Предположительно нерелевантные candidates анализируются отдельно в полном наборе шести документов
и в выборке 120 (рисунок 4). В census E3 сохраняет три отмеченных discourse-кандидата
при всех порогах от 0,35 до 0,55. В дополнительных 120 число draft IRRELEVANT снижается
с шести при 0,35 до четырёх при 0,45 и одного при 0,65, но одновременно теряются полезные
упоминания. Большинство не включённых в reference census-кандидатов осталось AMBIGUOUS;
поэтому малое число явно отмеченного шума не доказывает высокую истинную точность.

Высокие scores также не гарантируют подходящий онтологический hint. Например, E3 предлагает
CONCEPT для photon energy со score около 0,77 и MECHANISM для total knee arthroplasty
со score около 0,87. Такие ошибки сохраняются при всех рассматриваемых строгих порогах.
Существует и противоположный эффект: у E1 при 0,50 для medical records исчезает более
подходящий DATA_RESOURCE с низким score, тогда как OBJECT остаётся. Поэтому удаление
альтернативных типов нельзя автоматически считать улучшением предварительной классификации.

Фактические CPU-затраты представлены в таблице 6 и на рисунке 5. Непосредственный inference
занял 184,13 с для E1, 122,17 с для E2 и 42,15 с для E3; processing с нормализацией
и подготовкой входа — 186,63, 124,31 и 44,58 с соответственно. Пиковый RSS составлял
около 3,82 GiB для E1/E2 и 1,97 GiB для E3. Измерены 143 рабочих окна у E1 и 75
у E2/E3, без скрытого усечения текста. На модель выполнен один успешный основной проход,
поэтому дисперсия времени и статистическая значимость различий не оценены.

Нагрузка Qwen оценена без генерации и без загрузки её весов: использован только pinned
tokenizer Qwen3-4B. Единый planning prompt включает определения типов и полный abstract,
а source-exact кандидаты группируются по документу в batches не более 32 spans и 4096
input tokens. Все альтернативные hints сохраняются. При E3/0,35 получены 151572 input
tokens и 53 плановых вызова; при 0,45 — 129895 tokens и 48 вызовов. Экономия составляет
14,3% tokens при одновременной потере восьми текущих exact anchors. При 0,65 число tokens
снижается до 89207, но exact совпадения уменьшаются до 76. Эти counts, показанные на
рисунке 6, не являются измерением latency или стоимости Qwen в секундах.

Для рабочей точки применена заранее заданная осторожная политика: сохранить не менее 90%
baseline exact совпадений, не менее 80% в каждой дисциплине, не терять редкие совпавшие
draft-группы и не увеличивать census-шум. В рамках этой политики предварительно выбран
E3 + threshold 0,35 + N3. Контрольный более компактный выход предоставляет E2/0,35/N3.
Исключение защиты редких групп сдвигает рабочую точку E3 к 0,45, поэтому диапазон
0,35–0,45 обоснован как предмет последующего независимого аудита. Это условная инженерная
рекомендация, а не доказанный универсальный оптимум.

Исследование показывает, что глобальное повышение confidence до 0,45–0,65 не обеспечивает
бесплатного устранения шума: оно сокращает нагрузку на типизацию ценой конкретных научно
значимых терминов и не устраняет уверенные ошибки hints. Дальнейшее решение требует
человеческого согласования границ и типов, независимой выборки для threshold selection
и отдельной оценки реальной Qwen-цепочки. Все исходные тексты, предсказания, draft-оценки,
token counts и checksums сохранены для такой проверки.
'''
    Path('reports/ner_v2/NER_V2_003_THESIS_FRAGMENT.md').write_text(text,encoding='utf8',newline='\n')
    (ROOT/'NER_V2_003_THESIS_FRAGMENT.md').write_text(text,encoding='utf8',newline='\n')
    print('thesis fragment words',len(text.split()))


if __name__=='__main__':main()
