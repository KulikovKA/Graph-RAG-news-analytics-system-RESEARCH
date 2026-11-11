"""Render the 2F.0 report from analyze.py outputs; no inference or network access."""
from pathlib import Path
from collections import Counter
import csv
import json

OUT = Path(__file__).resolve().parent


def read(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def csv_out(name, rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def esc(value):
    return str(value).replace('|','\\|').replace('\n',' ')


def table(headers, rows):
    return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+[
        '| '+' | '.join(esc(x) for x in row)+' |' for row in rows]+['']


def f(value, digits=2):
    return f'{float(value):.{digits}f}'


def main():
    comp=read('comparison.csv')
    entities=read('entity_confidences.csv')
    source=read('source_entity_confidences.csv')
    summary=read('distribution_summary.csv')
    hist=read('confidence_histogram.csv')
    docs=read('retention_by_document.csv')
    disciplines=read('retention_by_discipline.csv')
    examples=read('examples.csv')
    # Context-led illustrative review, not new NER labels and not a precision estimate.
    notes={
        'approach':('generic_reference','Общее анафорическое обозначение нового подхода; без контекста название малоинформативно.'),
        'accurate':('incomplete_property','Один модификатор proposal boxes; слабое самостоятельное название вершины.'),
        'top-to-bottom back':('incomplete_span','Отрезан head attention modules; проблема границы span, а не редкий самостоятельный термин.'),
        'clear':('contextual_property','Клинически значимое состояние corneal grafts, но изолированное прилагательное имеет слабую самостоятельность. Не объявлять полностью бесполезным.'),
        'Internal Family Systems':('specific_term','В контексте прямо назван терапевтический подход IFS; содержательная потеря уже при 0.40.'),
        'magnetic field shear':('specific_term','Физическая величина/концепт, связанный с асимметрией турбулентности; низкий score не означает шум.'),
        'deuterium emission fluctuation':('specific_term','Конкретный объект измерений в диверторе; содержательная потеря при 0.40.'),
        'region proposal network':('specific_term','Прямое название RPN в object detection. Удаляется этот span, а не доказанно весь концепт во всех других формулировках.'),
        'optimised blockwise non-local means (OB-NLM)':('specific_term','Конкретный алгоритм GPU-denoising. Полное название теряется уже при 0.40; сокращённые варианты могут оставаться.'),
        'P2X(7) receptor antagonists':('specific_term','Конкретные антагонисты рецептора, использованные в эксперименте; потеря при 0.50.'),
        'synapomorphy':('specific_term','Точный эволюционный термин в выводе о потере acpP1; сохраняется при 0.50, исчезает при 0.60.'),
        'immunogenic hotspots':('specific_term','Области, распознаваемые HLA-DRB1*09, непосредственно исследуются в работе; потеря при 0.50.'),
        'diatom phylogeny':('specific_term','Предмет исследования эволюции диатомей; потеря при 0.50.'),
        'Pearson product-moment correlation coefficient':('specific_term','Назван конкретный коэффициент сравнения MD measurements; потеря полного названия при 0.50.'),
        'cardiac implantable electronic device (CIED)':('specific_term','Центральное устройство исследования; полное название теряется при 0.50. Отдельные CIED/CIEDs имеют другие scores.'),
        'green banking':('specific_term','Тема статьи и перечня финансовых инструментов; потеря этого span при 0.40.'),
        'advance notification':('specific_term','Конкретное предварительное уведомление о закрытии предприятий; содержательное экономическое понятие, потеря при 0.40.'),
        'co-investment':('specific_term_type_uncertain','В контексте конкретный финансовый инструмент; сомнения в OBJECT не делают PROCESS бессодержательным. Потеря при 0.50.'),
        'securitization':('specific_term','Конкретный финансовый процесс; сохраняется при 0.60 и теряется при 0.70.'),
        'clericalism':('specific_term','Конкретный термин в обсуждении церковного лидерства; сохраняется при 0.50, исчезает при 0.60.'),
        'symmetrical Lucas sine':('specific_term','Источник явно перечисляет эту форму решений; оценка математической корректности статьи не выполнялась. Потеря при 0.50.'),
        'X-ray magnetic circular dichroism (XMCD)':('specific_term','Название измерительного метода; сохраняется при 0.60, исчезает при 0.70.'),
        'Harvest Paradox':('specific_term','Экономический феномен прямо обсуждается в тексте. Разные spans одного названия имеют разные scores.'),
        'article':('generic_reference','Ссылка автора на саму статью. Max member поднимает score до 0.967406 и сохраняет общее название даже при 0.90.'),
        'ATP':('specific_term','Конкретная молекула и участник описанного механизма; max member сохраняет одну существующую группу при 0.50–0.80.'),
        'Western Ontario':('incomplete_span','В составе Western Ontario and McMaster Universities Arthritis Index scores это фрагмент названия шкалы; высокий max score не исправляет границы или тип ACTOR.'),
    }
    reviewed=[]
    for e in entities:
        name=e['name']
        if name.casefold()=='patients':
            category='contextual_cohort'
            note='Участники/когорта исследования, иногда анафорическое подмножество. Не собственное имя, но может быть полезной конечной вершиной клинического отношения; оценивать по контексту.'
        elif name=='RESULTS':
            category='section_heading'
            note='Заголовок секции перед двоеточием. Это структурная метка, а не конкретный научный результат; score сам по себе не устраняет такие метки.'
        elif name=='results':
            category='generic_result_reference'
            note='Общее обозначение результатов; фактическое содержание находится в предложении. Обычно малоинформативное самостоятельное название, но не доказанная ошибка NER.'
        elif name in notes:
            category,note=notes[name]
        else:
            continue
        reviewed.append(dict(canonical_entity_id=e['canonical_entity_id'],document_id=e['document_id'],discipline=e['discipline'],name=name,
                             representative=e['representative'],max_member=e['max_member'],
                             name_document_frequency_casefold=e['name_document_frequency_casefold'],assessment=category,reason=note,
                             **{f'max_kept_t{int(t*100):02}':float(e['max_member'])>=t for t in [.3,.4,.5,.6,.7,.8,.9]},
                             representative_context=e['representative_context'],max_member_context=e['max_member_context']))
    csv_out('qualitative_review.csv',reviewed)
    lines=['# Этап 2F.0 — NER Entity Confidence Threshold Analysis','',
           'Дата анализа: 11 ноября 2026, Europe/Moscow. Ветка: `design-v2-sequential-kg`. Только локальный анализ зафиксированного baseline; фильтр в рабочий граф не применён.','',
           '## Результат и выбор кандидатов','',
           '**Перспективные пороги: 0.40 и 0.50, основная стратегия — max member confidence.** 0.40 — предпочтительный осторожный кандидат: удаляет 110 вершин (9.80%), сохраняя 88.14% междокументных связей. 0.50 — более агрессивный кандидат: удаляет 292 вершины (26.02%), но сохраняет только 57.73% междокументных связей. Это кандидаты по компромиссу объёма, контекстных потерь и связности, а не доказанные оптимумы качества.','',
           'Порог 0.60 исключён из короткого списка: крупнейшая компонента уменьшается с 580 вершин / 23 документов до 116 / 6, остаётся 47.84% фактических рёбер; исчезают `synapomorphy` и `clericalism`. При 0.70–0.90 потери ещё сильнее; 0.90 оставляет лишь 3 из 194 междокументных связей и полностью опустошает документ W4381572413. Целевого числа вершин при выборе не использовано.','',
           '**Повышение confidence не отделяет надёжно бесполезные извлечения от научных терминов.** Уже при 0.40 исчезают полные названия OB-NLM, Internal Family Systems и `magnetic field shear`, тогда как общие `results` имеют score до 0.892064, а `article` — max member 0.967406. Уменьшение графа и сохранение его связности не равны улучшению precision или Graph-RAG.','',
           '## Данные, происхождение threshold и определения','',
           '- Вход: `../semantic_resolution/input/canonical_entities.jsonl`, 1122 канонические записи; `members.source_entity` точно восстанавливает все 1211 исходных сущностей и 1327 типизированных source mentions. 1211 — число исходных spans/entities, а не число независимых mention scores.','- У исходной сущности `confidence = max(source_mentions[].score)`: типы одного span уже агрегированы максимумом. Это проверено для всех записей. Здесь распределение 1211 confidence рассчитано по исходным сущностям; 1327 scores повторно не смешиваются с ними.','- `../.../reports/re_v2/relex_comparison/threshold035/e3_base/experiment_config.json` указывает SHA-256 именно `reports/ner_v2/research_run2/E3/mentions_035.jsonl`; хеш проверен. `scripts/build_relex_graphs.py:97–126` читает этот файл и группирует упоминания по span. **Применённый к baseline NER cutoff — 0.35, включительная граница E3.** `entity_threshold: null` означает внешние E3-сущности, а не отсутствие NER-отсечения.','- В `E3/config.json` historical `minimum_inference_threshold=0.15`. Это минимальный порог исторического инференса, а не cutoff текущего baseline. Нижние historical выгрузки не читались и не добавлялись. Строка 0.30 — тождественный baseline виртуальный подграф на наблюдаемом наборе, с эффективным нижним отсечением 0.35; оценить настоящий NER@0.30 по этим данным нельзя.','- Representative confidence — `confidence` того `members.source_entity`, чей `entity_id == canonical_entity_id`. Проверено совпадение с `context_source_entity_id`; это текущий представитель и плоский confidence в импорте 2E. Max member confidence — максимум `confidence` среди всех участников уже существующей канонической группы. Группы и представители не меняются.','- Из 1122 групп 62 содержат больше одного member; у 38 max member строго выше representative. Max member не является калиброванной вероятностью истинности: максимум может поддерживать и общий/ошибочно ограниченный span.','- Исходные 1760 RELEX_REL взяты из `relation_remapping/canonical_relations.jsonl`. Только готовые 685 SIMILAR_TO конфигурации `intfloat/e5-small-v2 / text_name / 0.95` выбраны из `semantic_links.csv`: 194 междокументных и 491 внутридокументное. Другие конфигурации исключены. Пороги ReLEx и embeddings не исследовались.','',
           'Для стратегии s и порога t: `V(s,t)={v: confidence_s(v)>=t}`, `E(s,t)={e∈E_baseline: head(e)∈V(s,t) и tail(e)∈V(s,t)}`. Удаление касается только вершин; все выжившие фактические и семантические отношения сохраняются с исходными ID и свойствами. Не проверяется confidence каждого member, создавшего исходное ребро: здесь фильтруется каноническая вершина целиком. Сохранившийся member не переносит score или тип на другие spans.','',
           'Связность рассчитана только на entity-вершинах и RELEX_REL/SIMILAR_TO: направления игнорируются для слабых компонент; документы и RELEX_MENTIONS не участвуют. SIMILAR_TO считается один раз на пару, параллельные фактические отношения считаются раздельно. Исходные 13 фактических петель сохраняются, пока жива вершина; вершина с петлёй не изолирована (степень > 0), хотя остаётся одноэлементной компонентой. Изолированная вершина — без единого инцидентного ребра. При равенстве размеров крупнейшей компоненты используется минимальный canonical ID.','',
           'Среднее число сущностей — на все 30 документов, включая опустевшие. В многотипной сущности каждый исходный тип учитывается по одному разу; поэтому суммы по типам могут превышать число вершин. Междокументные фактические отношения в baseline отсутствуют: соответствующий счётчик во всех вариантах равен 0, все междокументные связи здесь — SIMILAR_TO.','',
           '## Распределение confidence','']
    overall=[r for r in summary if r['scope']=='all']
    lines+=table(['Набор','N','min','p10','p25','median','mean','p75','p90','max'],[
        [r['level'],r['count']]+[f(r[k],6) for k in ['minimum','p10','p25','median','mean','p75','p90','maximum']] for r in overall])
    levels=['source_entities','canonical_representative','canonical_max_member']
    bins=[r for r in hist if r['scope']=='all' and r['level']==levels[0]]
    lines+=table(['Интервал','Исходные (1211)','Canonical representative (1122)','Canonical max member (1122)'],[
        [f"[{r['lower_inclusive']}; {r['upper']}{']' if r['upper_inclusive']=='True' else ')'}"]+
        [next(x['count'] for x in hist if x['scope']=='all' and x['level']==level and x['lower_inclusive']==r['lower_inclusive']) for level in levels] for r in bins])
    lines+=['Нули ниже 0.35 означают отсутствие записей в отсечённом baseline, а не отсутствие предсказаний модели в этих интервалах. Гистограммы, среднее и квантили по **каждому типу и каждому документу** для всех трёх наборов сохранены в `confidence_histogram.csv` и `distribution_summary.csv`. Квантили используют линейную интерполяцию по позиции `(n−1)p`.','',
            '### По типам: число / среднее / медиана','']
    types=sorted({r['group'] for r in summary if r['scope']=='type'})
    lines+=table(['Тип']+levels,[[t]+[next(f"{r['count']} / {f(r['mean'],4)} / {f(r['median'],4)}" for r in summary if r['scope']=='type' and r['group']==t and r['level']==level) for level in levels] for t in types])
    lines+=['### По документам: исходный / канонический набор и основная стратегия','',
            'В колонках mean/median: source entities → canonical max member. Все representative-квантили также доступны в CSV.','']
    doc_ids=sorted({e['document_id'] for e in entities})
    dr=[]
    for doc in doc_ids:
        a=next(r for r in summary if r['scope']=='document' and r['group']==doc and r['level']=='source_entities')
        b=next(r for r in summary if r['scope']=='document' and r['group']==doc and r['level']=='canonical_max_member')
        dr.append([doc,next(e['discipline'] for e in entities if e['document_id']==doc),a['count'],b['count'],f"{f(a['mean'],3)} → {f(b['mean'],3)}",f"{f(a['median'],3)} → {f(b['median'],3)}"]+
                  [next(r['retained'] for r in docs if r['document_id']==doc and r['strategy']=='max_member' and float(r['threshold'])==t) for t in [.4,.5,.6,.7,.8,.9]])
    lines+=table(['Документ','Дисциплина','Source N','Canon N','Mean','Median','≥.4','≥.5','≥.6','≥.7','≥.8','≥.9'],dr)
    lines+=['## Сравнение всех виртуальных подграфов','',
            'Полная машиночитаемая таблица: [comparison.csv](comparison.csv). В ней также есть проценты сохранения, петли, число непустых документов, охват междокументными связями и типы. Строка 0.30 помечена `below_original_cutoff=True`.','']
    for strategy in ['max_member','representative']:
        lines += [f'### {strategy}'+(' — основная стратегия' if strategy=='max_member' else ''),'']
        lines += table(['t','Вершин','На док.','Удалено','RELEX_REL','SIMILAR_TO','Междок.','Изоляты (%)','WCC','Max WCC','Док. в max'],[
            [f(r['threshold'],2),r['entities'],f(r['entities_per_document']),r['entities_removed'],r['relex_rel'],r['similar_to'],r['cross_document_links'],f"{r['isolated_vertices']} ({f(r['isolated_pct'])}%)",r['weak_components'],r['largest_component_vertices'],r['largest_component_documents']] for r in comp if r['strategy']==strategy])
        lines += ['Распределение оставшихся сущностей по типам:','']
        lines += table(['Тип']+[f(t,2) for t in [.3,.4,.5,.6,.7,.8,.9]],[
            [t]+[r['type_'+t] for r in comp if r['strategy']==strategy] for t in types])
    lines += ['Абсолютное число WCC и изолятов часто уменьшается вслед за удалением вершин, но это не улучшение связности: доля изолятов у max member возрастает с 17.83% до 37.55% при 0.90. Крупнейшая компонента может смениться; её размер и документы пересчитаны, не перенесены из baseline.','',
              '### Разница стратегий при одном пороге','']
    deltas=[]
    for a,b in zip(comp[:7],comp[7:]):
        assert a['threshold']==b['threshold']
        deltas.append([a['threshold']]+[int(a[k])-int(b[k]) for k in ['entities','relex_rel','similar_to','cross_document_links']])
    lines+=table(['t','Доп. вершины max','Доп. RELEX_REL','Доп. SIMILAR_TO','Доп. междок.'],deltas)
    lines+=['Max member сохраняет `ATP` (representative 0.435676 → max 0.899140), но также `article` (0.404989 → 0.967406) и фрагмент названия шкалы `Western Ontario` (0.360369 → 0.890749). При 0.70 всего 8 дополнительных вершин поддерживают 16 дополнительных междокументных связей; их наличие само по себе не доказывает полезность. Все расхождения, обе оценки и оба контекста — в `strategy_disagreements.csv`.','',
            '### Смещение по дисциплинам','']
    lines+=table(['Дисциплина','Baseline']+[f(t,2)+' осталось (%)' for t in [.4,.5,.6]],[
        [discipline,next(r['baseline'] for r in disciplines if r['discipline']==discipline)]+
        [next(f"{r['retained']} ({f(r['retention_pct'])}%)" for r in disciplines if r['discipline']==discipline and r['strategy']=='max_member' and float(r['threshold'])==t) for t in [.4,.5,.6]]
        for discipline in sorted({e['discipline'] for e in entities})])
    lines+=['Уже при 0.40 сохраняется 83.77% сущностей CS/ML против 93.28% Medicine. На уровне документов разброс ещё сильнее: W2130236240 сохраняет 9/13 (69.23%), W7215137876 — 18/25 (72.00%). При 0.50 W4312422299 сохраняет 12/23 (52.17%), W4389076493 — 27/51 (52.94%). Это наблюдаемая неоднородность score, а не доказательство худшего качества извлечений в этих дисциплинах.','',
            '## Контекстная проверка качества','',
            'Это целевой иллюстративный разбор локальных контекстов, выполненный при подготовке отчёта, без отдельного запуска LLM или любой модели. Он не является случайной выборкой, экспертной gold-разметкой или оценкой precision/recall. Низкий confidence, общий термин и сомнительный тип — разные свойства. Все разобранные canonical IDs, причины и контексты сохранены в [qualitative_review.csv](qualitative_review.csv).','',
            '### patients и results','',
            '`patients`: 16 канонических вершин из 24 исходных сущностей. Это могут быть реальные участники или подмножества когорт; generic surface name не делает узел бесполезным. В W2761455282 низкий span `dialysis patients` представлен извлечением `patients` с 0.368889 и исчезает при 0.40, но другие пациенты с 0.754086–0.898713 остаются. В W4391351014 группа пациентов имеет representative 0.657950 и max 0.931172; max member сохраняет её даже при 0.90. Контекст «sixteen patients with esophageal cancer» несёт клиническую информацию.','',
            '`results` (casefold): 8 канонических/исходных вершин — 3 заголовка `RESULTS` и 5 общих ссылок `results`. Заголовки со score 0.446378 и 0.449832 удаляются при 0.50, но заголовок 0.564522 — лишь при 0.60. Общие ссылки в содержательных предложениях имеют 0.707956, 0.765157, 0.784150, 0.892064 и переживают 0.70. Порог 0.40 не удаляет ни одной из восьми. Для заголовка есть явное структурное основание считать вершину малоценной; для анафорического слова это оценка самостоятельной информативности, а не ошибка распознавания.','']
    lines+=table(['t','patients max / rep','results casefold max / rep'],[
        [f(t,2)]+[' / '.join(str(sum(float(e[s])>=t for e in entities if e['name'].casefold()==name)) for s in ['max_member','representative']) for name in ['patients','results']] for t in [.3,.4,.5,.6,.7,.8,.9]])
    lines+=['### Редкие и специфические понятия из разных дисциплин','',
            '«Редкий в корпусе» здесь означает document frequency exact casefold-названия = 1 из 30, без синонимического объединения. Это не оценка мировой редкости или научной корректности термина. Для перечисленных ниже названий DF=1; дисциплины перенесены из метаданных корпуса (например, статья про IFS находится в History). Потеря конкретного span/названия не обязательно означает исчезновение всего понятия: другие формулировки, аббревиатуры и типы не объединялись.','']
    selected=['optimised blockwise non-local means (OB-NLM)','region proposal network','magnetic field shear','Internal Family Systems','advance notification','green banking','P2X(7) receptor antagonists','immunogenic hotspots','Pearson product-moment correlation coefficient','cardiac implantable electronic device (CIED)','co-investment','symmetrical Lucas sine','synapomorphy','clericalism','X-ray magnetic circular dichroism (XMCD)','securitization']
    for name in selected:
        r=next(e for e in reviewed if e['name']==name)
        first=next(t for t in [.4,.5,.6,.7,.8,.9] if float(r['max_member'])<t)
        lines += [f"- **{esc(name)}** — {r['discipline']}, `{r['document_id']}`, confidence **{f(r['max_member'],6)}**, впервые удаляется при **{f(first,2)}**. {r['reason']}",f"  Контекст: «{r['max_member_context']}»",'']
    lines+=['Полезные удаления также есть: `approach` (0.353786) — общее обозначение подхода; `accurate` (0.363876) — одиночный модификатор; `top-to-bottom back` (0.359056) — неполный span перед `attention modules`. Все исчезают при 0.40. Однако `clear` (0.362754/0.490462) описывает прозрачность corneal grafts: слабое самостоятельное название не означает отсутствия клинического смысла. Высокий score тоже не гарантирует хорошую границу span (`Western Ontario`). Поэтому утверждение «удалённые вершины — шум» данными опровергается.','',
            '## Примеры для каждого варианта','',
            'Ниже по два ближайших к границе удалённых и сохранённых примера. Выбор детерминированный, не случайный; при t=0.30 удалённых в доступном baseline нет. `examples.csv` дополнительно содержит крайние scores. Confidence указан по текущей стратегии, контекст — от member, задающего этот score. Полные ID в CSV разрешают одинаковые названия; в отчёте дан короткий префикс ID.','']
    for strategy in ['max_member','representative']:
        for t in [.3,.4,.5,.6,.7,.8,.9]:
            lines += [f'### {strategy}, t={f(t,2)}','']
            for status in ['removed','retained']:
                pool=[r for r in examples if r['strategy']==strategy and float(r['threshold'])==t and r['status']==status][:2]
                if not pool:
                    lines+=['Удалённых нет: доступный набор уже отсечён при 0.35.','']
                for e in pool:
                    label='Удалена' if status=='removed' else 'Сохранена'
                    lines += [f"- {label}: **{esc(e['name'])}**, confidence **{f(e['confidence'],6)}**, `{e['document_id']}`, `{e['canonical_entity_id'][:14]}…`; rep={f(e['representative'],6)}, max={f(e['max_member'],6)}.",f"  Контекст: «{e['context']}»",'']
    lines+=['## Ограничения и воспроизводимость','',
            'Это ретроспективные индуцированные подграфы зафиксированного снимка, не результат повторного NER. Confidence не откалиброван под качество/полезность; не оценены retrieval/answer quality, научная корректность публикаций и абсолютный recall. Утрата междокументной связи может удалять как полезный мост, так и связь общих слов; все рёбра оставлялись строго по наличию концов.','',
            'Проверены: точное восстановление 1211 source entities и 1327 mentions; происхождение mentions_035 по SHA-256; границы confidence и исходное отсечение; все offsets и контексты; существование концов рёбер; исходные петли; 1122/1760/685/194 и связность 200 изолятов / 258 WCC / 580 вершин / 23 документа; вложенность наборов при росте порога; включение representative-набора в max member-набор. Полный перечень проверок и SHA-256 входов/защищённых предыдущих артефактов — [validation.json](validation.json).','',
            'Команды из корня репозитория:','',
            '```powershell','.venv/Scripts/python.exe reports/entity_resolution/relex_base_t035/ner_threshold_analysis/analyze.py','.venv/Scripts/python.exe reports/entity_resolution/relex_base_t035/ner_threshold_analysis/write_report.py','```','',
            'Оба скрипта пишут только в текущий каталог отчёта. `analyze.py` использует стандартную библиотеку и не импортирует проектный код; `write_report.py` использует matplotlib только для рисунка. Ни NER, ни ReLEx, ни embeddings/LLM, ни cosine similarity не запускались. Entity Merge не выполнялся. Neo4j не открывался. Исходные данные и предыдущие отчёты не изменены; хеши защищённых файлов сверены до и после анализа.','',
            '### Артефакты','',
            '- `comparison.csv`, `metrics.json` — все 14 вариантов и baseline.','- `source_entity_confidences.csv` (1211), `entity_confidences.csv` (1122) — все оценки и контексты.','- `distribution_summary.csv`, `confidence_histogram.csv` — распределения overall / type / document / discipline для source, canonical representative и canonical max member.','- `retention_by_type.csv`, `retention_by_document.csv`, `retention_by_discipline.csv` — полные разрезы для всех вариантов.','- `virtual_subgraphs.jsonl` — точные сохранённые vertex IDs и два набора edge IDs по каждому варианту; исходные записи и свойства доступны в неизменённых входах.','- `components.csv` — все слабые компоненты каждого варианта с составом вершин и документов.','- `strategy_disagreements.csv`, `examples.csv`, `qualitative_review.csv` — расхождения и контекстные примеры.','- `analysis.png`, `analysis.svg` — распределения, сохранение рёбер и размер крупнейшей компоненты.','',
            '![Распределения confidence и последствия фильтрации](analysis.png)','',
            '**Этап 2F.0 завершён только анализом NER thresholds. Изменения KG или дальнейшие эксперименты не выполнялись.**','']
    # Correct relative source reference without tying the report to an absolute workstation path.
    lines=[line.replace('`../.../reports/re_v2/relex_comparison/threshold035/e3_base/experiment_config.json`','`reports/re_v2/relex_comparison/threshold035/e3_base/experiment_config.json`') for line in lines]
    position=lines.index('## Распределение confidence')
    lines[position:position]=[
        'Исторический NER checkpoint E3: `fastino/gliner2.5-base-v1`, revision `54785d51df8d86a0d8f2eb212fec15871aad2c4d` (из сохранённого `E3/config.json`). Его загрузка и выполнение в этом этапе не выполнялись. Текущее состояние Neo4j не запрашивалось: baseline воспроизведён по локальным артефактам 2E и проверен по всем указанным размерам и метрикам связности.','']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    plot(hist,comp)
    print(f'Wrote report.md; qualitative review: {len(reviewed)} records; 14 virtual variants.')


def plot(hist, comp):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,3,figsize=(16,4.7),layout='constrained')
    colors=['#6b7280','#2563eb','#059669']
    for i,(level,label) in enumerate([('source_entities','Source (1211)'),('canonical_representative','Representative (1122)'),('canonical_max_member','Max member (1122)')]):
        data=[r for r in hist if r['scope']=='all' and r['level']==level and float(r['lower_inclusive'])>=.35]
        axes[0].bar([j+(i-1)*.25 for j in range(len(data))],[float(r['pct']) for r in data],width=.24,color=colors[i],label=label)
    axes[0].set_xticks(range(7),['.35–.4','.4–.5','.5–.6','.6–.7','.7–.8','.8–.9','.9–1'],rotation=30)
    axes[0].set_ylabel('% of observed entities')
    axes[0].set_title('Confidence distribution (cutoff .35)')
    axes[0].legend(fontsize=8)
    data=[r for r in comp if r['strategy']=='max_member']
    xs=[float(r['threshold']) for r in data]
    for key,label,color in [('entities_retained_pct','Entities','#2563eb'),('relex_rel_retained_pct','RELEX_REL','#9333ea'),('similar_to_retained_pct','SIMILAR_TO','#059669'),('cross_document_links_retained_pct','Cross-document','#ea580c')]:
        axes[1].plot(xs,[float(r[key]) for r in data],marker='o',label=label,color=color)
    axes[1].set_title('Max member: retention vs baseline')
    axes[1].set_ylabel('% retained')
    axes[1].set_xlabel('NER confidence threshold')
    axes[1].legend(fontsize=8)
    for strategy,label,color in [('max_member','Max member','#059669'),('representative','Representative','#2563eb')]:
        data=[r for r in comp if r['strategy']==strategy]
        axes[2].plot(xs,[int(r['largest_component_vertices']) for r in data],marker='o',label=label,color=color)
    axes[2].set_title('Largest weak component')
    axes[2].set_ylabel('Entity vertices')
    axes[2].set_xlabel('NER confidence threshold')
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis='y',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    for ax in axes[1:]:
        ax.axvspan(.4,.5,color='#fde68a',alpha=.25)
        ax.set_xticks(xs)
    fig.suptitle('2F.0 • NER threshold analysis • 30 documents • fixed edges, no model reruns\n'
                 '.30 = observed baseline only; predictions below the original .35 cutoff are unavailable',fontsize=12)
    fig.savefig(OUT/'analysis.png',dpi=160)
    fig.savefig(OUT/'analysis.svg')
    plt.close(fig)


if __name__=='__main__':
    main()
