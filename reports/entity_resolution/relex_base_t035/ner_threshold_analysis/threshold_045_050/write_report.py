"""Summarize and independently verify the .45–.50 induced subgraphs."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[4]
ER=OUT.parents[1]
THRESHOLDS=[i/100 for i in range(45,51)]


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def jl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def csv_out(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)


def fmt(x,d=2):
    return f'{float(x):.{d}f}'


def table(headers,rows):
    esc=lambda v:str(v).replace('|','\\|').replace('\n',' ')
    return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+[
        '| '+' | '.join(esc(v) for v in row)+' |' for row in rows]+['']


def main():
    comp=read(OUT/'comparison.csv')
    entities=read(OUT/'entity_confidences.csv')
    by={e['canonical_entity_id']:e for e in entities}
    prior=read(OUT.parent/'comparison.csv')
    old_entities=jl(ER/'semantic_resolution/input/canonical_entities.jsonl')
    relations=jl(ER/'relation_remapping/canonical_relations.jsonl')
    links=[r for r in read(ER/'semantic_resolution/global_linking_preview/semantic_links.csv')
           if (r['model'],r['mode'],float(r['threshold']))==('intfloat/e5-small-v2','text_name',.95)]
    graphs=jl(OUT/'virtual_subgraphs.jsonl')
    components=read(OUT/'components.csv')
    score_by={}
    for strategy in ['max_member','representative']:
        score_by[strategy]={}
        for e in old_entities:
            source=[m['source_entity'] for m in e['members']]
            value=max(v['confidence'] for v in source) if strategy=='max_member' else next(v['confidence'] for v in source if v['entity_id']==e['canonical_entity_id'])
            score_by[strategy][e['canonical_entity_id']]=value
    # Rebuild all induced subgraphs independently of analyze.py and validate DFS
    # component counts with disjoint-set union. No node-removal simulations.
    for graph in graphs:
        strategy,t=graph['strategy'],graph['threshold']
        ids={eid for eid,value in score_by[strategy].items() if value>=t}
        assert ids==set(graph['entity_ids'])
        rs=[r for r in relations if r['head_id'] in ids and r['tail_id'] in ids]
        ls=[r for r in links if r['left_canonical_id'] in ids and r['right_canonical_id'] in ids]
        assert set(graph['relex_rel_ids'])=={r['canonical_triple_id'] for r in rs}
        assert set(graph['similar_to_ids'])=={r['semantic_link_id'] for r in ls}
        parent={v:v for v in ids}
        def find(v):
            while v!=parent[v]:
                parent[v]=parent[parent[v]]
                v=parent[v]
            return v
        touched=set()
        for a,b in [(r['head_id'],r['tail_id']) for r in rs]+[(r['left_canonical_id'],r['right_canonical_id']) for r in ls]:
            touched.update((a,b))
            parent[find(a)]=find(b)
        groups={}
        for v in ids:
            groups.setdefault(find(v),set()).add(v)
        largest=sorted(groups.values(),key=lambda g:(-len(g),min(g)))[0]
        expected=dict(entities=len(ids),relex_rel=len(rs),similar_to=len(ls),cross_document_links=sum(r['scope']=='cross_document' for r in ls),
                      isolated_vertices=len(ids-touched),weak_components=len(groups),largest_component_vertices=len(largest),
                      largest_component_documents=len({by[v]['document_id'] for v in largest}))
        actual=next(r for r in comp if r['strategy']==strategy and float(r['threshold'])==t)
        assert all(int(actual[k])==value for k,value in expected.items())
        if t==.5:
            old=next(r for r in prior if r['strategy']==strategy and float(r['threshold'])==t)
            assert actual==old,'Prior 0.50 comparison differs'
    validation=json.loads((OUT/'validation.json').read_text(encoding='utf-8'))
    for path,digest in validation['protected_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    (OUT/'independent_validation.json').write_text(json.dumps(dict(
        variants_checked=len(graphs),method='disjoint-set union independent of analyze.py DFS',
        exact_vertex_and_edge_membership=True,component_metrics_match=True,prior_050_rows_identical_both_strategies=True,
        protected_files_checked=len(validation['protected_sha256']),protected_files_unchanged=True,
        models_launched=False,neo4j_accessed=False),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    deltas=[]
    removed=[]
    for strategy in ['max_member','representative']:
        for lo,hi in zip(THRESHOLDS,THRESHOLDS[1:]):
            a=next(r for r in comp if r['strategy']==strategy and float(r['threshold'])==lo)
            b=next(r for r in comp if r['strategy']==strategy and float(r['threshold'])==hi)
            row=dict(strategy=strategy,from_threshold=lo,to_threshold=hi)
            row.update({k+'_lost':int(a[k])-int(b[k]) for k in ['entities','relex_rel','similar_to','cross_document_links','largest_component_vertices','largest_component_documents']})
            deltas.append(row)
            for e in entities:
                if lo<=float(e[strategy])<hi:
                    removed.append(dict(strategy=strategy,from_threshold=lo,to_threshold=hi,canonical_entity_id=e['canonical_entity_id'],
                                        name=e['name'],types=e['types'],document_id=e['document_id'],discipline=e['discipline'],confidence=e[strategy],
                                        representative=e['representative'],max_member=e['max_member'],context=e[strategy+'_context']))
    csv_out('step_deltas.csv',deltas)
    csv_out('marginal_removed_entities.csv',removed)
    cross_lost=[]
    for r in links:
        a,b=by[r['left_canonical_id']],by[r['right_canonical_id']]
        if r['scope']=='cross_document' and .47<=min(float(a['max_member']),float(b['max_member']))<.48:
            cross_lost.append(dict(semantic_link_id=r['semantic_link_id'],left_id=a['canonical_entity_id'],left_name=a['name'],left_document=a['document_id'],left_confidence=a['max_member'],
                                   right_id=b['canonical_entity_id'],right_name=b['name'],right_document=b['document_id'],right_confidence=b['max_member'],
                                   original_similarity=r['similarity'],left_context=a['max_member_context'],right_context=b['max_member_context']))
    assert len(cross_lost)==2
    csv_out('cross_document_losses_047_048.csv',cross_lost)
    old=next(c for c in components if c['strategy']=='max_member' and float(c['threshold'])==.47 and c['component_rank']=='1')
    old_ids=set(json.loads(old['entity_ids']))
    split=[]
    for c in components:
        if c['strategy']=='max_member' and float(c['threshold'])==.48:
            overlap=old_ids & set(json.loads(c['entity_ids']))
            if overlap:
                split.append(dict(component_rank=c['component_rank'],vertices=len(overlap),document_ids=c['document_ids']))
    assert sorted(r['vertices'] for r in split)==[59,301]
    csv_out('largest_component_split_047_048.csv',split)
    lines=['# Уточнение NER thresholds: 0.45–0.50, шаг 0.01','',
           'Продолжение этапа 2F.0 по запросу пользователя. Исходный coarse sweep и его отчёт сохранены без изменений. Используются те же 30 документов, 1122 canonical entities из 1211 source entities, 1760 RELEX_REL и 685 готовых SIMILAR_TO (194 междокументных). Исходный NER cutoff — 0.35.','',
           '## Вывод','',
           '**В заданном диапазоне предпочтителен осторожный кандидат 0.45 (max member): 925 вершин, удалено 197 (17.56%), сохраняются 1352 RELEX_REL, 510 SIMILAR_TO и 126 междокументных связей.** По сравнению с уже рассчитанным 0.40 это ещё 87 удалённых вершин и потеря 45 междокументных связей; поэтому даже 0.45 не является безусловно безопасным улучшением.','',
           '**0.46 — промежуточный кандидат; 0.47 — более агрессивный.** При 0.46 остаётся 905 вершин, сохраняется полный span `Pearson product-moment correlation coefficient` (0.464365). При 0.47 остаётся 886 вершин и те же 121 междокументная связь, но этот span, `endophenotypes` и `average recall (AR)` уже удалены. При переходе 0.46 → 0.47 удаляются 19 вершин и 62 фактических отношения: отсутствие потерь междокументных рёбер не означает отсутствие содержательных потерь.','',
           '**У 0.48 наблюдается структурный перелом**, но его нельзя использовать как самостоятельное доказательство качества порога: один из удалённых междокументных мостов связывает общие `paper`. Пороги 0.48–0.50 дают дальнейшее сокращение ценой потерь научных названий; преимущества по качеству извлечений не доказаны.','',
           '## Методика','',
           'Рассчитаны ровно шесть порогов 0.45, 0.46, 0.47, 0.48, 0.49, 0.50 для двух прежних стратегий. Representative — confidence source entity с canonical ID; max member — максимум source entity confidence в существующей группе. `V={v:score(v)>=t}`; исходное ребро сохраняется тогда и только тогда, когда присутствуют оба конца. Модели, Entity Merge, пересчёт cosine и независимый от вершин отбор рёбер не выполнялись; Neo4j не открывался.','',
           'Все сущности, исходные confidence, типы, provenance и контексты прочитаны из прежних файлов. Для каждого source entity confidence остаётся максимумом его исходных typed mention scores. В слабых компонентах игнорируются направления, учитываются изолированные вершины; документы и MENTIONS не входят. Петля даёт ненулевую степень. Среднее рассчитано на все 30 документов. Многотипная вершина учитывается в каждом её исходном типе.','',
           '## Полное сравнение','']
    for strategy in ['max_member','representative']:
        lines += [f'### {strategy}','']
        lines += table(['t','Вершин','На док.','Удалено','RELEX_REL','SIMILAR_TO','Междок.','Изоляты (%)','WCC','Max WCC','Док. max'],[
            [fmt(r['threshold']),r['entities'],fmt(r['entities_per_document']),r['entities_removed'],r['relex_rel'],r['similar_to'],r['cross_document_links'],
             f"{r['isolated_vertices']} ({fmt(r['isolated_pct'])}%)",r['weak_components'],r['largest_component_vertices'],r['largest_component_documents']]
            for r in comp if r['strategy']==strategy])
        types=sorted(k[5:] for k in comp[0] if k.startswith('type_'))
        lines+=table(['Тип']+[fmt(t) for t in THRESHOLDS],[
            [typ]+[r['type_'+typ] for r in comp if r['strategy']==strategy] for typ in types])
    lines+=['Междокументных RELEX_REL нет во всех вариантах: междокументный столбец содержит только SIMILAR_TO. Полные проценты сохранения и все дополнительные счётчики — [comparison.csv](comparison.csv). Все 30 документов сохраняют хотя бы одну вершину при каждом из шести порогов.','',
            '## Что меняется на каждом шаге','']
    lines+=table(['Переход','Удалено вершин','Потеря RELEX_REL','Потеря SIMILAR_TO','Потеря междок.','Убыль max WCC','Убыль док. max'],[
        [f"{fmt(r['from_threshold'])} → {fmt(r['to_threshold'])}"]+[r[k+'_lost'] for k in ['entities','relex_rel','similar_to','cross_document_links','largest_component_vertices','largest_component_documents']]
        for r in deltas if r['strategy']=='max_member'])
    lines+=['Разности для обеих стратегий — [step_deltas.csv](step_deltas.csv). Все конкретные удалённые вершины каждого шага с confidence и контекстами — [marginal_removed_entities.csv](marginal_removed_entities.csv).','',
            '### Разрыв при 0.47 → 0.48','',
            'Из 366 вершин прежней крупнейшей компоненты удаляются 6; оставшиеся разделяются на компоненты 301 вершина / 13 документов и 59 вершин / 4 документа. Ещё 17 из 23 удалённых на этом шаге вершин находились в других компонентах. Четыре документа переходят во вторую компоненту: W2317432785 (Physics), W2364973104 (Physics), W2906067930 (Economics), W4381572413 (History). Исчезновение документов из крупнейшей компоненты не означает удаления самих документов или всех их сущностей.','',
            'На этом шаге теряются две междокументные пары:','']
    lines+=table(['Левая вершина','Документ','NER confidence','Правая вершина','Документ','NER confidence'],[
        [r['left_name'],r['left_document'],fmt(r['left_confidence'],6),r['right_name'],r['right_document'],fmt(r['right_confidence'],6)] for r in cross_lost])
    lines+=['Полные ID, исходные similarity и контексты сохранены в `cross_document_losses_047_048.csv`; состав отделившихся частей — в `largest_component_split_047_048.csv` и `components.csv`. Связь `paper ↔ paper` соединяет общее самоупоминание публикаций разных дисциплин. Сохранение такого пути повышает формальную связность, но его содержательная полезность не подтверждена. Здесь не выполнялись дополнительные симуляции удаления отдельных вершин или фильтрация качества рёбер.','',
            '## patients и results','',
            'Весь диапазон не различает эти группы: при каждом пороге и обеих стратегиях остаются 14 из 16 canonical `patients` и 6 из 8 casefold `results`. Два заголовка `RESULTS` с 0.446378 и 0.449832 уже удалены при 0.45; третий (0.564522) остаётся даже при 0.50. Пять общих ссылок `results` остаются, включая 0.522291–0.892064. Наличие реальных участников клинической когорты у `patients` не делает их бесполезными; общий surface name и ошибка извлечения не тождественны.','',
            'Повышение 0.45 → 0.50 не даёт дополнительных удалений точных названий `patients` и `results`, но удаляет 95 других canonical entities. Это не означает, что все 95 полезны или все бесполезны; ниже приведены контекстные контрпримеры.','',
            '## Содержательные и малоинформативные потери','',
            'Разбор иллюстративный, по локальным контекстам, без внешней модели и без оценки precision/recall. Удаление конкретного названия не означает полного исчезновения концепта: сокращения и другие spans не объединялись.','']
    selected={
        'connexin':'Конкретный белок в описании gap junctions; потеря при 0.46.',
        'periodic wave solutions':'Перечисленная форма математических решений; потеря при 0.46.',
        'time varying parameter formulation':'Конкретная формулировка модели в исследовании конвергенции; потеря при 0.46.',
        'Pearson product-moment correlation coefficient':'Конкретный коэффициент для сравнения измерений; потеря при 0.47.',
        'endophenotypes':'Предмет исследования copy number variants; потеря при 0.47.',
        'average recall (AR)':'Названная метрика оценки detection; потеря при 0.47.',
        'nonlinear optics':'Научная область, из которой возникает исследуемая система; потеря при 0.48.',
        'co-investment':'Названный финансовый инструмент; потеря при 0.48.',
        'synthetic biology':'Область, обеспечивающая engineering microbial cell factories; потеря при 0.48.',
        "lunar regolith's thickness":'Измеряемая характеристика лунного реголита; потеря при 0.49.',
        'fare optimization':'Непосредственная тема транспортной модели; потеря при 0.49.',
        'diatom phylogeny':'Предмет обсуждения эволюции пластид; потеря при 0.50.',
        'magnetic resonance imaging':'Конкретная технология исследованного case study; потеря при 0.50.',
        'cardiac implantable electronic device (CIED)':'Полное название устройства исследования; потеря при 0.50, другие CIED-spans могут остаться.',
        'retrospectively':'Наречие из описания метода; слабая самостоятельная вершина, удаляется при 0.46.',
        'green':'Одиночный модификатор, слабое самостоятельное название; удаляется при 0.46.',
        'paper':'Самоупоминание статьи; удаляется при 0.48 и одновременно теряет формальный междокументный мост.',
        'clear':'В контексте прозрачность роговичного трансплантата: слабое самостоятельное имя, но клинический смысл есть; потеря при 0.50.',
    }
    reviewed=[]
    for name,reason in selected.items():
        candidates=[e for e in entities if e['name']==name and .45<=float(e['max_member'])<.5]
        assert candidates,name
        for e in candidates:
            first=next(t for t in THRESHOLDS if float(e['max_member'])<t)
            reviewed.append(dict(canonical_entity_id=e['canonical_entity_id'],name=name,document_id=e['document_id'],discipline=e['discipline'],
                                 confidence=e['max_member'],first_removed_threshold=first,reason=reason,context=e['max_member_context']))
            lines += [f"- **{name}**, `{e['document_id']}`, {e['discipline']}, **{fmt(e['max_member'],6)}**. {reason}",f"  Контекст: «{e['max_member_context']}»",'']
    csv_out('qualitative_review.csv',reviewed)
    lines+=['## По дисциплинам и документам','']
    ds=read(OUT/'retention_by_discipline.csv')
    lines+=table(['Дисциплина','Baseline']+[fmt(t) for t in THRESHOLDS],[
        [d,next(r['baseline'] for r in ds if r['discipline']==d)]+[next(r['retained'] for r in ds if r['discipline']==d and r['strategy']=='max_member' and float(r['threshold'])==t) for t in THRESHOLDS]
        for d in sorted({r['discipline'] for r in ds})])
    docs=read(OUT/'retention_by_document.csv')
    lines+=table(['Документ','Дисциплина','Baseline']+[fmt(t) for t in THRESHOLDS],[
        [d,next(r['discipline'] for r in docs if r['document_id']==d),next(r['baseline'] for r in docs if r['document_id']==d)]+
        [next(r['retained'] for r in docs if r['document_id']==d and r['strategy']=='max_member' and float(r['threshold'])==t) for t in THRESHOLDS]
        for d in sorted({r['document_id'] for r in docs})])
    lines+=['Таблицы выше используют max member; полные разрезы для обеих стратегий — `retention_by_discipline.csv` и `retention_by_document.csv`. Распределения исходных confidence не меняются; для удобства сохранены повторно тем же кодом в `distribution_summary.csv` и `confidence_histogram.csv`.','',
            '## Примеры для каждого варианта','',
            'По два ближайших к границе сохранённых и удалённых примера. Отбор детерминированный, не репрезентативная выборка качества. Confidence и контекст относятся к текущей стратегии. Полные ID и дополнительный крайний пример на группу — в `examples.csv`.','']
    examples=read(OUT/'examples.csv')
    for strategy in ['max_member','representative']:
        for t in THRESHOLDS:
            lines += [f'### {strategy}, t={fmt(t)}','']
            for status in ['removed','retained']:
                for e in [r for r in examples if r['strategy']==strategy and float(r['threshold'])==t and r['status']==status][:2]:
                    label='Удалена' if status=='removed' else 'Сохранена'
                    lines += [f"- {label}: **{e['name']}**, **{fmt(e['confidence'],6)}**, `{e['document_id']}`, `{e['canonical_entity_id'][:14]}…`.",f"  Контекст: «{e['context']}»",'']
    lines+=['## Проверки и воспроизведение','',
            'Независимый union-find подтвердил точные наборы вершин, рёбер и метрики компонент всех 12 вариантов. Обе строки 0.50 полностью совпали с предыдущим анализом по всем столбцам. SHA-256 всех защищённых входов и предыдущих результатов сверены; прежний coarse sweep не изменён. Подробности — `validation.json` и `independent_validation.json`.','',
            '```powershell','.venv/Scripts/python.exe reports/entity_resolution/relex_base_t035/ner_threshold_analysis/threshold_045_050/run.py','.venv/Scripts/python.exe reports/entity_resolution/relex_base_t035/ner_threshold_analysis/threshold_045_050/write_report.py','```','',
            '`run.py` переиспользует прежний стандартно-библиотечный `../analyze.py`, меняя только выходной каталог и список порогов. Он не вызывает прежний генератор отчёта, не меняет предыдущий вывод. `virtual_subgraphs.jsonl` содержит vertex IDs и исходные edge IDs каждого варианта; свойства остаются в неизменённых источниках. Все компоненты и их состав сохранены в `components.csv`.','',
            '**Выполнен только запрошенный анализ NER 0.45–0.50 с шагом 0.01. Фильтр к KG не применён.**','']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('PASS: all 12 subgraphs independently verified; 0.50 matches prior sweep; protected hashes unchanged.')
    print(f'Report saved; {len(reviewed)} contextual examples and {len(removed)} marginal-removal rows.')


if __name__=='__main__':
    main()
