"""Stage 2D: five in-memory SIMILAR_TO previews; no merges or graph writes.

Run: .venv/Scripts/python.exe scripts/preview_global_semantic_links.py
The unchanged 2C.1 worker supplies the exact 2B encoder and cosine matrix.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unicodedata

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding='utf-8')
import sweep_semantic_thresholds as sweep

p = sweep.protocol
ROOT, ER = p.ROOT, p.ER
OUTPUT = ER / "semantic_resolution/global_linking_preview"
VERIFIED = ER / "semantic_resolution/candidate_verification/verified_pairs.csv"
SWEEP = ER / "semantic_resolution/threshold_sweep/threshold_results.csv"
CONFIGS = [("BAAI/bge-small-en-v1.5", .90), ("BAAI/bge-small-en-v1.5", .95),
           ("BAAI/bge-base-en-v1.5", .95), ("intfloat/e5-small-v2", .95),
           ("thenlper/gte-small", .95)]


def label(model, threshold):
    return f"{model.split('/')[-1]} / name / {threshold:.2f}"


def graph(nodes, edges):
    """Weak projection retains every node; loops count for incidence only."""
    adjacency = {i: set() for i in nodes}
    touched = set()
    for a, b in edges:
        p.require(a in adjacency and b in adjacency, "Unknown graph endpoint")
        touched.update((a, b))
        if a != b:
            adjacency[a].add(b)
            adjacency[b].add(a)
    groups, component = [], {}
    for seed in sorted(nodes):
        if seed in component:
            continue
        stack, members = [seed], []
        component[seed] = len(groups)
        while stack:
            current = stack.pop()
            members.append(current)
            for other in adjacency[current]:
                if other not in component:
                    component[other] = len(groups)
                    stack.append(other)
        groups.append(sorted(members))
    groups.sort(key=lambda g: (-len(g), g[0]))
    p.require(sum(map(len, groups)) == len(nodes), "Incomplete component coverage")
    # An independent Union-Find checks BFS, including singleton components.
    ids = {eid: i for i, eid in enumerate(sorted(nodes))}
    uf = sweep.UnionFind(len(ids))
    for a, b in edges:
        uf.union(ids[a], ids[b])
    sizes = Counter(uf.find(i) for i in range(len(ids)))
    p.require(sorted(sizes.values()) == sorted(map(len, groups)), "BFS/UF mismatch")
    return dict(isolated=len(nodes)-len(touched), components=len(groups),
                largest=len(groups[0]) if groups else 0), groups, component, set(nodes)-touched


def reviews():
    result = {}
    ids = {r['canonical_entity_id'] for r in p.read_input()}
    with VERIFIED.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = tuple(sorted((row['left_canonical_id'], row['right_canonical_id'])))
            p.require(key[0] != key[1] and set(key) <= ids and key not in result, "Invalid 2C pair")
            decision = row['decision']
            reviewed = row['individually_reviewed'] == 'True'
            p.require(decision in ('ACCEPT','REJECT','REVIEW') and
                      (decision == 'REVIEW' or reviewed), "Invalid 2C decision")
            result[key] = dict(decision_2c=decision, individually_reviewed_2c=reviewed,
                               verification_pair_id=row['pair_id'], reason_2c=row['reason'],
                               review_method_2c=row['review_method'],
                               reason_entity_order_2c=row['reason_entity_order'],
                               referent_level_2c=row['referent_level'])
    counts = Counter(r['decision_2c'] for r in result.values())
    p.require(len(result)==49336 and counts['ACCEPT']==46 and counts['REJECT']==62, "Unexpected 2C snapshot")
    return result


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def analyze(scores, rows, relations, model, known):
    import numpy as np
    nodes = {r['canonical_entity_id']: r['document_id'] for r in rows}
    original = [(r['head_id'], r['tail_id']) for r in relations]
    base, _, base_components, isolated = graph(nodes, original)
    left, right = np.triu_indices(len(rows), 1)
    values = scores[left, right].astype(np.float64)
    p.require(len(values)==628881, "Not a full unordered search")
    output = []
    for selected_model, threshold in CONFIGS:
        if model != selected_model:
            continue
        links = []
        for k in np.flatnonzero(values >= threshold):
            a, b = rows[int(left[k])], rows[int(right[k])]
            if a['canonical_entity_id'] > b['canonical_entity_id']:
                a, b = b, a
            key = (a['canonical_entity_id'], b['canonical_entity_id'])
            link_id = 'sl:' + hashlib.sha256(json.dumps(key, separators=(',',':')).encode()).hexdigest()
            record = dict(configuration_id=label(model,threshold), model=model, mode='text_name',
                          threshold=threshold, semantic_link_id=link_id, relation_type='SIMILAR_TO',
                          directed=False, left_canonical_id=key[0], right_canonical_id=key[1],
                          left_name=a['text_name'], right_name=b['text_name'],
                          left_document_id=a['document_id'], right_document_id=b['document_id'],
                          scope='cross_document' if a['document_id'] != b['document_id'] else 'within_document',
                          similarity=float(values[k]), margin_above_threshold=float(values[k])-threshold,
                          left_types=a['entity_types'], right_types=b['entity_types'],
                          type_set_conflict=set(a['entity_types'])!=set(b['entity_types']),
                          types_disjoint=set(a['entity_types']).isdisjoint(b['entity_types']),
                          same_normalized_name=normalized(a['text_name'])==normalized(b['text_name']),
                          left_quality_flags=a['quality_flags'], right_quality_flags=b['quality_flags'],
                          damaged_name='damaged_name' in a['quality_flags']+b['quality_flags'],
                          partial_token='original_partial_token' in a['quality_flags']+b['quality_flags'],
                          baseline_same_component=base_components[key[0]]==base_components[key[1]],
                          left_source_entity_ids=a['source_entity_ids'], right_source_entity_ids=b['source_entity_ids'],
                          left_source_mention_ids=a['source_mention_ids'], right_source_mention_ids=b['source_mention_ids'],
                          left_context=a['representative_context'], right_context=b['representative_context'],
                          source_input=p.INPUT.relative_to(ROOT).as_posix(),
                          source_verification=VERIFIED.relative_to(ROOT).as_posix())
            record.update(known.get(key, dict(decision_2c='NOT_IN_2C', individually_reviewed_2c=False,
                                            verification_pair_id='', reason_2c='', review_method_2c='',
                                            reason_entity_order_2c='', referent_level_2c='')))
            links.append(record)
        links.sort(key=lambda r:(r['left_canonical_id'],r['right_canonical_id']))
        cross = [r for r in links if r['scope']=='cross_document']
        semantic = [(r['left_canonical_id'],r['right_canonical_id']) for r in links]
        after, groups, _, _ = graph(nodes, original+semantic)
        doc_edges = {tuple(sorted((r['left_document_id'],r['right_document_id']))) for r in cross}
        doc_nodes = set(nodes.values())
        doc_stats, _, _, _ = graph(doc_nodes, sorted(doc_edges))
        reached_doc_pairs = set()
        for g in groups:
            docs = sorted({nodes[eid] for eid in g})
            reached_doc_pairs.update((a,b) for i,a in enumerate(docs) for b in docs[i+1:])
        linked_isolated = isolated & {eid for pair in semantic for eid in pair}
        p.require(base['isolated']-after['isolated']==len(linked_isolated), "Isolation accounting mismatch")
        p.require(after['components']<=base['components'] and after['largest']>=base['largest'], "Connectivity regressed")
        stats = dict(threshold=threshold, remaining_canonical_vertices=len(nodes),
                     semantic_edges=len(links), within_edges=len(links)-len(cross), cross_edges=len(cross),
                     original_edges=len(original), total_edges=len(original)+len(links), **after,
                     rescued_isolated=len(linked_isolated), doc_incident=30-doc_stats['isolated'],
                     direct_doc_pairs=len(doc_edges), document_projection_components=doc_stats['components'],
                     document_projection_largest=doc_stats['largest'], reachable_doc_pairs=len(reached_doc_pairs),
                     largest_entity_component_docs=len({nodes[i] for i in groups[0]}),
                     max_docs_in_entity_component=max(len({nodes[i] for i in g}) for g in groups),
                     original_component_bridges=sum(not r['baseline_same_component'] for r in links),
                     decisions=dict(Counter(r['decision_2c'] for r in links)),
                     cross_decisions=dict(Counter(r['decision_2c'] for r in cross)),
                     cross_type_conflicts=sum(r['type_set_conflict'] for r in cross),
                     cross_disjoint_types=sum(r['types_disjoint'] for r in cross),
                     cross_damaged_names=sum(r['damaged_name'] for r in cross),
                     cross_partial_tokens=sum(r['partial_token'] for r in cross), links=links)
        p.require(len({r['semantic_link_id'] for r in links})==len(links), "Duplicate semantic edge")
        output.append(stats)
    return output


def worker(model):
    # Process-local adapter: reuse the unchanged 2C.1 encoder, restrict it to
    # text_name and replace its merge simulation with a non-merging preview.
    refs, known = sweep.reference(), reviews()
    sweep.reference = lambda: refs
    p.MODES = ('text_name',)
    sweep.simulate = lambda scores,rows,relations: analyze(scores,rows,relations,model,known)
    original_emit = p.emit
    def emit(kind, **data):
        if kind == 'sweep_result':
            kind = 'preview_result'
        elif data.get('stage') == 'five_thresholds_complete':
            data['stage'] = 'preview_complete'
        original_emit(kind, **data)
    p.emit = emit
    sweep.worker(model)


def run_worker(model):
    process = subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',model],
                               cwd=ROOT,stdout=subprocess.PIPE,text=True,encoding='utf-8')
    result = None
    try:
        for line in process.stdout:
            event = json.loads(line)
            if event['kind']=='preview_result':
                result = event['rows']
            else:
                print(json.dumps(event,ensure_ascii=False),flush=True)
        p.require(process.wait()==0 and result is not None, f'Preview failed: {model}')
    finally:
        if process.poll() is None:
            process.terminate(); process.wait()
    return result


def examples(results):
    chosen, used, names = [], set(), set()
    for result in results:
        cross = [r for r in result['links'] if r['scope']=='cross_document']
        high = sorted(cross,key=lambda r:(-r['similarity'],r['semantic_link_id']))
        boundary = sorted(cross,key=lambda r:(r['margin_above_threshold'],r['semantic_link_id']))
        rejected = [r for r in high if r['decision_2c']=='REJECT']
        for role,pool in [('high',high),('known_REJECT_or_high',rejected or high),
                          ('boundary',boundary),('boundary',boundary)]:
            available = [r for r in pool if r['semantic_link_id'] not in used]
            if not available:
                available = [r for r in high if r['semantic_link_id'] not in used]
            diverse = [r for r in available if (normalized(r['left_name']),normalized(r['right_name'])) not in names]
            # Diversify high examples; retain genuinely closest boundary cases.
            row = (diverse or available)[0] if role!='boundary' else available[0]
            chosen.append((role,row)); used.add(row['semantic_link_id'])
            names.add((normalized(row['left_name']),normalized(row['right_name'])))
    p.require(len(chosen)==len(used)==20, 'Need 20 distinct cross-document examples')
    return chosen


def cell(value):
    return str(value).replace('|','\\|').replace('\n',' ')


def report(results, baseline, hashes):
    lines = ['# Этап 2D — Global Semantic Linking Preview','',
             'Все сценарии рассчитаны только в памяти: сохраняются 1122 канонические вершины, 30 документов и все 1760 направленных типизированных отношений 1C (включая 13 петель). Добавляется отдельный неориентированный тип SIMILAR_TO; вершины и исходные отношения не объединяются и не переназначаются. В Neo4j ничего не записано.','',
             '## Методика','',
             'Использован неизменённый worker 2C.1 с адаптером только в памяти отдельного процесса: text_name и расчёт добавляемых связей вместо merge. Четыре локальные модели загружены последовательно; BGE-small переиспользован для двух порогов. Векторы 2B/2C.1 не сохранялись, поэтому получены заново. Все четыре полных SHA-256 embeddings совпали с 2B и 2C.1. Протокол: CPU FP32, 4 потока, batch=8, max_length=512, CLS для BGE; masked mean для E5/GTE; E5 query: с обеих сторон; L2-нормализация. E5-small — PyTorch, остальные — ONNX Runtime. Скачиваний и новых зависимостей нет.','',
             'Для каждой конфигурации проверены все 628881 неупорядоченная пара (без самопар и top-5). Cosine вычисляется в float32; сравнение его точного числового значения с десятичным порогом — в float64 без округления, как в 2C.1. Отбираются все similarity >= threshold, включая внутри- и междокументные пары, без фильтров типов и решений 2C. Один SIMILAR_TO на неупорядоченную пару в рамках конфигурации; связь добавляется даже при наличии другого предметного отношения между её концами. Сценарии не суммируются.','',
             'Слабая связность: неориентированная проекция всех исходных отношений плюс SIMILAR_TO; типы и направления игнорируются только для обхода. Каждая из 1122 вершин остаётся в графе. Изоляция — отсутствие любых инцидентных отношений; исходная петля считается связью, но не соединяет разные вершины. Новых петель нет. Компоненты дополнительно проверены независимым Union-Find.','',
             '## Пять сценариев','',
             'Количество вершин во всех строках — **1122**, исходных отношений — **1760**. Новые рёбра считаются отдельно, без удвоения неориентированных связей.','',
             '| Сценарий | Новых связей | Внутри документов | Между документами | Всего рёбер | Изолированные | Доля I, % | Слабые компоненты | Крупнейшая, V |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
             f"| Без SIMILAR_TO | 0 | 0 | 0 | 1760 | {baseline['isolated']} | {100*baseline['isolated']/1122:.2f} | {baseline['components']} | {baseline['largest']} |"]
    for r in results:
        lines.append(f"| {label(r['model'],r['threshold'])} | {r['semantic_edges']} | {r['within_edges']} | {r['cross_edges']} | {r['total_edges']} | {r['isolated']} | {100*r['isolated']/1122:.2f} | {r['components']} | {r['largest']} |")
    lines += ['', '## Междокументная связность','',
              '«Документов с прямой связью» — документы, инцидентные хотя бы одному междокументному SIMILAR_TO. Проекция документов имеет 30 вершин и одно ребро на связанную пару document IDs; включает документы без связей. Такая проекция может переоценивать сквозную связность: разные сущности внутри одного документа не обязательно связаны. Поэтому отдельно показаны реально достижимые пары документов (оба представлены в одной компоненте KG) и число документов в крупнейшей компоненте сущностей. Достижимость означает существование хотя бы одного пути между сущностями этих документов, не связность всех их сущностей.','',
              '| Сценарий | Документов с прямой связью / 30 | Прямых пар документов / 435 | Компонент проекции документов | Крупнейшая в проекции, docs | Достижимых пар документов в KG | Документов в крупнейшей компоненте KG |',
              '|---|---:|---:|---:|---:|---:|---:|',
              '| Без SIMILAR_TO | 0 | 0 | 30 | 1 | 0 | 1 |']
    for r in results:
        lines.append(f"| {label(r['model'],r['threshold'])} | {r['doc_incident']} | {r['direct_doc_pairs']} | {r['document_projection_components']} | {r['document_projection_largest']} | {r['reachable_doc_pairs']} | {r['largest_entity_component_docs']} |")
    lines += ['', '## Решения 2C и диагностические риски','',
              'ACCEPT/REJECT из 2C относятся к тождеству сущностей, а не к допустимости тематического SIMILAR_TO. REJECT — предупреждение против интерпретации перехода как эквивалентности, не доказанная ошибка новой связи. Типовые конфликты и повреждения также только сигналы риска. Решения не использованы как фильтры. Выборка 2C целевая и зависима от прежних моделей/top-5, не независимая gold-разметка; Precision/Recall/F1 и частота ошибок по этим числам не оцениваются. REVIEW и NOT_IN_2C не означают подтверждение.','',
              '| Сценарий | ACCEPT все / cross | REJECT все / cross | Cross REVIEW | Cross нет в 2C | Cross разные названия | Cross разные типы | Cross непересекающиеся типы | Cross повреждённое имя / partial_token |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        a,c = r['decisions'],r['cross_decisions']
        different_names = sum(l['scope']=='cross_document' and not l['same_normalized_name'] for l in r['links'])
        lines.append(f"| {label(r['model'],r['threshold'])} | {a.get('ACCEPT',0)} / {c.get('ACCEPT',0)} | {a.get('REJECT',0)} / {c.get('REJECT',0)} | {c.get('REVIEW',0)} | {c.get('NOT_IN_2C',0)} | {different_names} | {r['cross_type_conflicts']} | {r['cross_disjoint_types']} | {r['cross_damaged_names']} / {r['cross_partial_tokens']} |")
    lines += ['', '## 20 конкретных междокументных пар','',
              'По четыре примера на сценарий: высокий cosine; известный REJECT, если доступен, иначе высокий cosine; две пары с минимальным допущенным cosine. Повторы canonical-пары между сценариями исключены, поэтому минимальные выбираются среди ещё не показанных пар. Для высоких сходств предпочтены разные названия. boundary обозначает нижний край допущенных значений, а не обязательную близость к порогу: у BGE-base/0.95 все междокументные cosine >= 0.9999995, пограничных около 0.95 нет. У остальных сценариев показаны случаи около заданного порога. Отбор иллюстративный, не репрезентативный; полный состав и точные ID находятся в semantic_links.csv. Имена не исправлялись.','',
              '| Сценарий / отбор | Левая сущность | Документ | Правая сущность | Документ | Cosine | Решение 2C | Типы различаются | ID связи (префикс) |',
              '|---|---|---|---|---|---:|---|---|---|']
    for role,r in examples(results):
        lines.append('| '+' | '.join(map(cell,[label(r['model'],r['threshold'])+' / '+role,r['left_name'],r['left_document_id'],r['right_name'],r['right_document_id'],f"{r['similarity']:.8f}",r['decision_2c'],r['type_set_conflict'],r['semantic_link_id'][:15]]))+' |')
    by_config = {(r['model'],r['threshold']):r for r in results}
    e5 = by_config['intfloat/e5-small-v2',.95]
    gte = by_config['thenlper/gte-small',.95]
    base = by_config['BAAI/bge-base-en-v1.5',.95]
    small = by_config['BAAI/bge-small-en-v1.5',.95]
    relaxed = by_config['BAAI/bge-small-en-v1.5',.90]
    unknown = e5['cross_decisions'].get('REVIEW',0)+e5['cross_decisions'].get('NOT_IN_2C',0)
    lines += ['', '## Вывод и ограничения','',
              f"Для следующей проверки глобальных переходов наиболее перспективен **E5-small / text_name / 0.95**: {e5['cross_edges']} междокументные связи, {e5['largest_entity_component_docs']} документа в крупнейшей компоненте KG, {e5['reachable_doc_pairs']} достижимые пары документов, {e5['cross_decisions'].get('REJECT',0)} известных междокументных REJECT. По сравнению с BGE-small/0.95 число известных междокументных REJECT такое же, а охват крупнейшей компоненты выше ({small['largest_entity_component_docs']} → {e5['largest_entity_component_docs']} документов). Это условный выбор по связности и диагностическим рискам, не доказанный победитель по качеству: {unknown} из {e5['cross_edges']} междокументных пар остаются REVIEW или отсутствуют в 2C.", '',
              f"GTE-small/0.95 близок по глобальному эффекту: {gte['largest_entity_component_docs']} документа в крупнейшей компоненте, {gte['reachable_doc_pairs']} достижимых пар документов против {e5['reachable_doc_pairs']} у E5; документов с прямыми связями {gte['doc_incident']} против {e5['doc_incident']}. При этом известных междокументных REJECT {gte['cross_decisions'].get('REJECT',0)} против {e5['cross_decisions'].get('REJECT',0)} у E5. Во всём графе, наоборот, у E5 REJECT больше ({e5['decisions'].get('REJECT',0)} против {gte['decisions'].get('REJECT',0)}), поэтому рекомендация E5 ограничена целью междокументного linking и не означает минимального общего риска.", '',
              f"BGE-base/0.95 — наиболее консервативная контрольная точка по известным REJECT ({base['decisions'].get('REJECT',0)} всего, {base['cross_decisions'].get('REJECT',0)} междокументных), но крупнейшая компонента охватывает {base['largest_entity_component_docs']} документов. BGE-small/0.95 даёт близкий эффект ({small['largest_entity_component_docs']} документов, {small['cross_decisions'].get('REJECT',0)} междокументных REJECT). Ослабление BGE-small до 0.90 повышает охват до {relaxed['largest_entity_component_docs']} документов, но увеличивает междокументные REJECT до {relaxed['cross_decisions'].get('REJECT',0)} и конфликты наборов типов до {relaxed['cross_type_conflicts']}.", '',
              'Во всех пяти сценариях присутствуют 143 междокументные пары с одинаковым нормализованным названием. Дополнительных пар с разными названиями: BGE-small/0.90 — 15, BGE-small/0.95 — 2, BGE-base/0.95 — 0, E5-small/0.95 — 51, GTE-small/0.95 — 15. Нормализация здесь только диагностическая (NFKC, casefold, пробелы), вход модели не менялся. Значительная часть общей связности опирается на совпадающие слова вроде patients, study, results, cancer; разные исследования, когорты и смыслы от этого не становятся тождественными. Даже cosine около 1.0 встречается у известных REJECT.','',
              'Структурные показатели сравниваются при постоянном числе вершин: уменьшение числа изолированных вершин и компонент здесь обусловлено добавлением путей, а не сжатием сущностей. Эти пути являются гипотезами similarity, а не извлечёнными фактами или доказанной эквивалентностью. Полезность маршрутов, семантическая корректность SIMILAR_TO и качество Graph-RAG не проверялись.','',
              '## Проверки и воспроизводимость','',
              'Проверены 1122 уникальных canonical ID, 1211 исходных entity ID, 1327 mention ID и 30 документов; все 1760 отношений и их конечные точки; неизменность исходных типов/направлений/полей через контрольные суммы. Для пяти сценариев числа всех, внутри- и междокументных пар точно совпали с 2C.1. Самопар и дубликатов внутри сценария нет; все конечные точки присутствуют в 2A. BFS и Union-Find согласованы. Исходные документы, результаты предыдущих этапов, код инференса и файлы моделей проверены SHA-256 до и после работы. Новые файлы — только этот отчёт и semantic_links.csv; код — scripts/preview_global_semantic_links.py.','',
              'Повторный запуск: `.venv/Scripts/python.exe scripts/preview_global_semantic_links.py`. Порядок сценариев и пар детерминирован; названия, контексты, исходные ID и флаги сохранены в CSV со ссылками на 2A/2C. Проверки не превращают similarity в семантическое доказательство. Neo4j, mapping, отношения и артефакты 1A–2C.1 не изменяются.','',
              'SHA-256 входов и защищённых артефактов:','']
    lines += [f'- `{path.relative_to(ROOT).as_posix()}`: `{digest}`' for path,digest in hashes.items()]
    return '\n'.join(lines)+'\n'


def main():
    rows, relations = p.read_input(), sweep.read_jsonl(sweep.RELATIONS)
    nodes = {r['canonical_entity_id']:r['document_id'] for r in rows}
    p.require(len(relations)==len({r['canonical_triple_id'] for r in relations})==1760, 'Invalid relations')
    p.require(all(r['head_id'] in nodes and r['tail_id'] in nodes and
                  nodes[r['head_id']]==nodes[r['tail_id']]==r['document_id'] for r in relations), 'Invalid original endpoints')
    p.require(all(r['relation_type']!='SIMILAR_TO' for r in relations), 'Original semantic links already present')
    p.require(sum(r['head_id']==r['tail_id'] for r in relations)==13, 'Unexpected original loops')
    baseline,_,_,_ = graph(nodes,[(r['head_id'],r['tail_id']) for r in relations])
    p.require(baseline==dict(isolated=324,components=391,largest=61), 'Unexpected baseline')
    protected = {path for path in ER.rglob('*') if path.is_file() and not path.is_relative_to(OUTPUT)}
    protected.update(ROOT/'scripts'/name for name in ('benchmark_semantic_embeddings.py','sweep_semantic_thresholds.py'))
    protected.update(ROOT/name for name in ('reports/re_v2/relex_comparison/threshold035/e3_base/entities.jsonl',
                                          'reports/re_v2/relex_comparison/threshold035/e3_base/relations.jsonl',
                                          'reports/ner_v2/research_run2/input_abstracts.jsonl'))
    hashes = {path:p.sha(path) for path in sorted(protected)}
    refs = sweep.reference()
    models = list(dict.fromkeys(m for m,t in CONFIGS))
    manifests = [p.describe(m) for m in models]
    for desc in manifests:
        p.require(desc['files']==json.loads(refs[desc['model'],'text_name']['model_files']), 'Model differs from 2B')
        p.require(refs[desc['model'],'text_name']['input_sha256']==hashes[p.INPUT], 'Input differs from 2B')
    with SWEEP.open(encoding='utf-8-sig',newline='') as stream:
        prior = {(r['model'],float(r['threshold'])):r for r in csv.DictReader(stream) if r['mode']=='text_name'}
    results = [r for model in models for r in run_worker(model)]
    p.require([(r['model'],r['threshold']) for r in results]==CONFIGS, 'Incomplete configuration grid')
    all_links = []
    for r in results:
        old = prior[r['model'],r['threshold']]
        p.require((r['semantic_edges'],r['within_edges'],r['cross_edges'])==
                  tuple(int(old[k]) for k in ('pairs_above_threshold','within_document_pairs','cross_document_pairs')), '2C.1 pair counts differ')
        p.require(r['embedding_sha256']==old['embedding_sha256'] and r['remaining_canonical_vertices']==1122, 'Changed vectors/vertices')
        for link in r['links']:
            link['embedding_sha256'] = r['embedding_sha256']
            link['source_input_sha256'] = hashes[p.INPUT]
        all_links.extend(r['links'])
    text = report(results,baseline,hashes)
    p.require(text==report(results,baseline,hashes), 'Non-deterministic report')
    p.require(all(p.sha(path)==digest for path,digest in hashes.items()), 'Protected artifact changed')
    for desc in manifests:
        p.require(all(p.sha(Path(f['path']))==f['sha256'] for f in desc['files']), 'Model changed')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    p.write_csv(OUTPUT/'semantic_links.csv',all_links)
    (OUTPUT/'report.md').write_text(text,encoding='utf-8')
    p.require(all(p.sha(path)==digest for path,digest in hashes.items()), 'Protected artifact changed after writing')
    print(json.dumps([{k:v for k,v in r.items() if k not in ('links',)} for r in results],ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':
        worker(sys.argv[2])
    else:
        main()
