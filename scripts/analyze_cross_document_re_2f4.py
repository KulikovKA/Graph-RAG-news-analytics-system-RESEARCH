"""2F.4A-C offline research; the only database operations are READ queries.

Run --stage discover, then --stage pilot, --stage infer, --stage analyze.
Local fixed-span ReLEx only. No imports, new embeddings, NER or ER decisions.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='backslashreplace')
from tqdm import tqdm
import add_cross_document_relex as old

base, core, ROOT = old.base, old.core, old.ROOT
NS = 'relex_sem_e5_small_name_t095_ner_t046_er2f3'
OUT = base.ER / 'cross_document_re_2f4'
SOURCE = base.ER / 'global_entity_resolution'
THRESHOLDS = [x / 100 for x in range(35, 91, 5)]
GENERIC = {'results', 'result', 'study', 'studies', 'paper', 'author', 'authors',
           'patients', 'patient', 'dialysis patients', 'cancer patients', 'test', 'testing',
           'resolution', 'thickness', 'thicknesses', 'reliability', 'grain', 'grains',
           'cooperation', 'follow-up', 'genes', 'genetics'}
STOP = set('the a an and or of in on to for with by from is are was were this that these those '
           'as at it its be been can may we our their has have using used use study studies '
           'results result method methods analysis different based new proposed between '
           'effect effects data model models system systems performance patients'.split())
POLICY = dict(version='2f4-v1', inference_C_limit=160, inference_D_limit=40,
              generic_control_limit=6, paths_per_pair_in_model=1,
              generic_path_control_limit=6,
              max_saved_paths_per_pair=3, D_max_per_document_pair=2,
              D_min_shared_context_terms=3, D_min_shared_name_terms=1,
              D_min_context_jaccard=.045, min_threshold=.35,
              threshold_comparison='confidence >= threshold; stored adapter scores strictly > 0.35',
              candidate_pairs_unordered=True, inference_both_directions=True,
              path_search='one cross-document SIMILAR_TO or two HAS_SHARED_CONCEPT edges; one factual edge on each side',
              screening='frozen lexical/contextual ranking, no prediction score in selection')
require = base.require
dumps = base.dumps


def banner(s):
    print('\n' + '=' * 64 + '\n' + s + '\n' + '=' * 64, flush=True)


def progress(xs=None, **kw):
    return tqdm(xs, file=sys.stdout, ascii=True, mininterval=.5, dynamic_ncols=True, **kw)


def load(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def jsonl(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    core.write_jsonl(OUT / name, rows)


def csvout(name, rows, fields=None):
    rows = list(rows)
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def identity(prefix, obj):
    return prefix + core.digest(obj)[:24]


def terms(text):
    return {t for t in re.findall(r'[a-z][a-z0-9-]{2,}', text.casefold()) if t not in STOP}


def inputs():
    rows = core.read_jsonl(SOURCE / 'proposed_mapping.jsonl')
    original = {r['canonical_entity_id']: r['original_canonical_entity'] for r in rows}
    mapping = {r['canonical_entity_id']: r['proposed_final_id'] for r in rows}
    entities = {k: original[k] for k in sorted(set(mapping.values()))}
    docs = {d['document_id']: d for d in core.read_jsonl(base.DOCUMENTS)}
    facts = [f for bucket in core.read_jsonl(SOURCE / 'preview_factual_relations.jsonl') for f in bucket['original_relations']]
    semantic = core.read_jsonl(SOURCE / 'preview_semantic_links.jsonl')
    concepts = core.read_jsonl(SOURCE / 'preview_concept_links.jsonl')
    require((len(entities), len(facts), len(semantic), len(concepts)) == (811, 1307, 492, 118), 'Wrong 2F.3 source counts')
    old._entities, old._documents = original, docs
    profiles = {}
    for eid in entities:
        p = old.profile(eid, 240)
        group = [k for k, v in mapping.items() if v == eid]
        p['source_canonical_ids'] = group
        p['entity_types'] = sorted({t for k in group for t in original[k]['entity_types']})
        p['source_entity_ids'] = sorted({s for k in group for s in original[k]['source_entity_ids']})
        p['source_mention_ids'] = sorted({s for k in group for s in original[k]['source_mention_ids']})
        p['name'] = entities[eid]['text_name']
        p['discipline'] = docs[p['document_id']]['discipline']
        profiles[eid] = p
    return original, mapping, entities, docs, facts, semantic, concepts, profiles


def components(profiles, pairs, concept_ids):
    adjacency = {k: set() for k in set(profiles) | set(concept_ids)}
    for a, b in pairs:
        adjacency[a].add(b)
        adjacency[b].add(a)
    groups, seen = [], set()
    for seed in sorted(adjacency):
        if seed in seen:
            continue
        group, stack = [], [seed]
        seen.add(seed)
        while stack:
            a = stack.pop()
            group.append(a)
            for b in sorted(adjacency[a] - seen):
                seen.add(b)
                stack.append(b)
        groups.append(group)
    groups.sort(key=lambda g: (-len(g), min(g)))
    memberships = {k: i for i, g in enumerate(groups) for k in g}
    return dict(weak_components=len(groups), largest_component_vertices=len(groups[0]),
                largest_component_documents=len({profiles[k]['document_id'] for k in groups[0] if k in profiles})), memberships


def graph_pairs(mapping, facts, semantic, concepts):
    return ([(mapping[f['head_id']], mapping[f['tail_id']]) for f in facts] +
            [(s['head_id'], s['tail_id']) for s in semantic] +
            [(c['head_id'], c['tail_id']) for c in concepts])


def live_snapshot():
    """Streaming exact properties/endpoints hash, same contract as 2F.3."""
    from neo4j import READ_ACCESS
    driver, database = core.connect_neo4j()
    digest = hashlib.sha256()
    counts = {}
    base.NS = NS
    try:
        with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
            queries = dict(nodes='MATCH (n {namespace:$ns}) RETURN n.key AS key,labels(n) AS labels,properties(n) AS properties ORDER BY key',
                relationships='MATCH (h)-[r]->(t) WHERE r.namespace=$ns RETURN r.key AS key,type(r) AS type,h.key AS head,t.key AS tail,properties(r) AS properties ORDER BY key')
            for kind, query in queries.items():
                result = session.run(query, ns=NS)
                count = 0
                for row in result:
                    data = row.data()
                    digest.update((dumps(data) + '\n').encode())
                    count += 1
                require(not result.consume().counters.contains_updates, 'Read changed Neo4j')
                counts[kind] = count
            outside = base.outside_snapshot(session)
    finally:
        driver.close()
    expected = json.loads((base.ER / 'global_kg_2f3/verification.json').read_text(encoding='utf-8'))
    require(counts == dict(nodes=873, relationships=2728), 'Live baseline counts changed')
    require(digest.hexdigest() == expected['target_snapshot_after_second_import'], 'Live baseline properties changed')
    return dict(namespace=NS, database=database, **counts, sha256=digest.hexdigest(), outside=outside)


def protected_files():
    paths = [p for p in base.ER.rglob('*') if p.is_file() and not p.is_relative_to(OUT)]
    paths += [base.DOCUMENTS, old.CONFIG, old.ONTOLOGY, Path(old.__file__), Path(core.__file__), ROOT / 'scripts/reconstruct_global_kg_2f3.py']
    return {p.relative_to(ROOT).as_posix(): base.sha(p) for p in sorted(set(paths))}


def discover():
    banner('НАЧИНАЮ ПОИСК МЕЖДОКУМЕНТНЫХ КАНДИДАТОВ')
    original, mapping, entities, docs, facts, semantic, concepts, profiles = inputs()
    before = live_snapshot()
    protected = protected_files()
    baseline_pairs = graph_pairs(mapping, facts, semantic, concepts)
    concept_ids = {c['tail_id'] for c in concepts}
    baseline, membership = components(profiles, baseline_pairs, concept_ids)
    require(baseline == dict(weak_components=223, largest_component_vertices=369, largest_component_documents=17), 'Baseline connectivity mismatch')
    candidates, paths = {}, {}
    name_terms = {k: terms(p['name']) for k, p in profiles.items()}
    context_terms = {k: terms(p['source_context']['text']) for k, p in profiles.items()}

    def add(a, b, source, source_id, reason):
        if a == b or profiles[a]['document_id'] == profiles[b]['document_id']:
            return None
        a, b = sorted((a, b))
        pid = identity('pair:', [a, b])
        if pid not in candidates:
            pa, pb = profiles[a], profiles[b]
            shared = sorted(context_terms[a] & context_terms[b])
            jaccard = len(shared) / max(1, len(context_terms[a] | context_terms[b]))
            generic = any(p['name'].casefold() in GENERIC for p in (pa, pb))
            candidates[pid] = dict(pair_id=pid, left_id=a, right_id=b,
                left_name=pa['name'], right_name=pb['name'], left_document_id=pa['document_id'], right_document_id=pb['document_id'],
                sources={}, selection_reasons=[], evidence_path_ids=[], evidence_path_count=0,
                non_generic_evidence_path_count=0,
                generic_control=generic, same_baseline_component=membership[a] == membership[b],
                shared_context_terms=shared, context_jaccard=jaccard,
                shared_name_terms=sorted(name_terms[a] & name_terms[b]),
                same_discipline=pa['discipline'] == pb['discipline'],
                directions_evaluated=['left_to_right', 'right_to_left'])
        row = candidates[pid]
        row['sources'].setdefault(source, [])
        if source_id not in row['sources'][source]:
            row['sources'][source].append(source_id)
        if reason not in row['selection_reasons']:
            row['selection_reasons'].append(reason)
        return row

    bridges = []
    for s in progress(semantic, desc='SIMILAR_TO candidate discovery'):
        if s['original_cross_document']:
            r = add(s['head_id'], s['tail_id'], 'A', s['semantic_link_id'], 'Existing cross-document SIMILAR_TO; no identity assumed')
            bridges.append(dict(bridge_id=s['semantic_link_id'], kind='SIMILAR_TO', x1=s['head_id'], x2=s['tail_id'],
                                source_ids=[s['semantic_link_id']], decision_2f2=s['decision'],
                                similarity=float(s['original_link']['similarity']), identity=False))
    by_concept = defaultdict(list)
    for c in concepts:
        by_concept[c['tail_id']].append(c)
    for cid, links in sorted(by_concept.items()):
        for a, b in combinations(links, 2):
            if profiles[a['head_id']]['document_id'] == profiles[b['head_id']]['document_id']:
                continue
            bridge_id = identity('bridge:', [cid, a['concept_link_id'], b['concept_link_id']])
            add(a['head_id'], b['head_id'], 'B', bridge_id, 'Shared abstraction; instance identity and fact inheritance prohibited')
            bridges.append(dict(bridge_id=bridge_id, kind='HAS_SHARED_CONCEPT', x1=a['head_id'], x2=b['head_id'],
                concept_id=cid, source_ids=[a['concept_link_id'], b['concept_link_id']], identity=False))
    banner('НАЧИНАЮ ПОСТРОЕНИЕ EVIDENCE PATHS')
    neighbors = defaultdict(list)
    fact_index = {}
    for f in facts:
        fid = f['canonical_triple_id']
        h, t = mapping[f['head_id']], mapping[f['tail_id']]
        def endpoint(eid):
            p = old.profile(eid, 180)
            return {k: p[k] for k in ('canonical_entity_id', 'document_id', 'name', 'selected_source_entity_id',
                                     'mention_start', 'mention_end', 'mention_text', 'source_context', 'source_mention_ids')}
        evidence = []
        for sr in f['source_relations']:
            for e in sr['evidence']:
                require(docs[e['document_id']]['text'][e['start']:e['end']] == e['text'], 'Changed factual source passage')
                evidence.append(dict(triple_id=sr['triple_id'], window_id=e['window_id'], document_id=e['document_id'],
                                     start=e['start'], end=e['end'], text_sha256=hashlib.sha256(e['text'].encode()).hexdigest()))
        fact_index[fid] = dict(canonical_triple_id=fid, source_triple_ids=f['source_triple_ids'],
            original_head_id=f['head_id'], original_tail_id=f['tail_id'], head_id=h, tail_id=t,
            relation_type=f['relation_type'], document_id=f['document_id'],
            confidence=max(x['confidence'] for x in f['source_confidences']),
            source_endpoints=[endpoint(f['head_id']), endpoint(f['tail_id'])], evidence_passages=evidence,
            source_file='global_entity_resolution/preview_factual_relations.jsonl', source_claim_not_manually_validated=True)
        if h != t:
            neighbors[h].append((t, fid, 'forward'))
            neighbors[t].append((h, fid, 'reverse'))
    # Enumerate every one-hop factual flank. Save three deterministic representative
    # paths per endpoint pair; count all paths and retain all bridge source IDs.
    total_paths = 0
    for bridge in progress(bridges, desc='Evidence bridge expansion', unit='bridge'):
        x1, x2 = bridge['x1'], bridge['x2']
        bridge['generic_intermediate'] = any(profiles[x]['name'].casefold() in GENERIC for x in (x1, x2))
        for a, f1, d1 in sorted(neighbors[x1]):
            for b, f2, d2 in sorted(neighbors[x2]):
                row = add(a, b, 'C', bridge['bridge_id'], 'Factual flank / semantic-or-concept bridge / factual flank; non-transitive path')
                if row is None:
                    continue
                path_id = identity('path:', [a, f1, bridge['bridge_id'], f2, b])
                row['evidence_path_count'] += 1
                row['non_generic_evidence_path_count'] += not bridge['generic_intermediate']
                total_paths += 1
                path = dict(path_id=path_id, left_id=a, right_id=b, intermediate_entity_ids=[x1, x2],
                    factual_relation_ids=[f1, f2], bridge=bridge,
                    factual_traversals=[dict(from_id=a, to_id=x1, direction='reverse' if d1 == 'forward' else 'forward'),
                                        dict(from_id=x2, to_id=b, direction=d2)],
                    document_ids=[profiles[a]['document_id'], profiles[b]['document_id']],
                    factual_evidence_refs=[f1, f2], identity_assumed=False, entails_endpoint_relation=False)
                paths[path_id] = path
                row['evidence_path_ids'].append(path_id)
                row['evidence_path_ids'].sort(key=lambda k: (paths[k]['bridge']['generic_intermediate'],
                    paths[k]['bridge'].get('decision_2f2') == 'REJECT', k))
                row['evidence_path_ids'] = row['evidence_path_ids'][:POLICY['max_saved_paths_per_pair']]
    paths = {pid: paths[pid] for row in candidates.values() for pid in row['evidence_path_ids']}
    # D: a deliberately narrow, reproducible lexical search over different WCC.
    # Existing name/context tokens only; no cosine or embedding computation.
    d_pool = defaultdict(list)
    profile_ids = sorted(profiles)
    for i, a in enumerate(progress(profile_ids, desc='Cross-component lexical search', unit='entity')):
        for b in profile_ids[i + 1:]:
            if membership[a] == membership[b] or profiles[a]['document_id'] == profiles[b]['document_id']:
                continue
            if any(profiles[x]['name'].casefold() in GENERIC for x in (a, b)):
                continue
            shared_name = name_terms[a] & name_terms[b]
            shared_context = context_terms[a] & context_terms[b]
            j = len(shared_context) / max(1, len(context_terms[a] | context_terms[b]))
            compatible = bool(set(profiles[a]['entity_types']) & set(profiles[b]['entity_types']))
            same_field = profiles[a]['discipline'] == profiles[b]['discipline']
            if len(shared_name) < 1 or len(shared_context) < 3 or j < .045 or not compatible or not same_field:
                continue
            if not neighbors[a] or not neighbors[b]:
                continue
            ds = tuple(sorted((profiles[a]['document_id'], profiles[b]['document_id'])))
            score = j + .05 * len(shared_name) + .005 * min(len(neighbors[a]), len(neighbors[b]), 10)
            d_pool[ds].append((score, a, b))
    for ds, pool in sorted(d_pool.items()):
        for score, a, b in sorted(pool, key=lambda v: (-v[0], v[1], v[2]))[:2]:
            r = add(a, b, 'D', identity('lexical:', [a, b]), 'Different baseline WCC, shared name/context terms, compatible type, same discipline, factual neighbors')
            r['discovery_lexical_score'] = score

    def rank(r):
        # Specific endpoint names, same field and source-context vocabulary outrank
        # generic links. Baseline relation scores are not evidence entailment scores.
        return (2 * r['same_discipline'] + r['context_jaccard'] + .1 * len(r['shared_name_terms']) +
                .01 * min(r['evidence_path_count'], 5))
    for r in candidates.values():
        r['selection_rank_score'] = rank(r)
        r['selected_for_inference'] = False
        r['screening_reason'] = 'Discovery inventory; outside predeclared bounded research sample'
    nonspecific = [r for r in candidates.values() if not r['generic_control']]
    selected = {r['pair_id'] for r in nonspecific if {'A', 'B'} & r['sources'].keys()}
    for source, limit in [('C', POLICY['inference_C_limit']), ('D', POLICY['inference_D_limit'])]:
        pool = sorted([r for r in nonspecific if source in r['sources'] and
                       (source != 'C' or r['non_generic_evidence_path_count'])], key=lambda r: (-rank(r), r['pair_id']))
        # Round robin over document pairs prevents a single medical cohort dominating.
        grouped = defaultdict(list)
        for r in pool:
            grouped[tuple(sorted((r['left_document_id'], r['right_document_id'])))].append(r)
        order = []
        while any(grouped.values()):
            for ds in sorted(grouped):
                if grouped[ds]:
                    order.append(grouped[ds].pop(0))
        selected.update(r['pair_id'] for r in order[:limit])
    controls = []
    for term in ('results', 'patients', 'study'):
        pool = sorted([r for r in candidates.values() if 'A' in r['sources'] and
                       r['left_name'].casefold() == term and r['right_name'].casefold() == term], key=lambda r: r['pair_id'])
        controls.extend(pool[:2])
    selected.update(r['pair_id'] for r in controls)
    path_controls = sorted([r for r in nonspecific if r['evidence_path_count'] and not r['non_generic_evidence_path_count']
                            and r['pair_id'] not in selected], key=lambda r: (-rank(r), r['pair_id']))[:POLICY['generic_path_control_limit']]
    selected.update(r['pair_id'] for r in path_controls)
    for pid in selected:
        candidates[pid]['selected_for_inference'] = True
        candidates[pid]['screening_reason'] = ('Bounded diagnostic generic control' if candidates[pid]['generic_control'] else
            'Bounded diagnostic generic-intermediate path control' if candidates[pid] in path_controls else
            'Predeclared source/discipline/document-pair screening')
    rows = sorted(candidates.values(), key=lambda r: r['pair_id'])
    for r in rows:
        for ids in r['sources'].values():
            ids.sort()
        r['candidate_category'] = 'evidence_path_available' if r['evidence_path_ids'] else 'exploratory'
    jsonl('candidate_pairs.jsonl', progress(rows, desc='Saving candidate inventory'))
    jsonl('evidence_paths.jsonl', sorted(paths.values(), key=lambda p: p['path_id']))
    jsonl('source_factual_evidence.jsonl', sorted(fact_index.values(), key=lambda p: p['canonical_triple_id']))
    jsonl('entity_profiles.jsonl', profiles.values())
    summary = dict(discovered_pairs=len(rows), selected_for_inference=len(selected),
        source_original_cross_similar_to=121, cross_similar_unique_pairs=sum('A' in r['sources'] for r in rows),
        source_counts={s: sum(s in r['sources'] for r in rows) for s in 'ABCD'},
        selected_source_counts={s: sum(s in r['sources'] and r['selected_for_inference'] for r in rows) for s in 'ABCD'},
        discovered_documents=len({r[k] for r in rows for k in ('left_document_id', 'right_document_id')}),
        selected_documents=len({r[k] for r in rows if r['selected_for_inference'] for k in ('left_document_id', 'right_document_id')}),
        generic_controls=len(controls), generic_path_controls=len(path_controls),
        all_enumerated_paths=total_paths, saved_representative_paths=len(paths),
        baseline=baseline, policy=POLICY)
    manifest = dict(stage='2F.4A-C', created_utc=datetime.now(timezone.utc).isoformat(), namespace=NS,
        source_commit='e9f1994', source_files_sha256=protected, before_neo4j=before,
        selection=summary, model=old.MODEL_ID, model_path=str(old.MODEL_PATH.relative_to(ROOT)),
        runtime={k: version(k) for k in ('torch', 'gliner', 'transformers', 'tqdm', 'neo4j')},
        ner_threshold=.46, fixed_span_adapter_sha256=base.sha(Path(core.__file__)),
        relation_types=core.PREDICATES, thresholds=THRESHOLDS, neo4j_import=False)
    save('manifest.json', manifest)
    lines = ['# 2F.4A — Candidate discovery', '', '```json', json.dumps(summary, ensure_ascii=False, indent=2), '```', '',
        'Методика зафиксирована до inference. Все 121 SIMILAR_TO учтены, после ER это 113 уникальных пар. '
        'Список содержит весь обнаруженный пул; selected_for_inference задаёт ограниченную исследовательскую выборку. '
        'Источники A–D перекрываются. Граф, включая общие слова и все старые рёбра, не меняется.', '',
        'C: только одна междокументная семантическая перемычка либо один общий концепт и по одному factual ребру с каждой стороны. '
        'Более длинные цепочки не исследуются. Полный счёт путей сохранён, тексты/provenance вынесены в source_factual_evidence.jsonl и entity_profiles.jsonl; '
        'до трёх реальных путей на пару сохранены для проверки. Все bridge IDs сохранены в sources.C. '
        'Сам путь не доказывает транзитивности и не разрешает перенос фактов через концепт.', '',
        'D: узкий поиск по пересечению слов названий и контекстов, дисциплине, типам и наличию factual окрестностей. '
        'Это ограничивает полноту: парафразы без общих слов не обнаруживаются; broad discipline не гарантирует научной совместимости.', '',
        'В модель подаются два исходных фрагмента с документными границами, точные spans и один сохранённый путь с двумя source excerpts. '
        'Путь явно помечен как non-entailing; synthetic input не source evidence. Два направления оцениваются одним вызовом адаптера.', '',
        '## Реальные примеры до запуска модели', '', '| Категория | Сущности / документы | Причина и ограничение |', '|---|---|---|']
    examples = []
    lines += ['Сильных в смысле доказанного научного вывода до аудита нет. В первой группе — лишь относительно перспективные кандидаты для проверки. '
              'Первичный вариант отбора выявил доминирование patients как промежуточного узла: основной C-отбор исправлен до полного inference; '
              'пути только через общие промежуточные термины ограничены шестью диагностическими парами.', '']
    for kind, condition in [('Перспективный для проверки, не доказанный', lambda r: not r['generic_control'] and r['non_generic_evidence_path_count'] and r['same_discipline']),
                            ('Слабый / общий термин', lambda r: r['generic_control']),
                            ('Сомнительный cross-component', lambda r: 'D' in r['sources'])]:
        for r in sorted(filter(condition, rows), key=lambda r: (-rank(r), r['pair_id']))[:3]:
            examples.append((kind, r))
            lines.append(f"| {kind} | {r['left_name']} ({r['left_document_id']}) ↔ {r['right_name']} ({r['right_document_id']}) | {r['selection_reasons'][0]}; слова: {', '.join(r['shared_context_terms'][:12])}; это приоритет проверки, не поддержанное отношение |")
    (OUT / 'candidate_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def checkpoint():
    path = OUT / 'relex_predictions.jsonl'
    if not path.exists():
        return {}
    # A killed final append is recoverable; a damaged complete line is an error.
    data = path.read_bytes()
    if data and not data.endswith(b'\n'):
        prefix = data[:data.rfind(b'\n') + 1]
        save('checkpoint_recovery.json', dict(discarded_incomplete_tail_bytes=len(data) - len(prefix)))
        path.write_bytes(prefix)
    rows = core.read_jsonl(path)
    require(len({r['pair_id'] for r in rows}) == len(rows), 'Duplicate checkpoint pair')
    return {r['pair_id']: r for r in rows}


def build_input(pair, profiles, paths, fact_index, radius, path_radius):
    endpoints = [profiles[pair[s + '_id']] for s in ('left', 'right')]
    text, members, fragments = '', [], []
    for p in endpoints:
        doc = old._documents[p['document_id']]
        context = p['source_context']
        lo, hi = max(context['start'], p['mention_start'] - radius), min(context['end'], p['mention_end'] + radius)
        text += f"[DOCUMENT {p['document_id']}]\n"
        offset = len(text)
        a, b = offset + p['mention_start'] - lo, offset + p['mention_end'] - lo
        fragment = doc['text'][lo:hi]
        text += fragment + f"\n[END DOCUMENT {p['document_id']}]\n"
        members.append(dict(entity_id=p['canonical_entity_id'], start=a, end=b, surface_text=p['mention_text'], entity_types=p['entity_types']))
        fragments.append(dict(document_id=p['document_id'], start=lo, end=hi, text=fragment,
            revision_id=doc['revision_id'], selected_source_entity_id=p['selected_source_entity_id']))
    chosen_path = paths[pair['evidence_path_ids'][0]] if pair['evidence_path_ids'] else None
    path_fragments = []
    if chosen_path:
        text += '\n[GRAPH PATH: NOT AN ENTAILED CLAIM]\n'
        for fid in chosen_path['factual_relation_ids']:
            fact = fact_index[fid]
            names = [p['name'] for p in fact['source_endpoints']]
            text += f"{names[0]} --{core.PREDICATES[fact['relation_type']]}--> {names[1]}\n"
            # A compact exact source excerpt centred on the original factual head.
            p = fact['source_endpoints'][0]
            context = p['source_context']
            lo = max(context['start'], p['mention_start'] - path_radius)
            hi = min(context['end'], p['mention_end'] + path_radius)
            fragment = old._documents[p['document_id']]['text'][lo:hi]
            text += f"[SOURCE {p['document_id']}] {fragment}\n"
            path_fragments.append(dict(factual_relation_id=fid, document_id=p['document_id'], start=lo, end=hi, text=fragment))
        text += f"[BRIDGE {chosen_path['bridge']['kind']}; NOT IDENTITY; NO FACT INHERITANCE]\n"
    return text, members, fragments, chosen_path, path_fragments


def infer_pair(pair, profiles, paths, facts, signature):
    model, cfg = old._model, old._cfg
    for radius, path_radius in [(180, 90), (120, 70), (90, 50), (60, 35), (40, 20), (20, 10)]:
        text, members, fragments, path, path_fragments = build_input(pair, profiles, paths, facts, radius, path_radius)
        tokens, starts, ends = model.prepare_inputs([text])
        for m in members:
            require(text[m['start']:m['end']] == m['surface_text'], 'Joint source span mismatch')
            require(m['start'] in starts[0] and m['end'] in ends[0], 'Unaligned fixed span')
        processor = model.data_processor
        prompted, _ = processor.prepare_inputs(tokens, cfg['entity_labels'], relations=list(core.PREDICATES.values()))
        count = len(processor.transformer_tokenizer(prompted, is_split_into_words=True, truncation=False, padding=False)['input_ids'][0])
        if count <= cfg['token_limit']:
            break
    else:
        raise ValueError('Prepared evidence path and complete entity spans exceed token budget')
    window = dict(window_id=identity('joint:', [pair['pair_id'], text]), document_id='synthetic:' + pair['pair_id'], start=0, end=len(text), text=text)
    for m in members:
        m['document_id'] = window['document_id']
    raw, audit = core.predict_window(model, window, members, cfg['entity_labels'], .35, cfg['token_limit'])
    require(audit['spans'] == 2 and audit['pairs'] == 2 and audit['tokens'] == count, 'Adapter truncation or span count mismatch')
    lookup = {(m['start'], m['end']): m for m in members}
    reverse = {label: code for code, label in core.PREDICATES.items()}
    predictions = []
    for r in raw:
        h, t = [lookup[(r[s]['start'], r[s]['end'])] for s in ('head', 'tail')]
        require(all(r[s]['text'] == m['surface_text'] for s, m in zip(('head', 'tail'), (h, t))), 'Wrong prediction endpoint')
        code = reverse[r['relation']]
        predictions.append(dict(candidate_id=identity('xre2f4:', [h['entity_id'], code, t['entity_id']]),
            head_id=h['entity_id'], tail_id=t['entity_id'], relation_type=code, confidence=r['score'],
            direction='left_to_right' if h['entity_id'] == pair['left_id'] else 'right_to_left',
            evidence_path_ids=pair['evidence_path_ids'], model_path_id=path['path_id'] if path else None,
            source_factual_relation_ids=path['factual_relation_ids'] if path else [],
            experimental=True, synthetic_input_is_evidence=False, status='UNVERIFIED_PREDICTION'))
    return dict(record_type='pair_result', pair_id=pair['pair_id'], inference_signature=signature,
        model_input=dict(**window, synthetic=True, is_source_evidence=False), fixed_spans=members,
        source_fragments=fragments, path_fragments=path_fragments, included_path_id=path['path_id'] if path else None,
        context_radius=radius, path_radius=path_radius, audit=audit,
        relations=sorted(predictions, key=lambda r: r['candidate_id']), status='predicted' if predictions else 'no_predictions')


def inference(stage):
    banner('НАЧИНАЮ SMOKE TEST GLINER RELEX' if stage == 'pilot' else 'НАЧИНАЮ ПОЛНЫЙ INFERENCE GLINER RELEX')
    _, _, _, docs, _, _, _, profiles = inputs()
    pairs = [p for p in core.read_jsonl(OUT / 'candidate_pairs.jsonl') if p['selected_for_inference']]
    paths = {p['path_id']: p for p in core.read_jsonl(OUT / 'evidence_paths.jsonl')}
    facts = {p['canonical_triple_id']: p for p in core.read_jsonl(OUT / 'source_factual_evidence.jsonl')}
    manifest = load('manifest.json')
    cfg = json.loads(old.CONFIG.read_text(encoding='utf-8'))
    model_hashes = {name: base.sha(old.MODEL_PATH / name) for name in cfg['models']['base']}
    require(model_hashes == cfg['models']['base'], 'Local model differs from previous experiment')
    ontology = json.loads(old.ONTOLOGY.read_text(encoding='utf-8'))
    require(set(ontology['relation_types']) == set(core.PREDICATES), 'Ontology mismatch')
    signature_data = dict(model=old.MODEL_ID, checkpoint_sha256=model_hashes,
        candidate_sha256=base.sha(OUT / 'candidate_pairs.jsonl'), evidence_sha256=base.sha(OUT / 'evidence_paths.jsonl'),
        facts_sha256=base.sha(OUT / 'source_factual_evidence.jsonl'), profile_sha256=base.sha(OUT / 'entity_profiles.jsonl'),
        adapter_sha256=base.sha(Path(core.__file__)), runner_sha256=base.sha(Path(__file__)),
        token_limit=512, threads=cfg['threads'], seed=0, device='cpu', min_threshold=.35,
        entity_labels=cfg['entity_labels'], relation_labels=core.PREDICATES, runtime=manifest['runtime'])
    signature = core.digest(signature_data)
    cached = checkpoint()
    require(all(r['inference_signature'] == signature for r in cached.values()), 'Checkpoint signature differs; do not mix experiments')
    require(set(cached) <= {p['pair_id'] for p in pairs}, 'Checkpoint has unknown candidate pairs')
    print('ЗАГРУЖАЮ ТОЛЬКО ЛОКАЛЬНЫЙ CHECKPOINT; DOWNLOADS DISABLED.', flush=True)
    started = time.perf_counter()
    old.initialize(cfg, old._entities, docs)
    print(f'МОДЕЛЬ ЗАГРУЖЕНА ЗА {time.perf_counter() - started:.1f} СЕК.', flush=True)
    if stage == 'infer':
        gate = load('pilot_validation.json')
        require(gate['passed'] and gate['inference_signature'] == signature, 'Pilot gate not passed')
        tasks = pairs
    else:
        # Diverse pilot: sources, paths/no paths, generic controls, domains, WCC.
        tasks = []
        conditions = [lambda p: p['generic_control'] and p['left_name'].casefold() == 'results',
                      lambda p: p['generic_control'] and p['left_name'].casefold() == 'patients',
                      lambda p: 'D' in p['sources'], lambda p: 'B' in p['sources'] and not p['generic_control'],
                      lambda p: p['evidence_path_ids'] and not p['same_discipline'] and not p['generic_control']]
        for condition in conditions:
            found = next((p for p in pairs if condition(p) and p not in tasks), None)
            if found:
                tasks.append(found)
        for discipline in sorted({p['discipline'] for p in profiles.values()}):
            found = next((p for p in pairs if profiles[p['left_id']]['discipline'] == discipline and p not in tasks and not p['generic_control']), None)
            if found:
                tasks.append(found)
        tasks += [p for p in sorted(pairs, key=lambda r: (-r['selection_rank_score'], r['pair_id'])) if p not in tasks][:max(0, 16 - len(tasks))]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'relex_predictions.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
        for pair in progress(tasks, desc='Fixed-span pilot' if stage == 'pilot' else 'Fixed-span inference', unit='pair'):
            if pair['pair_id'] in cached:
                continue
            result = infer_pair(pair, profiles, paths, facts, signature)
            stream.write(dumps(result) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            cached[pair['pair_id']] = result
    if stage == 'pilot':
        first = cached[tasks[0]['pair_id']]
        repeated = infer_pair(tasks[0], profiles, paths, facts, signature)
        require(first == repeated, 'Non-deterministic smoke')
        pilot = dict(passed=False, technical_smoke_passed=True, pending_content_review=True,
            inference_signature=signature, pairs=[p['pair_id'] for p in tasks], pair_count=len(tasks),
            exact_repeat=True, tokens_max=max(cached[p['pair_id']]['audit']['tokens'] for p in tasks),
            positive_pairs=sum(bool(cached[p['pair_id']]['relations']) for p in tasks),
            predictions=sum(len(cached[p['pair_id']]['relations']) for p in tasks),
            fixed_spans_and_two_directions_verified=True, untruncated_paths=True)
        save('pilot_validation.json', pilot)
    else:
        require(len(cached) == len(pairs), 'Incomplete inference')
    manifest['inference'] = dict(**signature_data, signature=signature, completed_pairs=len(cached),
        selected_pairs=len(pairs), positive_pairs=sum(bool(r['relations']) for r in cached.values()),
        predictions=sum(len(r['relations']) for r in cached.values()), checkpoint_resume=True,
        elapsed_current_run_seconds=time.perf_counter() - started)
    save('manifest.json', manifest)
    print(json.dumps({k: manifest['inference'][k] for k in ('completed_pairs', 'selected_pairs', 'positive_pairs', 'predictions')}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['discover', 'pilot', 'infer'], required=True)
    args = parser.parse_args()
    if args.stage == 'discover':
        discover()
    else:
        inference(args.stage)


if __name__ == '__main__':
    main()
