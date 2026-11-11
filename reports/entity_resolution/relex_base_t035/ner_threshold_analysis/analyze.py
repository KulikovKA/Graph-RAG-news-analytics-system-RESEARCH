"""Stage 2F.0: offline induced-subgraph analysis; standard library only.

Run from any directory: .venv/Scripts/python.exe <path-to-this-file>
Writes only beside this script. Never imports project/model/database code.
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import statistics

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
ER = ROOT / 'reports/entity_resolution/relex_base_t035'
THRESHOLDS = [.3, .4, .5, .6, .7, .8, .9]
STRATEGIES = ['max_member', 'representative']
INPUTS = {
    'entities': ER / 'semantic_resolution/input/canonical_entities.jsonl',
    'relations': ER / 'relation_remapping/canonical_relations.jsonl',
    'links': ER / 'semantic_resolution/global_linking_preview/semantic_links.csv',
    'original_entities': ROOT / 'reports/re_v2/relex_comparison/threshold035/e3_base/entities.jsonl',
    'mentions': ROOT / 'reports/ner_v2/research_run2/E3/mentions_035.jsonl',
    'ner_config': ROOT / 'reports/ner_v2/research_run2/E3/config.json',
    're_config': ROOT / 'reports/re_v2/relex_comparison/threshold035/e3_base/experiment_config.json',
    'documents': ROOT / 'reports/ner_v2/research_run2/input_abstracts.jsonl',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_csv(name, rows):
    assert rows, name
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()} for row in rows)


def quantile(values, p):
    values = sorted(values)
    pos = (len(values) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def stats(values):
    return dict(count=len(values), minimum=min(values), p10=quantile(values, .1),
                p25=quantile(values, .25), median=statistics.median(values),
                mean=statistics.mean(values), p75=quantile(values, .75),
                p90=quantile(values, .9), maximum=max(values))


def connectivity(ids, edges, entity_by_id):
    adjacency = {eid: set() for eid in ids}
    touched = set()
    for a, b in edges:
        assert a in ids and b in ids
        touched.update((a, b))
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen, groups = set(), []
    for seed in sorted(ids):
        if seed in seen:
            continue
        stack, group = [seed], []
        seen.add(seed)
        while stack:
            current = stack.pop()
            group.append(current)
            for other in sorted(adjacency[current] - seen):
                seen.add(other)
                stack.append(other)
        groups.append(sorted(group))
    groups.sort(key=lambda group: (-len(group), group[0]))
    largest = groups[0] if groups else []
    return dict(isolated_vertices=len(ids - touched), weak_components=len(groups),
                largest_component_vertices=len(largest),
                largest_component_documents=len({entity_by_id[eid]['document_id'] for eid in largest})), groups


def main():
    # Guard all pre-existing report artifacts and all directly consumed sources.
    protected = sorted({p for p in ER.rglob('*') if p.is_file() and OUT not in p.parents} | set(INPUTS.values()))
    before = {p: sha(p) for p in protected}
    entities = jsonl(INPUTS['entities'])
    originals = jsonl(INPUTS['original_entities'])
    mentions = jsonl(INPUTS['mentions'])
    documents = {d['document_id']: d for d in jsonl(INPUTS['documents'])}
    relations = jsonl(INPUTS['relations'])
    with INPUTS['links'].open(encoding='utf-8-sig', newline='') as stream:
        links = [r for r in csv.DictReader(stream)
                 if (r['model'], r['mode'], float(r['threshold'])) == ('intfloat/e5-small-v2', 'text_name', .95)]
    by_id = {e['canonical_entity_id']: e for e in entities}
    assert len(entities) == len(by_id) == 1122 and len(documents) == 30
    assert len(originals) == 1211 and len(mentions) == 1327 and len(relations) == 1760 and len(links) == 685
    assert len({r['canonical_triple_id'] for r in relations}) == len(relations)
    assert len({r['semantic_link_id'] for r in links}) == len(links)
    assert all(r['source_input_sha256'] == before[INPUTS['entities']] for r in links)
    raw, canonical = [], []
    for e in entities:
        eid = e['canonical_entity_id']
        rep = next(m for m in e['members'] if m['source_entity']['entity_id'] == eid)
        best = sorted(e['members'], key=lambda m: (-m['source_entity']['confidence'], m['source_entity']['entity_id']))[0]
        assert e['context_source_entity_id'] == eid
        for m in e['members']:
            source = m['source_entity']
            assert source['document_id'] == e['document_id']
            assert .35 <= source['confidence'] <= 1
            assert source['confidence'] == max(x['score'] for x in source['source_mentions'])
            assert all(x['score'] >= .35 for x in source['source_mentions'])
            assert source['surface_text'] == documents[e['document_id']]['text'][source['start']:source['end']]
            ctx = m['context']
            assert ctx['text'] == documents[e['document_id']]['text'][ctx['start']:ctx['end']]
            raw.append(dict(entity_id=source['entity_id'], canonical_entity_id=eid,
                            document_id=e['document_id'], discipline=e['provenance']['document']['discipline'],
                            name=source['surface_text'], types=source['entity_types'],
                            confidence=source['confidence'], start=source['start'], end=source['end'],
                            mention_count=len(source['source_mentions']), context=ctx['text']))
        canonical.append(dict(canonical_entity_id=eid, document_id=e['document_id'],
                              discipline=e['provenance']['document']['discipline'],
                              title=e['provenance']['document']['title'], name=e['surface_text'], types=e['entity_types'],
                              member_count=len(e['members']), representative=rep['source_entity']['confidence'],
                              max_member=best['source_entity']['confidence'],
                              max_source_entity_id=best['source_entity']['entity_id'],
                              representative_context=rep['context']['text'], max_member_context=best['context']['text']))
    flat_sources = {m['source_entity']['entity_id']: m['source_entity'] for e in entities for m in e['members']}
    assert len(raw) == len(flat_sources) == 1211
    assert flat_sources == {e['entity_id']: e for e in originals}
    flat_mentions = {m['mention_id']: m for e in originals for m in e['source_mentions']}
    assert flat_mentions == {m['mention_id']: m for m in mentions}
    all_types = sorted({t for e in entities for t in e['entity_types']})
    name_docs = defaultdict(set)
    for e in canonical:
        name_docs[e['name'].casefold()].add(e['document_id'])
    for e in canonical:
        e['name_document_frequency_casefold'] = len(name_docs[e['name'].casefold()])
    scores = {s: {e['canonical_entity_id']: e[s] for e in canonical} for s in STRATEGIES}
    re_pairs = [(r['head_id'], r['tail_id']) for r in relations]
    sem_pairs = [(r['left_canonical_id'], r['right_canonical_id']) for r in links]
    assert all(a in by_id and b in by_id for a, b in re_pairs + sem_pairs)
    assert all(by_id[a]['document_id'] == by_id[b]['document_id'] for a, b in re_pairs)
    assert sum(a == b for a, b in re_pairs) == 13
    base, _ = connectivity(set(by_id), re_pairs + sem_pairs, by_id)
    assert base == dict(isolated_vertices=200, weak_components=258, largest_component_vertices=580, largest_component_documents=23)
    assert sum(by_id[a]['document_id'] != by_id[b]['document_id'] for a, b in sem_pairs) == 194
    comparison, per_type, per_document, per_discipline, components, membership, examples, disagreements = [], [], [], [], [], [], [], []
    index = {e['canonical_entity_id']: e for e in canonical}
    for strategy in STRATEGIES:
        previous = set(by_id)
        for threshold in THRESHOLDS:
            ids = {eid for eid, score in scores[strategy].items() if score >= threshold}
            assert ids <= previous
            previous = ids
            re_kept = [r for r in relations if r['head_id'] in ids and r['tail_id'] in ids]
            sem_kept = [r for r in links if r['left_canonical_id'] in ids and r['right_canonical_id'] in ids]
            edges = [(r['head_id'], r['tail_id']) for r in re_kept] + [(r['left_canonical_id'], r['right_canonical_id']) for r in sem_kept]
            conn, groups = connectivity(ids, edges, by_id)
            cross = [r for r in sem_kept if r['left_document_id'] != r['right_document_id']]
            retained = [index[eid] for eid in sorted(ids)]
            count_types = Counter(t for e in retained for t in e['types'])
            row = dict(strategy=strategy, threshold=threshold, effective_observed_floor=max(.35, threshold),
                       below_original_cutoff=threshold < .35, entities=len(ids), entities_per_document=len(ids)/30,
                       entities_removed=1122-len(ids), entities_retained_pct=100*len(ids)/1122,
                       relex_rel=len(re_kept), relex_rel_retained_pct=100*len(re_kept)/1760,
                       similar_to=len(sem_kept), similar_to_retained_pct=100*len(sem_kept)/685,
                       cross_document_relex_rel=0, cross_document_similar_to=len(cross), cross_document_links=len(cross),
                       cross_document_links_retained_pct=100*len(cross)/194,
                       within_document_similar_to=len(sem_kept)-len(cross), kg_edges=len(edges),
                       factual_self_loops=sum(r['head_id']==r['tail_id'] for r in re_kept),
                       documents_with_entities=len({e['document_id'] for e in retained}),
                       documents_with_cross_document_links=len({r[side+'_document_id'] for r in cross for side in ('left','right')}),
                       isolated_pct=100*conn['isolated_vertices']/len(ids) if ids else 0, **conn)
            row.update({'type_'+t: count_types[t] for t in all_types})
            comparison.append(row)
            for t in all_types:
                per_type.append(dict(strategy=strategy, threshold=threshold, entity_type=t,
                                     baseline=sum(t in e['types'] for e in canonical), retained=count_types[t]))
            for doc_id in sorted(documents):
                original = [e for e in canonical if e['document_id']==doc_id]
                kept = [e for e in retained if e['document_id']==doc_id]
                per_document.append(dict(strategy=strategy, threshold=threshold, document_id=doc_id,
                                         discipline=original[0]['discipline'], title=original[0]['title'],
                                         baseline=len(original), retained=len(kept), removed=len(original)-len(kept),
                                         retention_pct=100*len(kept)/len(original),
                                         relex_rel=sum(r['document_id']==doc_id for r in re_kept),
                                         within_document_similar_to=sum(r['left_document_id']==r['right_document_id']==doc_id for r in sem_kept),
                                         incident_cross_document_links=sum(doc_id in (r['left_document_id'],r['right_document_id']) for r in cross)))
            for discipline in sorted({e['discipline'] for e in canonical}):
                n = sum(e['discipline']==discipline for e in canonical)
                k = sum(e['discipline']==discipline for e in retained)
                per_discipline.append(dict(strategy=strategy, threshold=threshold, discipline=discipline,
                                           baseline=n, retained=k, removed=n-k, retention_pct=100*k/n))
            for rank, group in enumerate(groups, 1):
                doc_ids = sorted({by_id[eid]['document_id'] for eid in group})
                components.append(dict(strategy=strategy, threshold=threshold, component_rank=rank,
                                       vertices=len(group), documents=len(doc_ids), document_ids=doc_ids, entity_ids=group))
            membership.append(dict(strategy=strategy, threshold=threshold, entity_ids=sorted(ids),
                                   relex_rel_ids=sorted(r['canonical_triple_id'] for r in re_kept),
                                   similar_to_ids=sorted(r['semantic_link_id'] for r in sem_kept)))
            for status, pool in [('removed',[e for e in canonical if e['canonical_entity_id'] not in ids]), ('retained',retained)]:
                # Boundary examples and extremes; deterministic, explicitly not a random quality sample.
                ordered = sorted(pool, key=lambda e: (abs(e[strategy]-threshold), e['canonical_entity_id']))
                selected = ordered[:2]
                if pool:
                    extreme = min(pool, key=lambda e:e[strategy]) if status=='removed' else max(pool,key=lambda e:e[strategy])
                    if extreme not in selected:
                        selected.append(extreme)
                for e in selected:
                    examples.append(dict(strategy=strategy, threshold=threshold, status=status,
                                         canonical_entity_id=e['canonical_entity_id'], document_id=e['document_id'], discipline=e['discipline'],
                                         name=e['name'], types=e['types'], confidence=e[strategy],
                                         representative=e['representative'], max_member=e['max_member'],
                                         context=e[strategy+'_context']))
    for threshold in THRESHOLDS:
        a={eid for eid,s in scores['max_member'].items() if s>=threshold}
        b={eid for eid,s in scores['representative'].items() if s>=threshold}
        assert b <= a
        for eid in sorted(a-b):
            disagreements.append(dict(threshold=threshold, **index[eid]))
    distributions, histogram = [], []
    bins = [0,.3,.35,.4,.5,.6,.7,.8,.9,1.0]
    for level, records, field in [('source_entities',raw,'confidence'), ('canonical_representative',canonical,'representative'), ('canonical_max_member',canonical,'max_member')]:
        scopes = [('all','all',records)]
        scopes += [('type',t,[e for e in records if t in e['types']]) for t in all_types]
        scopes += [('document',d,[e for e in records if e['document_id']==d]) for d in sorted(documents)]
        scopes += [('discipline',d,[e for e in records if e['discipline']==d]) for d in sorted({e['discipline'] for e in records})]
        for scope,key,group in scopes:
            values = [e[field] for e in group]
            distributions.append(dict(level=level, scope=scope, group=key, **stats(values)))
            for lo,hi in zip(bins,bins[1:]):
                n=sum(lo<=v and (v<hi or hi==1 and v==1) for v in values)
                histogram.append(dict(level=level,scope=scope,group=key,lower_inclusive=lo,upper=hi,
                                      upper_inclusive=hi==1,count=n,pct=100*n/len(values)))
    write_csv('comparison.csv', comparison)
    write_csv('entity_confidences.csv', canonical)
    write_csv('source_entity_confidences.csv', raw)
    write_csv('distribution_summary.csv', distributions)
    write_csv('confidence_histogram.csv', histogram)
    write_csv('retention_by_type.csv', per_type)
    write_csv('retention_by_document.csv', per_document)
    write_csv('retention_by_discipline.csv', per_discipline)
    write_csv('components.csv', components)
    write_csv('examples.csv', examples)
    write_csv('strategy_disagreements.csv', disagreements)
    with (OUT/'virtual_subgraphs.jsonl').open('w',encoding='utf-8') as stream:
        for row in membership:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    write_json('metrics.json',dict(baseline=dict(documents=30,source_entities=1211,canonical_entities=1122,relex_rel=1760,similar_to=685,cross_document_links=194,**base),comparison=comparison))
    after = {p: sha(p) for p in protected}
    assert before == after, 'Protected artifacts changed'
    config=json.loads(INPUTS['re_config'].read_text(encoding='utf-8'))
    ner_config=json.loads(INPUTS['ner_config'].read_text(encoding='utf-8'))
    assert any(p.replace('\\','/').endswith('E3/mentions_035.jsonl') and h==before[INPUTS['mentions']] for p,h in config['source_sha256'].items())
    write_json('validation.json',dict(original_ner_cutoff=.35,minimum_historical_inference_threshold=ner_config['minimum_inference_threshold'],
                                     relex_config_entity_threshold=config['entity_threshold'],
                                     source_entity_confidence_equals_max_source_mention_score=True,
                                     source_entity_reconstruction_exact=True,source_mention_reconstruction_exact=True,
                                     baseline_connectivity_matched=True,canonical_representative_ids_matched=True,
                                     all_contexts_offsets_verified=True,all_threshold_sets_nested=True,
                                     representative_subset_of_max_member=True,protected_files_unchanged=True,
                                     protected_file_count=len(protected),sources={k:dict(path=str(p.relative_to(ROOT)).replace('\\','/'),sha256=before[p]) for k,p in INPUTS.items()},
                                     protected_sha256={str(p.relative_to(ROOT)).replace('\\','/'):h for p,h in before.items()}))
    print(json.dumps(dict(baseline=base,comparison=[{k:r[k] for k in ('strategy','threshold','entities','entities_removed','relex_rel','similar_to','cross_document_links','isolated_vertices','weak_components','largest_component_vertices','largest_component_documents')} for r in comparison]),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
