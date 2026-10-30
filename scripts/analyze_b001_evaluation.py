"""Offline evaluator supplement. Never imported by extraction; no model/prompt changes."""
from collections import Counter
import argparse
from pathlib import Path
from cgraph.common import file_sha256, read_json, write_json
from cgraph.inputs import load_inputs
from cgraph.labels import load_labels
from cgraph.evaluation import entity_key, prf

cfg = read_json('config/b001.json')
docs = load_inputs(cfg, 'all')
gold = load_labels(cfg, docs)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--predictions', default='artifacts/b001/dev_primary')
args = parser.parse_args()
directory = Path(args.predictions)
run = read_json(directory / 'run.json')
totals, funnel, fp_categories, rejected_stage = Counter(), Counter(), Counter(), Counter()
by_kind = {k: {'gold': [], 'predicted': []} for k in read_json(cfg['ontology'])['entities']}
fn_examples, final_examples = {}, []
for doc in docs:
    code = doc['document_code']
    path = directory / f'{code}.json'
    assert file_sha256(path) == run['prediction_files_sha256'][code]
    pred, g = read_json(path), gold[code]
    gm = {m['mention_id']: m for m in g['mentions']}
    pm = {m['mention_id']: dict(m, chunk_id=c['chunk_id']) for c in pred['chunks'] for m in c['entities']}
    typed = {entity_key(m['chunk_id'], m) for m in pm.values()}
    untyped = {key[:3] for key in typed}

    def edge(a, mentions):
        h, t = mentions[a['head_mention_id']], mentions[a['tail_mention_id']]
        return entity_key(h['chunk_id'], h), entity_key(t['chunk_id'], t), a['relation_type']

    ga = [a for a in g['assertions'] if a['support_status'] == 'SUPPORTED']
    raw = [a for c in pred['chunks'] for a in c['raw_proposals']]
    final = [a for c in pred['chunks'] for a in c['assertions']]
    ge, re, fe = Counter(edge(a, gm) for a in ga), Counter(edge(a, pm) for a in raw), Counter(edge(a, pm) for a in final)
    pairs = {(entity_key(pm[p['head_mention_id']]['chunk_id'], pm[p['head_mention_id']]),
              entity_key(pm[p['tail_mention_id']]['chunk_id'], pm[p['tail_mention_id']]))
             for c in pred['chunks'] for p in c['candidate_pairs']}
    funnel.update(gold=len(ga), untyped_endpoints=sum(n for e,n in ge.items() if e[0][:3] in untyped and e[1][:3] in untyped),
                  typed_endpoints=sum(n for e,n in ge.items() if e[0] in typed and e[1] in typed),
                  candidate_pairs=sum(n for e,n in ge.items() if e[:2] in pairs),
                  raw=sum((ge & re).values()), final=sum((ge & fe).values()))
    for m in gm.values(): by_kind[m['kind']]['gold'].append(entity_key(m['chunk_id'], m))
    for m in pm.values(): by_kind[m['kind']]['predicted'].append(entity_key(m['chunk_id'], m))
    remaining = ge.copy()
    gs = {entity_key(m['chunk_id'],m) for m in gm.values()}
    gu = {x[:3] for x in gs}
    for a in final:
        e = edge(a, pm)
        match = remaining[e] > 0
        if match: remaining[e] -= 1
        else:
            category = 'boundary' if e[0][:3] not in gu or e[1][:3] not in gu else 'kind' if e[0] not in gs or e[1] not in gs else 'relation_or_unannotated'
            fp_categories[category] += 1
        final_examples.append(dict(document=code, head=pm[a['head_mention_id']]['surface_text'],
                                   tail=pm[a['tail_mention_id']]['surface_text'], relation=a['relation_type'],
                                   matched=match, polarity=a['polarity'], modality=a['modality'],
                                   evidence=a['evidence'][0]['text'], qualifiers=a['qualifiers']))
    for a in ga:
        e = edge(a, gm)
        if e in fe: continue
        stage = ('boundary' if e[0][:3] not in untyped or e[1][:3] not in untyped else
                 'kind' if e[0] not in typed or e[1] not in typed else
                 'candidate' if e[:2] not in pairs else 'model' if e not in re else 'validator')
        if stage not in fn_examples:
            h,t = gm[a['head_mention_id']],gm[a['tail_mention_id']]
            ch = next(c for c in doc['chunks'] if c['chunk_id']==h['chunk_id'])
            lo,hi = max(0,min(h['start'],t['start'])-60),min(len(ch['text']),max(h['end'],t['end'])+100)
            fn_examples[stage] = dict(document=code, head=h['surface_text'], head_kind=h['kind'],
                                     tail=t['surface_text'], tail_kind=t['kind'], relation=a['relation_type'],
                                     source_excerpt=ch['text'][lo:hi])
    for c in pred['chunks']:
        totals.update(entities=len(c['entities']), mapped_raw=len(c['raw_proposals']),
                      structural_records=len(c['structurally_validated']), accepted=len(c['assertions']),
                      rejected_records=len(c['rejected']), unresolved_qualifiers=len(c['unresolved']),
                      accepted_qualifiers=sum(len(a['qualifiers']) for a in c['assertions']))
        rejected_stage.update(x['stage'] for x in c['rejected'])
for relative, digest in run['raw_files_sha256'].items():
    path = directory / relative
    assert file_sha256(path)==digest
    record = read_json(path)
    if record['parsed'] is not None:
        totals.update(llm_entity_records=len(record['parsed']['entities']),
                      llm_relation_records=len(record['parsed']['relations']),
                      llm_explicit_abstain=sum(a['d']=='ABSTAIN' for a in record['parsed']['relations']))
    else:
        totals['failed_windows'] += 1
result = dict(counts=dict(totals), correct_gold_edges_remaining=dict(funnel), final_fp_categories=dict(fp_categories),
              rejected_stages=dict(rejected_stage), entity_by_kind={k:prf(v['gold'],v['predicted']) for k,v in by_kind.items()},
              fn_examples=fn_examples, final_examples=final_examples,
              candidate_policy_actual='One-sentence B windows and the unchanged one-sentence C candidate generator; no wider contexts in the primary run.',
              note='Post-inference offline evaluator supplement; no changes to frozen extraction, prompt, decoding or predictions. FP means unmatched to annotations, not independent semantic adjudication.')
write_json('reports/b001/error_analysis.json', result)
write_json('reports/b001/run.json', run)
print({k:result[k] for k in ('counts','correct_gold_edges_remaining','final_fp_categories','rejected_stages')})
