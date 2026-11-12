"""Stage 2F.4 ontology experiment. Offline fixed-span inference; no graph IO.

Run with .venv/Scripts/python.exe --stage prepare, pilot, infer.
Labels and input bytes freeze before the first forward. Resume never retunes labels.
"""
import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import time
import types

import compare_cross_document_prompts as prior

core, previous, ROOT = prior.core, prior.previous, prior.ROOT
SOURCE = previous.OUT / 'positive_candidate_audit'
OUT = previous.OUT / 'ontology_experiment'
PILOT = ['P01', 'P03', 'P06', 'N01', 'N07', 'W04']
require, banner, progress = prior.require, prior.banner, prior.progress


def load(name):
    return json.loads((OUT/name).read_text(encoding='utf-8'))


def save(name, value):
    core.atomic_json(OUT/name, value)


def configs():
    original = json.loads((prior.OUT/'configurations.json').read_text(encoding='utf-8'))
    labels = {k: v['relation_labels'] for k,v in original.items()}
    labels['D'] = {r['code']: r['label'] for r in load('cross_document_ontology.json')['relations']}
    return labels


def protected(paths=None):
    if paths is None:
        paths = subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
        paths = [p for p in paths if p and not p.startswith(OUT.relative_to(ROOT).as_posix()+'/')
                 and p != Path(__file__).relative_to(ROOT).as_posix()]
    return {p: prior.base.sha(ROOT/p) for p in progress(paths,desc='Protected input hashes',unit='file')}


def data():
    docs = {d['document_id']:d for d in core.read_jsonl(prior.base.DOCUMENTS)}
    profiles = {p['canonical_entity_id']:p for p in core.read_jsonl(previous.OUT/'entity_profiles.jsonl')}
    facts = {f['canonical_triple_id']:f for f in core.read_jsonl(previous.OUT/'source_factual_evidence.jsonl')}
    return docs,profiles,facts


def fragment(doc, start, end, radius):
    # Preserve every target character, crop only at GLiNER word boundaries.
    bounds = sorted({0,len(doc['text'])} | {m.start() for m in core.WORDS.finditer(doc['text'])})
    low = max(b for b in bounds if b <= max(0,start-radius))
    high = min(b for b in bounds if b >= min(len(doc['text']),end+radius))
    return low,high


def context(spec, docs, profiles, radius):
    text,spans,fragments = '',[],[]
    endpoints = [profiles[spec[s+'_id']] for s in ('E2','E3')]
    if spec['pool']=='main':
        chunks = [[(i,p)] for i,p in enumerate(endpoints)]
    else:
        chunks = [[(i,p) for i,p in enumerate(endpoints)]]
    for chunk in chunks:
        doc = docs[chunk[0][1]['document_id']]
        low,high = fragment(doc,min(p['mention_start'] for _,p in chunk),max(p['mention_end'] for _,p in chunk),radius)
        text += f"[DOCUMENT {doc['document_id']}]\n"
        offset = len(text)
        text += doc['text'][low:high] + f"\n[END DOCUMENT {doc['document_id']}]\n"
        for i,p in chunk:
            spans.append(dict(entity_id=p['canonical_entity_id'],target_slot=['E2','E3'][i],
                start=offset+p['mention_start']-low,end=offset+p['mention_end']-low,
                surface_text=p['mention_text'],entity_types=p['entity_types']))
        fragments.append(dict(document_id=doc['document_id'],revision_id=doc['revision_id'],
            source_url=doc['source_url'],start=low,end=high,text=doc['text'][low:high]))
    spans.sort(key=lambda s:s['target_slot'])
    return dict(spec,text=text,fixed_spans=spans,source_fragments=fragments,context_radius=radius,
        synthetic=spec['pool']=='main',is_source_evidence=False,local_relations_inserted=False,
        input_sha256=core.digest([text,spans]))


def prepare():
    banner('НАЧИНАЮ ПРОВЕРКУ RELATION LABELS И ФИКСАЦИЮ ОБЩИХ КОНТЕКСТОВ')
    require(not (OUT/'experiment_pairs.jsonl').exists(),'Inputs already frozen; use pilot/infer resume')
    labels=configs()
    require([len(v) for v in labels.values()]==[20,9,20,9],'Unexpected ontology sizes')
    docs,profiles,facts=data()
    records=core.read_jsonl(SOURCE/'candidate_evidence.jsonl')
    specs=[]
    for r in records:
        specs.append(dict(pair_id=r['audit_id'],pool='main',sample_role=r['record_type'],
            E2_id=r['E2']['profile']['canonical_entity_id'],E3_id=r['E3']['profile']['canonical_entity_id'],
            endpoints={s:r[s]['profile'] for s in ('E2','E3')},
            source_audit_id=r['audit_id'],evidence_ids=r['evidence_ids'],
            original_control_pair_id=r.get('previous_experiment_pair_id'),
            local_factual_ids={s:[f['canonical_triple_id'] for f in r['local_factual_neighborhood'][s]] for s in ('E2','E3')}))
    require(len(specs)==28 and len({tuple(sorted((s['E2_id'],s['E3_id']))) for s in specs})==28,'Main pair duplicates')
    sanity_ids=[
        ('6bc9f298194e668d817bbf558c70c2fa38cfc4cf6e599b001855313d1d846164','SUBTYPE_OF','E2_TO_E3'),
        ('187f615fc985a99c9659b0bf0eedb6a50544aeb056520f203786f710f70f4fff','HAS_PROPERTY','E2_TO_E3'),
        ('1fd476fd02f25a2a83966170882f7d94ef3e3dadc882e0a400bf8de1b49e331d','HAS_PROPERTY','E2_TO_E3'),
        ('2ec3cc4dbe1b26100264c6949be25bb07a16a5e6dfb2cc65cc392ca74c989e07','DERIVED_FROM','E2_TO_E3'),
        ('0f72780579f43fb283b728e626c8d3fd23bb7774f23cfeaae0b00ad832b3eee0','APPLIED_TO','E2_TO_E3'),
        ('1079d408f3e9abffb7fce2534c6619e8b2ffba37965e5de3484f72350b80e941','USED_FOR','E2_TO_E3')]
    for i,(fid,expected,direction) in enumerate(sanity_ids,1):
        f=facts['ct:'+fid]
        specs.append(dict(pair_id=f'S{i:02}',pool='sanity',sample_role='source_checked_intra_document',
            E2_id=f['head_id'],E3_id=f['tail_id'],endpoints={s:profiles[eid] for s,eid in [('E2',f['head_id']),('E3',f['tail_id'])]},
            evidence_ids=[f['canonical_triple_id']]+f['source_triple_ids'],source_factual_relation=f,
            expected_D_code=expected,expected_direction=direction,source_checked_before_inference=True))
    model,cfg,model_hashes=prior.initialize()
    prepared=[]
    for spec in progress(specs,desc='Common A/B/C/D tokenizer+collator audit',unit='pair'):
        for radius in [360,280,220,180,140,100,60,20,0]:
            item=context(spec,docs,profiles,radius)
            counts={}
            for name,rel in labels.items():
                n,starts,ends=prior.token_audit(model,cfg,item['text'],rel)
                counts[name]=n
                for sp in item['fixed_spans']:
                    require(item['text'][sp['start']:sp['end']]==sp['surface_text'],'Target text mismatch')
                    require(sp['start'] in starts and sp['end'] in ends,'Target not word aligned')
            if max(counts.values())<=512:
                item['token_counts']=counts
                prepared.append(item)
                break
        else:raise ValueError('Cannot fit complete targets in every prompt budget')
    for pool,name in [('main','experiment_pairs.jsonl'),('sanity','sanity_pairs.jsonl')]:
        core.write_jsonl(OUT/name,[r for r in prepared if r['pool']==pool])
    common=dict(model_id=prior.old.MODEL_ID,checkpoint_sha256=model_hashes,
        adapter_sha256=prior.base.sha(Path(core.__file__)),runner_sha256=prior.base.sha(Path(__file__)),
        ontology_sha256=prior.base.sha(OUT/'cross_document_ontology.json'),
        contexts_sha256={n:prior.base.sha(OUT/n) for n in ['experiment_pairs.jsonl','sanity_pairs.jsonl']},
        configurations=labels,entity_labels=cfg['entity_labels'],device='cpu',threads=cfg['threads'],
        seed=0,deterministic=True,token_limit=512,raw_storage_cutoff=0.0,positive_cutoff_strictly_greater=.35,
        runtime={k:version(k) for k in ['torch','gliner','transformers','tqdm']})
    signatures={k:core.digest(dict(common,configuration=k,ordered_labels=list(v.items()))) for k,v in labels.items()}
    save('manifest.json',dict(stage='2F.4 ontology experiment',source_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        created_utc=datetime.now(timezone.utc).isoformat(),design_set_used=True,independent_gold=False,
        protected_files_sha256=protected(),common_inference=common,signatures=signatures,
        pilot_ids=PILOT,model_forwards=0,neo4j_calls=0,ner_calls=0,embedding_calls=0,
        preforward_artifacts_sha256={n:prior.base.sha(OUT/n) for n in ['cross_document_ontology.json','ontology_design.md','ontology_coverage.csv','experiment_pairs.jsonl','sanity_pairs.jsonl']}))
    save('validation.json',dict(preparation_passed=True,pairs=28,sanity_pairs=6,exact_spans=True,
        label_maps_verified=True,max_tokens={k:max(r['token_counts'][k] for r in prepared) for k in labels},
        no_expected_label_or_audit_status_in_model_text=True,local_predictions_not_inserted=True))


def checked_manifest():
    manifest=load('manifest.json');common=manifest['common_inference']
    require(protected(manifest['protected_files_sha256'])==manifest['protected_files_sha256'],'Protected source changed')
    require(prior.base.sha(Path(__file__))==common['runner_sha256'],'Frozen runner changed')
    require(configs()==common['configurations'],'Frozen labels changed')
    for n,h in manifest['preforward_artifacts_sha256'].items():require(prior.base.sha(OUT/n)==h,'Frozen input changed: '+n)
    return manifest


def checkpoint(path,signature=None):
    if not path.exists():return {}
    payload=path.read_bytes()
    if payload and not payload.endswith(b'\n'):
        end=payload.rfind(b'\n')+1
        core.atomic_json(path.with_suffix('.recovery.json'),dict(discarded_bytes=len(payload)-end))
        path.write_bytes(payload[:end])
    rows=core.read_jsonl(path)
    require(len({(r['configuration'],r['pair_id']) for r in rows})==len(rows),'Duplicate checkpoint')
    if signature:require(all(r['inference_signature']==signature for r in rows),'Checkpoint signature mismatch')
    return {(r['configuration'],r['pair_id']):r for r in rows}


def prediction(model,cfg,item,name,signature,labels):
    window=dict(text=item['text'],start=0,end=len(item['text']))
    raw,audit=core.predict_window(model,window,item['fixed_spans'],cfg['entity_labels'],0.0,512,relation_labels=labels)
    require(audit['spans']==audit['pairs']==2 and audit['tokens']==item['token_counts'][name],'Head/span/token mismatch')
    require(audit['relation_prompt_count']==len(labels) and audit['relation_prompts']==list(labels.values()),'Head labels differ')
    lookup={(p['start'],p['end']):p for p in item['fixed_spans']}
    reverse={v:k for k,v in labels.items()}
    scores=[]
    for x in raw:
        h,t=[lookup[(x[s]['start'],x[s]['end'])] for s in ['head','tail']]
        require(h!=t and all(x[s]['text']==p['surface_text'] for s,p in [('head',h),('tail',t)]),'Unexpected target')
        code=reverse[x['relation']]
        require(math.isfinite(x['score']) and 0<x['score']<=1,'Bad raw score')
        scores.append(dict(prediction_id='onto:'+core.digest([name,item['pair_id'],h['entity_id'],code,t['entity_id']])[:24],
            head_id=h['entity_id'],tail_id=t['entity_id'],head_slot=h['target_slot'],tail_slot=t['target_slot'],
            relation_code=code,relation_prompt=x['relation'],confidence=x['score']))
    require(len(scores)==2*len(labels),'Incomplete all-direction raw scores')
    scores.sort(key=lambda r:r['prediction_id'])
    return dict(configuration=name,pair_id=item['pair_id'],pool=item['pool'],input_sha256=item['input_sha256'],
        inference_signature=signature,audit=audit,raw_scores=scores,relations=[x for x in scores if x['confidence']>.35])


def append(path,row):
    with path.open('a',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def infer(stage):
    manifest=checked_manifest();labels=configs();validation=load('validation.json')
    if stage=='infer':
        gate=validation.get('pilot',{})
        require(gate.get('technical_passed') and gate.get('content_reviewed'),'Pilot review required')
        require(prior.base.sha(OUT/'pilot_review.csv')==gate['content_review_sha256'],'Pilot review changed')
    banner('НАЧИНАЮ PILOT CROSS-DOCUMENT RELEX' if stage=='pilot' else 'НАЧИНАЮ СРАВНЕНИЕ ОНТОЛОГИЙ A/B/C/D')
    started=time.perf_counter();model,cfg,hashes=prior.initialize()
    require(hashes==manifest['common_inference']['checkpoint_sha256'],'Checkpoint changed')
    main=core.read_jsonl(OUT/'experiment_pairs.jsonl');sanity=core.read_jsonl(OUT/'sanity_pairs.jsonl')
    forwards=0
    if stage=='pilot' and not validation.get('adapter_compatibility'):
        item=main[0];window=dict(text=item['text'],start=0,end=len(item['text']))
        historical=types.ModuleType('historical_adapter');historical.__file__=str(core.__file__)
        exec(compile(subprocess.check_output(['git','show','2e7c740:scripts/build_relex_graphs.py']).decode('utf-8'),str(core.__file__),'exec'),historical.__dict__)
        old_raw,old_audit=historical.predict_window(model,window,item['fixed_spans'],cfg['entity_labels'],0.0,512)
        default_raw,default_audit=core.predict_window(model,window,item['fixed_spans'],cfg['entity_labels'],0.0,512)
        explicit_raw,explicit_audit=core.predict_window(model,window,item['fixed_spans'],cfg['entity_labels'],0.0,512,relation_labels=labels['A'])
        require(old_raw==default_raw==explicit_raw and old_audit==default_audit,'Default adapter regression')
        validation['adapter_compatibility']=dict(passed=True,historical_commit='2e7c740',pair_id=item['pair_id'],
            all_raw_scores_identical=True,default_audit_identical=True,forwards=3)
        forwards+=3
        save('validation.json',validation)
    for name,rels in labels.items():
        path=OUT/f'predictions_{name}.jsonl';cached=checkpoint(path,manifest['signatures'][name])
        targets=[p for p in main if stage=='infer' or p['pair_id'] in PILOT]
        pending=[p for p in targets if (name,p['pair_id']) not in cached]
        for item in progress(pending,desc=f'{name} {stage} fixed spans',unit='pair'):
            result=prediction(model,cfg,item,name,manifest['signatures'][name],rels);append(path,result);forwards+=1
        if stage=='pilot':
            path=OUT/'sanity_predictions.jsonl';cached=checkpoint(path)
            require(all(r['inference_signature']==manifest['signatures'][r['configuration']] for r in cached.values()),'Sanity resume signature')
            for item in progress([s for s in sanity if (name,s['pair_id']) not in cached],desc=f'{name} source sanity',unit='pair'):
                append(path,prediction(model,cfg,item,name,manifest['signatures'][name],rels));forwards+=1
    manifest['model_forwards']+=forwards
    manifest.setdefault('runs',[]).append(dict(stage=stage,new_forwards=forwards,elapsed_seconds=time.perf_counter()-started,
        completed_utc=datetime.now(timezone.utc).isoformat()))
    save('manifest.json',manifest)
    validation=load('validation.json')
    validation[stage+'_technical_passed']=True
    save('validation.json',validation)
    checked_manifest()
    print(json.dumps(dict(stage=stage,new_forwards=forwards,protected_unchanged=True)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','pilot','infer'],required=True)
    args=parser.parse_args()
    if args.stage=='prepare':prepare()
    else:infer(args.stage)
