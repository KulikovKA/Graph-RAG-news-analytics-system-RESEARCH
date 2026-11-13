"""Local GLiREL evaluation on the frozen Stage 2F.4 corpus; no graph access.

Install the missing loguru dependency outside the main environment, if needed:
  .venv/Scripts/python.exe -m pip install --target .cache/glirel_runtime loguru
Run --stage sanity, inspect technical/content results, then --stage cross.
The two model files are read in place, never downloaded, rewritten or committed.
"""
import argparse
import copy
from datetime import datetime, timezone
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from unittest.mock import patch

import experiment_split_context as split

ROOT, core = split.ROOT, split.core
OUT = split.onto.previous.OUT / 'glirel_evaluation'
SANITY = split.onto.OUT / 'sanity_pairs.jsonl'
CROSS = split.OUT / 'contexts_S2.jsonl'
DEPENDENCIES = ROOT / '.cache/glirel_runtime'
BACKBONE = ROOT / 'artifacts/models/deberta-v3-large'
TOKEN_RE = re.compile(r'\w+(?:[-_]\w+)*|\S')  # Actual GLiREL 1.2.1 string API tokenizer.
THRESHOLDS = [.35, .50, .70]
require, sha, banner, progress = split.require, split.sha, split.banner, split.progress


def save(obj):
    core.atomic_json(OUT/'manifest.json', obj)


def load():
    return json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))


def model_path():
    for path in (ROOT/'model', Path.home()/'glirel-test/model'):
        if all((path/n).is_file() for n in ('glirel_config.json','pytorch_model.bin')):
            return path
    raise FileNotFoundError('No complete local GLiREL checkpoint in either authorized location')


def initialize(path):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    if DEPENDENCIES.exists():
        sys.path.insert(0, str(DEPENDENCIES))
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from glirel import GLiREL
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    tokenizer_loader, hub_loader = AutoTokenizer.from_pretrained, GLiREL._from_pretrained

    @classmethod
    def compatible_loader(cls, **kwargs):
        # hub >=1 strips deprecated kwargs; GLiREL 1.2.1 still requires them.
        kwargs.setdefault('proxies', None)
        kwargs.setdefault('resume_download', False)
        return hub_loader(**kwargs)

    def local_tokenizer(name, **kwargs):
        require(name == 'microsoft/deberta-v3-large', 'Unexpected backbone request')
        return tokenizer_loader(str(BACKBONE), local_files_only=True, **kwargs)

    def backbone_from_config(name, **kwargs):
        require(name == 'microsoft/deberta-v3-large', 'Unexpected backbone request')
        # The complete encoder is in GLiREL's state_dict. No base weights needed.
        return AutoModel.from_config(AutoConfig.from_pretrained(str(BACKBONE),local_files_only=True))

    with patch.object(GLiREL, '_from_pretrained', compatible_loader), \
         patch.object(AutoTokenizer, 'from_pretrained', side_effect=local_tokenizer), \
         patch.object(AutoModel, 'from_pretrained', side_effect=backbone_from_config):
        model = GLiREL.from_pretrained(str(path), local_files_only=True, strict=True)
    model.to('cpu').eval()
    model.device=torch.device('cpu')
    require(not model.training and model.config.fixed_relation_types and model.config.max_entity_pair_distance is None, 'Model eval configuration')
    require(not getattr(model.config,'add_entity_markers',False), 'Unexpected span-shifting marker preprocessing')
    import glirel
    api = dict(predict_relations_signature=str(inspect.signature(GLiREL.predict_relations)),
        state_dict_entries=len(model.state_dict()),parameters=sum(p.numel() for p in model.parameters()),
        strict_load=True,missing_keys=[],unexpected_keys=[],
        runtime={p:version(p) for p in ['glirel','torch','transformers','huggingface-hub','tqdm','loguru']},
        python=sys.executable,tokenizer_class=type(model.token_rep_layer.bert_layer.tokenizer).__name__,
        tokenizer_size=len(model.token_rep_layer.bert_layer.tokenizer),max_source_words=model.config.max_len,
        conservative_transformer_budget=512,offline=True,threads=2,seed=0,deterministic=True,
        loader_compatibility='process-local legacy Hub kwargs; local tokenizer; encoder from local config then complete strict GLiREL state_dict load',
        package_sources_sha256={p.relative_to(Path(glirel.__file__).parent).as_posix():sha(p)
            for p in [Path(glirel.__file__).parent/n for n in ['model.py','modules/base.py','modules/token_rep.py','modules/transformer_embeddings.py']]})
    return model, api


def tokenize_context(item):
    matches=list(TOKEN_RE.finditer(item['text']))
    tokens=[m.group() for m in matches]
    starts={m.start():i for i,m in enumerate(matches)}
    ends={m.end():i for i,m in enumerate(matches)}
    ner, mapped = [], []
    require(len(item['fixed_spans'])==2,'Expected exactly two fixed targets')
    for sp in item['fixed_spans']:
        require(item['text'][sp['start']:sp['end']]==sp['surface_text'],'Bad original character span')
        require(sp['start'] in starts and sp['end'] in ends,'Target not GLiREL token-aligned')
        a,b=starts[sp['start']],ends[sp['end']]
        require(tokens[a:b+1]==TOKEN_RE.findall(sp['surface_text']),'Token/mention mismatch')
        ner.append([a,b,'ENTITY',sp['surface_text']])  # Inclusive end in input.
        mapped.append(dict(sp,token_start=a,token_end_inclusive=b,mention_tokens=tokens[a:b+1]))
    require(len({(e[0],e[1]) for e in ner})==2,'Collapsed identical-surface mentions')
    return dict(tokens=tokens,token_offsets=[[m.start(),m.end()] for m in matches],ner=ner,targets=mapped)


def preflight(model,item):
    a=tokenize_context(item)
    if len(a['tokens'])>model.config.max_len:
        return dict(a,fits=False,source_word_count=len(a['tokens']))
    rels=split.labels()
    batch=model.collate_fn([dict(tokenized_text=a['tokens'],ner=copy.deepcopy(a['ner']))],
        relation_types=list(rels.values()),device='cpu')
    require(batch['tokens'][0]==a['tokens'] and batch['seq_length'].item()==len(a['tokens']),'GLiREL source token truncation')
    require(batch['span_idx'].shape==(1,2,2) and batch['span_mask'].all().item(),'Wrong supplied entity spans')
    require(batch['span_idx'][0].tolist()==[e[:2] for e in a['ner']],'Shifted inclusive entity spans')
    pairs=batch['relations_idx'][0].tolist()
    expected=[[a['ner'][0][:2],a['ner'][1][:2]],[a['ner'][1][:2],a['ner'][0][:2]]]
    require(sorted(pairs)==sorted(expected),'Both directions not prepared')
    require(list(batch['classes_to_id'][0])==list(rels.values()),'Prompt order/label mapping')
    prompted=[value for label in rels.values() for value in [model.rel_token,label]]+[model.sep_token]+a['tokens']
    enc=model.token_rep_layer.bert_layer.tokenizer([prompted],is_split_into_words=True,
        truncation=False,padding=False)
    count=len(enc['input_ids'][0])
    require(set(w for w in enc.word_ids(0) if w is not None)==set(range(len(prompted))),'Missing subword-to-word alignment')
    a.update(source_word_count=len(a['tokens']),transformer_tokens=count,prompt_words=len(prompted)-len(a['tokens']),
        prepared_pairs=pairs,label_index_to_prompt=batch['id_to_classes'][0],
        fits=(len(a['tokens'])<=model.config.max_len and count<=512))
    return a


def crop(item, fraction):
    """Shrink each original source block by the same fraction around its targets."""
    result=copy.deepcopy(item)
    text, spans, fragments='',[],[]
    for f in item['source_fragments']:
        endpoints=[(s,p) for s,p in item['endpoints'].items() if p['document_id']==f['document_id']]
        require(bool(endpoints),'Fragment without target document')
        lo=min(p['mention_start'] for s,p in endpoints);hi=max(p['mention_end'] for s,p in endpoints)
        low=lo-int((lo-f['start'])*fraction);high=hi+int((f['end']-hi)*fraction)
        words=list(TOKEN_RE.finditer(f['text']))
        lefts=[f['start']+w.start() for w in words if f['start']+w.start()<=low]
        rights=[f['start']+w.end() for w in words if f['start']+w.end()>=high]
        low=max(lefts);high=min(rights)
        part=f['text'][low-f['start']:high-f['start']]
        text+=f"[DOCUMENT {f['document_id']}]\n";offset=len(text);text+=part
        for slot,p in endpoints:
            spans.append(dict(entity_id=p['canonical_entity_id'],target_slot=slot,
                start=offset+p['mention_start']-low,end=offset+p['mention_end']-low,
                surface_text=p['mention_text'],entity_types=p['entity_types']))
        text+=f"\n[END DOCUMENT {f['document_id']}]\n\n"
        fragments.append(dict(f,start=low,end=high,text=part,model_text_start=offset,model_text_end=offset+len(part)))
    result.update(text=text,fixed_spans=sorted(spans,key=lambda s:s['target_slot']),source_fragments=fragments,
        glirel_crop_fraction=fraction)
    result['input_sha256']=core.digest([text,result['fixed_spans']])
    return result


def prepare(model, source):
    a=preflight(model,source)
    if a['fits']:
        return dict(source), a, True
    for fraction in [.95,.90,.85,.80,.75,.70,.65,.60,.55,.50,.45,.40,.35,.30,.25,.20,.15,.10,.05,0]:
        item=crop(source,fraction);a=preflight(model,item)
        if a['fits']:
            return item,a,False
    raise ValueError('Complete target spans and labels cannot fit GLiREL budget')


def convert(raw, audit, rels):
    # GLiREL output positions use exclusive end, unlike its inclusive ner input.
    targets={(sp['token_start'],sp['token_end_inclusive']+1):sp for sp in audit['targets']}
    reverse={p:c for c,p in rels.items()};scores=[]
    for r in raw:
        h,t=targets[tuple(r['head_pos'])],targets[tuple(r['tail_pos'])]
        require(h!=t,'Self-pair output')
        for side,sp in [('head',h),('tail',t)]:
            require(r[side+'_text']==sp['mention_tokens'],'Returned mention tokens shifted')
        require(r['label'] in reverse and 0<=r['score']<=1,'Bad output label/score')
        scores.append(dict(head_id=h['entity_id'],tail_id=t['entity_id'],head_slot=h['target_slot'],tail_slot=t['target_slot'],
            relation_code=reverse[r['label']],relation_prompt=r['label'],confidence=r['score'],
            head_token_span_exclusive=r['head_pos'],tail_token_span_exclusive=r['tail_pos']))
    keys={(s['head_slot'],s['tail_slot'],s['relation_code']) for s in scores}
    require(len(keys)==len(scores),'Duplicate API output')
    missing=[dict(head_slot=h,tail_slot=t,relation_code=code,score=None)
        for h,t in [('E2','E3'),('E3','E2')] for code in rels if (h,t,code) not in keys]
    return scores,missing


def checkpoint(path, signature):
    if not path.exists():return {}
    require(not path.read_bytes() or path.read_bytes().endswith(b'\n'),'Incomplete checkpoint tail; preserve and repair before resume')
    rows=core.read_jsonl(path)
    require(len({r['pair_id'] for r in rows})==len(rows),'Duplicate checkpoint pair')
    require(all(r['inference_signature']==signature for r in rows),'Foreign checkpoint signature')
    return {r['pair_id']:r for r in rows}


def main(stage):
    OUT.mkdir(parents=True,exist_ok=True)
    banner('НАЧИНАЮ ЛОКАЛЬНУЮ GLIREL: ПРОВЕРКА HASHES И ЗАГРУЗКА БЕЗ СКАЧИВАНИЯ ВЕСОВ')
    path=model_path()
    hashes={n:sha(path/n) for n in ['glirel_config.json','pytorch_model.bin']}
    sources={str(p.relative_to(ROOT)):sha(p) for p in [SANITY,CROSS,split.onto.OUT/'cross_document_ontology.json',
        split.OUT/'predictions_S2.jsonl',split.OUT/'pair_scope_review.csv']}
    if (OUT/'manifest.json').exists():
        m=load()
        require(m['model_sha256']==hashes and m['source_sha256']==sources,'Frozen source/model changed')
        require(m['runner_sha256']==sha(Path(__file__)),'Inference runner changed after freeze')
        require(split.protected(m['protected_files_sha256'])==m['protected_files_sha256'],'Previous tracked files changed')
    else:
        require(stage=='sanity','Run sanity first')
        m=dict(stage='2F.4 local GLiREL evaluation',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
            created_utc=datetime.now(timezone.utc).isoformat(),client_date='2026-11-13',
            model_id='jackboyla/glirel-large-v0',model_path=str(path),model_sha256=hashes,source_sha256=sources,
            backbone_files_sha256={n:sha(BACKBONE/n) for n in ['config.json','tokenizer_config.json','spm.model']},
            runner_sha256=sha(Path(__file__)),relation_labels=split.labels(),thresholds=THRESHOLDS,
            retrieval_threshold=0.0,top_k=-1,threshold_operator='>=',
            design='Frozen S2 bytes, two fixed targets, official GLiREL regex and inclusive ner end; all available API scores; conservative 512 subword budget; same-fraction symmetric source crop only if required; no label or score-based changes.',
            protected_files_sha256=split.protected(),independent_gold=False,neo4j_calls=0,ner_calls=0,embedding_calls=0,training_calls=0,
            missing_dependency='loguru installed with dependencies only under .cache/glirel_runtime; main environment unchanged')
    model,api=initialize(path)
    if 'api' in m:require(m['api']==api,'Runtime/package API changed')
    m['api']=api
    signature=core.digest(dict(model_sha256=hashes,sources=sources,runner=m['runner_sha256'],api=api,
        labels=m['relation_labels'],threshold=0.0,top_k=-1,budget=512))
    if 'inference_signature' in m:require(m['inference_signature']==signature,'Signature changed')
    m['inference_signature']=signature
    save(m)  # Freeze settings before the first sanity forward.
    if stage=='cross':
        gate=m.get('sanity',{})
        require(gate.get('technical_passed') and gate.get('reviewed'), 'Sanity technical and source review required')
        require(sha(OUT/'sanity_results.jsonl')==gate['results_sha256'],'Reviewed sanity changed')
    source=SANITY if stage=='sanity' else CROSS
    output=OUT/('sanity_results.jsonl' if stage=='sanity' else 'cross_document_predictions.jsonl')
    cached=checkpoint(output,signature)
    items=core.read_jsonl(source)
    require(len(items)==(6 if stage=='sanity' else 28),'Incorrect pair count')
    require(set(cached)<=set(i['pair_id'] for i in items),'Foreign checkpoint pair')
    banner(f'НАЧИНАЮ GLIREL {stage.upper()}: {len(items)} ПАР, CPU, TQDM, CHECKPOINT ПОСЛЕ КАЖДОЙ ПАРЫ')
    started=time.perf_counter();forwards=0
    for src in progress(items,desc=f'GLiREL {stage}',unit='pair'):
        if src['pair_id'] in cached:continue
        item,a,identical=prepare(model,src)
        transformer_counts=[]
        def check_encoder(module,args,kwargs):
            ids=kwargs.get('input_ids',args[0] if args else None)
            transformer_counts.append(ids.shape[1])
            require(ids.shape[1]==a['transformer_tokens'],'Actual encoder input differs from preflight')
        hook=model.token_rep_layer.bert_layer.model.register_forward_pre_hook(check_encoder,with_kwargs=True)
        try:
            raw=model.predict_relations(a['tokens'],list(split.labels().values()),ner=copy.deepcopy(a['ner']),
                threshold=0.0,top_k=-1)
        finally:hook.remove()
        require(transformer_counts==[a['transformer_tokens']],'Unexpected encoder forward count')
        scores,missing=convert(raw,a,split.labels())
        row=dict(pair_id=item['pair_id'],pool=stage,inference_signature=signature,
            source_input_sha256=src['input_sha256'],actual_input_sha256=core.digest([item['text'],item['fixed_spans']]),
            unchanged_source_input=identical,actual_input=dict(text=item['text'],fixed_spans=item['fixed_spans'],
                source_fragments=item['source_fragments'],endpoints=item['endpoints']),token_audit=a,
            raw_api_predictions=raw,raw_scores=scores,unreturned_scores=missing,
            relations=[s for s in scores if s['confidence']>=.35],actual_encoder_forwards=1)
        if stage=='sanity':row.update(expected_D_code=src['expected_D_code'],expected_direction=src['expected_direction'])
        split.onto.append(output,row);forwards+=1
    rows=core.read_jsonl(output)
    require(len(rows)==len(items),'Incomplete run')
    m[stage]=dict(technical_passed=True,reviewed=False,pairs=len(rows),results_sha256=sha(output),
        available_scores=sum(len(r['raw_scores']) for r in rows),unreturned_score_slots=sum(len(r['unreturned_scores']) for r in rows),
        identical_inputs=sum(r['unchanged_source_input'] for r in rows),elapsed_seconds=time.perf_counter()-started,
        newly_computed_forwards=forwards)
    m['model_forwards']=sum(len(core.read_jsonl(OUT/n)) for n in ['sanity_results.jsonl','cross_document_predictions.jsonl'] if (OUT/n).exists())
    require({n:sha(path/n) for n in hashes}==hashes,'Local model files changed')
    save(m)
    print(json.dumps(m[stage]),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['sanity','cross'],required=True)
    main(parser.parse_args().stage)
