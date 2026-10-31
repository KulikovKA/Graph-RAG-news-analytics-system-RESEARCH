"""NER-V2-003: frozen corpus acquisition and candidate-quality experiments.

Historical research modules and results are read-only inputs to this experiment.
"""
import argparse
import hashlib
import json
import os
import random
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from cgraph.common import file_sha256, fingerprint, read_json, write_json
from .io import read_documents, validate_span, write_jsonl
from .openalex import OpenAlexClient, SELECT, reconstruct_abstract, source_record
from .research import Resources, SHORT, environment, jsonl

ROOT = Path('reports/ner_v2/research_run2')
THRESHOLDS = [.15, .25, .35, .45, .50, .55, .60, .65]
CORPUS_SEED = 4303


def prepare_corpus(root=ROOT):
    root=Path(root)
    snapshot=root/'openalex_snapshot.json'
    if snapshot.exists():
        saved=read_json(snapshot)
        records=[source_record(x['work'],x['field']) for x in saved['selected']]
        if (root/'input_abstracts.jsonl').exists():
            assert read_documents(root/'input_abstracts.jsonl')==records
        else:
            write_jsonl(root/'input_abstracts.jsonl',records)
        return records
    root.mkdir(parents=True,exist_ok=True)
    client=OpenAlexClient()
    fields=read_json('contracts/ner_v2/openalex_snapshot.json')['fields']
    old=read_documents('reports/ner_v2/benchmark_run1/input_abstracts.jsonl')
    used_ids={d['openalex_id'] for d in old}
    used_dois={d['doi'].lower() for d in old if d.get('doi')}
    used_hashes={d['text_sha256'] for d in old}
    selected=[];pools=[]
    title_exclusion=re.compile(r'\b(editorial|erratum|corrigendum|retraction|book review|book notice|isbn|edition)\b|^correction:|\bEd\.\s*\d',re.I)
    for field in fields:
        params={'filter':field['filter']+',has_abstract:true,language:en,type:article,is_retracted:false',
                'sample':150,'seed':CORPUS_SEED,'per_page':150,'select':SELECT+',is_retracted'}
        path=Path('artifacts/ner_v2/openalex_run2')/(field['discipline'].split('/')[0].strip()+'.json')
        if path.exists():payload=read_json(path)
        else:
            payload=client.get('works',params)
            write_json(path,payload)
        pool=[];rejected=Counter()
        for work in sorted(payload['results'],key=lambda x:x['id']):
            try:text=reconstruct_abstract(work.get('abstract_inverted_index'))
            except ValueError:
                rejected['invalid_abstract']+=1;continue
            source=(work.get('primary_location') or {}).get('source') or {}
            topic=work.get('primary_topic') or {}
            digest=hashlib.sha256(text.encode('utf8')).hexdigest()
            doi=(work.get('doi') or '').lower()
            letters=[c for c in text if c.isalpha()]
            latin_fraction=sum('a'<=c.lower()<='z' for c in letters)/len(letters) if letters else 0
            english_words={'the','and','of','in','to','a','is','with','for','we','was','were','that','this'}
            function_fraction=sum(w.strip('.,:;()').lower() in english_words for w in text.split())/len(text.split())
            reason=None
            if work['id'] in used_ids or (doi and doi in used_dois) or digest in used_hashes:reason='duplicate_or_old'
            elif work.get('language')!='en' or work.get('type')!='article' or work.get('is_retracted'):reason='language_type_retracted'
            elif source.get('type') not in {'journal','conference'}:reason='source_type'
            elif not 100<=len(text.split())<=400:reason='word_count'
            elif latin_fraction<.90 or function_fraction<.035:reason='abstract_language_preflight'
            elif not work.get('title') or title_exclusion.search(work['title']):reason='editorial_title'
            elif re.search(r'\bthis (?:book|volume|history of pediatrics) is\b|\bthe book under review\b',text,re.I):reason='book_review_abstract'
            elif not topic.get('id') or topic.get('field',{}).get('id')!=field['field']['id']:reason='field'
            elif field.get('subfield') and topic.get('subfield',{}).get('id')!=field['subfield']['id']:reason='subfield'
            if reason:rejected[reason]+=1
            else:pool.append(work)
        random.Random(CORPUS_SEED).shuffle(pool)
        chosen=[];topics=set()
        for work in pool:
            topic=work['primary_topic']['id']
            text=reconstruct_abstract(work['abstract_inverted_index'])
            digest=hashlib.sha256(text.encode('utf8')).hexdigest()
            doi=(work.get('doi') or '').lower()
            if topic in topics or work['id'] in used_ids or (doi and doi in used_dois) or digest in used_hashes:continue
            chosen.append(work);topics.add(topic)
            used_ids.add(work['id']);used_hashes.add(digest)
            if doi:used_dois.add(doi)
            if len(chosen)==5:break
        if len(chosen)!=5:raise ValueError('Insufficient eligible distinct-topic works; corpus not frozen')
        selected.extend({'field':field,'work':w} for w in chosen)
        pools.append(dict(discipline=field['discipline'],params=params,candidate_ids=[w['id'] for w in payload['results']],
                          response_sha256=file_sha256(path),eligible=len(pool),rejected=dict(rejected)))
        print(field['discipline'],[w['id'].rsplit('/',1)[-1] for w in chosen],flush=True)
    records=[source_record(x['work'],x['field']) for x in selected]
    assert len(records)==len({d['document_id'] for d in records})==30
    assert all(v==5 for v in Counter(d['discipline'] for d in records).values())
    write_json(snapshot,dict(seed=CORPUS_SEED,retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
                            selection_policy='sample150-sort-id-shuffle4303-first5-distinct-topics-100to400words-exclude-old-id-doi-sha',
                            selected=selected,pools=pools,requests=client.requests))
    write_jsonl(root/'input_abstracts.jsonl',records)
    audit_ids=[]
    for discipline in sorted({d['discipline'] for d in records}):
        subset=sorted([d for d in records if d['discipline']==discipline],key=lambda x:x['document_id'])
        audit_ids.append(random.Random(CORPUS_SEED).choice(subset)['document_id'])
    write_json(root/'corpus_manifest.json',dict(status='FROZEN_BEFORE_INFERENCE',corpus_seed=CORPUS_SEED,
               input_sha256=file_sha256(root/'input_abstracts.jsonl'),snapshot_sha256=file_sha256(snapshot),
               words=sum(len(d['text'].split()) for d in records),audit_document_ids=audit_ids,
               audit_selection='Random(4303).choice(sorted-by-id five documents) independently in each discipline',
               documents=[{k:d[k] for k in ['document_id','discipline','title','year','doi','field','primary_topic','source_url','text_sha256']} for d in records]))
    write_jsonl(root/'audit_abstracts_before_predictions.jsonl',[d for d in records if d['document_id'] in audit_ids])
    return records


def retained(score,threshold,model):
    """Respect the strict v2.1 decoder and inclusive boundary decoder."""
    return score>=threshold if model=='E3' else score>threshold


def validate_frozen(root):
    root=Path(root)
    corpus=read_json(root/'corpus_manifest.json');reference=read_json(root/'reference_lock.json')
    if file_sha256(root/'input_abstracts.jsonl')!=corpus['input_sha256']:
        raise ValueError('Frozen corpus SHA-256 mismatch')
    if file_sha256(root/'assisted_reference_frozen.jsonl')!=reference['reference_sha256']:
        raise ValueError('Frozen assisted reference SHA-256 mismatch')
    if reference['corpus_sha256']!=corpus['input_sha256']:
        raise ValueError('Reference belongs to a different corpus')
    return corpus,reference


def configuration(root,name,*,native=False):
    corpus,reference=validate_frozen(root)
    checkpoint=(read_json('contracts/ner_v2/gliner25_research_lock.json') if name=='E3' else
                read_json('reports/ner_v2/benchmark_run1/manifest.json')['models']['large']['model_identifier'])
    return dict(experiment='NER-V2-003',configuration=name,corpus_seed=CORPUS_SEED,seed=42,
                cpu_threads=4,device='cpu',dtype='float32',window_words=60 if name=='E1' else 110,
                overlap_words=20 if name=='E1' else 25,max_input_tokens=512,thresholds=THRESHOLDS,
                minimum_inference_threshold=.15,native_all_thresholds=native,
                labels=SHORT,normalization='N3',flat_ner=False,multi_label=True,overlap_policy='allow',
                checkpoint=checkpoint,input_sha256=corpus['input_sha256'],
                reference_sha256=reference['reference_sha256'],ontology_sha256=file_sha256('contracts/ner_v2/ontology.v2.json'),
                normalizer_sha256=file_sha256('src/cgraph/ner_v2/scientific_normalization.py'),
                runner_sha256=file_sha256(__file__),timing_repetitions=1,warmup='none; first forward included',
                native_check_document_ids=corpus['audit_document_ids'],native_check_thresholds=[.35,.50,.65])


def load_checkpoint(path,config_hash,doc):
    """Only reuse a complete, source-exact document with matching frozen config."""
    value=read_json(path)
    if value.get('status')!='complete' or value.get('config_hash')!=config_hash or value.get('text_sha256')!=doc['text_sha256']:
        raise ValueError('Resume checkpoint source/config mismatch')
    for record in value['mentions']:
        validate_span(doc['text'],record['start'],record['end'],record['surface_text'])
    return value


def atomic_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.partial')
    write_json(temporary,value);temporary.replace(path)


def run_inference(root,name,*,native=False,stop_after=None):
    """One model per process, atomic document checkpoints, fail-closed thresholds.

    If a native-equivalence check fails, this mode stops. Re-run in a NEW output
    root with native_all_thresholds=True; all eight thresholds then use native
    inference, with its extra calls timed explicitly. No incompatible cache reuse.
    """
    import torch
    import numpy as np
    from cgraph.ner import windows
    from .detection import postprocess,check_token_budget
    from .model_setup import load_detector
    from .scientific_normalization import ScientificNormalizer
    from .normalization import detect_abbreviations
    root=Path(root)
    config=configuration(root,name,native=native);config_hash=fingerprint(config)
    out=root/name;out.mkdir(parents=True,exist_ok=True)
    config_path=out/'config.json'
    if config_path.exists() and read_json(config_path)!=config:
        raise ValueError('Existing output has a different configuration; select a new output root')
    write_json(config_path,config)
    docs=read_documents(root/'input_abstracts.jsonl')
    directory=out/'documents';directory.mkdir(exist_ok=True)
    completed={}
    for doc in docs:
        path=directory/(doc['document_id']+'.json')
        if path.exists():completed[doc['document_id']]=load_checkpoint(path,config_hash,doc)
    if len(completed)==len(docs):
        aggregate_run(root,name,config,completed,docs)
        print(name,'resume: all 30 documents verified, zero model loads / forward calls',flush=True)
        return
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['OMP_NUM_THREADS']='4';torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    invocation_id=len(list(out.glob('invocation_*.json')))+1
    invocation=dict(status='running',config_hash=config_hash,environment=environment(),
                    reused_document_ids=sorted(completed),processed_document_ids=[],checkpoint_resume=True)
    invocation_path=out/f'invocation_{invocation_id:02}.json'
    atomic_json(invocation_path,invocation)
    with Resources() as resources:
        started=time.perf_counter()
        if name=='E3':
            from gliner2 import AutoExtractor
            model=AutoExtractor.from_pretrained('artifacts/ner_v2/models/gliner25',map_location='cpu',
                                                local_files_only=True,attention_backend='standard')
        else:model=load_detector('large')
        model.to('cpu').float().eval()
        invocation['model_load_seconds']=time.perf_counter()-started
        started=time.perf_counter();normalizer=ScientificNormalizer()
        invocation['normalizer_load_seconds']=time.perf_counter()-started
        counter={'calls':0,'lengths':[]}
        def hook(module,args,kwargs):
            ids=kwargs.get('input_ids',args[0] if args else None)
            if ids is None:raise ValueError('Encoder input cannot be instrumented')
            length=int(ids.shape[-1])
            if length>512:raise ValueError('Encoder token budget exceeded')
            counter['calls']+=1;counter['lengths'].append(length)
        encoders=[m for m in model.modules() if type(m).__name__=='DebertaV2Model']
        if len(encoders)!=1:raise ValueError('Unexpected encoder layout')
        encoders[0].register_forward_pre_hook(hook,with_kwargs=True)
        schema=model.create_schema().entities(SHORT) if name=='E3' else None
        descriptions={v:k for k,v in SHORT.items()}
        def predict(text,threshold):
            with torch.inference_mode():
                if name=='E3':
                    result=model.extract_entities(text,SHORT,threshold=threshold,include_confidence=True,
                                                  include_spans=True,overlap_policy='allow',max_len=None)
                    return [dict(start=p['start'],end=p['end'],text=p['text'],label=SHORT[label],score=p['confidence'])
                            for label,items in result['entities'].items() for p in items]
                return model.predict_entities(text,list(descriptions),threshold=threshold,flat_ner=False,multi_label=True)
        try:
            for doc in docs:
                if doc['document_id'] in completed:continue
                t0=time.perf_counter();encoder_before=counter['calls'];length_before=len(counter['lengths'])
                raw=[];budgets=[];checks=[];raw_at={t:[] for t in THRESHOLDS}
                infer=preflight=normalization=validation_seconds=0.
                text=doc['text'];queue=list(windows(text,config['window_words'],config['overlap_words']))
                index=0
                while index<len(queue):
                    a,b=queue[index];index+=1;window=text[a:b]
                    t=time.perf_counter()
                    try:
                        if name=='E3':
                            tokens=len(model.processor.transform_record(window,schema,max_len=None).input_ids)
                            if tokens>512:raise ValueError('Budget exceeded')
                        else:tokens=check_token_budget(model,window,descriptions,512)
                    except ValueError:
                        words=list(re.finditer(r'\S+',window))
                        if len(words)<8:raise ValueError('Cannot fit the frozen prompt')
                        sentence=[m.end() for m in re.finditer(r'[.!?]\s+',window)]
                        mid=min(sentence,key=lambda x:abs(x-len(window)/2)) if sentence else len(window)//2
                        pivot=min(range(1,len(words)),key=lambda x:abs(words[x].start()-mid))
                        overlap=min(config['overlap_words'],max(1,len(words)//6))
                        left=min(len(words)-1,pivot+overlap//2);right=max(1,pivot-overlap//2)
                        parts=[(a,a+words[left-1].end()),(a+words[right].start(),b)]
                        if any(y-x>=b-a for x,y in parts):raise ValueError('Subdivision made no progress')
                        queue[index:index]=parts;budgets.append(dict(start=a,end=b,status='split',parts=parts))
                        preflight+=time.perf_counter()-t;continue
                    preflight+=time.perf_counter()-t
                    budgets.append(dict(start=a,end=b,tokens=tokens,status='accepted'))
                    t=time.perf_counter();output=predict(window,.15);infer+=time.perf_counter()-t
                    raw.append(dict(document_id=doc['document_id'],window_start=a,window_end=b,predictions=output))
                    raw_at[.15].append((a,output))
                    if not checks and doc['document_id'] in config['native_check_document_ids']:
                        for threshold in config['native_check_thresholds']:
                            t=time.perf_counter();actual=predict(window,threshold);seconds=time.perf_counter()-t
                            validation_seconds+=seconds
                            direct={(p['start'],p['end'],p['label']):p['score'] for p in actual}
                            filtered={(p['start'],p['end'],p['label']):p['score'] for p in output if retained(p['score'],threshold,name)}
                            equal=direct.keys()==filtered.keys() and all(abs(direct[k]-filtered[k])<1e-6 for k in direct)
                            checks.append(dict(threshold=threshold,matches=equal,native_count=len(direct),filtered_count=len(filtered),seconds=seconds))
                            if not equal and not native:
                                atomic_json(out/'threshold_equivalence_FAILURE.json',dict(document_id=doc['document_id'],window=[a,b],checks=checks))
                                raise ValueError('Native threshold equivalence failed; use --native-all-thresholds in a new run root')
                    for threshold in THRESHOLDS[1:]:
                        if native:
                            t=time.perf_counter();predictions=predict(window,threshold);infer+=time.perf_counter()-t
                        else:predictions=[p for p in output if retained(p['score'],threshold,name)]
                        raw_at[threshold].append((a,predictions))
                t=time.perf_counter();analyzed=normalizer.analyze(text);definitions=detect_abbreviations(text)
                all_mentions={}
                for threshold in (THRESHOLDS if native else [.15]):
                    mentions=[]
                    for item in postprocess(text,raw_at[threshold],descriptions):
                        value=normalizer.normalize(text,item['start'],item['end'],analyzed=analyzed,abbreviations=definitions)
                        identity=[doc['document_id'],doc['revision_id'],item['start'],item['end'],item['ner_label_hint'],name]
                        mentions.append({**item,**value,'document_id':doc['document_id'],'revision_id':doc['revision_id'],
                                         'mention_id':'m:'+fingerprint(identity),'configuration':name,
                                         'source_reference':doc['source_reference']})
                    all_mentions[f'{threshold:.2f}']=mentions
                normalization=time.perf_counter()-t
                if not native:
                    for threshold in THRESHOLDS[1:]:
                        all_mentions[f'{threshold:.2f}']=[m for m in all_mentions['0.15'] if retained(m['score'],threshold,name)]
                record=dict(status='complete',config_hash=config_hash,text_sha256=doc['text_sha256'],document_id=doc['document_id'],
                            mentions=all_mentions['0.15'],native_mentions_by_threshold=all_mentions if native else None,
                            raw_windows=raw,abbreviations=definitions,budgets=budgets,threshold_checks=checks,
                            performance=dict(inference_seconds=infer,normalization_seconds=normalization,preflight_seconds=preflight,
                            native_check_seconds=validation_seconds,processing_seconds=time.perf_counter()-t0-validation_seconds,
                            encoder_calls=counter['calls']-encoder_before-len(checks),native_check_encoder_calls=len(checks),
                            sequence_lengths=counter['lengths'][length_before:],windows=sum(b['status']=='accepted' for b in budgets)))
                atomic_json(directory/(doc['document_id']+'.json'),record)
                completed[doc['document_id']]=record
                invocation['processed_document_ids'].append(doc['document_id'])
                atomic_json(invocation_path,invocation)
                print(name,f'{len(completed)}/30',doc['document_id'],'mentions',len(record['mentions']),flush=True)
                if stop_after and len(invocation['processed_document_ids'])>=stop_after:break
            invocation['status']='complete' if len(completed)==len(docs) else 'stopped_after_checkpoint'
        except Exception as exc:
            invocation['status']='failed';invocation['error_type']=type(exc).__name__
            raise
        finally:
            invocation.update(peak_rss_bytes=resources.peak,baseline_rss_bytes=resources.baseline,
                              measured_encoder_calls=counter['calls'])
            atomic_json(invocation_path,invocation)
    if len(completed)==len(docs):aggregate_run(root,name,config,completed,docs)


def aggregate_run(root,name,config,completed,docs):
    out=Path(root)/name
    ordered=[completed[d['document_id']] for d in docs]
    mentions=[m for r in ordered for m in r['mentions']]
    write_jsonl(out/'raw_windows.jsonl',[w for r in ordered for w in r['raw_windows']])
    for threshold in THRESHOLDS:
        values=([m for r in ordered for m in r['native_mentions_by_threshold'][f'{threshold:.2f}']]
                if config['native_all_thresholds'] else [m for m in mentions if retained(m['score'],threshold,name)])
        write_jsonl(out/f'mentions_{int(threshold*100):03}.jsonl',values)
    write_jsonl(out/'document_abbreviations.jsonl',[dict(document_id=r['document_id'],**a) for r in ordered for a in r['abbreviations']])
    invocations=[read_json(p) for p in sorted(out.glob('invocation_*.json'))]
    summed={key:sum(r['performance'][key] for r in ordered) for key in ['inference_seconds','normalization_seconds','preflight_seconds','processing_seconds','native_check_seconds','encoder_calls','native_check_encoder_calls','windows']}
    seconds=summed['processing_seconds'];words=sum(len(d['text'].split()) for d in docs)
    write_json(out/'performance.json',dict(status='complete',documents=len(docs),words=words,**summed,
               model_load_seconds=sum(x.get('model_load_seconds',0) for x in invocations),
               normalizer_load_seconds=sum(x.get('normalizer_load_seconds',0) for x in invocations),
               peak_rss_bytes=max(x.get('peak_rss_bytes',0) for x in invocations),
               documents_per_second=len(docs)/seconds,words_per_second=words/seconds,
               timing_repetitions=1,warmup='none',invocations=len(invocations),
               note='One minimum-threshold inference; filtered thresholds share this measurement. Checkpoint I/O and final exports excluded.',
               total_phase_seconds=seconds+sum(x.get('model_load_seconds',0)+x.get('normalizer_load_seconds',0) for x in invocations),
               maximum_encoder_tokens=max(length for r in ordered for length in r['performance']['sequence_lengths'])))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['corpus','infer'])
    parser.add_argument('--output-root',default=str(ROOT))
    parser.add_argument('--config',choices=['E1','E2','E3'])
    parser.add_argument('--native-all-thresholds',action='store_true')
    parser.add_argument('--stop-after',type=int)
    args=parser.parse_args()
    if args.stage=='corpus':prepare_corpus(args.output_root)
    else:
        if not args.config:parser.error('infer requires --config')
        run_inference(args.output_root,args.config,native=args.native_all_thresholds,stop_after=args.stop_after)


if __name__=='__main__':main()
