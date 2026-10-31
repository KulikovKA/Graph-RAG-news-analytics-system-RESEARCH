"""Reproducible NORM-V2-001 / NER-V2-002 CPU experiment support."""
import argparse
import importlib.metadata
import json
import os
import platform
import random
import re
import threading
import time
from pathlib import Path

import psutil

from cgraph.common import read_json, write_json, file_sha256
from .io import read_documents, write_jsonl, validate_span
from .normalization import LinguisticNormalizer, safe_unicode, detect_abbreviations
from .scientific_normalization import ScientificNormalizer

ROOT = Path('reports/ner_v2/research_run1')
HIST = Path('reports/ner_v2/benchmark_run1')
SHORT = dict(OBJECT='physical, biological or digital object', ACTOR='person or organization',
             CONCEPT='scientific concept or theory', METHOD='algorithm or research method',
             PROCESS='event or scientific process', DATA_RESOURCE='dataset or research corpus',
             ATTRIBUTE='property, parameter or metric', RESULT='measured or observed result',
             CONDITION='experimental or contextual condition', TASK='research task or application',
             MECHANISM='mechanism explaining a phenomenon')
BROAD = ['scientific object or entity','scientific concept','method or technique',
         'process or event','property or result']


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines()]


class Resources:
    """Sample process working set every 10 ms; absolute RSS includes loaded libraries."""
    def __enter__(self):
        self.process = psutil.Process()
        self.baseline = self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        def sample():
            while not self.stop.wait(.01):
                self.peak = max(self.peak, self.process.memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()


def environment():
    versions = {}
    for name in ['torch','gliner','gliner2','transformers','spacy','en-core-web-sm','numpy',
                 'psutil','huggingface-hub','tokenizers','peft','matplotlib']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=platform.python_version(), platform=platform.platform(), processor=platform.processor(),
                logical_cpu=psutil.cpu_count(), physical_cpu=psutil.cpu_count(logical=False),
                ram_bytes=psutil.virtual_memory().total, versions=versions,
                cpu_threads=4, seed=42, device='cpu', rss_sampling_interval_seconds=.01)


def normalize_form(variant, engine, text, a, b, analyzed=None, definitions=None):
    if variant == 'N0':
        return dict(surface_text=text[a:b], normalized_form=safe_unicode(text[a:b]),
                    lemma=safe_unicode(text[a:b]), tokens=[], abbreviation_candidates=[], normalizer_version='nfc-whitespace-v1')
    return engine.normalize(text,a,b,analyzed=analyzed,
                            abbreviations=definitions if variant in ('N1','N3') else [])


def run_norm():
    out = ROOT / 'normalization'
    out.mkdir(parents=True, exist_ok=False)
    specification = read_json('contracts/ner_v2/research_cases.json')
    docs = read_documents(HIST / 'input_abstracts.jsonl')
    by_id = {d['document_id']:d for d in docs}
    metrics, errors = [], []
    # Load N1 and measure it before constructing or executing N2.
    for variant in ['N0','N1','N2','N3']:
        start = time.perf_counter()
        with Resources() as resources:
            load_start=time.perf_counter()
            engine = None if variant=='N0' else (LinguisticNormalizer() if variant=='N1' else ScientificNormalizer())
            load_seconds=time.perf_counter()-load_start
            start=time.perf_counter()
            results=[]
            for case in specification['cases']:
                value=normalize_form(variant,engine,case['text'],case['start'],case['end'])
                actual=value['normalized_form']
                baseline=safe_unicode(case['surface'])
                correct=actual==case['expected']
                error_type=None if correct else ('incorrect_transformation' if actual!=baseline else 'missed_transformation')
                row={**case, 'variant':variant, 'actual':actual, 'correct':correct,
                     'changed':actual!=baseline,'error_type':error_type, 'tokens':value['tokens']}
                results.append(row)
                if not correct:
                    errors.append({**row,'reason': 'N0 preserves inflection' if variant=='N0' else
                                   ('contextual POS/lemma or modifier lemmatization' if variant=='N1' else
                                    'conservative rule rejected change or contextual POS/lemma mismatch')})
            case_seconds=time.perf_counter()-start
            pair_results=[]
            for kind in ['distinct_pairs','equivalent_pairs']:
                for a,b in specification[kind]:
                    forms=[]
                    for surface in [a,b]:
                        text='We investigated '+surface+' in this study.'
                        forms.append(normalize_form(variant,engine,text,16,16+len(surface))['normalized_form'])
                    pair_results.append(dict(kind=kind, left=a,right=b,left_form=forms[0],right_form=forms[1], equal=forms[0]==forms[1]))
            abbreviation_results=[]
            for case in specification['abbreviations']:
                definitions=detect_abbreviations(case['text']) if variant in ('N1','N3') else []
                expected={tuple(p) for p in case['pairs']}
                actual={(d['short_form'],d['long_form']) for d in definitions}
                for definition in definitions:
                    assert case['text'][definition['long_start']:definition['long_end']]==definition['long_form']
                    assert case['text'][definition['short_start']:definition['short_end']]==definition['short_form']
                abbreviation_results.append({**case,'definitions':definitions,'matched':len(actual&expected),
                                             'expected_count':len(expected),'predicted_count':len(actual),
                                             'false_matches':len(actual-expected)})
            definition_rows=[]
            doc_analyses={d['document_id']: (engine.analyze(d['text']) if engine else None) for d in docs}
            doc_definitions={d['document_id']:detect_abbreviations(d['text']) if variant in ('N1','N3') else [] for d in docs}
            for d in docs:
                for definition in doc_definitions[d['document_id']]:
                    definition_rows.append(dict(document_id=d['document_id'],revision_id=d['revision_id'],
                                                **definition,persisted_by_historical_pipeline=False,
                                                persisted_by_variant=variant=='N3'))
            applied=[]
            for size in ['large','medium','small']:
                rows=[]
                for record in jsonl(HIST/f'{size}.jsonl'):
                    d=by_id[record['document_id']]
                    value=normalize_form(variant,engine,d['text'],record['start'],record['end'],
                                         doc_analyses[d['document_id']],doc_definitions[d['document_id']])
                    row={**record, **value, 'normalization_variant':variant,'historical_model':size}
                    validate_span(d['text'],row['start'],row['end'],row['surface_text'])
                    rows.append(row)
                write_jsonl(out/f'{variant}_{size}.jsonl',rows)
                applied.extend(rows)
            processing_seconds=time.perf_counter()-start
        expected_total=sum(x['expected_count'] for x in abbreviation_results)
        predicted_total=sum(x['predicted_count'] for x in abbreviation_results)
        matched=sum(x['matched'] for x in abbreviation_results)
        metric=dict(variant=variant,cases=len(results),correct=sum(x['correct'] for x in results),
                    incorrect_transformations=sum(x['error_type']=='incorrect_transformation' for x in results),
                    missed_transformations=sum(x['error_type']=='missed_transformation' for x in results),
                    correct_rate=sum(x['correct'] for x in results)/len(results),
                    incorrect_transformation_rate=sum(x['error_type']=='incorrect_transformation' for x in results)/len(results),
                    by_kind={kind:dict(total=sum(x['kind']==kind for x in results),correct=sum(x['kind']==kind and x['correct'] for x in results)) for kind in ['technical_control','publication_assisted']},
                    false_collisions=sum(x['equal'] for x in pair_results if x['kind']=='distinct_pairs'), distinct_pairs=5,
                    unified_variants=sum(x['equal'] for x in pair_results if x['kind']=='equivalent_pairs'), equivalent_pairs=5,
                    abbreviation_expected=expected_total,abbreviation_detected=predicted_total,abbreviation_matched=matched,
                    abbreviation_coverage=matched/expected_total,abbreviation_matching_correctness=matched/predicted_total if predicted_total else None,
                    source_spans_checked=len(applied),source_spans_preserved=len(applied),
                    historical_mentions_changed=sum(x['normalized_form']!=safe_unicode(x['surface_text']) for x in applied),
                    document_definitions=len(definition_rows),document_definitions_persisted=len(definition_rows) if variant=='N3' else 0,
                    load_seconds=load_seconds,case_seconds=case_seconds,processing_seconds=processing_seconds,
                    peak_rss_bytes=resources.peak,baseline_rss_bytes=resources.baseline)
        write_jsonl(out/f'{variant}_cases.jsonl',results)
        write_json(out/f'{variant}_pairs.json',pair_results)
        write_json(out/f'{variant}_abbreviation_checks.json',abbreviation_results)
        write_jsonl(out/f'{variant}_document_abbreviations.jsonl',definition_rows)
        metrics.append(metric)
        write_json(out/'metrics.json',metrics)
        print(variant,metric['correct'],len(results),'errors',metric['incorrect_transformations'],flush=True)
    write_jsonl(out/'errors.jsonl',errors)
    write_json(ROOT/'environment.json',environment())


def ner_config(name):
    historical = read_json(HIST/'manifest.json')
    return dict(experiment='NER-V2-002',name=name,seed=42,cpu_threads=4,device='cpu',
                window_words=60 if name=='E1' else 110,overlap_words=20 if name=='E1' else 25,
                threshold=.15,exploratory_thresholds=[.15,.25,.35],max_input_tokens=512,
                flat_ner=False,multi_label=True,overlap_policy='allow',normalization='N3',
                checkpoint=(read_json('contracts/ner_v2/gliner25_research_lock.json') if name=='E3'
                            else historical['models']['large']['model_identifier']),
                labels=SHORT if name!='E4' else {f'CANDIDATE_{i}':s for i,s in enumerate(BROAD)})


def run_ner(name):
    from cgraph.ner import windows
    from .detection import check_token_budget, postprocess
    from .model_setup import load_detector
    import torch
    import numpy as np
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    config=ner_config(name)
    out=ROOT/name
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'config.json',config)
    docs=read_documents(HIST/'input_abstracts.jsonl')
    descriptions={v:k for k,v in config['labels'].items()}
    started=time.perf_counter()
    with Resources() as resources:
        load_started=time.perf_counter()
        if name=='E3':
            from gliner2 import AutoExtractor
            model=AutoExtractor.from_pretrained('artifacts/ner_v2/models/gliner25',map_location='cpu',
                                                local_files_only=True,attention_backend='standard')
        else:
            model=load_detector('large')
        model.to('cpu').float().eval()
        model_load_seconds=time.perf_counter()-load_started
        engine_started=time.perf_counter()
        normalizer=ScientificNormalizer()
        normalizer_load_seconds=time.perf_counter()-engine_started
        counter={'encoder_calls':0,'actual_sequence_lengths':[]}
        def on_encoder(module,args,kwargs):
            counter['encoder_calls']+=1
            ids=kwargs.get('input_ids',args[0] if args else None)
            if ids is not None:
                length=int(ids.shape[-1]);counter['actual_sequence_lengths'].append(length)
                if length>512:raise ValueError('Actual encoder budget exceeded')
        encoders=[m for m in model.modules() if type(m).__name__=='DebertaV2Model']
        if len(encoders)!=1:raise ValueError('Unexpected encoder layout')
        encoders[0].register_forward_pre_hook(on_encoder,with_kwargs=True)
        raw,mentions,budgets,definitions=[],[],[],[]
        inference_seconds=normalization_seconds=preflight_seconds=0.
        native_threshold_check=None
        schema=model.create_schema().entities(SHORT) if name=='E3' else None
        processing_start=time.perf_counter()
        for doc in docs:
            text=doc['text'];raw_windows=[]
            queue=list(windows(text,config['window_words'],config['overlap_words']))
            # Budget failure triggers sentence boundaries, then overlapping word
            # subdivision for a single overlong sentence. Every source word stays covered.
            i=0
            while i<len(queue):
                a,b=queue[i];i+=1
                window=text[a:b]
                preflight_start=time.perf_counter()
                try:
                    if name=='E3':
                        record=model.processor.transform_record(window,schema,max_len=None)
                        tokens=len(record.input_ids)
                        if tokens>512:raise ValueError('Token budget exceeded')
                    else:
                        tokens=check_token_budget(model,window,descriptions,512)
                except ValueError as exc:
                    count=len(window.split())
                    if count<8:raise ValueError('Cannot fit the frozen labels') from exc
                    # Prefer a sentence boundary near the midpoint; preserve overlap.
                    boundaries=[m.end() for m in re.finditer(r'[.!?]\s+',window)]
                    middle=min(boundaries,key=lambda x:abs(x-len(window)/2)) if boundaries else len(window)//2
                    words=list(re.finditer(r'\S+',window))
                    pivot=min(range(1,len(words)),key=lambda x:abs(words[x].start()-middle))
                    overlap=min(config['overlap_words'],max(1,len(words)//6))
                    left_end=min(len(words)-1,pivot+overlap//2)
                    right_start=max(1,pivot-overlap//2)
                    parts=[(a,a+words[left_end-1].end()),(a+words[right_start].start(),b)]
                    if any(y-x>=b-a for x,y in parts):raise ValueError('Budget subdivision made no progress')
                    queue[i:i]=parts
                    budgets.append(dict(document_id=doc['document_id'],start=a,end=b,status='split_over_budget',
                                        reason=str(exc),new_spans=parts))
                    preflight_seconds+=time.perf_counter()-preflight_start
                    continue
                preflight_seconds+=time.perf_counter()-preflight_start
                budgets.append(dict(document_id=doc['document_id'],start=a,end=b,status='accepted',tokens=tokens))
                t=time.perf_counter()
                with torch.inference_mode():
                    if name=='E3':
                        result=model.extract_entities(window,SHORT,threshold=.15,include_confidence=True,
                                                      include_spans=True,overlap_policy='allow',max_len=None)
                        output=[dict(start=p['start'],end=p['end'],text=p['text'],label=SHORT[label],score=p['confidence'])
                                for label,items in result['entities'].items() for p in items]
                    else:
                        output=model.predict_entities(window,list(descriptions),threshold=.15,flat_ner=False,multi_label=True)
                inference_seconds+=time.perf_counter()-t
                # Validate score-filter equivalence on the first accepted source window.
                # These extra calls are measured separately and not included in throughput.
                if native_threshold_check is None:
                    check_start=time.perf_counter();checks=[]
                    before=counter['encoder_calls']
                    for threshold in [.25,.35]:
                        with torch.inference_mode():
                            if name=='E3':
                                native=model.extract_entities(window,SHORT,threshold=threshold,include_confidence=True,
                                                              include_spans=True,overlap_policy='allow',max_len=None)
                                direct={(p['start'],p['end'],SHORT[label]) for label,items in native['entities'].items() for p in items}
                            else:
                                native=model.predict_entities(window,list(descriptions),threshold=threshold,flat_ner=False,multi_label=True)
                                direct={(p['start'],p['end'],p['label']) for p in native}
                        filtered={(p['start'],p['end'],p['label']) for p in output if p['score']>=threshold}
                        if filtered!=direct:raise ValueError('Threshold filtering is not equivalent')
                        checks.append(dict(threshold=threshold,spans=len(direct),matches=True))
                    native_threshold_check=dict(document_id=doc['document_id'],start=a,end=b,checks=checks,
                                                seconds=time.perf_counter()-check_start,encoder_calls=counter['encoder_calls']-before)
                raw.append(dict(document_id=doc['document_id'],window_start=a,window_end=b,predictions=output))
                raw_windows.append((a,output))
            t=time.perf_counter()
            analyzed=normalizer.analyze(text);abbreviations=detect_abbreviations(text)
            definitions.extend(dict(document_id=doc['document_id'],revision_id=doc['revision_id'],**d) for d in abbreviations)
            for item in postprocess(text,raw_windows,descriptions):
                value=normalizer.normalize(text,item['start'],item['end'],analyzed=analyzed,abbreviations=abbreviations)
                mentions.append(dict(document_id=doc['document_id'],revision_id=doc['revision_id'],
                                     model_identifier=config['checkpoint'] if name!='E3' else
                                     config['checkpoint']['repo_id']+'@'+config['checkpoint']['revision'],
                                     source_reference=doc['source_reference'],**{**item,**value}))
            normalization_seconds+=time.perf_counter()-t
            print(name,doc['document_id'],'mentions',sum(m['document_id']==doc['document_id'] for m in mentions),flush=True)
        processing_wall_seconds=time.perf_counter()-processing_start-native_threshold_check['seconds']
        write_jsonl(out/'raw_windows.jsonl',raw)
        write_jsonl(out/'mentions_015.jsonl',mentions)
        write_jsonl(out/'document_abbreviations.jsonl',definitions)
        write_json(out/'token_budgets.json',budgets)
        for threshold in [.25,.35]:
            write_jsonl(out/f'mentions_{int(threshold*100):03}.jsonl',[m for m in mentions if m['score']>=threshold])
    words=sum(len(d['text'].split()) for d in docs)
    model_path=Path('artifacts/ner_v2/models/gliner25' if name=='E3' else 'artifacts/ner_v2/models/runtime_large')
    model_files={str(p.relative_to(model_path)):{'bytes':p.stat().st_size,'sha256':file_sha256(p)}
                 for p in model_path.rglob('*') if p.is_file() and '.cache' not in p.parts}
    write_json(out/'performance.json',dict(status='complete',documents=len(docs),words=words,
               model_load_seconds=model_load_seconds,normalizer_load_seconds=normalizer_load_seconds,
               inference_seconds=inference_seconds,normalization_seconds=normalization_seconds,
               preflight_seconds=preflight_seconds,processing_wall_seconds=processing_wall_seconds,
               total_seconds=model_load_seconds+normalizer_load_seconds+processing_wall_seconds,
               documents_per_second=len(docs)/processing_wall_seconds,words_per_second=words/processing_wall_seconds,
               peak_rss_bytes=resources.peak,baseline_rss_bytes=resources.baseline,
               model_disk_bytes=sum(p['bytes'] for p in model_files.values()),model_files=model_files,
               encoder_calls=counter['encoder_calls']-native_threshold_check['encoder_calls'],
               actual_sequence_lengths=counter['actual_sequence_lengths'],
               threshold_equivalence_check=native_threshold_check,
               library_source_sha256={str(p):file_sha256(p) for p in (
                   Path('src/cgraph/ner_v2/research.py'),Path('src/cgraph/ner_v2/normalization.py'),
                   Path('src/cgraph/ner_v2/scientific_normalization.py'))},environment=environment()))


def main():
    global ROOT
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['norm','ner'])
    parser.add_argument('--config',choices=['E1','E2','E3','E4'])
    parser.add_argument('--output-root',default=str(ROOT))
    args=parser.parse_args()
    ROOT=Path(args.output_root)
    lock=read_json('contracts/ner_v2/research_input_lock.json')
    if file_sha256(lock['input'])!=lock['input_sha256']:
        raise ValueError('Frozen corpus changed')
    if file_sha256('contracts/ner_v2/research_cases.json')!=lock['cases_sha256']:
        raise ValueError('Frozen normalization specifications changed')
    if args.stage=='ner' and not args.config:
        parser.error('ner requires --config')
    os.environ['OMP_NUM_THREADS']='4'
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(42)
    random.seed(42)
    if args.stage=='norm':run_norm()
    else:run_ner(args.config)


if __name__=='__main__': main()
