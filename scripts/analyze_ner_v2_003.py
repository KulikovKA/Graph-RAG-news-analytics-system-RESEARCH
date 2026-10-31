"""NER-V2-003 descriptive analysis, stratified audit, and tokenizer-only costing."""
import csv
import itertools
import json
import math
import random
import re
from collections import Counter,defaultdict
from pathlib import Path

from cgraph.common import fingerprint,read_json,write_json,file_sha256
from cgraph.ner_v2.io import read_documents,write_jsonl
from cgraph.ner_v2.quality_audit import ROOT,THRESHOLDS,retained,validate_frozen
from cgraph.ner_v2.research import SHORT,jsonl

MODELS=['E1','E2','E3']
NOISE={'we','our','us','they','it','this','these','those','paper','article','study','work','results',
       'abstract','background','introduction','conclusion','conclusions','methods','objectives',
       'this paper','this study','this work','the present study','the author'}


def csv_out(path,rows):
    rows=list(rows)
    if rows:
        with Path(path).open('w',encoding='utf8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(rows)


def key(m):return m['document_id'],m['start'],m['end']


def score_bin(score):
    for a,b in zip(THRESHOLDS,THRESHOLDS[1:]+[1.00001]):
        if a<=score<b:return f'{a:.2f}-{min(b,1):.2f}'
    raise ValueError('Score outside audit bins')


def source_candidates(outputs,documents):
    by_id={d['document_id']:d for d in documents}
    grouped=defaultdict(lambda:defaultdict(list))
    for name,rows in outputs.items():
        for m in rows:grouped[key(m)][name].append(m)
    result={}
    for identity,by_model in sorted(grouped.items()):
        doc_id,a,b=identity;doc=by_id[doc_id]
        exemplars=[m for rows in by_model.values() for m in rows]
        representative=max(exemplars,key=lambda x:x['score'])
        predictions={name:[{'label':m['ner_label_hint'],'score':m['score']} for m in by_model.get(name,[])] for name in MODELS}
        result[identity]=dict(candidate_id='c:'+fingerprint(list(identity))[:20],document_id=doc_id,
              discipline=doc['discipline'],source_url=doc['source_url'],text_sha256=doc['text_sha256'],
              start=a,end=b,surface=doc['text'][a:b],normalized_form=representative['normalized_form'],
              source_text=doc['text'],context=doc['text'][max(0,a-100):min(len(doc['text']),b+100)],
              predictions=predictions,models_at_015=len(by_model),
              retained_by_threshold={name:{f'{t:.2f}':any(retained(p['score'],t,name) for p in predictions[name]) for t in THRESHOLDS} for name in MODELS},
              token_evidence=[{k:t[k] for k in ['text','pos','tag']} for t in representative['tokens']])
    return result


def draft_annotation(candidate,references):
    identity=key(candidate)
    exact=[r for r in references if key(r)==identity]
    surface=candidate['surface'];lower=surface.strip(' .,:;()[]').lower()
    a,b=candidate['start'],candidate['end'];text=candidate['source_text']
    if lower in NOISE:
        category='IRRELEVANT';significance='LOW_DRAFT';reason='Closed discourse/header watchlist; context must be reviewed'
    elif ((a and text[a-1].isalnum() and surface[0].isalnum()) or
          (b<len(text) and text[b].isalnum() and surface[-1].isalnum())):
        category='WRONG_BOUNDARY';significance='POSSIBLY_RELEVANT';reason='Inside-word boundary in immutable source'
    elif exact:
        category='VALID_SPAN';significance='HIGH_DRAFT';reason='Exact span in source-authored assisted reference'
    elif re.search(r'\b(of|with|and|for|the|a|an)$',lower):
        category='PARTIAL_SPAN';significance='POSSIBLY_RELEVANT';reason='Dangling function word suggests incomplete phrase'
    else:
        category='AMBIGUOUS';significance='UNDETERMINED';reason='Not in incomplete reference; cannot infer irrelevance from absence'
    suggested_types=sorted({r['suggested_type'] for r in exact})
    type_evaluations={}
    for name,predictions in candidate['predictions'].items():
        type_evaluations[name]=[dict(label=p['label'],score=p['score'],
                  proposed_category=('TYPE_HINT_CORRECT' if p['label'] in suggested_types else 'TYPE_HINT_INCORRECT')
                                     if suggested_types else 'TYPE_HINT_AMBIGUOUS') for p in predictions]
    return dict(status='UNREVIEWED',origin='assistant_rules_plus_frozen_reference',
                boundary_category=category,scientific_significance=significance,reason=reason,
                suggested_types=suggested_types,type_evaluations=type_evaluations,
                independent_human_review=False,
                human_review={'span_category':None,'type_hint_category':None,'scientific_significance':None,
                              'reviewer':None,'notes':None})


def stratified_sample(candidates,audit_ids):
    rng=random.Random(4303)
    chosen=[];used=set()
    for discipline in sorted({c['discipline'] for c in candidates.values()}):
        pool=[]
        for identity,candidate in candidates.items():
            if candidate['document_id'] in audit_ids or candidate['discipline']!=discipline:continue
            for name,predictions in candidate['predictions'].items():
                for p in predictions:
                    pool.append((identity,name,p['label'],score_bin(p['score']),
                                 'shared' if candidate['models_at_015']>1 else 'model_specific',p['score']))
        rng.shuffle(pool)
        counts=[Counter() for _ in range(5)]
        for _ in range(20):
            eligible=[item for item in pool if item[0] not in used]
            if not eligible:raise ValueError('Not enough unique candidates for discipline sample')
            def priority(item):
                identity,model,label,confidence,agreement,score=item
                return (counts[0][model],counts[1][label],counts[2][confidence],counts[3][agreement],counts[4][identity[0]])
            item=min(eligible,key=priority)
            identity,model,label,confidence,agreement,score=item;used.add(identity)
            for counter,value in zip(counts,[model,label,confidence,agreement,identity[0]]):counter[value]+=1
            chosen.append({**candidates[identity],'audit_partition':'stratified_120',
                           'focal_model':model,'focal_type':label,'focal_score':score,
                           'confidence_stratum':confidence,'agreement_stratum':agreement})
    assert len(chosen)==len(used)==120
    return chosen


def prepare_audit(candidates,documents,references,manifest):
    ids=set(manifest['audit_document_ids'])
    full=[{**c,'audit_partition':'six_document_census'} for c in candidates.values() if c['document_id'] in ids]
    sample=stratified_sample(candidates,ids)
    # Apply only explicitly stored assistant review amendments; human status stays UNREVIEWED.
    overrides_path=ROOT/'assistant_audit_amendments.json'
    overrides=read_json(overrides_path) if overrides_path.exists() else {}
    rows=[]
    for row in full+sample:
        annotation=draft_annotation(row,references)
        amendment=overrides.get(row['candidate_id'])
        if amendment:
            annotation.update(amendment)
            annotation['status']='UNREVIEWED';annotation['independent_human_review']=False
            annotation['origin']='assistant_context_review_amendment'
            types=annotation['suggested_types']
            annotation['type_evaluations']={name:[dict(label=p['label'],score=p['score'],proposed_category=
                  ('TYPE_HINT_CORRECT' if len(types)==1 and p['label'] in types else
                   'TYPE_HINT_AMBIGUOUS' if not types or p['label'] in types else 'TYPE_HINT_INCORRECT'))
                   for p in predictions] for name,predictions in row['predictions'].items()}
        rows.append({**row,'draft_assessment':annotation})
    write_jsonl(ROOT/'audit_candidates_UNREVIEWED.jsonl',rows)
    write_jsonl(ROOT/'audit_stratified_120.jsonl',[r for r in rows if r['audit_partition']=='stratified_120'])
    coverage={field:dict(Counter(str(r[field]) for r in sample)) for field in ['discipline','focal_model','focal_type','confidence_stratum','agreement_stratum']}
    write_json(ROOT/'audit_sampling_manifest.json',dict(seed=4303,total=len(rows),census_candidates=len(full),
               stratified_candidates=len(sample),coverage=coverage,
               ordered_sample_fingerprint=fingerprint([[r['candidate_id'],r['document_id'],r['start'],r['end']] for r in sample]),
               selection='Greedy balanced round-robin model,type,confidence,agreement,document; seeded tie order; unique source spans',
               statistical_limit='Purposive multiway stratification; no population error-rate extrapolation or confidence intervals'))
    docs_by_id={d['document_id']:d for d in documents}
    write_jsonl(ROOT/'audit_full_documents.jsonl',[dict(document=docs_by_id[doc_id],
                  candidates=[r for r in rows if r['document_id']==doc_id]) for doc_id in manifest['audit_document_ids']])
    return rows


def counts(rows):
    spans={key(m) for m in rows};nested=crossing=0
    for doc in {m[0] for m in spans}:
        subset=sorted((a,b) for d,a,b in spans if d==doc)
        for (a,b),(c,d) in itertools.combinations(subset,2):
            if max(a,c)<min(b,d):
                if (a<=c and d<=b) or (c<=a and b<=d):nested+=1
                else:crossing+=1
    return dict(mentions=len(rows),unique_spans=len(spans),nested_pairs=nested,crossing_pairs=crossing,
                alternative_hints=len(rows)-len(spans))


def reference_analysis(outputs_by_threshold,references):
    rows=[];details=[];by_type=[]
    for name in MODELS:
        baseline={key(m) for m in outputs_by_threshold[(name,.35)]}
        base_exact=sum(key(r) in baseline for r in references)
        for threshold in THRESHOLDS:
            predictions=outputs_by_threshold[(name,threshold)];spans={key(m) for m in predictions}
            for ref in references:
                identity=key(ref);a,b=ref['start'],ref['end']
                exact=identity in spans
                overlap=any(doc==ref['document_id'] and max(a,c)<min(b,d) for doc,c,d in spans)
                details.append(dict(configuration=name,threshold=threshold,reference_id=ref['reference_id'],
                    document_id=ref['document_id'],discipline=ref['discipline'],surface=ref['surface'],
                    start=a,end=b,suggested_type=ref['suggested_type'],exact_match=exact,overlap_match=overlap,
                    was_exact_at_035=identity in baseline,status='ASSISTED_REFERENCE_DRAFT'))
            matching=[r for r in details if r['configuration']==name and r['threshold']==threshold]
            exact=sum(r['exact_match'] for r in matching);overlap=sum(r['overlap_match'] for r in matching)
            rows.append(dict(configuration=name,threshold=threshold,reference_spans=len(references),exact_matches=exact,
                 overlap_matches=overlap,missing_exact=len(references)-exact,missing_overlap=len(references)-overlap,
                 exact_matching_fraction=exact/len(references),overlap_matching_fraction=overlap/len(references),
                 exact_retention_vs_035=exact/base_exact if base_exact else None))
            for label in SHORT:
                subset=[r for r in matching if r['suggested_type']==label]
                base=sum(key(r) in baseline for r in references if r['suggested_type']==label)
                by_type.append(dict(configuration=name,threshold=threshold,reference_type=label,total=len(subset),
                    exact_matches=sum(r['exact_match'] for r in subset),overlap_matches=sum(r['overlap_match'] for r in subset),
                    baseline_035_exact=base))
    csv_out(ROOT/'reference_metrics.csv',rows);csv_out(ROOT/'reference_by_type.csv',by_type)
    write_jsonl(ROOT/'reference_matching_details.jsonl',details)
    return rows,details,by_type


def build_metrics(outputs,documents,audit):
    rows=[];discipline_rows=[];labels=[];audit_metrics=[];type_metrics=[];document_rows=[]
    by_id={d['document_id']:d for d in documents};words=sum(len(d['text'].split()) for d in documents)
    for name in MODELS:
        low={key(m) for m in outputs[(name,.15)]};baseline={key(m) for m in outputs[(name,.35)]}
        for threshold in THRESHOLDS:
            predictions=outputs[(name,threshold)];current={key(m) for m in predictions}
            others=set().union(*[{key(m) for m in outputs[(other,threshold)]} for other in MODELS if other!=name])
            row=dict(configuration=name,threshold=threshold,**counts(predictions),
                    spans_per_document=len(current)/len(documents),spans_per_100_words=100*len(current)/words,
                    documents_without_spans=len(documents)-len({d for d,a,b in current}),
                    spans_lost_vs_015=len(low-current),retained_fraction_vs_035=len(current&baseline)/len(baseline),
                    count_ratio_vs_035=len(current)/len(baseline),model_specific_spans=len(current-others),
                    watchlist_suspected_spans=len({key(m) for m in predictions if m['surface_text'].strip(' .,:;()[]').lower() in NOISE}))
            rows.append(row)
            for doc in documents:
                subset=[m for m in predictions if m['document_id']==doc['document_id']]
                document_rows.append(dict(configuration=name,threshold=threshold,document_id=doc['document_id'],
                    discipline=doc['discipline'],words=len(doc['text'].split()),mentions=len(subset),
                    unique_spans=len({key(m) for m in subset}),
                    spans_per_100_words=100*len({key(m) for m in subset})/len(doc['text'].split())))
            for discipline in sorted({d['discipline'] for d in documents}):
                docs=[d for d in documents if d['discipline']==discipline]
                subset=[m for m in predictions if by_id[m['document_id']]['discipline']==discipline]
                discipline_rows.append(dict(configuration=name,threshold=threshold,discipline=discipline,
                       **counts(subset),spans_per_document=len({key(m) for m in subset})/len(docs),
                       documents_without_spans=len(docs)-len({m['document_id'] for m in subset})))
            count=Counter(m['ner_label_hint'] for m in predictions)
            labels.extend(dict(configuration=name,threshold=threshold,label=label,mentions=count[label]) for label in SHORT)
            for partition in ['six_document_census','stratified_120']:
                checked=[a for a in audit if a['audit_partition']==partition and a['retained_by_threshold'][name][f'{threshold:.2f}']]
                categories=Counter(a['draft_assessment']['boundary_category'] for a in checked)
                audit_metrics.append(dict(configuration=name,threshold=threshold,partition=partition,present=len(checked),
                    **{c:categories[c] for c in ['VALID_SPAN','PARTIAL_SPAN','WRONG_BOUNDARY','IRRELEVANT','AMBIGUOUS']},
                    draft_irrelevant_fraction=categories['IRRELEVANT']/len(checked) if checked else None,status='UNREVIEWED'))
                type_counts=Counter(p['proposed_category'] for a in checked if a['draft_assessment']['boundary_category']=='VALID_SPAN'
                    for p in a['draft_assessment']['type_evaluations'][name] if retained(p['score'],threshold,name))
                type_metrics.append(dict(configuration=name,threshold=threshold,partition=partition,
                    boundary_filter='VALID_SPAN_ONLY',**{c:type_counts[c] for c in ['TYPE_HINT_CORRECT','TYPE_HINT_INCORRECT','TYPE_HINT_AMBIGUOUS']},status='UNREVIEWED'))
    csv_out(ROOT/'threshold_metrics.csv',rows);csv_out(ROOT/'discipline_metrics.csv',discipline_rows)
    csv_out(ROOT/'label_metrics.csv',labels);csv_out(ROOT/'audit_quality_metrics.csv',audit_metrics)
    csv_out(ROOT/'type_hint_audit_metrics.csv',type_metrics);csv_out(ROOT/'document_metrics.csv',document_rows)
    write_json(ROOT/'threshold_metrics.json',rows)
    return rows,audit_metrics


SYSTEM_PROMPT='''You assign scientific entity types to source-exact candidate mentions.
Use the supplied abstract as context. Do not change offsets, invent mentions, merge entities, or extract relations.
For each candidate_id return typing_status ENTITY, NON_ENTITY, or UNKNOWN. For ENTITY choose one allowed type.
Treat NER hints as fallible evidence. Return JSON only: {"decisions":[{"candidate_id":"...","typing_status":"ENTITY","entity_type":"METHOD"}]}.
Allowed types and definitions:
'''


def qwen_cost(outputs,documents):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('artifacts/ner_v2/qwen3_4b_tokenizer',local_files_only=True)
    ontology=read_json('contracts/ner_v2/ontology.v2.json')
    system=SYSTEM_PROMPT+'\n'.join(label+': '+spec['definition_ru'] for label,spec in ontology['entity_types'].items())
    write_json(ROOT/'qwen_prompt_template.json',dict(system=system,payload_fields=['document_id','abstract','candidates'],
               max_candidates=32,max_prompt_tokens=4096,enable_thinking=False,tokenizer_lock='qwen_tokenizer_lock.json'))
    def payload(doc,candidates):return dict(document_id=doc['document_id'],abstract=doc['text'],candidates=candidates)
    def serialized(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'))
    def token_count(value):
        return len(tokenizer.apply_chat_template([{'role':'system','content':system},{'role':'user','content':serialized(value)}],
                    tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False))
    summary=[];batches=[]
    for name in MODELS:
        for threshold in THRESHOLDS:
            predictions=outputs[(name,threshold)];batch_records=[];unbatched_bytes=0;candidate_count=0
            for doc in documents:
                grouped=defaultdict(list)
                for m in predictions:
                    if m['document_id']==doc['document_id']:grouped[(m['start'],m['end'])].append(m)
                candidates=[]
                for (a,b),items in sorted(grouped.items()):
                    candidates.append(dict(candidate_id='c:'+fingerprint([doc['document_id'],a,b])[:20],start=a,end=b,
                           surface=doc['text'][a:b],normalized_form=items[0]['normalized_form'],
                           ner_hints=[dict(type=m['ner_label_hint'],score=round(m['score'],6)) for m in sorted(items,key=lambda x:x['ner_label_hint'])]))
                candidate_count+=len(candidates)
                if not candidates:continue
                unbatched_bytes+=len(serialized(payload(doc,candidates)).encode('utf8'))
                batch=[]
                for candidate in candidates:
                    trial=batch+[candidate]
                    if len(trial)>32 or token_count(payload(doc,trial))>4096:
                        if not batch:raise ValueError('One candidate cannot fit the planning prompt')
                        value=payload(doc,batch)
                        batch_records.append(dict(configuration=name,threshold=threshold,document_id=doc['document_id'],
                                     input_tokens=token_count(value),json_bytes=len(serialized(value).encode('utf8')),payload=value))
                        batch=[candidate]
                        if token_count(payload(doc,batch))>4096:raise ValueError('Candidate exceeds planning budget')
                    else:batch=trial
                if batch:
                    value=payload(doc,batch)
                    batch_records.append(dict(configuration=name,threshold=threshold,document_id=doc['document_id'],
                                  input_tokens=token_count(value),json_bytes=len(serialized(value).encode('utf8')),payload=value))
            batches.extend(batch_records)
            summary.append(dict(configuration=name,threshold=threshold,unique_candidates=candidate_count,
                    candidates_per_document=candidate_count/len(documents),unbatched_json_bytes=unbatched_bytes,
                    batched_json_bytes=sum(b['json_bytes'] for b in batch_records),
                    input_tokens=sum(b['input_tokens'] for b in batch_records),batch_calls=len(batch_records),
                    maximum_prompt_tokens=max((b['input_tokens'] for b in batch_records),default=0),
                    estimate_status='EXACT_TOKENIZATION_OF_PLANNING_PROMPT_NO_GENERATION'))
            print('Qwen tokenizer-only',name,threshold,summary[-1]['input_tokens'],flush=True)
    for row in summary:
        baseline=next(x for x in summary if x['configuration']==row['configuration'] and x['threshold']==.35)
        row['token_reduction_vs_035']=1-row['input_tokens']/baseline['input_tokens']
        row['candidate_reduction_vs_035']=1-row['unique_candidates']/baseline['unique_candidates']
    csv_out(ROOT/'qwen_cost_estimates.csv',summary);write_json(ROOT/'qwen_cost_estimates.json',summary)
    write_jsonl(ROOT/'qwen_planned_batches.jsonl',batches)
    return summary


def threshold_examples(outputs,documents,references,audit):
    events=[];missed=[];doc_by_id={d['document_id']:d for d in documents}
    for name in MODELS:
        low={key(m) for m in outputs[(name,.15)]}
        for index,threshold in enumerate(THRESHOLDS):
            current={key(m) for m in outputs[(name,threshold)]}
            previous={key(m) for m in outputs[(name,THRESHOLDS[index-1])]} if index else low
            retained_refs=[r for r in references if key(r) in current]
            newly_lost=[r for r in references if key(r) in previous-current]
            removed_noise=[a for a in audit if a['draft_assessment']['boundary_category']=='IRRELEVANT' and
                           key(a) in low-current]
            alternative=[]
            for a in audit:
                types=a['draft_assessment']['suggested_types'];predictions=a['predictions'][name]
                correct=[p for p in predictions if p['label'] in types and retained(p['score'],threshold,name)]
                wrong_removed=[p for p in predictions if p['label'] not in types and not retained(p['score'],threshold,name)]
                if correct and wrong_removed:alternative.append(dict(candidate_id=a['candidate_id'],surface=a['surface'],
                        document_id=a['document_id'],start=a['start'],end=a['end'],retained_hints=correct,removed_hints=wrong_removed,
                        status='UNREVIEWED_TYPE_INTERPRETATION'))
            empty=[d['document_id'] for d in documents if not any(k[0]==d['document_id'] for k in current)]
            observations={
                'useful_term_retained':[dict(surface=r['surface'],document_id=r['document_id'],start=r['start'],end=r['end'],suggested_type=r['suggested_type']) for r in retained_refs[:5]],
                'draft_irrelevant_removed':[dict(surface=a['surface'],document_id=a['document_id'],start=a['start'],end=a['end']) for a in removed_noise[:5]],
                'newly_lost_assisted_terms':[dict(surface=r['surface'],document_id=r['document_id'],start=r['start'],end=r['end'],suggested_type=r['suggested_type']) for r in newly_lost],
                'incorrect_alternative_removed_correct_hint_retained':alternative[:5],
                'documents_without_candidates':empty,
            }
            for category,examples in observations.items():
                events.append(dict(configuration=name,threshold=threshold,event=category,
                    status='OBSERVED_UNREVIEWED' if examples else 'NO_OBSERVED_EVENT',examples=examples))
            for ref in references:
                if key(ref) not in current:
                    overlap=any(d==ref['document_id'] and max(a,ref['start'])<min(b,ref['end']) for d,a,b in current)
                    missed.append({**ref,'configuration':name,'threshold':threshold,
                         'proposed_audit_category':'PARTIAL_SPAN' if overlap else 'MISSED_ENTITY',
                         'interpretation':'Exact reference missing; overlapping candidate exists' if overlap else 'No overlapping candidate',
                         'source_text':doc_by_id[ref['document_id']]['text']})
    write_jsonl(ROOT/'threshold_examples.jsonl',events)
    write_jsonl(ROOT/'audit_missing_reference_UNREVIEWED.jsonl',missed)


def recommend(metrics,reference_details,reference_types,quality,costs):
    protocol=read_json(ROOT/'protocol.json')['recommendation_policy']
    results=[]
    for name in MODELS:
        base=[r for r in reference_details if r['configuration']==name and r['threshold']==.35]
        base_count=sum(r['exact_match'] for r in base)
        noise_base=next(q['IRRELEVANT'] for q in quality if q['configuration']==name and q['threshold']==.35 and q['partition']=='six_document_census')
        decisions=[]
        for threshold in THRESHOLDS:
            subset=[r for r in reference_details if r['configuration']==name and r['threshold']==threshold]
            now=sum(r['exact_match'] for r in subset)
            retention=now/base_count if base_count else 0
            per_discipline=[]
            for discipline in sorted({r['discipline'] for r in base}):
                denom=sum(r['exact_match'] for r in base if r['discipline']==discipline)
                numer=sum(r['exact_match'] for r in subset if r['discipline']==discipline)
                per_discipline.append(numer/denom if denom else 1.)
            rare=[r for r in reference_types if r['configuration']==name and r['threshold']==threshold and 1<=r['baseline_035_exact']<=4]
            rare_preserved=all(r['exact_matches']>=r['baseline_035_exact'] for r in rare)
            noise=next(q['IRRELEVANT'] for q in quality if q['configuration']==name and q['threshold']==threshold and q['partition']=='six_document_census')
            allowed=threshold>=.35 and retention>=.9 and min(per_discipline)>=.8 and rare_preserved and noise<=noise_base
            decisions.append(dict(threshold=threshold,eligible=allowed,exact_reference_retention=retention,
                                min_discipline_retention=min(per_discipline),rare_groups_preserved=rare_preserved,
                                irrelevant_census=noise))
        chosen=max(d['threshold'] for d in decisions if d['eligible'])
        matching=next(m for m in metrics if m['configuration']==name and m['threshold']==chosen)
        cost=next(c for c in costs if c['configuration']==name and c['threshold']==chosen)
        results.append(dict(configuration=name,recommended_threshold=chosen,unique_spans=matching['unique_spans'],
                     reference_exact_matches=sum(r['exact_match'] for r in reference_details if r['configuration']==name and r['threshold']==chosen),
                     qwen_input_tokens=cost['input_tokens'],qwen_batch_calls=cost['batch_calls'],decisions=decisions))
    # Rank on the draft reference first; final report discusses the full trade-off,
    # rather than pretending this is independently validated model optimization.
    best=max(results,key=lambda x:(x['reference_exact_matches'],-x['qwen_input_tokens']))
    write_json(ROOT/'recommendation.json',dict(status='PRELIMINARY_ASSISTED_WORKING_POINT',policy=protocol,
                       per_model=results,primary=best['configuration'],primary_threshold=best['recommended_threshold'],
                       limitation='Selected on the assisted audit set itself; not an independently validated optimum'))
    write_json(ROOT/'recommendation_sensitivity.json',{
       r['configuration']:dict(strict_policy_threshold=r['recommended_threshold'],
           all_eight_thresholds_permitted=max(d['threshold'] for d in r['decisions'] if
               d['exact_reference_retention']>=.9 and d['min_discipline_retention']>=.8 and d['rare_groups_preserved'] and
               d['irrelevant_census']<=next(v['irrelevant_census'] for v in r['decisions'] if v['threshold']==.35)),
           without_rare_group_protection=max(d['threshold'] for d in r['decisions'] if d['threshold']>=.35 and
               d['exact_reference_retention']>=.9 and d['min_discipline_retention']>=.8 and
               d['irrelevant_census']<=next(v['irrelevant_census'] for v in r['decisions'] if v['threshold']==.35))) for r in results})
    return results


def figures(metrics,quality,reference_metrics,costs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    series=[
        ('01_candidate_counts','Число уникальных spans','Threshold → число кандидатов',metrics,'unique_spans'),
        ('02_candidate_retention','Сохранено от spans при 0,35','Сохранение исходного набора 0,35',metrics,'retained_fraction_vs_035'),
        ('03_draft_irrelevant','Предположительно нерелевантные spans','Шесть документов: оценки UNREVIEWED',
             [q for q in quality if q['partition']=='six_document_census'],'IRRELEVANT'),
        ('04_assisted_reference','Доля exact совпадений с draft reference','158 reference spans; не истинный Recall',reference_metrics,'exact_matching_fraction'),
        ('05_qwen_load','Input tokens планового шаблона','Qwen: только tokenization, без inference',costs,'input_tokens'),
    ]
    for filename,ylabel,title,data,field in series:
        fig,ax=plt.subplots(figsize=(8,4.3),layout='constrained')
        for name in MODELS:
            subset=sorted([r for r in data if r['configuration']==name],key=lambda x:x['threshold'])
            ax.plot([r['threshold'] for r in subset],[r[field] for r in subset],marker='o',label=name)
        ax.set(title=title,xlabel='Score threshold',ylabel=ylabel,xticks=THRESHOLDS)
        ax.grid(alpha=.2);ax.legend()
        fig.savefig(folder/(filename+'.png'),dpi=170);fig.savefig(folder/(filename+'.svg'));plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
    for ax,partition,title in zip(axes,['six_document_census','stratified_120'],['Полный набор шести документов','Стратифицированные 120 кандидатов']):
        for name in MODELS:
            subset=sorted([r for r in quality if r['configuration']==name and r['partition']==partition],key=lambda x:x['threshold'])
            ax.plot([r['threshold'] for r in subset],[r['IRRELEVANT'] for r in subset],marker='o',label=name)
        ax.set(title=title,xlabel='Threshold',ylabel='IRRELEVANT (draft, UNREVIEWED)',xticks=THRESHOLDS)
        ax.tick_params(axis='x',labelsize=8);ax.legend();ax.grid(alpha=.2)
    fig.savefig(folder/'03_draft_irrelevant.png',dpi=170);fig.savefig(folder/'03_draft_irrelevant.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.3),layout='constrained')
    for name in MODELS:
        seconds=read_json(ROOT/name/'performance.json')['inference_seconds']
        subset=[r for r in metrics if r['configuration']==name]
        ax.plot([r['unique_spans'] for r in subset],[seconds]*len(subset),marker='o',label=name)
        for r in subset:
            ax.annotate(f"{r['threshold']:.2f}",(r['unique_spans'],seconds),xytext=(0,5 if THRESHOLDS.index(r['threshold'])%2 else -14),
                        textcoords='offset points',ha='center',fontsize=7)
    ax.set(title='Один фактический NER timing на модель; thresholds — постфильтр',
           xlabel='Уникальные spans после фильтра',ylabel='NER при 0,15, с')
    ax.legend();ax.grid(alpha=.2)
    ax.set_ylim(0,max(read_json(ROOT/name/'performance.json')['inference_seconds'] for name in MODELS)*1.15)
    fig.savefig(folder/'06_candidates_ner_time.png',dpi=170);fig.savefig(folder/'06_candidates_ner_time.svg');plt.close(fig)
    for path in folder.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8',newline='\n')


def main():
    global ROOT
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--skip-qwen',action='store_true')
    parser.add_argument('--output-root',default=str(ROOT))
    args=parser.parse_args()
    ROOT=Path(args.output_root)
    manifest,_=validate_frozen(ROOT)
    docs=read_documents(ROOT/'input_abstracts.jsonl');references=jsonl(ROOT/'assisted_reference_frozen.jsonl')
    outputs={(name,t):jsonl(ROOT/name/f'mentions_{int(t*100):03}.jsonl') for name in MODELS for t in THRESHOLDS}
    candidates=source_candidates({name:outputs[(name,.15)] for name in MODELS},docs)
    audit=prepare_audit(candidates,docs,references,manifest)
    metrics,quality=build_metrics(outputs,docs,audit)
    reference_metrics,details,by_type=reference_analysis(outputs,references)
    costs=read_json(ROOT/'qwen_cost_estimates.json') if args.skip_qwen else qwen_cost(outputs,docs)
    threshold_examples(outputs,docs,references,audit)
    recommend(metrics,details,by_type,quality,costs)
    figures(metrics,quality,reference_metrics,costs)
    print('audit rows',len(audit),'references',len(references),'metric configurations',len(metrics),flush=True)


if __name__=='__main__':main()
