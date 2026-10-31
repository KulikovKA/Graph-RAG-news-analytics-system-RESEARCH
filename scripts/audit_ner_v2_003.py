"""Final independent structural audit of NER-V2-003 artifacts (not semantic Gold)."""
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from cgraph.common import read_json,write_json,file_sha256
from cgraph.ner_v2.io import read_documents,validate_span
from cgraph.ner_v2.quality_audit import ROOT,THRESHOLDS,retained,validate_frozen
from cgraph.ner_v2.research import jsonl


def main():
    manifest,reference_lock=validate_frozen(ROOT)
    docs=read_documents(ROOT/'input_abstracts.jsonl');by_id={d['document_id']:d for d in docs}
    old=read_documents('reports/ner_v2/benchmark_run1/input_abstracts.jsonl')
    assert len(docs)==30 and all(n==5 for n in Counter(d['discipline'] for d in docs).values())
    assert not {d['document_id'] for d in old}&set(by_id)
    assert not {d['text_sha256'] for d in old}&{d['text_sha256'] for d in docs}
    assert not {d['doi'].lower() for d in old if d.get('doi')}&{d['doi'].lower() for d in docs if d.get('doi')}
    assert all(100<=len(d['text'].split())<=400 for d in docs)
    for discipline in {d['discipline'] for d in docs}:
        assert len({d['primary_topic']['id'] for d in docs if d['discipline']==discipline})==5
    references=jsonl(ROOT/'assisted_reference_frozen.jsonl')
    assert len(references)==158 and {r['document_id'] for r in references}==set(manifest['audit_document_ids'])
    for r in references:
        validate_span(by_id[r['document_id']]['text'],r['start'],r['end'],r['surface'])
        assert r['status']=='ASSISTED_REFERENCE_DRAFT' and r['review_status']=='UNREVIEWED'
    result={'status':'PASSED','corpus_documents':30,'words':manifest['words'],'reference_spans':158,'models':{}}
    for name in ['E1','E2','E3']:
        config=read_json(ROOT/name/'config.json');performance=read_json(ROOT/name/'performance.json')
        assert config['input_sha256']==manifest['input_sha256'] and config['reference_sha256']==reference_lock['reference_sha256']
        assert config['runner_sha256']==file_sha256('src/cgraph/ner_v2/quality_audit.py')
        records=jsonl(ROOT/name/'mentions_015.jsonl')
        for m in records:
            validate_span(by_id[m['document_id']]['text'],m['start'],m['end'],m['surface_text'])
            assert m['ner_label_hint'] in config['labels'] and 0<=m['score']<=1
        validations=raw_count=words_covered=0;maximum=0
        for d in docs:
            checkpoint=read_json(ROOT/name/'documents'/f"{d['document_id']}.json")
            assert checkpoint['status']=='complete' and checkpoint['text_sha256']==d['text_sha256']
            for check in checkpoint['threshold_checks']:
                assert check['matches'] and check['native_count']==check['filtered_count'];validations+=1
            for raw in checkpoint['raw_windows']:
                for p in raw['predictions']:
                    validate_span(d['text'],raw['window_start']+p['start'],raw['window_start']+p['end'],p['text'])
                    assert 0<=p['start']<p['end']<=raw['window_end']-raw['window_start'];raw_count+=1
            import re
            for word in re.finditer(r'\S+',d['text']):
                assert any(w['window_start']<=word.start() and word.end()<=w['window_end'] for w in checkpoint['raw_windows'])
                words_covered+=1
            maximum=max(maximum,max(checkpoint['performance']['sequence_lengths']))
        assert validations==18 and words_covered==5744 and maximum<=512
        for t in THRESHOLDS:
            rows=jsonl(ROOT/name/f'mentions_{int(t*100):03}.jsonl')
            assert rows==[m for m in records if retained(m['score'],t,name)]
            assert len({(m['document_id'],m['start'],m['end'],m['ner_label_hint']) for m in rows})==len(rows)
        result['models'][name]=dict(minimum_threshold_mentions=len(records),raw_window_predictions=raw_count,
              source_words_covered=words_covered,native_equivalence_checks=validations,max_encoder_tokens=maximum,
              windows=performance['windows'],inference_encoder_calls=performance['encoder_calls'],
              score_threshold_equalities=sum(m['score'] in THRESHOLDS for m in records))
        assert performance['windows']==performance['encoder_calls']
    audit=jsonl(ROOT/'audit_candidates_UNREVIEWED.jsonl')
    assert len(audit)==553 and sum(a['audit_partition']=='stratified_120' for a in audit)==120
    assert len({a['candidate_id'] for a in audit})==len(audit)
    assert all(a['draft_assessment']['status']=='UNREVIEWED' and not a['draft_assessment']['independent_human_review'] for a in audit)
    sample=[a for a in audit if a['audit_partition']=='stratified_120']
    assert all(sum(a['discipline']==d for a in sample)==20 for d in {a['discipline'] for a in sample})
    assert all(a['document_id'] not in manifest['audit_document_ids'] for a in sample)
    assert all(a['draft_assessment']['origin']=='assistant_context_review_amendment' for a in sample)
    result['audit']={'census_candidates':433,'stratified_candidates':120,'assistant_context_reviews':120,'human_reviewed':0}
    batches=jsonl(ROOT/'qwen_planned_batches.jsonl')
    costs=read_json(ROOT/'qwen_cost_estimates.json')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('artifacts/ner_v2/qwen3_4b_tokenizer',local_files_only=True)
    system=read_json(ROOT/'qwen_prompt_template.json')['system']
    for cost in costs:
        selected=[b for b in batches if b['configuration']==cost['configuration'] and b['threshold']==cost['threshold']]
        assert len(selected)==cost['batch_calls'] and sum(b['input_tokens'] for b in selected)==cost['input_tokens']
        identities=[c['candidate_id'] for b in selected for c in b['payload']['candidates']]
        assert len(set(identities))==len(identities)==cost['unique_candidates']
        assert all(len(b['payload']['candidates'])<=32 and b['input_tokens']<=4096 for b in selected)
        b=selected[0]
        tokens=tokenizer.apply_chat_template([{'role':'system','content':system},
            {'role':'user','content':json.dumps(b['payload'],ensure_ascii=False,separators=(',',':'))}],
            tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        assert len(tokens)==b['input_tokens'] and len(tokens)>100
    result['qwen']={'configurations':24,'batches_saved':len(batches),'independent_retokenization_checks':24,
                    'max_prompt_tokens':max(b['input_tokens'] for b in batches),'model_loads':0,'generation_calls':0}
    baseline='0366724d620d076adc8035114424cb06a57ff5d1'
    tracked=subprocess.check_output(['git','ls-tree','-r','--name-only',baseline],text=True).splitlines()
    protected=[p for p in tracked if p.startswith(('reports/ner_v2/','contracts/ner_v2/','src/cgraph/ner_v2/')) or
               p in ['config/ner_v2.json','config/ner_v2_research.json','pyproject.toml','uv.lock']]
    changed=[]
    for path in protected:
        original=subprocess.check_output(['git','show',baseline+':'+path])
        if original.replace(b'\r\n',b'\n')!=Path(path).read_bytes().replace(b'\r\n',b'\n'):changed.append(path)
    assert not changed,changed
    result['historical_protection']={'baseline_commit':baseline,'files_checked':len(protected),'changed':changed}
    write_json(ROOT/'integrity_audit.json',result)
    paths=[p for p in ROOT.rglob('*') if p.is_file() and p.name!='sha256_manifest.json']
    paths += [Path('src/cgraph/ner_v2/quality_audit.py'),Path('tests/test_ner_v2_003.py')]
    paths += list(Path('scripts').glob('*ner_v2_003*.py'))
    paths += [Path('reports/ner_v2/NER_V2_003_CANDIDATE_QUALITY_AUDIT.md'),Path('reports/ner_v2/NER_V2_003_THESIS_FRAGMENT.md')]
    write_json(ROOT/'sha256_manifest.json',{str(p).replace('\\','/'):{'sha256':file_sha256(p),'bytes':p.stat().st_size}
                                           for p in sorted(set(paths)) if p.exists()})
    print('integrity audit passed; protected historical files',len(protected),'NER native checks',54,'audited candidates',len(audit))


if __name__=='__main__':main()
