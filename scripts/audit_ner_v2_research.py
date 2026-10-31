"""Audit provenance, source coverage, exact offsets and frozen historical files."""
import hashlib
import importlib.util
import itertools
import re
import subprocess
import tomllib
from pathlib import Path

from cgraph.common import file_sha256,write_json,read_json
from cgraph.ner_v2.research import ROOT,HIST,jsonl
from cgraph.ner_v2.io import validate_span


def main():
    docs=jsonl(HIST/'input_abstracts.jsonl');by_id={d['document_id']:d for d in docs}
    baseline='9a8556772a0c24ce80b61e8704683037bcdf150f'
    audit={'status':'complete','baseline_commit':baseline,'historical_files':{},'normalization':{},'ner':{}}
    # Compare all historical benchmark files, original normalizer, ontology and
    # earlier experiment directories using Git's committed bytes, not timestamps.
    tracked=subprocess.check_output(['git','ls-tree','-r','--name-only',baseline],text=True).splitlines()
    protected=[p for p in tracked if p.startswith('reports/ner_v2/benchmark_run1/') or
               p in ['src/cgraph/ner_v2/normalization.py','contracts/ner_v2/ontology.v2.json',
                     'config/ner_v2.json','contracts/ner_v2/model_lock.json']]
    for name in protected:
        original=subprocess.check_output(['git','show',f'{baseline}:{name}'])
        # Git may store LF while checkout uses CRLF; compare logical UTF-8 text
        # and record both byte hashes explicitly. No rewriting is performed.
        current=Path(name).read_bytes()
        same=original.replace(b'\r\n',b'\n')==current.replace(b'\r\n',b'\n')
        assert same,name
        audit['historical_files'][name]={'git_sha256':hashlib.sha256(original).hexdigest(),
                                        'checkout_sha256':file_sha256(name),'unchanged_text':same,
                                        'same_bytes':original==current}
    for size in ['large','medium','small']:
        historical=jsonl(HIST/f'{size}.jsonl')
        n1=jsonl(ROOT/f'normalization/N1_{size}.jsonl')
        same=sum(a['normalized_form']==b['normalized_form'] for a,b in zip(historical,n1))
        assert len(historical)==len(n1) and same==len(historical)
        for variant in ['N0','N1','N2','N3']:
            records=jsonl(ROOT/f'normalization/{variant}_{size}.jsonl')
            assert len(records)==len(historical)
            for old,new in zip(historical,records):
                for key in ['document_id','revision_id','start','end','surface_text','mention_id','score','ner_label_hint','source_reference']:
                    assert old[key]==new[key],key
                validate_span(by_id[new['document_id']]['text'],new['start'],new['end'],new['surface_text'])
        audit['normalization'][size]={'records':len(historical),'N1_matches_historical_forms':same,
                                      'all_variants_preserve_identity_surface_offsets':True}
    for name in ['E1','E2','E3','E4']:
        raw=jsonl(ROOT/name/'raw_windows.jsonl');budgets=read_json(ROOT/name/'token_budgets.json')
        perf=read_json(ROOT/name/'performance.json')
        accepted=[b for b in budgets if b['status']=='accepted']
        assert len(raw)==len(accepted)==perf['encoder_calls']
        covered=0;raw_count=0;equal_scores=0
        for doc in docs:
            source_windows=[r for r in raw if r['document_id']==doc['document_id']]
            for word in re.finditer(r'\S+',doc['text']):
                assert any(r['window_start']<=word.start() and word.end()<=r['window_end'] for r in source_windows)
                covered+=1
            for window in source_windows:
                for m in window['predictions']:
                    validate_span(doc['text'],window['window_start']+m['start'],window['window_start']+m['end'],m['text'])
                    assert window['window_start']<=window['window_start']+m['start']<window['window_start']+m['end']<=window['window_end']
                    raw_count+=1
                    equal_scores+=m['score'] in [.15,.25,.35]
        assert equal_scores==0
        rows=jsonl(ROOT/name/'mentions_015.jsonl')
        for threshold in [.25,.35]:
            filtered=[m for m in rows if m['score']>=threshold]
            saved=jsonl(ROOT/name/f'mentions_{int(threshold*100):03}.jsonl')
            assert filtered==saved
        definitions=jsonl(ROOT/name/'document_abbreviations.jsonl')
        for item in definitions:
            d=by_id[item['document_id']]
            for form in ['short','long']:
                validate_span(d['text'],item[form+'_start'],item[form+'_end'],item[form+'_form'])
        assert max(perf['actual_sequence_lengths'])<=512
        assert max(b['tokens'] for b in accepted)<=512
        audit['ner'][name]={'raw_records_validated':raw_count,'words_covered':covered,
                            'accepted_windows':len(accepted),'split_over_budget':len(budgets)-len(accepted),
                            'max_preflight_tokens':max(b['tokens'] for b in accepted),
                            'max_actual_encoder_tokens':max(perf['actual_sequence_lengths']),
                            'scores_equal_threshold':equal_scores,'filtered_files_match':True,
                            'exact_definition_spans':len(definitions)}
    old=tomllib.loads(subprocess.check_output(['git','show',f'{baseline}:uv.lock'],text=True))
    new=tomllib.loads(Path('uv.lock').read_text(encoding='utf8'))
    before={x['name']:x['version'] for x in old['package']}
    after={x['name']:x['version'] for x in new['package']}
    changed={k:(v,after.get(k)) for k,v in before.items() if after.get(k)!=v}
    assert not changed
    audit['lockfile']={'historical_packages_version_changed':changed,'new_packages':{k:v for k,v in after.items() if k not in before}}
    libraries={}
    for package in ['gliner','gliner2','spacy']:
        spec=importlib.util.find_spec(package)
        root=Path(spec.origin).parent
        libraries[package]={str(p.relative_to(root)):file_sha256(p) for p in root.rglob('*.py')}
    write_json(ROOT/'library_source_hashes.json',libraries)
    write_json(ROOT/'integrity_audit.json',audit)
    paths=[p for p in ROOT.rglob('*') if p.is_file() and p.name!='sha256_manifest.json']
    paths += [p for folder in ['src/cgraph/ner_v2','contracts/ner_v2'] for p in Path(folder).glob('*') if p.is_file()]
    paths += [Path(p) for p in ['pyproject.toml','uv.lock','config/ner_v2_research.json',
                              'reports/ner_v2/NER_V2_002_NORM_V2_001_RESEARCH_REPORT.md',
                              'reports/ner_v2/NER_V2_002_NORM_V2_001_THESIS_FRAGMENT.md']]
    paths += list(Path('scripts').glob('*ner_v2_research*.py'))+[Path('scripts/prepare_gliner25_research.py'),Path('tests/test_ner_v2_research.py')]
    write_json(ROOT/'sha256_manifest.json',{str(p).replace('\\','/'):{'sha256':file_sha256(p),'bytes':p.stat().st_size} for p in sorted(set(paths))})
    print('audit complete; all historical packages and files unchanged;',len(paths),'hashed files')


if __name__=='__main__':main()
