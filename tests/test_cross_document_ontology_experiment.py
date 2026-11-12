"""Safety checks for the separate research runner; no model or database calls."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('ontology_experiment', SCRIPTS/'experiment_cross_document_ontology.py')
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def test_resume_recovers_only_incomplete_tail_and_preserves_completed_result(tmp_path):
    path=tmp_path/'predictions.jsonl'
    row=dict(configuration='D',pair_id='P01',inference_signature='frozen',raw_scores=[.123])
    completed=(json.dumps(row)+'\n').encode()
    path.write_bytes(completed+b'{"configuration":')
    cached=run.checkpoint(path,'frozen')
    assert cached[('D','P01')]==row
    assert path.read_bytes()==completed
    assert json.loads(path.with_suffix('.recovery.json').read_text())['discarded_bytes']==len(b'{"configuration":')


def test_resume_refuses_foreign_signature_and_duplicate_result(tmp_path):
    path=tmp_path/'predictions.jsonl'
    row=dict(configuration='D',pair_id='P01',inference_signature='foreign')
    path.write_text(json.dumps(row)+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='signature'):
        run.checkpoint(path,'frozen')
    path.write_text((json.dumps(row)+'\n')*2,encoding='utf-8')
    with pytest.raises(ValueError,match='Duplicate'):
        run.checkpoint(path)


def test_cross_document_context_preserves_original_mentions_with_identical_surface_forms():
    docs={'a':dict(document_id='a',text='The results improved.',revision_id='ra',source_url='a'),
          'b':dict(document_id='b',text='Other results declined.',revision_id='rb',source_url='b')}
    profiles={'x':dict(document_id='a',canonical_entity_id='x',mention_start=4,mention_end=11,
                       mention_text='results',entity_types=['RESULT']),
              'y':dict(document_id='b',canonical_entity_id='y',mention_start=6,mention_end=13,
                       mention_text='results',entity_types=['RESULT'])}
    item=run.context(dict(pool='main',pair_id='test',E2_id='x',E3_id='y'),docs,profiles,100)
    assert [s['entity_id'] for s in item['fixed_spans']]==['x','y']
    assert len({(s['start'],s['end']) for s in item['fixed_spans']})==2
    for span in item['fixed_spans']:
        assert item['text'][span['start']:span['end']]=='results'
    assert '[END DOCUMENT a]\n[DOCUMENT b]' in item['text']
    assert all(docs[f['document_id']]['text'][f['start']:f['end']]==f['text'] for f in item['source_fragments'])
