"""Source integrity, resumability, strict thresholds, and audit sampling tests."""
from pathlib import Path
import pytest

from cgraph.common import write_json,file_sha256
from cgraph.ner_v2.quality_audit import retained,load_checkpoint,validate_frozen,atomic_json


@pytest.mark.parametrize('name,expected',[('E1',False),('E2',False),('E3',True)])
def test_threshold_boundary_matches_model_decoder(name,expected):
    assert retained(.50,.50,name) is expected
    assert retained(.500001,.50,name)
    assert not retained(.499999,.50,name)


def test_resume_only_accepts_complete_source_exact_checkpoint(tmp_path):
    path=tmp_path/'doc.json'
    doc={'text':'pH and β-cells','text_sha256':'source'}
    value={'status':'complete','config_hash':'config','text_sha256':'source',
           'mentions':[dict(start=7,end=14,surface_text='β-cells')]}
    write_json(path,value)
    assert load_checkpoint(path,'config',doc)==value
    with pytest.raises(ValueError,match='mismatch'):load_checkpoint(path,'other',doc)
    value['mentions'][0]['start']=6;write_json(path,value)
    with pytest.raises(ValueError,match='Offsets'):load_checkpoint(path,'config',doc)


def test_frozen_reference_and_corpus_have_independent_locks(tmp_path):
    (tmp_path/'input_abstracts.jsonl').write_text('source',encoding='utf8')
    (tmp_path/'assisted_reference_frozen.jsonl').write_text('reference',encoding='utf8')
    source=file_sha256(tmp_path/'input_abstracts.jsonl')
    write_json(tmp_path/'corpus_manifest.json',dict(input_sha256=source))
    write_json(tmp_path/'reference_lock.json',dict(corpus_sha256=source,
               reference_sha256=file_sha256(tmp_path/'assisted_reference_frozen.jsonl')))
    validate_frozen(tmp_path)
    (tmp_path/'assisted_reference_frozen.jsonl').write_text('changed',encoding='utf8')
    with pytest.raises(ValueError,match='reference'):validate_frozen(tmp_path)


def test_atomic_checkpoint_does_not_replace_complete_file_on_write_failure(tmp_path,monkeypatch):
    import cgraph.ner_v2.quality_audit as quality
    path=tmp_path/'checkpoint.json';write_json(path,{'status':'complete'})
    before=path.read_bytes()
    def fail(path,value):
        Path(path).write_text('incomplete',encoding='utf8')
        raise OSError('Simulated disk failure')
    monkeypatch.setattr(quality,'write_json',fail)
    with pytest.raises(OSError):atomic_json(path,{'status':'new'})
    assert path.read_bytes()==before


def test_sampling_is_unique_balanced_seeded_and_excludes_census():
    import importlib.util
    spec=importlib.util.spec_from_file_location('audit_analysis','scripts/analyze_ner_v2_003.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    candidates={}
    for field in range(6):
        for document in range(5):
            for item in range(10):
                doc=f'D{field}-{document}';identity=(doc,item*10,item*10+5)
                candidates[identity]=dict(document_id=doc,discipline=str(field),start=identity[1],end=identity[2],
                    models_at_015=3,predictions={name:[dict(label=['OBJECT','METHOD','ATTRIBUTE'][item%3],score=.16+.07*item)] for name in ['E1','E2','E3']})
    census={f'D{field}-0' for field in range(6)}
    a=module.stratified_sample(candidates,census);b=module.stratified_sample(candidates,census)
    assert a==b and len(a)==120
    assert len({module.key(x) for x in a})==120
    assert all(x['document_id'] not in census for x in a)
    assert all(sum(x['discipline']==str(i) for x in a)==20 for i in range(6))
