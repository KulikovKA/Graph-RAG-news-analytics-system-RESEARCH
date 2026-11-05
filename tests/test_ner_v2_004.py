"""Unit checks for NER-V2-004 without model inference."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('prompt_optimization', Path(__file__).resolve().parents[1] / 'scripts/ner_v2_004_prompt_optimization.py')
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_schemas_match_ontology_and_e3():
    assert mod.SCHEMAS['V0'] == mod.read_json(mod.HIST/'E3/config.json')['labels']
    for schema in mod.SCHEMAS.values():
        assert set(schema) == set(mod.read_json(mod.ONTOLOGY)['entity_types'])
        assert len(set(schema.values())) == 11
    for t in mod.TYPES:
        assert mod.SCHEMAS['V2'][t].startswith(mod.SCHEMAS['V1'][t])


def test_metrics_missed_confusions_new_and_duplicate_labels():
    gold = {('d',0,1):'OBJECT', ('d',2,3):'METHOD', ('d',4,5):'METHOD'}
    predictions = [dict(document_id='d',start=0,end=1,ner_label_hint='OBJECT',score=.8),
                   dict(document_id='d',start=0,end=1,ner_label_hint='METHOD',score=.7),
                   dict(document_id='d',start=2,end=3,ner_label_hint='OBJECT',score=.9),
                   dict(document_id='d',start=6,end=7,ner_label_hint='RESULT',score=.9)]
    summary, per, matrix, best = mod.score(gold,predictions)
    assert summary['coverage'] == pytest.approx(2/3)
    assert summary['typing_accuracy'] == .5
    assert summary['correct_fraction'] == pytest.approx(1/3)
    assert summary['new_spans'] == 1
    assert matrix['METHOD']['MISSED'] == 1 and matrix['METHOD']['OBJECT'] == 1
    obj = next(x for x in per if x['type'] == 'OBJECT')
    assert obj['f1'] == pytest.approx(2/3)
    assert summary['macro_f1'] == pytest.approx(1/3)


def test_non_entities_are_false_positives_not_an_ontology_class():
    gold = {('d',0,1):'OBJECT', ('d',2,3):'NON_ENTITY', ('d',4,5):'NON_ENTITY', ('d',6,7):'METHOD'}
    predictions = [dict(document_id='d',start=0,end=1,ner_label_hint='OBJECT',score=.9),
                   dict(document_id='d',start=2,end=3,ner_label_hint='OBJECT',score=.8)]
    summary, per, matrix, _ = mod.score(gold, predictions)
    assert summary['coverage'] == .5
    assert summary['typing_accuracy'] == 1
    assert summary['typing_accuracy_all_matched'] == .5
    assert summary['correct_fraction'] == .25
    assert summary['correct_entity_fraction'] == .5
    assert (summary['gold_nonentity'], summary['matched_nonentity']) == (2, 1)
    assert matrix['NON_ENTITY']['OBJECT'] == 1 and matrix['NON_ENTITY']['MISSED'] == 1
    obj = next(row for row in per if row['type'] == 'OBJECT')
    assert (obj['tp'], obj['fp'], obj['fn']) == (1, 1, 0)
    assert len(per) == 11 and 'NON_ENTITY' not in mod.SCHEMAS['V0']


def test_split_is_deterministic_and_document_level():
    docs = [{'document_id':str(i), 'discipline':str(i//5)} for i in range(30)]
    split = mod.split_docs(docs)
    assert split == mod.split_docs(list(reversed(docs)))
    assert list(split.values()).count('test') == 6
    assert len(split) == 30


def test_gold_rejects_missing_extra_and_wrong_surface(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, 'identity', lambda r: (r['document_id'],r['start'],r['end']))
    # The real contract requires all 1211 exact spans.
    docs = [dict(document_id='d', text='x'*1211)]
    candidates = [dict(document_id='d',start=i,end=i+1) for i in range(1211)]
    rows = [dict(**r, entity_type='OBJECT',surface_text='x') for r in candidates]
    path = tmp_path/'gold.jsonl'
    path.write_text('\n'.join(map(json.dumps,rows)),encoding='utf8')
    assert len(mod.load_gold(path,docs,candidates)) == 1211
    rows[-1]['surface_text'] = 'y'
    path.write_text('\n'.join(map(json.dumps,rows)),encoding='utf8')
    with pytest.raises(ValueError, match='surface mismatch'): mod.load_gold(path,docs,candidates)
    rows.pop()
    path.write_text('\n'.join(map(json.dumps,rows)),encoding='utf8')
    with pytest.raises(ValueError, match='missing=1'): mod.load_gold(path,docs,candidates)


def test_adaptive_windows_share_all_schema_budgets():
    class Record:
        def __init__(self, n, words, text):
            self.input_ids = [0]*n
            self.text = text
            self.start_token_idx = [w[1] for w in words]
            self.end_token_idx = [w[2] for w in words]
    class Processor:
        def word_splitter(self, text, lower=True): return [(m.group(),m.start(),m.end()) for m in mod.re.finditer(r'\w+|[^\w\s]',text)]
        def transform_record(self,text,schema,max_len=None):
            text = text if text.endswith(('.', '!', '?')) else text + '.'
            words = self.word_splitter(text)
            return Record(len(words)*5 + schema, words, text)
    text = ' '.join('word' for _ in range(110))
    spans = mod.shared_windows(text,Processor(),[0,10,20])
    assert len(spans) > 1
    assert all(max(sizes) <= 512 for _,_,sizes in spans)
    assert all(any(a <= w.start() and w.end() <= b for a,b,_ in spans) for w in mod.re.finditer(r'\S+',text))


def test_window_validation_rejects_real_truncation():
    class Record:
        text = 'first second.'
        input_ids = [1, 2]
        start_token_idx = [0]
        end_token_idx = [5]
    class Processor:
        def word_splitter(self, text, lower=True):
            return [(m.group(), m.start(), m.end()) for m in mod.re.finditer(r'\w+|[^\w\s]', text)]
        def transform_record(self, text, schema, max_len=None): return Record()
    with pytest.raises(ValueError, match='truncated'):
        mod.shared_windows('first second', Processor(), [{}])


def test_source_prediction_drops_only_gliner_added_period():
    result = lambda a, b, s: {'start': a, 'end': b, 'text': s, 'confidence': .8}
    assert mod.source_prediction('alpha', result(0, 6, 'alpha.'), 'object') is None
    assert mod.source_prediction('alpha', result(5, 6, '.'), 'object') is None
    assert mod.source_prediction('alpha.', result(0, 6, 'alpha.'), 'object')['end'] == 6
    assert mod.source_prediction('alpha', result(0, 5, 'alpha'), 'object')['end'] == 5


def test_source_prediction_checks_text_and_adjusts_stripped_whitespace():
    raw = {'start': 0, 'end': 7, 'text': 'alpha', 'confidence': .8}
    result = mod.source_prediction(' alpha ', raw, 'object')
    assert (result['start'], result['end'], result['text']) == (1, 6, 'alpha')
    raw['text'] = 'other'
    with pytest.raises(ValueError, match='inconsistent'):
        mod.source_prediction(' alpha ', raw, 'object')
    raw.update(start=0, end=50, text='alpha')
    with pytest.raises(ValueError, match='inconsistent'):
        mod.source_prediction(' alpha ', raw, 'object')


def test_resume_preserves_compatible_completed_checkpoints(tmp_path):
    old = dict(script_sha256='a0fda94f0835fbc16178d0deb19636681f42bbcb40f2d2a4a2ecb9b35317f88f', config_hash='old', threshold=.35)
    mod.write_json(tmp_path/'manifest.json', old)
    checkpoint = tmp_path/'dev/V0/doc.json'
    mod.write_json(checkpoint, {'config_hash': 'old'})
    current = dict(script_sha256='new', config_hash='irrelevant', threshold=.35)
    migrated = mod.freeze_manifest(tmp_path, current, True)
    assert migrated['compatible_checkpoint_hashes'] == ['old']
    assert mod.read_json(checkpoint)['config_hash'] == 'old'
    assert mod.read_json(tmp_path/'manifest_preflight_old.json') == old
    resumed = mod.freeze_manifest(tmp_path, dict(script_sha256='new', threshold=.35), True)
    assert resumed == migrated
    with pytest.raises(ValueError, match='manifest mismatch'):
        mod.freeze_manifest(tmp_path, dict(script_sha256='new', threshold=.50), True)


def test_resume_keeps_all_prior_compatible_hashes(tmp_path):
    old = dict(script_sha256='a374d5e9cabed5675e3783b18baece123235405381482b25e476bddb72e1e87f',
               config_hash='previous', compatible_checkpoint_hashes=['earlier'], threshold=.35)
    mod.write_json(tmp_path/'manifest.json', old)
    mod.write_json(tmp_path/'dev/V0/doc.json', {'config_hash': 'earlier'})
    result = mod.freeze_manifest(tmp_path, dict(script_sha256='new', threshold=.35), True)
    assert result['compatible_checkpoint_hashes'] == ['previous', 'earlier']
