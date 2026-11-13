"""Source-provenance and fixed-target checks, with no network/model/database."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import experiment_split_context as run


@pytest.fixture
def pair():
    docs = {}
    record = dict(audit_id='test', evidence_ids=['source-ref'])
    for slot, did, eid, text in [('E2', 'a', 'x', 'First. The results improved. More.'),
                                  ('E3', 'b', 'y', 'Other results declined.')]:
        docs[did] = dict(document_id=did, text=text, revision_id='r'+did,
                         source_url=did, source_reference={'provider': 'fixture'})
        start = text.index('results')
        record[slot] = dict(profile=dict(canonical_entity_id=eid, document_id=did,
            selected_source_entity_id=eid, source_entity_ids=[eid], source_mention_ids=['m'+eid],
            mention_start=start, mention_end=start+7, mention_text='results', entity_types=['RESULT']))
    return record, docs


def test_disjoint_passages_preserve_native_and_joint_offsets_and_distinct_targets(pair):
    record, docs = pair
    ranges = {'E2': [(0, 6), (7, len(docs['a']['text']))], 'E3': [(0, len(docs['b']['text']))]}
    item = run.build(record, docs, ranges, 'S3')
    assert '[PASSAGE BREAK]' in item['text']
    assert len(item['fixed_spans']) == 2
    assert [s['entity_id'] for s in item['fixed_spans']] == ['x', 'y']
    assert len({(s['start'], s['end']) for s in item['fixed_spans']}) == 2
    for sp in item['fixed_spans']:
        assert item['text'][sp['start']:sp['end']] == 'results'
    for f in item['source_fragments']:
        assert docs[f['document_id']]['text'][f['start']:f['end']] == f['text']
        assert item['text'][f['model_text_start']:f['model_text_end']] == f['text']
    assert '[END DOCUMENT a]\n\n[DOCUMENT b]' in item['text']


def test_missing_target_is_rejected_and_overlap_is_not_a_second_target(pair):
    record, docs = pair
    with pytest.raises(ValueError, match='exactly one'):
        run.build(record, docs, {'E2': [(0, 6)], 'E3': [(0, len(docs['b']['text']))]}, 'S1')
    ranges = {'E2': [(0, len(docs['a']['text'])), (7, 28)], 'E3': [(0, len(docs['b']['text']))]}
    item = run.build(record, docs, ranges, 'S3')
    assert len(item['fixed_spans']) == 2
    assert len(item['source_fragments']) == 2


def test_metadata_answers_never_enter_model_text_or_input_hash(pair):
    record, docs = pair
    ranges = {s: [(0, len(docs[record[s]['profile']['document_id']]['text']))] for s in ('E2', 'E3')}
    first = run.build(record, docs, ranges, 'S1')
    changed = copy.deepcopy(record)
    changed.update(classification='STRONG_CANDIDATE', expected_relation='DERIVED_FROM', audit_reason='ANSWER_LEAK')
    second = run.build(changed, docs, ranges, 'S3', [{'canonical_triple_id': 'ct:secret'}])
    assert first['text'] == second['text']
    assert first['input_sha256'] == second['input_sha256']
    assert 'ANSWER_LEAK' not in second['text'] and 'ct:secret' not in second['text']


def test_sentence_and_tiny_window_keep_complete_wide_target(pair):
    record, docs = pair
    p = record['E2']['profile']
    a, b = run.minimal_range(docs['a'], p)
    assert docs['a']['text'][a:b].strip() == 'The results improved.'
    a, b = run.minimal_range(docs['a'], p, limit=0)
    assert docs['a']['text'][a:b] == 'results'
