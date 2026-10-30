"""Behavioral checks on invented text, not DEV or Gold."""
import pytest
from cgraph.common import read_json, sha256
from cgraph.inputs import load_ontology
from bgraph.contract import schema
from bgraph.mapping import process
from bgraph.windows import anchor, windows


def entity(s, k, n=0, o=1):
    return dict(s=s, k=k, n=n, o=o, r='UNKNOWN')


def relation(h, t, r='DETECTS', p='POSITIVE', q=None, d='ASSERT'):
    return dict(h=h, t=t, r=r, e=[0], p=p, m='ASSERTED', q=q or [], d=d)


def run(text, entities, relations, unresolved=None):
    chunk = dict(chunk_id='chunk', text=text, text_sha256=sha256(text.encode()), section='abstract',
                 ordinal=0, section_start=0, section_end=len(text), language='en')
    doc = dict(document_id='doc', revision_id='revision')
    rows = [dict(id=0, start=0, end=len(text), text=text)]
    response = dict(entities=entities, relations=relations, unresolved=unresolved or [])
    return process(doc, chunk, [(rows, response)], load_ontology(read_json('config/b001.json')))


def test_repeat_and_nested_are_exact():
    text = 'NO2 and NO2 on graphene oxide and oxide.'
    rows = [dict(id=0, start=0, end=len(text), text=text)]
    assert anchor(dict(s='NO2', n=0, o=2), rows) == (8, 11)
    assert anchor(dict(s='graphene oxide', n=0, o=22), rows) == (15, 29)
    with pytest.raises(ValueError):
        anchor(dict(s='NO2', n=0, o=3), rows)
    out = run(text, [entity('graphene oxide', 'MATERIAL'), entity('oxide', 'MATERIAL')], [])
    assert len(out['entities']) == 2
    assert out['entities'][1]['start'] > out['entities'][0]['start']


def test_chemical_no_is_not_negation_and_negation_conflict_abstains():
    out = run('Sensor detects NO.', [entity('Sensor', 'DEVICE'), entity('NO', 'ANALYTE')], [relation(0, 1)])
    assert len(out['assertions']) == 1
    out = run('Sensor does not detect NO.', [entity('Sensor', 'DEVICE'), entity('NO', 'ANALYTE')], [relation(0, 1)])
    assert not out['assertions']
    assert 'EXPLICIT_NEGATION_CONFLICT' in out['rejected'][0]['reason']
    out = run('Sensor does not detect NO.', [entity('Sensor', 'DEVICE'), entity('NO', 'ANALYTE')], [relation(0, 1, p='NEGATIVE')])
    assert out['assertions'][0]['polarity'] == 'NEGATIVE'


def test_direction_and_unknown_qualifier_are_not_invented():
    text = 'Sensor detects NH3 at 25 °C.'
    entities = [entity('Sensor', 'DEVICE'), entity('NH3', 'ANALYTE')]
    q = dict(s='25 °C', n=0, o=1, k='temperature', v='wrong', u='°C')
    out = run(text, entities, [relation(0, 1, q=[q]), relation(1, 0)])
    assert len(out['assertions']) == 1
    assert out['assertions'][0]['qualifier_status'] == 'UNKNOWN'
    assert out['assertions'][0]['qualifiers'] == []
    assert out['unresolved'][0]['reason'] == 'UNSUPPORTED'
    assert any('ONTOLOGY_DOMAIN_RANGE' in x['reason'] for x in out['rejected'])


def test_model_abstention_preserved():
    out = run('Sensor detects NH3.', [entity('Sensor', 'DEVICE'), entity('NH3', 'ANALYTE')],
              [relation(0, 1, d='ABSTAIN')])
    assert len(out['raw_proposals']) == 1 and not out['assertions']
    assert out['rejected'][0]['reason'] == 'MODEL_ABSTAIN'


def test_windows_cover_sentences_with_overlap_and_schema_frozen():
    text = 'One sentence. Another sentence. Third sentence. Fourth sentence.'
    result = list(windows(text, dict(sentences=3, overlap=1, max_chars=1000)))
    assert [r['id'] for r in result[0]] == [0, 1, 2]
    assert [r['id'] for r in result[1]] == [2, 3]
    cfg = read_json('config/b001.json')
    assert schema(load_ontology(cfg)) == read_json(cfg['schema'])
