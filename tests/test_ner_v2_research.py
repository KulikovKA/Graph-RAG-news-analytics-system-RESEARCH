"""Meaningful source-integrity and conservative-normalization regression checks."""
import pytest

from cgraph.ner_v2.normalization import LinguisticNormalizer, detect_abbreviations
from cgraph.ner_v2.scientific_normalization import ScientificNormalizer
from cgraph.ner_v2.detection import postprocess


@pytest.fixture(scope='module')
def scientific():
    return ScientificNormalizer()


@pytest.mark.parametrize('surface,expected', [
    ('odds ratio','odds ratio'), ('weakly coupled plasmas','weakly coupled plasma'),
    ('reduced graphene oxide','reduced graphene oxide'), ('convolutional neural networks','convolutional neural network'),
    ('pH','pH'),('NH₃','NH₃'),('BERT','BERT'),('COVID-19','COVID-19'),('children','child'),
])
def test_conservative_semantics_and_source(scientific,surface,expected):
    text='We investigated '+surface+' in this study.'
    a=text.index(surface);b=a+len(surface)
    value=scientific.normalize(text,a,b)
    assert value['normalized_form']==expected
    assert value['surface_text']==text[a:b]==surface
    for token in value['tokens']:
        assert text[token['start']:token['end']]==token['text']


def test_partial_token_never_inflects(scientific):
    text='The networks were studied.'
    a=text.index('networks')
    result=scientific.normalize(text,a+1,a+8)
    assert result['normalized_form']=='etworks'
    assert result['tokens'][0]['partial_token']


def test_document_definitions_survive_ner_absence_and_ambiguity():
    text='principal component analysis (PCA) differs from patient-controlled analgesia (PCA).'
    definitions=detect_abbreviations(text)
    assert {x['long_form'] for x in definitions}=={'principal component analysis','patient-controlled analgesia'}
    for x in definitions:
        assert text[x['short_start']:x['short_end']]==x['short_form']
        assert text[x['long_start']:x['long_end']]==x['long_form']
        assert x['scope']=='document_candidate_only'
    assert detect_abbreviations('PCA without a definition.')==[]


def test_overlap_and_same_span_alternate_hints_are_preserved():
    text='neural networks'
    rows=[dict(start=0,end=15,text=text,label='object',score=.5),
          dict(start=0,end=15,text=text,label='method',score=.6),
          dict(start=7,end=15,text='networks',label='object',score=.7)]
    result=postprocess(text,[(0,rows),(0,[{**rows[0],'score':.8}])],{'object':'OBJECT','method':'METHOD'})
    assert len(result)==3
    assert result[1]['score']==.8


def test_historical_algorithm_still_exhibits_regression():
    baseline=LinguisticNormalizer()
    text='The odds ratio was recorded.'
    a=text.index('odds ratio')
    assert baseline.normalize(text,a,a+10)['normalized_form']=='odd ratio'


def test_research_cli_refuses_changed_frozen_input(monkeypatch,tmp_path):
    import cgraph.ner_v2.research as research
    import sys
    source=tmp_path/'source.jsonl'
    source.write_text('changed input',encoding='utf8')
    monkeypatch.setattr(research,'read_json',lambda path:{'input':str(source),'input_sha256':'0'*64})
    monkeypatch.setattr(sys,'argv',['research','ner','--config','E1','--output-root',str(tmp_path/'output')])
    with pytest.raises(ValueError,match='Frozen corpus changed'):
        research.main()
    assert not (tmp_path/'output').exists()
