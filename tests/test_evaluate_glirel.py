"""GLiREL inclusive-input/exclusive-output and source-crop regression checks."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import evaluate_glirel as run


def item():
    text='[DOCUMENT A]\nT-cells show activity.\n[END DOCUMENT A]\n[DOCUMENT B]\nOther T-cells respond.\n[END DOCUMENT B]'
    spans=[]
    for slot,start in [('E2',text.index('T-cells')),('E3',text.rindex('T-cells'))]:
        spans.append(dict(entity_id=slot,target_slot=slot,start=start,end=start+7,surface_text='T-cells',entity_types=['OBJECT']))
    return dict(text=text,fixed_spans=spans)


def test_same_surface_forms_keep_two_distinct_inclusive_token_spans():
    c=item();a=run.tokenize_context(c)
    assert len(a['ner'])==2 and a['ner'][0][:2]!=a['ner'][1][:2]
    for ent,sp in zip(a['ner'],c['fixed_spans']):
        assert a['tokens'][ent[0]:ent[1]+1]==['T-cells']
        assert c['text'][a['token_offsets'][ent[0]][0]:a['token_offsets'][ent[1]][1]]==sp['surface_text']


def test_api_exclusive_output_end_maps_back_to_inclusive_ner_and_missing_scores_stay_absent():
    a=run.tokenize_context(item());h,t=a['ner']
    raw=[dict(head_pos=[h[0],h[1]+1],tail_pos=[t[0],t[1]+1],head_text=['T-cells'],tail_text=['T-cells'],label='is part of',score=.1)]
    scores,missing=run.convert(raw,a,{'PART_OF':'is part of','SUBTYPE_OF':'is a type of'})
    assert scores[0]['head_slot']=='E2' and scores[0]['tail_slot']=='E3'
    assert len(missing)==3 and all(m['score'] is None for m in missing)
    raw[0]['head_pos'][1]+=1
    with pytest.raises(KeyError):run.convert(raw,a,{'PART_OF':'is part of'})


def test_non_aligned_character_span_fails_before_model_call():
    c=item();c['fixed_spans'][0].update(start=c['fixed_spans'][0]['start']+2,surface_text='cells')
    with pytest.raises(ValueError,match='token-aligned'):run.tokenize_context(c)


def test_symmetric_crop_preserves_both_complete_targets_and_source_substrings():
    source={}
    text='';spans=[];fragments=[];endpoints={}
    for s,d in [('E2','A'),('E3','B')]:
        passage='Many earlier words. T-cells respond to signals. More words later.'
        at=passage.index('T-cells')
        endpoints[s]=dict(document_id=d,canonical_entity_id=s,mention_start=at,mention_end=at+7,mention_text='T-cells',entity_types=['OBJECT'])
        fragments.append(dict(document_id=d,start=0,end=len(passage),text=passage))
        source[d]=passage
    c=dict(text=text,fixed_spans=spans,source_fragments=fragments,endpoints=endpoints)
    cropped=run.crop(c,.25);a=run.tokenize_context(cropped)
    assert len(a['ner'])==2
    for sp in cropped['fixed_spans']:assert cropped['text'][sp['start']:sp['end']]=='T-cells'
    for f in cropped['source_fragments']:assert source[f['document_id']][f['start']:f['end']]==f['text']
    assert cropped['text'].count('[DOCUMENT ')==2 and cropped['text'].count('[END DOCUMENT ')==2
