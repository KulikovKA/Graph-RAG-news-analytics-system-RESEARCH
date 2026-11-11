"""Stage 2F.2: local Global ER proposal on the fixed NER .46 graph.

No inference, cosine calculation, database writes or import. This script reuses
the existing JSONL/CSV readers, normalization and connectivity implementation.
"""
import argparse
import csv
from collections import Counter, defaultdict
import itertools
import importlib.util
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode=True
# A whole exact-name group includes lossless member provenance in one CSV field.
csv.field_size_limit(16 * 1024 * 1024)
from exact_matching_audit import normalize, read_jsonl, sha256, ROOT
from conservative_entity_resolution import read_csv
from import_global_semantic_graph import connectivity

ER=ROOT/'reports/entity_resolution/relex_base_t035'
OUT=ER/'global_entity_resolution'
KG=ER/'ner_threshold_analysis/global_kg_t046_max'
ENTITIES=ER/'semantic_resolution/input/canonical_entities.jsonl'
RELATIONS=ER/'relation_remapping/canonical_relations.jsonl'
LINKS=ER/'semantic_resolution/global_linking_preview/semantic_links.csv'
DOCS=ROOT/'reports/ner_v2/research_run2/input_abstracts.jsonl'
PRIOR=ER/'semantic_resolution/candidate_verification/verified_pairs.csv'
NS='relex_sem_e5_small_name_t095_ner_t046_max'


def require(ok,message):
    if not ok:raise ValueError(message)


def write_json(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def write_jsonl(name,rows):
    with (OUT/name).open('w',encoding='utf-8') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')


def write_csv(name,rows):
    import csv
    require(bool(rows),'Empty CSV '+name)
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)


def load():
    membership=json.loads((KG/'membership.json').read_text(encoding='utf-8'))
    ids=set(membership['entity_ids'])
    entities=[e for e in read_jsonl(ENTITIES) if e['canonical_entity_id'] in ids]
    entities.sort(key=lambda e:(e['document_id'],min(x['start'] for x in e['source_offsets']),e['canonical_entity_id']))
    by={e['canonical_entity_id']:e for e in entities}
    aliases={eid:f'N{i:03}' for i,eid in enumerate(by,1)}
    facts=[r for r in read_jsonl(RELATIONS) if r['canonical_triple_id'] in set(membership['relex_rel_ids'])]
    links=[r for r in read_csv(LINKS) if r['semantic_link_id'] in set(membership['similar_to_ids'])
           and (r['model'],r['mode'],float(r['threshold']))==('intfloat/e5-small-v2','text_name',.95)]
    documents={d['document_id']:d for d in read_jsonl(DOCS)}
    require(len(by)==len(entities)==905 and len(facts)==1307 and len(links)==492 and len(documents)==30,'Wrong baseline')
    require(all(r['head_id'] in ids and r['tail_id'] in ids for r in facts),'Invalid fact endpoints')
    require(all(r['left_canonical_id'] in ids and r['right_canonical_id'] in ids and r['model']=='intfloat/e5-small-v2' and float(r['threshold'])==.95 for r in links),'Invalid semantic baseline')
    nodes={e['canonical_entity_id']:e['document_id'] for e in entities}
    baseline=connectivity(nodes,[(r['head_id'],r['tail_id']) for r in facts]+[(r['left_canonical_id'],r['right_canonical_id']) for r in links])
    require(baseline==dict(entities=905,isolated=175,weak_components=230,largest_component_vertices=378,largest_component_documents=17),'Baseline connectivity mismatch')
    require(sum(r['scope']=='cross_document' for r in links)==121,'Baseline cross-link mismatch')
    for e in entities:
        require(max(m['source_entity']['confidence'] for m in e['members'])>=.46,'Entity below current NER threshold')
        for m in e['members']:
            source=m['source_entity'];ctx=m['context'];doc=documents[e['document_id']]
            require(source['document_id']==e['document_id'] and source['surface_text']==doc['text'][source['start']:source['end']],'Source span mismatch')
            require(ctx['text']==doc['text'][ctx['start']:ctx['end']],'Source context mismatch')
            require(all(mention['revision_id']==doc['revision_id'] for mention in source['source_mentions']),'Source revision mismatch')
    grouped=defaultdict(list)
    for e in entities:grouped[normalize(e['surface_text'])].append(e)
    groups={name:members for name,members in sorted(grouped.items()) if len(members)>1}
    norm={eid:normalize(e['surface_text']) for eid,e in by.items()}
    exact_links=[r for r in links if norm[r['left_canonical_id']]==norm[r['right_canonical_id']]]
    audit=dict(duplicate_groups=len(groups),vertices_in_duplicate_groups=sum(len(g) for g in groups.values()),
               exact_name_similar_to=len(exact_links),cross_document_exact_name_similar_to=sum(r['scope']=='cross_document' for r in exact_links))
    require(audit==dict(duplicate_groups=96,vertices_in_duplicate_groups=259,exact_name_similar_to=349,cross_document_exact_name_similar_to=91),'Duplicate preliminary counts differ')
    return entities,by,aliases,facts,links,documents,groups,baseline,audit


def inventory():
    entities,by,aliases,facts,links,docs,groups,baseline,audit=load()
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for e in entities:
        eid=e['canonical_entity_id']
        rows.append(dict(alias=aliases[eid],canonical_entity_id=eid,document_id=e['document_id'],name=e['surface_text'],entity_types=e['entity_types'],
                         confidence=max(m['source_entity']['confidence'] for m in e['members']),source_entity_ids=e['source_entity_ids'],
                         source_mention_ids=e['source_mention_ids'],source_offsets=e['source_offsets'],quality_flags=e['quality_flags'],
                         source_revision=e['provenance']['document']['revision_id'],discipline=e['provenance']['document']['discipline'],
                         member_contexts=[dict(source_entity_id=m['source_entity']['entity_id'],name=m['source_entity']['surface_text'],
                                              confidence=m['source_entity']['confidence'],types=m['source_entity']['entity_types'],
                                              start=m['source_entity']['start'],end=m['source_entity']['end'],context=m['context']['text']) for m in e['members']],
                         incident_facts=[dict(id=r['canonical_triple_id'],head=aliases[r['head_id']],relation=r['relation_type'],tail=aliases[r['tail_id']]) for r in facts if eid in (r['head_id'],r['tail_id'])]))
    write_csv('entity_inventory.csv',rows)
    write_json('baseline_audit.json',dict(namespace=NS,counts=dict(documents=30,entities=905,relex_rel=1307,similar_to=492,cross_document_similar_to=121),connectivity=baseline,duplicate_audit=audit,
                                         sources={p.relative_to(ROOT).as_posix():sha256(p) for p in [ENTITIES,RELATIONS,LINKS,DOCS,KG/'membership.json',KG/'manifest.json',PRIOR]}))
    candidates={}
    def add(a,b,origin,evidence=None):
        if a==b:return
        key=tuple(sorted((a,b)))
        entry=candidates.setdefault(key,dict(origins=set(),evidence=[]))
        entry['origins'].add(origin)
        if evidence:entry['evidence'].append(evidence)
    for name,g in groups.items():
        for a,b in itertools.combinations(g,2):add(a['canonical_entity_id'],b['canonical_entity_id'],'exact_name')
    for r in links:add(r['left_canonical_id'],r['right_canonical_id'],'existing_SIMILAR_TO',dict(semantic_link_id=r['semantic_link_id'],similarity=r['similarity'],scope=r['scope']))
    for r in read_csv(PRIOR):
        a,b=r['left_canonical_id'],r['right_canonical_id']
        if a in by and b in by and r['individually_reviewed']=='True':add(a,b,'prior_2C_review',dict(decision=r['decision'],reason=r['reason'],referent_level=r['referent_level']))
    abbrevs=set()
    for e in entities:
        for m in e['members']:
            for mention in m['source_entity']['source_mentions']:
                for a in mention.get('abbreviation_candidates',[]):
                    abbrevs.add((e['document_id'],a['long_form'],a['short_form'],a['long_start'],a['long_end'],a['short_start'],a['short_end']))
    for doc,long,short,ls,le,ss,se in sorted(abbrevs):
        longs=[e for e in entities if e['document_id']==doc and (normalize(e['surface_text']) in {normalize(long),normalize(f'{long} ({short})')}
               or any(x['start']==ls and x['end']==le for x in e['source_offsets']))]
        shorts=[e for e in entities if e['document_id']==doc and (normalize(e['surface_text'])==normalize(short) or any(x['start']==ss and x['end']==se for x in e['source_offsets']))]
        for a,b in itertools.product(longs,shorts):add(a['canonical_entity_id'],b['canonical_entity_id'],'explicit_abbreviation_candidate',dict(document_id=doc,long_form=long,short_form=short,long_offsets=[ls,le],short_offsets=[ss,se]))
    rows=[]
    for (a,b),r in sorted(candidates.items(),key=lambda p:(aliases[p[0][0]],aliases[p[0][1]])):
        rows.append(dict(left_alias=aliases[a],right_alias=aliases[b],left_id=a,right_id=b,left_name=by[a]['surface_text'],right_name=by[b]['surface_text'],
                         left_document=by[a]['document_id'],right_document=by[b]['document_id'],origins=sorted(r['origins']),evidence=r['evidence']))
    write_csv('candidate_inventory.csv',rows)
    print(json.dumps(dict(**audit,candidate_pairs=len(rows),candidate_origins=Counter(o for row in rows for o in row['origins'])),ensure_ascii=False))


def plan_module():
    spec=importlib.util.spec_from_file_location('global_er_context_review',OUT/'review_plan.py')
    plan=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plan)
    for path,digest in plan.SNAPSHOT_SHA256.items():require(sha256(ROOT/path)==digest,'Review source changed: '+path)
    return plan


def graph_metrics(node_documents,edges):
    # Reuse the existing DFS with component membership; concept nodes have no
    # document of their own and must not add a fictitious document to counts.
    spec=importlib.util.spec_from_file_location('ner_confidence_connectivity',ER/'ner_threshold_analysis/analyze.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    props={eid:dict(document_id=next(iter(sorted(documents)),None)) for eid,documents in node_documents.items()}
    counts,groups=module.connectivity(set(props),edges,props)
    counts['largest_component_documents']=len(set().union(*(node_documents[eid] for eid in groups[0]))) if groups else 0
    return dict(vertices=len(props),**counts)


def resolve():
    plan=plan_module()
    protected={p for p in ER.rglob('*') if p.is_file() and not p.is_relative_to(OUT)}|{DOCS,Path(__file__)}
    before={p:sha256(p) for p in sorted(protected)}
    entity_input_hash=before[ENTITIES]
    entities,by,aliases,facts,links,documents,groups,baseline,audit=load()
    require(set(groups)==set(plan.GROUPS),'Not all exact-name groups were reviewed')
    inventory()
    inv={r['alias']:r for r in read_csv(OUT/'entity_inventory.csv')}
    id_by_alias={v:k for k,v in aliases.items()}
    candidate={tuple(sorted((r['left_alias'],r['right_alias']))):r for r in read_csv(OUT/'candidate_inventory.csv')}
    manual_pairs={}
    for left,right,decision,reason in plan.PAIR_RULES:
        for a,b in itertools.product(left.split(),right.split()):
            if a==b:continue
            pair=tuple(sorted((a,b)))
            if pair in manual_pairs:
                require(manual_pairs[pair][0]==decision,'Conflicting manual pair rules '+str(pair))
                manual_pairs[pair]=(decision,manual_pairs[pair][1]+' '+reason)
            else:manual_pairs[pair]=(decision,reason)
    overrides={tuple(sorted((a,b))):(d,r) for a,b,d,r in plan.PAIR_OVERRIDES}
    merge_by_alias={}
    merge_pairs={}
    merge_groups=[]
    mapping={eid:eid for eid in by}
    for number,(member_text,level,reason,proposed_type) in enumerate(plan.MERGES,1):
        members=member_text.split()
        require(len(members)>1 and len(set(members))==len(members),'Invalid clique')
        require(not set(members)&set(merge_by_alias),'Overlapping merge cliques; no transitive closure allowed')
        require(all(a in inv for a in members),'Unknown alias in clique')
        ids=[id_by_alias[a] for a in members]
        require(len({by[eid]['document_id'] for eid in ids})==1,'This proposal has no cross-document identity proof')
        representative=min(ids,key=lambda eid:(by[eid]['document_id'],min(x['start'] for x in by[eid]['source_offsets']),-len(by[eid]['surface_text']),eid))
        gid=f'M{number:03}'
        row=dict(merge_group_id=gid,referent_level=level,reason=reason,proposed_primary_type=proposed_type or None,
                 representative_id=representative,representative_alias=aliases[representative],member_aliases=members,member_canonical_ids=ids,
                 source_entity_ids=sorted({x for eid in ids for x in by[eid]['source_entity_ids']}),
                 source_mention_ids=sorted({x for eid in ids for x in by[eid]['source_mention_ids']}),
                 document_id=by[representative]['document_id'],
                 original_type_sets={a:by[id_by_alias[a]]['entity_types'] for a in members},
                 type_set_conflict=len({tuple(by[eid]['entity_types']) for eid in ids})>1,
                 original_multitype=any(len(by[eid]['entity_types'])>1 for eid in ids))
        merge_groups.append(row)
        for a,eid in zip(members,ids):merge_by_alias[a]=row;mapping[eid]=representative
        for pair in itertools.combinations(sorted(members),2):
            require(pair not in overrides or overrides[pair][0]=='MERGE','Clique contradicts explicit pair review')
            require(pair not in manual_pairs or manual_pairs[pair][0] in ('MERGE','SHARED_CONCEPT'),'Clique contradicts a specific negative/uncertain review')
            merge_pairs[pair]=row
            if pair not in candidate:
                a,b=pair
                candidate[pair]=dict(left_alias=a,right_alias=b,left_id=id_by_alias[a],right_id=id_by_alias[b],
                                     left_name=inv[a]['name'],right_name=inv[b]['name'],left_document=inv[a]['document_id'],right_document=inv[b]['document_id'],
                                     origins=json.dumps(['explicit_reviewed_clique']),evidence='[]')
    require(all(mapping[mapping[eid]]==mapping[eid] for eid in mapping),'Mapping is not idempotent')
    concepts=[]
    concepts_for_alias=defaultdict(set)
    concept_members={}
    for name,member_text,reason in plan.CONCEPTS:
        members=member_text.split()
        require(all(a in inv for a in members),'Unknown concept member')
        cid='concept:2f2:'+re.sub('[^a-z0-9]+','_',name.lower()).strip('_')
        require(cid not in concept_members,'Concept ID collision')
        concept_members[cid]=members
        targets=sorted({mapping[id_by_alias[a]] for a in members})
        require(len(targets)>1,'Unnecessary singleton concept after merge')
        for a in members:concepts_for_alias[a].add(cid)
        concepts.append(dict(concept_id=cid,label=name,decision='SHARED_CONCEPT',reason=reason,relation_type='HAS_SHARED_CONCEPT',
                             source_aliases=members,source_canonical_ids=[id_by_alias[a] for a in members],
                             contextual_target_ids=targets,contextual_entity_count=len(targets),
                             document_ids=sorted({inv[a]['document_id'] for a in members}),
                             source_type_sets={a:json.loads(inv[a]['entity_types']) for a in members},
                             fact_inheritance_allowed=False,external_ontology_id=None))
    for pair,row in candidate.items():
        if pair in merge_pairs:
            old=[r for r in json.loads(row['evidence']) if r.get('decision')=='REJECT']
            require(not old,'Merge contradicts previous 2C explicit REJECT: '+str(pair))
    decisions=[]
    unresolved=[]
    for pair,row in sorted(candidate.items()):
        a,b=pair
        ea,eb=by[id_by_alias[a]],by[id_by_alias[b]]
        evidence=json.loads(row['evidence'])
        priors=[r for r in evidence if 'decision' in r]
        common_concepts=sorted(concepts_for_alias[a]&concepts_for_alias[b])
        same_name=normalize(ea['surface_text'])==normalize(eb['surface_text'])
        if pair in merge_pairs:
            decision,reason,level='MERGE',merge_pairs[pair]['reason'],merge_pairs[pair]['referent_level']
            method='explicit_all_pairs_context_clique'
        elif pair in overrides:
            decision,reason=overrides[pair];level='explicit_pair_scope';method='specific_pair_context_review'
        elif pair in manual_pairs:
            decision,reason=manual_pairs[pair];level='explicit_pair_scope';method='specific_pair_context_review'
        elif same_name:
            decision,level,reason=plan.GROUPS[normalize(ea['surface_text'])]
            require(decision!='MERGE','Exact-name MERGE missing explicit clique '+str(pair))
            method='complete_exact_group_context_review'
        elif priors:
            prior=priors[0]
            decision='REJECT' if prior['decision']=='REJECT' else 'REVIEW'
            reason='Сохранено отдельное контекстное основание 2C: '+prior['reason']
            if prior['decision']=='ACCEPT':reason+=' В текущей MERGE-симуляции тождество на выбранном уровне отдельно не подтверждено.'
            level=prior['referent_level'];method='prior_2C_evidence_reconsidered'
        else:
            decision='REVIEW';method='specific_pair_insufficient_evidence';level='unresolved'
            reason=f"В сохранённых контекстах «{ea['surface_text']}» и «{eb['surface_text']}» нет явного определения или идентификатора, подтверждающего равенство референтов с сохранением модификаторов. Cosine не является доказательством."
            unresolved.append(pair)
        require(decision in {'MERGE','SHARED_CONCEPT','RELATED','REJECT','REVIEW'},'Unknown decision')
        require(decision!='SHARED_CONCEPT' or common_concepts,'Shared-concept pair without a supported common concept '+str(pair))
        ctx=lambda e:[dict(source_entity_id=m['source_entity']['entity_id'],start=m['source_entity']['start'],end=m['source_entity']['end'],
                           confidence=m['source_entity']['confidence'],entity_types=m['source_entity']['entity_types'],
                           context_start=m['context']['start'],context_end=m['context']['end'],text=m['context']['text']) for m in e['members']]
        decisions.append(dict(candidate_id='gerpair:'+hashlib.sha256('|'.join(sorted((row['left_id'],row['right_id']))).encode()).hexdigest()[:24],
                              left_alias=a,right_alias=b,left_canonical_id=id_by_alias[a],right_canonical_id=id_by_alias[b],
                              left_name=ea['surface_text'],right_name=eb['surface_text'],left_document_id=ea['document_id'],right_document_id=eb['document_id'],
                              scope='within_document' if ea['document_id']==eb['document_id'] else 'cross_document',same_normalized_name=same_name,
                              decision=decision,referent_level=level,reason=reason,review_method=method,
                              common_concept_ids=common_concepts,merge_group_id=merge_pairs[pair]['merge_group_id'] if pair in merge_pairs else None,
                              left_target_id=mapping[id_by_alias[a]],right_target_id=mapping[id_by_alias[b]],
                              left_types=ea['entity_types'],right_types=eb['entity_types'],type_set_conflict=set(ea['entity_types'])!=set(eb['entity_types']),
                              types_disjoint=not set(ea['entity_types'])&set(eb['entity_types']),
                              left_max_confidence=max(m['source_entity']['confidence'] for m in ea['members']),
                              right_max_confidence=max(m['source_entity']['confidence'] for m in eb['members']),
                              left_offsets=ea['source_offsets'],right_offsets=eb['source_offsets'],left_quality_flags=ea['quality_flags'],right_quality_flags=eb['quality_flags'],
                              left_contexts=ctx(ea),right_contexts=ctx(eb),left_source_entity_ids=ea['source_entity_ids'],right_source_entity_ids=eb['source_entity_ids'],
                              left_source_mention_ids=ea['source_mention_ids'],right_source_mention_ids=eb['source_mention_ids'],
                              left_revision=ea['provenance']['document']['revision_id'],right_revision=eb['provenance']['document']['revision_id'],
                              left_source_reference=ea['provenance']['document']['source_reference'],right_source_reference=eb['provenance']['document']['source_reference'],
                              left_incident_factual_ids=[r['id'] for r in json.loads(inv[a]['incident_facts'])],
                              right_incident_factual_ids=[r['id'] for r in json.loads(inv[b]['incident_facts'])],
                              candidate_origins=json.loads(row['origins']),prior_evidence=evidence,
                              source_input=ENTITIES.relative_to(ROOT).as_posix(),source_sha256=entity_input_hash))
    print('Pairs needing additional individual review:',unresolved,flush=True)
    write_csv('resolution_decisions.csv',decisions)
    write_csv('concept_candidates.csv',concepts)
    write_json('merge_groups.json',merge_groups)
    decision_by_pair={tuple(sorted((r['left_canonical_id'],r['right_canonical_id']))):r for r in decisions}
    duplicates=[]
    for i,(name,members) in enumerate(groups.items(),1):
        ids=[e['canonical_entity_id'] for e in members]
        pairs=[decision_by_pair[tuple(sorted(p))] for p in itertools.combinations(ids,2)]
        doc_counts=Counter(e['document_id'] for e in members)
        incident_rels=[r for r in facts if r['head_id'] in ids or r['tail_id'] in ids]
        exact_sem=[r for r in links if r['left_canonical_id'] in ids and r['right_canonical_id'] in ids]
        duplicates.append(dict(group_id=f'G{i:03}',normalized_name=name,vertices=len(members),document_count=len(doc_counts),documents=dict(doc_counts),
                               within_document_repetition=any(n>1 for n in doc_counts.values()),cross_document_repetition=len(doc_counts)>1,
                               category=plan.GROUPS[name][1],group_reason=plan.GROUPS[name][2],decision_counts=dict(Counter(p['decision'] for p in pairs)),
                               member_aliases=[aliases[eid] for eid in ids],member_ids=ids,source_entity_ids=sorted({x for e in members for x in e['source_entity_ids']}),
                               source_mention_ids=sorted({x for e in members for x in e['source_mention_ids']}),
                               proposed_target_count=len({mapping[eid] for eid in ids}),eliminated_within_group=len(ids)-len({mapping[eid] for eid in ids}),
                               original_type_sets={aliases[eid]:by[eid]['entity_types'] for eid in ids},
                               type_set_conflict=len({tuple(e['entity_types']) for e in members})>1,original_multitype=any(len(e['entity_types'])>1 for e in members),
                               shared_concepts=sorted(set().union(*(concepts_for_alias[aliases[eid]] for eid in ids))),
                               similar_to=len(exact_sem),cross_document_similar_to=sum(r['scope']=='cross_document' for r in exact_sem),
                               incident_factual_count=len(incident_rels),incident_factual_ids=[r['canonical_triple_id'] for r in incident_rels],
                               members=[dict(alias=aliases[eid],canonical_entity=by[eid],referent_note=plan.ENTITY_NOTES.get(aliases[eid])) for eid in ids]))
    write_csv('duplicate_groups.csv',duplicates)
    proposed=[]
    for e in entities:
        eid=e['canonical_entity_id'];alias=aliases[eid]
        merge=merge_by_alias.get(alias)
        proposed.append(dict(canonical_entity_id=eid,alias=alias,proposed_final_id=mapping[eid],representative_id=mapping[eid],
                             identity_decision='MERGE' if merge else 'KEEP_SEPARATE',merge_group_id=merge['merge_group_id'] if merge else None,
                             merge_members=merge['member_canonical_ids'] if merge else [eid],
                             identity_reason=merge['reason'] if merge else 'Подтверждённая MERGE-группа не предложена; исходная contextual entity сохраняется отдельно.',
                             referent_level=merge['referent_level'] if merge else None,referent_note=plan.ENTITY_NOTES.get(alias),
                             concept_ids=sorted(concepts_for_alias[alias]),concept_relation_type='HAS_SHARED_CONCEPT',fact_inheritance_allowed=False,
                             proposed_primary_type=merge['proposed_primary_type'] if merge else None,original_entity_types_unchanged=e['entity_types'],
                             source_entity_ids=e['source_entity_ids'],source_mention_ids=e['source_mention_ids'],source_offsets=e['source_offsets'],
                             original_canonical_entity=e,source_input=ENTITIES.relative_to(ROOT).as_posix(),source_input_sha256=entity_input_hash,
                             candidate_decision_ids=[r['candidate_id'] for r in decisions if eid in (r['left_canonical_id'],r['right_canonical_id'])]))
    write_jsonl('proposed_mapping.jsonl',proposed)
    require(len(proposed)==905 and {r['canonical_entity_id'] for r in proposed}==set(by),'Incomplete mapping')
    require({r['canonical_entity_id']:r['original_canonical_entity'] for r in proposed}==by,'Source provenance changed in mapping')
    # Re-map facts only in memory. Preserve every current canonical relation AND
    # every raw relation already embedded in its source_relations provenance.
    fact_buckets=defaultdict(list)
    for r in facts:fact_buckets[(r['document_id'],mapping[r['head_id']],r['relation_type'],mapping[r['tail_id']])].append(r)
    fact_preview=[]
    loop_diagnostics=[]
    for (doc,head,typ,tail),source in sorted(fact_buckets.items()):
        fact_preview.append(dict(preview_relation_id='gerfact:'+hashlib.sha256(json.dumps([doc,head,typ,tail]).encode()).hexdigest()[:24],
                                 document_id=doc,head_id=head,relation_type=typ,tail_id=tail,is_self_loop=head==tail,
                                 original_canonical_relation_count=len(source),original_canonical_triple_ids=[r['canonical_triple_id'] for r in source],
                                 new_loop_source_ids=[r['canonical_triple_id'] for r in source if head==tail and r['head_id']!=r['tail_id']],
                                 original_relations=source))
        if head==tail:
            for r in source:
                if r['head_id']!=r['tail_id']:
                    a,b=aliases[r['head_id']],aliases[r['tail_id']]
                    loop_diagnostics.append(dict(canonical_triple_id=r['canonical_triple_id'],head_alias=a,head_name=by[r['head_id']]['surface_text'],
                                                 tail_alias=b,tail_name=by[r['tail_id']]['surface_text'],relation_type=typ,document_id=doc,
                                                 merge_group_id=merge_by_alias[a]['merge_group_id'],
                                                 identity_reason=merge_by_alias[a]['reason'],source_confidences=r['source_confidences'],
                                                 relation_review_required=True,
                                                 note='Предсказанное исходное отношение становится петлёй при контекстно подтверждённой идентичности. Не удалять и не объявлять корректным фактом: сохранить provenance для отдельной проверки отношения.'))
    restored={r['canonical_triple_id']:r for bucket in fact_preview for r in bucket['original_relations']}
    require(restored=={r['canonical_triple_id']:r for r in facts},'Factual provenance not exactly restorable')
    require(all(not r['head_id'].startswith('concept:') and not r['tail_id'].startswith('concept:') for r in fact_preview),'Facts inherited by concept')
    write_jsonl('preview_factual_relations.jsonl',fact_preview)
    write_csv('new_loop_diagnostics.csv',loop_diagnostics)
    semantic_preview=[]
    for r in links:
        a,b=r['left_canonical_id'],r['right_canonical_id']
        d=decision_by_pair[tuple(sorted((a,b)))]
        head,tail=mapping[a],mapping[b]
        status=('identity_redundant' if head==tail else 'replaceable_by_concept_path' if d['decision']=='SHARED_CONCEPT'
                else 'rejected_candidate' if d['decision']=='REJECT' else 'review_pending' if d['decision']=='REVIEW' else 'related_retained')
        semantic_preview.append(dict(semantic_link_id=r['semantic_link_id'],head_id=head,tail_id=tail,
                                     decision=d['decision'],status=status,decision_id=d['candidate_id'],common_concept_ids=d['common_concept_ids'],
                                     same_normalized_name=normalize(by[a]['surface_text'])==normalize(by[b]['surface_text']),
                                     original_cross_document=r['scope']=='cross_document',original_link=r))
    require(len(semantic_preview)==492,'Lost semantic provenance')
    write_jsonl('preview_semantic_links.jsonl',semantic_preview)
    concept_links=[]
    for c in concepts:
        for target in c['contextual_target_ids']:
            source_ids=[eid for eid in c['source_canonical_ids'] if mapping[eid]==target]
            concept_links.append(dict(concept_link_id='gerconcept:'+hashlib.sha256((target+'|'+c['concept_id']).encode()).hexdigest()[:24],
                                      relation_type='HAS_SHARED_CONCEPT',head_id=target,tail_id=c['concept_id'],
                                      source_canonical_ids=source_ids,document_ids=sorted({by[eid]['document_id'] for eid in source_ids}),
                                      reason=c['reason'],semantic_layer='context_to_abstraction',factual=False,identity=False,fact_inheritance_allowed=False))
    write_jsonl('preview_concept_links.jsonl',concept_links)
    nodes=defaultdict(set)
    for eid,e in by.items():nodes[mapping[eid]].add(e['document_id'])
    fact_pairs=[(mapping[r['head_id']],mapping[r['tail_id']]) for r in facts]
    sim_pairs=[(r['head_id'],r['tail_id']) for r in semantic_preview]
    concept_pairs=[(r['head_id'],r['tail_id']) for r in concept_links]
    concept_nodes={**nodes,**{c['concept_id']:set() for c in concepts}}
    retained_sim=[r for r in semantic_preview if r['status'] in ('related_retained','review_pending')]
    metrics=dict(baseline=graph_metrics({eid:{e['document_id']} for eid,e in by.items()},[(r['head_id'],r['tail_id']) for r in facts]+[(r['left_canonical_id'],r['right_canonical_id']) for r in links]),
                 merge_only_lossless=graph_metrics(nodes,fact_pairs+sim_pairs),
                 merge_plus_concepts_lossless_similar_to=graph_metrics(concept_nodes,fact_pairs+sim_pairs+concept_pairs),
                 proposed_display=graph_metrics(concept_nodes,fact_pairs+[(r['head_id'],r['tail_id']) for r in retained_sim]+concept_pairs))
    # Cases have distinct meanings: exact-identity redundancy, replaceable class
    # links, rejected links and unreviewed links are NOT conflated.
    summary=dict(namespace=NS,baseline_documents=30,baseline_entities=905,baseline_facts=1307,baseline_similar_to=492,
                 duplicate_audit=audit,reviewed_exact_groups=len(groups),candidate_pair_count=len(decisions),
                 candidate_decisions=dict(Counter(r['decision'] for r in decisions)),
                 merge_groups=len(merge_groups),merge_members=sum(len(r['member_canonical_ids']) for r in merge_groups),
                 eliminated_vertices=905-len(nodes),contextual_vertices_after_merge=len(nodes),concept_nodes=len(concepts),total_vertices_with_concepts=len(concept_nodes),
                 cross_document_merge_groups=sum(len({by[eid]['document_id'] for eid in r['member_canonical_ids']})>1 for r in merge_groups),
                 merge_groups_with_type_set_conflict=sum(r['type_set_conflict'] for r in merge_groups),merge_groups_with_multitype=sum(r['original_multitype'] for r in merge_groups),
                 exact_groups_with_type_set_conflict=sum(r['type_set_conflict'] for r in duplicates),exact_groups_with_multitype=sum(r['original_multitype'] for r in duplicates),
                 facts=dict(original_records=1307,unique_structural_relations=len(fact_preview),parallel_structural_groups=sum(len(v)>1 for v in fact_buckets.values()),
                            parallel_record_excess=1307-len(fact_preview),original_self_loop_records=sum(r['head_id']==r['tail_id'] for r in facts),
                            self_loop_records_after_merge=sum(a==b for a,b in fact_pairs),
                            original_raw_relation_count=sum(r['source_relation_count'] for r in facts),
                            original_raw_relation_ids=len({tid for r in facts for tid in r['source_triple_ids']})),
                 semantic_statuses=dict(Counter(r['status'] for r in semantic_preview)),
                 exact_name_semantic_statuses=dict(Counter(r['status'] for r in semantic_preview if r['same_normalized_name'])),
                 cross_document_semantic_decisions=dict(Counter(r['decision'] for r in semantic_preview if r['original_cross_document'])),
                 cross_document_supported_existing_transitions=sum(r['original_cross_document'] and r['decision'] in ('RELATED','SHARED_CONCEPT','MERGE') for r in semantic_preview),
                 cross_document_supported_exact_transitions=sum(r['original_cross_document'] and r['same_normalized_name'] and r['decision'] in ('RELATED','SHARED_CONCEPT','MERGE') for r in semantic_preview),
                 cross_document_factual_relations_after_merge=sum(by[r['head_id']]['document_id']!=by[r['tail_id']]['document_id'] for r in facts),
                 concept_links=len(concept_links),concepts_spanning_documents=sum(len(c['document_ids'])>1 for c in concepts),
                 retained_display_similar_to=len(retained_sim),connectivity=metrics,
                 unknown_nonexact_pair_reviews=[list(p) for p in unresolved])
    summary['facts']['self_loop_records_after_merge']=sum(len(b['original_relations']) for b in fact_preview if b['is_self_loop'])
    summary['facts']['new_loop_records']=sum(len(b['new_loop_source_ids']) for b in fact_preview)
    summary['facts']['unique_self_loop_structures']=sum(b['is_self_loop'] for b in fact_preview)
    summary['facts']['merge_groups_creating_loops']=len({r['merge_group_id'] for r in loop_diagnostics})
    summary['source_entities_covered']=len({sid for e in entities for sid in e['source_entity_ids']})
    summary['source_mentions_covered']=len({mid for e in entities for mid in e['source_mention_ids']})
    summary['candidate_entities_covered']=len({eid for r in decisions for eid in (r['left_canonical_id'],r['right_canonical_id'])})
    summary['within_document_repetition_groups']=sum(r['within_document_repetition'] for r in duplicates)
    summary['cross_document_repetition_groups']=sum(r['cross_document_repetition'] for r in duplicates)
    summary['exact_name_redundant_identity_or_concept']=sum(r['same_normalized_name'] and r['status'] in ('identity_redundant','replaceable_by_concept_path') for r in semantic_preview)
    write_json('simulation_summary.json',summary)
    type_rows=[]
    for e in entities:
        eid=e['canonical_entity_id'];a=aliases[eid]
        if len(e['entity_types'])>1 or (a in merge_by_alias and merge_by_alias[a]['type_set_conflict']):
            type_rows.append(dict(alias=a,canonical_entity_id=eid,name=e['surface_text'],original_types=e['entity_types'],source_types_unchanged=True,
                                  merge_group_id=merge_by_alias[a]['merge_group_id'] if a in merge_by_alias else None,
                                  proposed_primary_type=merge_by_alias[a]['proposed_primary_type'] if a in merge_by_alias else None,
                                  reason=merge_by_alias[a]['reason'] if a in merge_by_alias else 'Множественная исходная типизация не разрешена автоматически; исходные типы сохранены.',
                                  source_contexts=json.loads(inv[a]['member_contexts'])))
    write_csv('typing_audit.csv',type_rows)
    require(all(sha256(p)==h for p,h in before.items()),'Protected artifact changed')
    write_json('validation.json',dict(all_905_ids_mapped_once=True,mapping_idempotent=True,explicit_merge_cliques_disjoint=True,
                                     every_merge_pair_explicitly_recorded=True,all_96_exact_groups_reviewed=True,
                                     all_492_semantic_edges_have_decisions=True,all_1307_facts_exactly_restorable=True,source_canonical_entities_exactly_restorable=True,
                                     original_entity_types_not_modified=True,no_factual_edge_on_concept=True,no_cross_document_identity_assumed=True,
                                     protected_files_unchanged=True,protected_sha256={p.relative_to(ROOT).as_posix():h for p,h in before.items()},
                                     source_contract=plan.SNAPSHOT_SHA256,review_plan_sha256=sha256(OUT/'review_plan.py'),
                                     models_launched=False,neo4j_write_access_used=False,graph_imported=False))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def verify_live(env_file=None):
    """MATCH/RETURN only, transaction access mode READ, no model imports."""
    from build_relex_graphs import connect_neo4j
    entities,by,aliases,facts,links,documents,groups,baseline,audit=load()
    expected_facts={r['canonical_triple_id']:r for r in facts}
    expected_links={r['semantic_link_id']:r for r in links}
    digest=hashlib.sha256()
    manifest=json.loads((KG/'manifest.json').read_text(encoding='utf-8'))
    driver,database=connect_neo4j(env_file)
    try:
        with driver.session(database=database,default_access_mode='READ') as session:
            nodes=session.run('MATCH (n {namespace:$ns}) RETURN n.key AS key,labels(n) AS labels,properties(n) AS properties ORDER BY key',ns=NS)
            seen_entities,seen_documents=set(),set()
            for row in nodes:
                data=row.data();data['labels'].sort()
                props=data['properties']
                digest.update((json.dumps(data,ensure_ascii=False,sort_keys=True)+'\n').encode())
                require(props['fingerprint']==manifest['fingerprint'],'Live graph fingerprint mismatch')
                if 'RELEXEntity' in row['labels']:
                    eid=props['canonical_entity_id'];seen_entities.add(eid)
                    require(eid in by and json.loads(props['source_json'])==by[eid],'Live entity differs from source')
                    require(props['confidence']==max(m['source_entity']['confidence'] for m in by[eid]['members']),'Live confidence mismatch')
                else:
                    doc=props['document_id'];seen_documents.add(doc)
                    require('RELEXDocument' in row['labels'] and json.loads(props['source_json'])==documents[doc],'Live document mismatch')
            require(not nodes.consume().counters.contains_updates,'Read query performed updates')
            require(seen_entities==set(by) and seen_documents==set(documents),'Live vertex coverage mismatch')
            result=session.run('MATCH (h)-[r]->(t) WHERE r.namespace=$ns RETURN r.key AS key,type(r) AS type,h.key AS head,t.key AS tail,properties(r) AS properties ORDER BY key',ns=NS)
            counts=Counter();seen_facts,seen_links,seen_mentions=set(),set(),set()
            for row in result:
                data=row.data();props=data['properties'];counts[row['type']]+=1
                digest.update((json.dumps(data,ensure_ascii=False,sort_keys=True)+'\n').encode())
                require(props['fingerprint']==manifest['fingerprint'],'Live edge fingerprint mismatch')
                if row['type']=='RELEX_REL':
                    rid=props['canonical_triple_id'];seen_facts.add(rid)
                    require(rid in expected_facts and json.loads(props['source_json'])==expected_facts[rid],'Live factual provenance mismatch')
                    require(row['head']==NS+':'+expected_facts[rid]['head_id'] and row['tail']==NS+':'+expected_facts[rid]['tail_id'],'Live fact endpoints mismatch')
                elif row['type']=='SIMILAR_TO':
                    rid=props['semantic_link_id'];seen_links.add(rid)
                    require(rid in expected_links and json.loads(props['source_json'])==expected_links[rid],'Live semantic provenance mismatch')
                    require(row['head']==NS+':'+expected_links[rid]['left_canonical_id'] and row['tail']==NS+':'+expected_links[rid]['right_canonical_id'],'Live semantic endpoints mismatch')
                elif row['type']=='RELEX_MENTIONS':
                    eid=row['tail'][len(NS)+1:];seen_mentions.add(eid)
                    require(eid in by and row['head']==NS+':'+by[eid]['document_id'],'Live mention endpoints mismatch')
                else:raise ValueError('Unexpected live relationship type')
            require(not result.consume().counters.contains_updates,'Read query performed updates')
            require(counts==Counter(RELEX_REL=1307,SIMILAR_TO=492,RELEX_MENTIONS=905),'Live edge counts mismatch')
            require(seen_facts==set(expected_facts) and seen_links==set(expected_links) and seen_mentions==set(by),'Live edge coverage mismatch')
    finally:driver.close()
    output=dict(namespace=NS,database=database,documents=30,entities=905,relationship_counts=dict(counts),
                source_properties_and_provenance_exact=True,snapshot_sha256=digest.hexdigest(),access_mode='READ',
                contains_updates=False,queries=['MATCH nodes RETURN properties','MATCH relationships RETURN properties'])
    target=OUT/'live_baseline_validation.json'
    if target.exists():
        previous=json.loads(target.read_text(encoding='utf-8'))
        require(previous['snapshot_sha256']==output['snapshot_sha256'],'Live graph changed since initial read')
        output['same_snapshot_before_after_analysis']=True
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(target.name,output)
    print('READ-ONLY live baseline verified: '+json.dumps({k:v for k,v in output.items() if k not in ['queries']}))


def validate_preview():
    """Independent DSU verification of exported artifacts and all four views."""
    entities,by,aliases,facts,links,documents,groups,baseline,audit=load()
    summary=json.loads((OUT/'simulation_summary.json').read_text(encoding='utf-8'))
    rows=read_jsonl(OUT/'proposed_mapping.jsonl')
    mapping={r['canonical_entity_id']:r['proposed_final_id'] for r in rows}
    require(len(rows)==len(mapping)==905 and set(mapping)==set(by),'Mapping coverage validation failed')
    require(all(mapping[mapping[eid]]==mapping[eid] for eid in mapping),'Mapping idempotence failed')
    require({r['canonical_entity_id']:r['original_canonical_entity'] for r in rows}==by,'Entity provenance restoration failed')
    decisions=read_csv(OUT/'resolution_decisions.csv')
    dp={tuple(sorted((r['left_canonical_id'],r['right_canonical_id']))):r for r in decisions}
    require(len(dp)==len(decisions),'Duplicate pair decision')
    for r in decisions:
        same=mapping[r['left_canonical_id']]==mapping[r['right_canonical_id']]
        require(same==(r['decision']=='MERGE'),'Unjustified transitive merge or contradictory decision')
    merge_groups=json.loads((OUT/'merge_groups.json').read_text(encoding='utf-8'))
    for g in merge_groups:
        for pair in itertools.combinations(g['member_canonical_ids'],2):
            require(dp[tuple(sorted(pair))]['decision']=='MERGE','Merge clique not complete')
    factual=read_jsonl(OUT/'preview_factual_relations.jsonl')
    restored={r['canonical_triple_id']:r for bucket in factual for r in bucket['original_relations']}
    require(restored=={r['canonical_triple_id']:r for r in facts},'Not all 1307 factual objects restorable')
    require(sum(len(b['original_relations']) for b in factual)==1307,'Canonical factual records duplicated/lost')
    for bucket in factual:
        require(not bucket['head_id'].startswith('concept:') and not bucket['tail_id'].startswith('concept:'),'Fact ends on concept')
        for r in bucket['original_relations']:
            require(mapping[r['head_id']]==bucket['head_id'] and mapping[r['tail_id']]==bucket['tail_id'] and r['relation_type']==bucket['relation_type'],'Wrong factual remapping')
    semantic=read_jsonl(OUT/'preview_semantic_links.jsonl')
    require({r['semantic_link_id']:r['original_link'] for r in semantic}=={r['semantic_link_id']:r for r in links},'Not all 492 semantic objects restorable')
    concept_rows=read_csv(OUT/'concept_candidates.csv')
    concept_links=read_jsonl(OUT/'preview_concept_links.jsonl')
    concept_ids={r['concept_id'] for r in concept_rows}
    require(len(concept_ids)==32 and not concept_ids&set(mapping),'Concept identity collision')
    require(len({(r['head_id'],r['tail_id']) for r in concept_links})==len(concept_links),'Duplicate abstraction links')
    require(all(r['head_id'] in set(mapping.values()) and r['tail_id'] in concept_ids and not r['factual'] and not r['fact_inheritance_allowed'] for r in concept_links),'Concept link violation')
    origin_docs=defaultdict(set)
    for eid in by:origin_docs[mapping[eid]].add(by[eid]['document_id'])
    f_edges=[(mapping[r['head_id']],mapping[r['tail_id']]) for r in facts]
    s_edges=[(r['head_id'],r['tail_id']) for r in semantic]
    c_edges=[(r['head_id'],r['tail_id']) for r in concept_links]
    with_concepts={**origin_docs,**{cid:set() for cid in concept_ids}}
    scenarios={
        'baseline':({eid:{e['document_id']} for eid,e in by.items()},[(r['head_id'],r['tail_id']) for r in facts]+[(r['left_canonical_id'],r['right_canonical_id']) for r in links]),
        'merge_only_lossless':(origin_docs,f_edges+s_edges),
        'merge_plus_concepts_lossless_similar_to':(with_concepts,f_edges+s_edges+c_edges),
        'proposed_display':(with_concepts,f_edges+[(r['head_id'],r['tail_id']) for r in semantic if r['status'] in ('related_retained','review_pending')]+c_edges),
    }
    independent={}
    for label,(ndocs,edges) in scenarios.items():
        parent={n:n for n in ndocs};touched=set()
        def find(n):
            while parent[n]!=n:parent[n]=parent[parent[n]];n=parent[n]
            return n
        for a,b in edges:
            require(a in parent and b in parent,'Unknown simulated endpoint')
            touched.update((a,b));parent[find(a)]=find(b)
        comps=defaultdict(set)
        for n in ndocs:comps[find(n)].add(n)
        largest=sorted(comps.values(),key=lambda g:(-len(g),min(g)))[0]
        metrics=dict(vertices=len(ndocs),isolated_vertices=len(set(ndocs)-touched),weak_components=len(comps),
                     largest_component_vertices=len(largest),largest_component_documents=len(set().union(*(ndocs[n] for n in largest))))
        require(metrics==summary['connectivity'][label],'Independent connectivity mismatch '+label)
        independent[label]=metrics
    validation=json.loads((OUT/'validation.json').read_text(encoding='utf-8'))
    require(all(sha256(ROOT/path)==digest for path,digest in validation['protected_sha256'].items()),'Prior artifact changed')
    require(summary['unknown_nonexact_pair_reviews']==[],'Unreviewed candidate fallback remains')
    write_json('independent_validation.json',dict(algorithm='disjoint_set_union',all_four_connectivity_views_match=independent,
                                                all_merge_cliques_complete=True,no_negative_or_review_pair_merged=True,
                                                all_905_original_entity_objects_restorable=True,all_1307_factual_objects_restorable=True,
                                                all_492_semantic_objects_restorable=True,original_raw_relations=1490,
                                                no_fact_inheritance_by_concepts=True,source_files_and_prior_reports_unchanged=True))
    print('Independent exported-artifact/DSU validation passed.',flush=True)


def write_report():
    s=json.loads((OUT/'simulation_summary.json').read_text(encoding='utf-8'))
    audit=json.loads((OUT/'baseline_audit.json').read_text(encoding='utf-8'))
    inv={r['alias']:r for r in read_csv(OUT/'entity_inventory.csv')}
    groups=read_csv(OUT/'duplicate_groups.csv')
    decisions=read_csv(OUT/'resolution_decisions.csv')
    merges=json.loads((OUT/'merge_groups.json').read_text(encoding='utf-8'))
    concepts=read_csv(OUT/'concept_candidates.csv')
    sem=read_jsonl(OUT/'preview_semantic_links.jsonl')
    loops=read_csv(OUT/'new_loop_diagnostics.csv')
    mapping={r['alias']:r for r in read_jsonl(OUT/'proposed_mapping.jsonl')}
    plan=plan_module()
    def table(headers,rows):
        esc=lambda v:str(v).replace('|','\\|').replace('\n',' ')
        return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(esc(v) for v in row)+' |' for row in rows]+['']
    def excerpt(alias):
        e=inv[alias]
        contexts=json.loads(e['member_contexts'])
        # The canonical representative is not necessarily the first member.
        source_id=e['canonical_entity_id']
        return next((c['context'] for c in contexts if c['source_entity_id']==source_id),contexts[0]['context'])
    lines=['# Stage 2F.2 — Global Entity Resolution: proposed mapping и preview','',
           'Дата: 11 ноября 2026, Europe/Moscow. Ветка `design-v2-sequential-kg`. Статус: **предложение для проверки, не импорт и не новая версия KG**.','',
           f'Актуальный namespace: `{NS}`. NER ≥0.46, confidence=max_member; E5-small / text_name / similarity≥0.95. Все счётчики ниже относятся к **905**, а не историческим 1122 вершинам.','',
           '## Основной результат','',
           f"Контекстно обоснованы **{s['merge_groups']} proposed MERGE-групп** из {s['merge_members']} текущих canonical IDs. Это позволяет устранить {s['eliminated_vertices']} повторов: **905 → {s['contextual_vertices_after_merge']} contextual entities**. Предлагаются **{s['concept_nodes']} отдельных научных концепта**, итоговая схема — **{s['total_vertices_with_concepts']} вершин**, без документов. Все подтверждённые identity-группы находятся внутри одного документа; **междокументного тождества конкретных референтов не доказано**. Глобальный слой связывает контексты общими концептами.",'',
           f"Все **1307 исходных RELEX_REL** сохранены в восстанавливаемом виде. После переназначения это {s['facts']['unique_structural_relations']} уникальных структурных троек: {s['facts']['parallel_structural_groups']} групп параллельных отношений, {s['facts']['parallel_record_excess']} дополнительных записей с теми же концами/типом. Петли: 12 исходных → {s['facts']['self_loop_records_after_merge']} записей, из них {s['facts']['new_loop_records']} новые. Это диагностические последствия ER, а не новые достоверные факты.",'',
           f"Из 349 одноимённых SIMILAR_TO **254 можно считать структурно избыточными в предложенной двухслойной схеме**: 49 схлопываются по доказанной идентичности, 205 заменимы путём contextual entity → concept ← contextual entity. Ещё 27 отклонены по содержанию, 42 требуют REVIEW, 26 остаются RELATED. Отклонение связи и избыточность — разные причины.",'',
           f"Из 121 существующего междокументного перехода **90 имеют положительное контекстное основание на заявленном семантическом уровне**: 77 SHARED_CONCEPT и 13 RELATED. Это **не 90 подтверждённых фактических отношений**. 30 отклонены, 1 требует REVIEW. Все исходные 492 semantic records сохранены для аудита.",'',
           'Числа MERGE — результат текущей аналитической проверки контекстов агентом, не независимая человеческая gold-разметка и не разрешение применить mapping. 18 групп имеют несовпадающие исходные наборы типов; основания исключений явно записаны. Перед применением потребуются проверка предлагаемой политики типизации и проверка отношений, которые становятся петлями. Этап 2F.3 не начат.','',
           '## Baseline и независимый аудит повторов','']
    lines+=table(['Показатель','Baseline'],[['Документы',30],['Canonical entities',905],['Source entities / typed mentions в их provenance',f"{s['source_entities_covered']} / {s['source_mentions_covered']}"],['RELEX_REL / SIMILAR_TO','1307 / 492'],['Междокументные SIMILAR_TO / RELEX_REL','121 / 0'],['Изоляты / слабые компоненты','175 / 230'],['Крупнейшая компонента: entities / documents','378 / 17'],['Группы нормализованных имён / вершины в них','96 / 259'],['Одноимённые SIMILAR_TO / междокументные одноимённые','349 / 91']])
    lines += [f"Нормализация переиспользует 1A: NFC → casefold → NFC → сжатие Unicode-пробелов; пунктуация, числа, дефисы, квалификаторы и повреждения не стираются. **87 групп** содержат повторы внутри документа, **11** — между документами; 2 группы (`patients`, `energy consumption`) входят в обе категории. Все 349 неупорядоченных exact-name пар текущего снимка уже имеют SIMILAR_TO. Сам факт совпадения вектора имени не используется как доказательство идентичности.",'',
              'Baseline проверен полным read-only чтением Neo4j: все 935 физических узлов (30+905), 2704 отношения (905 MENTIONS +1307 facts +492 semantic), source_json, max-member confidence, endpoints и fingerprint сопоставлены с локальными источниками. Снимок перед заключительной симуляцией и после неё сверяется SHA-256; подробности — `live_baseline_validation.json`.','',
              '## Что учтено из предыдущих этапов','',
              '- [1A](../exact_matching_audit/report.md): совпадение имён — только генератор кандидатов; различать геномный/таксономический order, разные множества genes/eyes/patients и измерения в разные моменты.',
              '- [1B](../conservative_resolution/report.md): существующие source-member группы не разбираются и не переписываются. В mapping каждая текущая canonical запись сохранена целиком, включая прежние решения и все исходные member IDs. Контекстная идентичность и типизация исследуются отдельно.',
              '- [2C](../semantic_resolution/candidate_verification/proposed_merge_rules.md): проверять уровень референта и внешние модификаторы, не голосовать cosine. 113 индивидуально разобранных прежних пар, чьи концы сохранились при NER .46, включены как evidence; старые ACCEPT не импортированы автоматически как MERGE. 2C REJECT внутри proposed MERGE-клик отсутствуют. REJECT на тождество из 2C отдельно переосмыслен: например, разные strain IDs, XANES/XMCD, hip/knee arthroplasty и AP/AR могут получить RELATED, не MERGE; самостоятельная полезная близость и identity различаются.',
              '- [2E](../semantic_resolution/global_kg_import/report.md) и [2F.1](../ner_threshold_analysis/global_kg_t046_max/report.md): используют lossless provenance, один SIMILAR_TO на неупорядоченную пару и отдельные document-mentions. Актуальный набор берётся из membership/manifest 2F.1; старый граф не подменяет baseline.','',
              '## Правила Global Entity Resolution','',
              '1. **MERGE** требует одного референта на явно выбранном уровне: именованный вид/метод/молекулярный вид, документно определённый объект/набор или одна явно определённая когорта/оценка. Один объектный класс не равен физическим экземплярам; область применения процедуры не равна операциям разных пациентов. Для каждой группы заданы полный состав и положительное основание; каждая пара внутри группы имеет собственную строку MERGE. Клики не пересекаются, транзитивного замыкания похожести нет.',
              '2. **SHARED_CONCEPT** сохраняет contextual IDs и добавляет отдельный `concept:2f2:*`. Связь `HAS_SHARED_CONCEPT` — контекстная принадлежность научной теме/классу, **не identity, не доказанная IS_A-инстанциация, не факт пациента и не наследование фактов**. Фактические рёбра никогда не перенаправляются на concept_id.',
              '3. **RELATED** означает обоснованную близость при разных уровнях, носителях или условиях: болезнь/риск, класс/подтип, параметр/значение, действие/объект. Идентификаторы не объединяются; новых типизированных фактических отношений не создаётся.',
              '4. **REJECT** отклоняет предлагаемый прямой кандидат как ошибочную/малоинформативную связь: полисемия, generic self-reference разных публикаций, heading, modifier fragment или подтверждённое различие референтов без полезного прямого основания. Это не удаление исходного узла или его фактов.',
              '5. **REVIEW** оставляет ID раздельно при неизвестном авторстве/составе/времени, повреждённом имени, неоднозначной границе или неразрешённом уровне gene/allele, instrument/scores, class/instance. REVIEW-вершины сохраняются; их старые semantic links остаются явно pending.',
              '6. Для аббревиатур использовать раскрытие в **том же документе**, проверить все member contexts и модификаторы. Сокращение может задавать один named kind, но IOP до/после операции не становится одним измерением. PK/RT объединяются на уровне procedure kind, а не individual operations/fractions.',
              '7. Между документами одинаковое имя/высокий cosine не доказывают идентичность конкретного объекта. Требуются общий устойчивый идентификатор и совпадение уровня/спецификации либо прямое документальное подтверждение. В этом корпусе таких cross-document MERGE-групп не предложено.',
              '8. Тип — диагностический сигнал. Равенство типов не доказывает identity; несовместимость типов может быть NER-ошибкой. В отличие от строгого gate 1B/2C, 2F.2 допускает **явно обоснованное исключение** для определения имени/аббревиатуры или точно указанной когорты. Source types сохраняются; proposed_primary_type — только предложение для проверки, не применённая коррекция.',
              '9. Представитель — существующий ID: минимальный document_id/start, затем наиболее полное имя при равном start, затем canonical ID. Это не выбор по NER confidence. Все source scores/offsets/provenance сохраняются. У концептов нет выдуманной calibrated confidence.',
              '10. Existing RELEX_REL учитываются как диагностические последствия решения, **не независимое доказательство identity**. Автоматически извлечённые HAS_PART/COMPARED_WITH между двумя именами одного объекта могут стать петлями; они сохраняются с пометкой relation_review_required.','',
              '## Покрытие кандидатов и решения','',
              f"В итоговой таблице **{s['candidate_pair_count']} пар** и **{s['candidate_entities_covered']} различных contextual IDs**. Исходный pool: 492 SIMILAR_TO, включая все 349 exact-name пар; 113 прежних индивидуальных 2C-проверок и 30 пар из сохранённых abbreviation candidates (множества пересекаются). Получено 579 уникальных пар. Ещё 14 пар добавлены только для полного явного попарного рассмотрения предложенных клик. Это не новый semantic search и не полный перебор 905×904/2.",'']
    lines+=table(['Решение','Пар'],[[k,s['candidate_decisions'][k]] for k in ['MERGE','SHARED_CONCEPT','RELATED','REJECT','REVIEW']])
    lines+=['MERGE-пар больше, чем MERGE-групп: например, clique из четырёх участников имеет шесть пар. Все 96 exact-name групп имеют конкретные group-level основания и полный состав в `duplicate_groups.csv`; вся существующая semantic-сеть имеет решение в `resolution_decisions.csv`. Узлы вне выбранного candidate pool сохранены отдельными и **не объявлены глобально уникальными**. Необоснованных fallback-решений без отдельного контекстного основания в итоговой таблице нет.','',
            '## patients: когорты, подгруппы и общий Patients','',
            'В baseline 14 одноимённых вершин из пяти документов. Пять документов **не означают пять одинаково определённых клинических когорт**:','',
            '- W2761455282: TKA-серия **21 ESRD-пациент / 26 knees, 2007–2015**; только её явно анафорические упоминания входят в одну MERGE-группу N398/N399/N400. Background ESRD-популяция в тексте про hip arthroplasty и these patients в постановке вопроса не автоматически равны этой enrolled серии.',
            '- В том же документе: **other 20** после смерти одного, **6** с осложнениями и **19** с improved outcomes — самостоятельные подгруппы. Пересечения групп событий не выводятся. **Most patients** в заключении может означать те же 19, но это REVIEW, а не доказанное тождество.',
            '- W2791492120: **10** CRSwNP-пациентов, полипоидная ткань/кровь. N455/N463/N474 — одна явная серия; CONDITION у qualified patient phrase ошибочен для людей, но не делает их диагнозом CRSwNP.',
            '- W3101922761: **40 пациентов / 40 глаз** с GDD after PK. Пациенты и глаза не объединяются, даже когда их количество совпадает.',
            '- W4391351014: **16** пациентов с esophageal cancer, radiotherapy 2012–2022, Heidelberg — отдельная серия; background cancer patients relying on CIED не приравниваются enrolled группе.',
            '- W3123056951: **общая популяция rHuEPO recipients** и **some patients с anti-rHuEPO neutralizing antibodies/PRCA**. Численность собственной клинической когорты не указана; работа описывает дизайн и in-vitro проверку белков. Нельзя выдумывать enrolled cohort.','']
    patient_aliases='N386 N391 N399 N400 N417 N418 N426 N436 N455 N582 N622 N626 N864 N869'.split()
    lines+=table(['Alias / ID prefix','Документ','Offset(s)','Типы','Max confidence','Референт / решение'],[
        [a+' / '+inv[a]['canonical_entity_id'][:14]+'…',inv[a]['document_id'],[x['start'] for x in json.loads(inv[a]['source_offsets'])],
         '/'.join(json.loads(inv[a]['entity_types'])),f"{float(inv[a]['confidence']):.6f}",plan.ENTITY_NOTES[a]+(' MERGE '+mapping[a]['merge_group_id'] if mapping[a]['merge_group_id'] else ' Сохранить отдельно; Patients context link.')]
        for a in patient_aliases])
    patient_concept=next(c for c in concepts if c['label']=='Patients')
    lines += [f"Предлагаемый `{patient_concept['concept_id']}` охватывает 20 исходных contextual IDs, включая qualified/dialysis/cancer-patient forms; после identity MERGE остаются **{patient_concept['contextual_entity_count']}** context→concept связей в пяти документах. Никакие HAS_RESULT/CAUSES/USES, описывающие отдельную когорту, не переносятся на Patients. При обходе такой путь означает общую категорию, а не «другая статья подтверждает тот же результат у тех же пациентов».",'',
              'Три показательных контекста:','',f"- N400, 21 patients: «{excerpt('N400')}»",f"- N418, 6 patients: «{excerpt('N418')}»",f"- N869, 16 patients: «{excerpt('N869')}»",'',
              '## results: общий концепт не предлагается','',
              '6 вершин / 6 документов: пять анафорических указателей и один заголовок. Все 15 exact-name попарных связей отклонены как полезная глобальная идентификация; существующие factual edges исходных узлов не удаляются. N725 имеет локальную информационную роль в PNM, однако это не основание для общего научного Results.','']
    lines+=table(['Alias','Документ','Offset','Max confidence','Контекстная оценка'],[[a,inv[a]['document_id'],[x['start'] for x in json.loads(inv[a]['source_offsets'])],f"{float(inv[a]['confidence']):.6f}",plan.ENTITY_NOTES[a]] for a in 'N030 N195 N258 N600 N725 N793'.split()])
    lines += [f"- Пример heading N600: «{excerpt('N600')}»",f"- Пример generic reference N793: «{excerpt('N793')}»",'',
              '## Другие обязательные кейсы','']
    cases=[
        ('convergence','N170 N172 N174 N176 N181 N183 N189 N193 N194','Вводятся разные definitions/hypothesis, затем отдельно выводы об exchange, interest и inflation rates. Один общий Convergence связывает контексты без MERGE этих результатов. Факт положительной сходимости nominal exchange rates нельзя переносить на inflation rates, где convergence can be rejected.'),
        ('glaucoma','N573 N580 N587 N592 N604 N606 N609 N612','Различать диагноз и слово внутри compound names. Только N573/N609 предлагаются как одна disease-kind entity этого контекста. Device/outcome/surgery/medications/success не исправляются и не сливаются с диагнозом по одному head glaucoma. Full intractable glaucoma остаётся RELATED к head-span, квалификатор не стирается.'),
        ('eyes','N571 N579 N586 N602 N610 N614','N579/N586 подтверждают исходную 40-eye серию. 20 long-followed eyes и доли глаз с разными outcomes не та же группа. Eyes — отдельный концепт, а не Patients даже при 40/40.'),
        ('ATP','N089 N106 N109','MERGE на уровне именованного молекулярного вида. OBJECT/ATTRIBUTE — несовместимая исходная типизация; предлагаем OBJECT отдельно. Endogenous/reagent pools, exposure times и процессы secretion не отождествляются.'),
        ('FGF-1','N085 N108 N113 N117','Явное Fibroblast growth factor 1 (FGF-1) и устойчивое имя во всех contexts. MERGE по именованному фактору с сохранением всех ACTOR/ATTRIBUTE/CONCEPT/OBJECT source labels. Условия before/after и 2h/7h не становятся одним событием.'),
        ('TKA','N392 N397 N402 N430','Полное имя и class-level TKA объединяются на уровне типа процедуры. Primary TKA в конкретной серии N402 сохраняется отдельно и связан с концептом Total knee arthroplasty. Total hip arthroplasty — другой вид операции и не объединяется.'),
        ('genes','N671 N678 N680 N684','82 up-regulated, восемь tolerance-enhancing и пять successful candidates — разные наборы. Только these genes / tolerance-enhancing genes в одном предложении обозначают восемь; proposed MERGE N678/N680. Genetics как дисциплина — RELATED, не один набор genes.'),
        ('breast cancer','N002 N007','N002 — болезнь; N007 вложен в breast cancer risk, граница требует REVIEW. Breast cancer vs breast cancer risk — RELATED, а general cancer vs esophageal cancer не равны. Одинаковый CONDITION не разрешает identity.'),
    ]
    for title,member_text,reason in cases:
        lines += ['### '+title,'',reason,'']
        for a in member_text.split():
            e=inv[a]
            lines += [f"- {a}, `{e['canonical_entity_id'][:14]}…`, `{e['document_id']}`, offsets {[x['start'] for x in json.loads(e['source_offsets'])]}, types {'/'.join(json.loads(e['entity_types']))}, max={float(e['confidence']):.6f}. "+plan.ENTITY_NOTES.get(a,''),f"  Контекст: «{excerpt(a)}»",'']
    lines += ['## Синонимы, аббревиатуры и корректные/ошибочные объединения','']
    selected=[('N001','N006'),('N083','N111'),('N845','N846'),('N833','N840'),('N387','N444'),('N446','N464'),('N716','N735'),('N620','N648'),('N641','N651'),('N636','N651'),('N392','N402'),('N575','N603'),('N603','N605'),('N044','N045'),('N010','N009'),('N782','N783'),('N299','N485'),('N067','N075')]
    pair_rows=[]
    for a,b in selected:
        d=next((r for r in decisions if {r['left_alias'],r['right_alias']}=={a,b}),None)
        if d:pair_rows.append([a+' '+inv[a]['name'],b+' '+inv[b]['name'],d['decision'],d['reason']])
    lines += table(['Первый референт','Второй','Решение','Основание'],pair_rows)
    lines += ['Раскрытие аббревиатуры само по себе задаёт лексическое значение, но не identity любых численных результатов или наборов. Поэтому class-level MD/FGF-1/ARPN и именованное учреждение PNM могут иметь положительное основание, а pre-/postoperative IOP, instrument vs scores и HLA gene vs allele — разные результаты. Полные evidence, source mention IDs и точные source fragments для каждой пары — в `resolution_decisions.csv`.','',
              '## Влияние типизации и старых factual edges','',
              f"Из 96 exact-name групп **28** имеют различающиеся наборы исходных типов, **29** — хотя бы одно многотипное извлечение. В предложенных 69 MERGE-группах **18** имеют type-set conflict, **15** содержат multi-type members (счётчики пересекаются). Strict same-single-type gate 1B заблокировал бы такие случаи; считать их разными объектами только по label тоже неверно.",'',
              'Явные примеры: FGF-1 помечен ACTOR/ATTRIBUTE, ARPN — ACTOR/CONCEPT, PNM — DATA_RESOURCE, PBMC — DATA_RESOURCE/OBJECT, CRSwNP — ACTOR/CONDITION/OBJECT, qualified patients with CRSwNP — CONDITION. Текст определяет соответственно фактор, модель, вуз, клетки, заболевание и людей. Original types не исправлены в provenance; proposed_primary_type указан отдельно в MERGE-группе. Неизвестный тип ERM оставлен без обязательной предлагаемой primary label.','',
              f"**14 MERGE-групп создают 27 новых factual-loop records.** Проверены исходные neighbourhoods и все новые loop cases; `new_loop_diagnostics.csv` содержит каждую запись. Среди примеров — PNM HAS_MEMBER Politeknik Negeri Madiun, HSR COMPARED_WITH high-speed rail и ARPN COMPARED_WITH ARPN. Контекстное тождество имени не доказывает корректность этих машинных отношений. Их нельзя молча удалить, объявить новыми фактами или использовать как доказательство плохого/хорошего ER.",'',
              'Пример genes: source HAS_PART между tolerance-enhancing genes и limonene synthase gene подозрителен, поскольку текст говорит coexpressed with; ER сохраняет его как исходное предсказание, а не превращает в факт общего концепта Genes. Исправление ReLEx в этот этап не входит.','',
              '## Предлагаемые общие концепты','',
              'Это локальные concept_id проекта без внешнего ontology ID. Все новые связи имеют отдельный тип HAS_SHARED_CONCEPT, factual=false, identity=false, fact_inheritance_allowed=false. У концепта нет собственного document_id; число документов в компонентах считает только реальные source documents.','']
    lines += table(['Concept','Contextual targets после MERGE','Документов','Назначение'],[[c['label'],c['contextual_entity_count'],len(json.loads(c['document_ids'])),c['reason']] for c in concepts])
    lines += [f"Всего **118** context→concept связей; только **3** концепта охватывают несколько документов: Patients (5), Energy consumption (2) и Medical complications (2). Остальные сохраняют различия контекстов внутри одной работы и могут быть проверены как будущие глобальные классы. Общих Results, Paper, Study, Author, Reliability или Resolution не создаётся.",'',
              '## Симуляция структуры','',
              'Все сценарии — только in-memory bookkeeping; ни один граф не импортирован. Направления игнорируются для слабых компонент; петли сохраняются и дают ненулевую степень; документы/Mentions не входят. Для ties используется минимальный ID, а число документов вычисляется из provenance contextual members, без фиктивного документа концепта.','']
    labels={'baseline':'Текущий baseline','merge_only_lossless':'Только MERGE, все старые SIM','merge_plus_concepts_lossless_similar_to':'MERGE + concepts, все старые SIM','proposed_display':'Proposed display + concepts'}
    lines+=table(['Сценарий','Вершин','Изоляты','WCC','Крупнейшая','Документов в ней'],[[labels[k]]+[v[t] for t in ['vertices','isolated_vertices','weak_components','largest_component_vertices','largest_component_documents']] for k,v in s['connectivity'].items()])
    lines+=[f"Последний display-view сохраняет все 1307 factual records (1076 структурных buckets), добавляет 118 conceptual edges и оставляет **{s['retained_display_similar_to']}** SIMILAR_TO ({s['semantic_statuses']['related_retained']} RELATED +{s['semantic_statuses']['review_pending']} pending REVIEW). Остальные direct semantic edges представлены audit-статусами: {s['semantic_statuses']['identity_redundant']} identity-redundant, {s['semantic_statuses']['replaceable_by_concept_path']} replaceable-by-concept-path, {s['semantic_statuses']['rejected_candidate']} rejected. Все 492 оригинальных semantic records остаются восстановимы; это предложение отображения, не фактическое удаление из Neo4j.",'',
            'Верхние два proposed views намеренно разделены: один показывает чистый эффект adding concepts при всех старых semantic edges, другой — отдельный эффект предложенной содержательной классификации direct links. Уменьшение числа WCC при добавлении hub-концепта — топологический эффект абстракции; оно не создаёт новых подтверждений фактов. Рост изоляции в proposed display показывает, что часть прежней связанности давали отклонённые generic связи. Ни 905→811, ни рост/падение largest component не являются оценкой улучшения качества.','',
            '### Сохранность фактических отношений','']
    lines+=table(['Метрика','Значение'],[['Сохранённых canonical RELEX_REL records',1307],['Вложенных raw source triple IDs',1490],['Уникальных (document,head,type,tail) после mapping',1076],['Групп параллельных отношений',164],['Дополнительных записей в этих группах',231],['Исходных / итоговых loop records','12 / 39'],['Новых loop records / уникальных loop structures','27 / 26'],['Фактических отношений на concept nodes',0],['Междокументных factual edges после mapping',0]])
    lines+=['`preview_factual_relations.jsonl` хранит full original_relations для каждого структурного bucket: исходные canonical IDs, source triples, confidence, evidence и provenance. Распаковка строго восстанавливает все 1307 исходных объектов и 1490 raw source relations. Структурная дедупликация не удаляет записи, не суммирует confidence и не порождает новые scientific facts.','',
            '### Смысл междокументных переходов','']
    lines+=table(['Решение по существующим cross-document SIMILAR_TO','Число'],[[k,s['cross_document_semantic_decisions'].get(k,0)] for k in ['MERGE','SHARED_CONCEPT','RELATED','REJECT','REVIEW']])
    lines+=['90 положительно обоснованных переходов — это 77 классово-концептуальных и 13 тематически RELATED кандидатов; 66 из них одноимённые. RELATED, например genetics↔genes или breast cancer↔cancer, сохраняет различие референтов/специфичности, а не поддерживает MERGE. Один pending cross-document case — неизвестные author из разных публикаций: без ORCID/имён нельзя доказать ни тождество, ни различие людей.','',
            'В proposed display 77 прямых SHARED_CONCEPT links заменимы путями через соответствующие абстракции; они не считаются 77 фактическими междокументными отношениями. 13 RELATED links остаются прямыми, REVIEW author остаётся отдельно pending. Число всех возможных пар через концептуальный hub не выдаётся за число независимых доказательств. Старые CROSS_DOC_RE_CANDIDATE из отдельного эксперимента не входят в baseline и не добавлялись.','',
            '## Полный каталог 96 групп','',
            'M/S/R/X/? в колонке решений означают MERGE / SHARED_CONCEPT / RELATED / REJECT / REVIEW и показывают число пар, а не узлов. Полные ID, все member contexts, source offsets/types/confidence/provenance и incident factual IDs сохранены в duplicate_groups.csv.','']
    def counts(r):
        c=json.loads(r['decision_counts'])
        return ', '.join(f'{short}:{c[label]}' for label,short in [('MERGE','M'),('SHARED_CONCEPT','S'),('RELATED','R'),('REJECT','X'),('REVIEW','?')] if c.get(label))
    lines+=table(['Группа','Имя','Вершин / док.','Решения пар','Конкретное основание'],[[r['group_id'],r['normalized_name'],r['vertices']+' / '+r['document_count'],counts(r),r['group_reason']] for r in groups])
    lines+=['## Риски, границы и проверки','',
            '- Это контекстная аналитическая классификация, не независимая экспертная gold-разметка. Не оценивались precision/recall, retrieval quality, answer faithfulness или истинность самих научных публикаций.',
            '- Не все возможные синонимы среди 905 вершин перебраны. Candidate pool ограничен существующими связями, exact names, сохранёнными явными раскрытиями и прежними индивидуальными 2C evidence. Источники с редкими формулировками могут остаться вне pool; KEEP_SEPARATE не означает доказанную уникальность.',
            '- Один canonical node 1B уже может содержать несколько source mentions, в том числе low-confidence members. Они не расщепляются; max_member policy исходного графа сохранена. При NER .46 присутствуют 993 source entities /1101 mentions, остальные старые пороговые предсказания не возвращались.',
            '- Повреждённые Co–Fe–Zr формулы, uncertainty gene/allele, границы instrument/scores, whole/subset и unspecified authors не разрешаются similarity. Низкий confidence не отменяет научное содержание, высокий не доказывает identity.',
            '- Применение 18 type-conflicted MERGE-групп потребовало бы отдельного принятия описанной политики semantic identity vs source label. Proposed_primary_type — справочное предложение, raw types не исправлялись. 14 групп, создающих новые loops, требуют проверки отношений до практической перестройки.',
            '- Обобщённые concept nodes не получают factual edges или агрегированную достоверность исследований. Маршрут через Patients означает общую категорию и должен сохранять document/cohort context; результаты independent cohorts не наследуются.',
            '- Модели NER/ReLEx/embeddings/LLM, embeddings/cosine и новый поиск similarity не запускались. Не создавались новые factual relations. Neo4j только читался; MERGE, смена схемы/типов и создание новой версии Global KG не выполнялись.','',
            'Автоматические проверки: все 905 ID ровно один раз; target idempotence; disjoint explicit cliques; решение MERGE по каждой паре клики; отсутствие negative/REVIEW pair, попавшей в одно identity; покрытие всех 96 групп и всех 492 semantic records; точное восстановление всех factual/source objects; отсутствие factual ends на concept nodes; source offsets/revisions и неизменность SHA-256 предыдущих артефактов. Независимый DSU повторяет все четыре component views из экспортированных файлов.','',
            'Команда воспроизведения (только локальные новые артефакты):','',
            '```powershell','.venv/Scripts/python.exe scripts/preview_global_entity_resolution.py','```','',
            'Для отдельной read-only проверки фактического baseline: тот же скрипт с `--verify-live` и существующими NEO4J_* либо `--neo4j-env <existing-file>`. Учётные данные не записываются. Подробности проверок: `validation.json`, `independent_validation.json`, `live_baseline_validation.json`.','',
            'Артефакты: обязательные `duplicate_groups.csv`, `resolution_decisions.csv`, `concept_candidates.csv`, `proposed_mapping.jsonl`, `preview_report.md`; дополнительные `review_plan.py` (явный source-bound review), `entity_inventory.csv`, `candidate_inventory.csv`, `merge_groups.json`, `typing_audit.csv`, `new_loop_diagnostics.csv`, `preview_factual_relations.jsonl`, `preview_semantic_links.jsonl`, `preview_concept_links.jsonl`, `simulation_summary.json` и файлы валидации.','',
            '**Этап 2F.2 завершён только proposed mapping и симуляцией. Изменений Neo4j и запуска этапа 2F.3 нет.**','']
    (OUT/'preview_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Full preview_report.md generated.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',action='store_true')
    parser.add_argument('--verify-live',action='store_true')
    parser.add_argument('--neo4j-env',type=Path)
    args=parser.parse_args()
    if args.verify_live:verify_live(args.neo4j_env)
    elif args.inventory:inventory()
    else:
        resolve()
        validate_preview()
        write_report()


if __name__=='__main__':main()
