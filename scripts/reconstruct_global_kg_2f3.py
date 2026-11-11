"""Stage 2F.3: apply ONLY the approved 2F.2 mapping and import a lossless KG.

Run with existing NEO4J_* credentials, or --neo4j-env <existing-file>.
--prepare-only reads the source Neo4j namespace but does not write to Neo4j.
No ER decisions, model loading, embeddings, edge filtering or new facts.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
for stream in (sys.stdout,sys.stderr):
    if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8',errors='backslashreplace')
from tqdm import tqdm
import import_global_semantic_graph as base

ROOT=base.ROOT
ER=base.ER
SOURCE_NS='relex_sem_e5_small_name_t095_ner_t046_max'
TARGET_NS='relex_sem_e5_small_name_t095_ner_t046_er2f3'
SOURCE=ER/'global_entity_resolution'
SOURCE_KG=ER/'ner_threshold_analysis/global_kg_t046_max'
OUT=ER/'global_kg_2f3'
BATCH=100
STAGE='ИНИЦИАЛИЗАЦИЯ'
LAST_BATCH='НЕТ'
IMPORT_COMMITTED=False
base.NS=TARGET_NS
base.OUT=OUT
require=base.require
dumps=base.dumps
csv.field_size_limit(16*1024*1024)
LABELS=('RELEXDocument','RELEXEntity','RELEXConcept')
REL_TYPES=('RELEX_MENTIONS','RELEX_REL','SIMILAR_TO','HAS_SHARED_CONCEPT')
EXPECTED=dict(documents=30,entities=811,concepts=32,kg_vertices=843,total_nodes=873,
              factual_relations=1307,semantic_links=492,concept_links=118,document_mentions=811,
              kg_edges=1917,total_relationships=2728)
INPUT_NAMES=('merge_groups.json','proposed_mapping.jsonl','concept_candidates.csv','preview_concept_links.jsonl',
             'preview_factual_relations.jsonl','preview_semantic_links.jsonl','simulation_summary.json','validation.json','preview_report.md')


def banner(title):
    global STAGE,LAST_BATCH
    STAGE=title;LAST_BATCH='НЕТ В ТЕКУЩЕМ ЭТАПЕ'
    print('\n'+'='*70+'\n'+title+'\n'+'='*70,flush=True)


def progress(items=None,*,total=None,desc):
    return tqdm(items,total=total,desc=desc,unit='шт.',file=sys.stdout,ascii=True,
                dynamic_ncols=True,mininterval=.3,leave=True)


def key(identity):return TARGET_NS+':'+identity


def write_json(name,value):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8',newline='\n')


def read_json(name):return json.loads((SOURCE/name).read_text(encoding='utf-8'))


def source_snapshot(session):
    banner('НАЧИНАЮ ЧТЕНИЕ И ПРОВЕРКУ ИСХОДНОГО NAMESPACE 2F.1')
    nodes={};edges={};digest=hashlib.sha256()
    result=session.run('MATCH (n {namespace:$ns}) RETURN n.key AS key,labels(n) AS labels,properties(n) AS properties ORDER BY key',ns=SOURCE_NS)
    for row in progress(result,total=935,desc='ЧТЕНИЕ ИСХОДНЫХ УЗЛОВ'):
        data=row.data();data['labels'].sort()
        require(data['key'] not in nodes,'Duplicate source node key')
        nodes[data['key']]=data
        digest.update((json.dumps(data,ensure_ascii=False,sort_keys=True)+'\n').encode())
    require(not result.consume().counters.contains_updates,'Source read updated data')
    result=session.run('MATCH (h)-[r]->(t) WHERE r.namespace=$ns RETURN r.key AS key,type(r) AS type,h.key AS head,t.key AS tail,properties(r) AS properties ORDER BY key',ns=SOURCE_NS)
    for row in progress(result,total=2704,desc='ЧТЕНИЕ ИСХОДНЫХ ОТНОШЕНИЙ'):
        data=row.data()
        require(data['key'] not in edges,'Duplicate source relationship key')
        require(data['head'] in nodes and data['tail'] in nodes,'Source cross-namespace relationship')
        edges[data['key']]=data
        digest.update((json.dumps(data,ensure_ascii=False,sort_keys=True)+'\n').encode())
    require(not result.consume().counters.contains_updates,'Source read updated data')
    require(len(nodes)==935 and len(edges)==2704,'Incorrect live 2F.1 baseline')
    require(Counter(r['type'] for r in edges.values())==Counter(RELEX_MENTIONS=905,RELEX_REL=1307,SIMILAR_TO=492),'Source edge types mismatch')
    require(Counter(label for row in nodes.values() for label in row['labels'])==Counter(RELEXDocument=30,RELEXEntity=905),'Source labels mismatch')
    expected=json.loads((SOURCE/'live_baseline_validation.json').read_text(encoding='utf-8'))
    require(digest.hexdigest()==expected['snapshot_sha256'],'Live source differs from approved 2F.2 snapshot')
    print('ИСХОДНЫЙ ГРАФ ПРОВЕРЕН: 30 ДОКУМЕНТОВ, 905 СУЩНОСТЕЙ, 2704 ОТНОШЕНИЯ.',flush=True)
    return nodes,edges,digest.hexdigest()


def metrics(nodes,edges):
    """Use the existing exact connectivity; concepts add no document identity."""
    kg={k:r for k,r in nodes.items() if r['label']!='RELEXDocument'}
    pairs=[(r['head'],r['tail']) for r in edges.values() if r['type']!='RELEX_MENTIONS']
    docs={k:r['properties'].get('document_id') for k,r in kg.items()}
    calculated=base.connectivity(docs,pairs)
    # Existing helper counts None as a document. Recover components explicitly
    # to count real context documents only, without assigning concepts a document.
    adjacency={k:set() for k in kg};touched=set();external=set();loop_counts=Counter()
    for a,b in pairs:
        require(a in kg and b in kg,'KG endpoint missing')
        adjacency[a].add(b);adjacency[b].add(a);touched.update((a,b))
        if a==b:loop_counts[a]+=1
        else:external.update((a,b))
    seen=set();groups=[]
    for seed in sorted(kg):
        if seed in seen:continue
        stack=[seed];seen.add(seed);group=[]
        while stack:
            n=stack.pop();group.append(n)
            for other in sorted(adjacency[n]-seen):seen.add(other);stack.append(other)
        groups.append(group)
    groups.sort(key=lambda g:(-len(g),min(g)))
    calculated['largest_component_documents']=len({docs[n] for n in groups[0] if docs[n] is not None})
    only_loops=sorted(set(loop_counts)-external)
    calculated.update(kg_vertices=calculated.pop('entities'),self_loop_only_vertices=len(only_loops),
                      without_external_edges=len(set(kg)-external),
                      contextual_entities=sum(r['label']=='RELEXEntity' for r in kg.values()),
                      concepts=sum(r['label']=='RELEXConcept' for r in kg.values()))
    require(calculated['without_external_edges']==calculated['isolated']+len(only_loops),'Isolation partition mismatch')
    return calculated,only_loops


def build(source_nodes,source_edges,source_digest,hashes):
    banner('НАЧИНАЮ ЧТЕНИЕ УТВЕРЖДЁННЫХ АРТЕФАКТОВ 2F.2')
    rows=base.core.read_jsonl(SOURCE/'proposed_mapping.jsonl')
    merges=read_json('merge_groups.json')
    with (SOURCE/'concept_candidates.csv').open(encoding='utf-8-sig',newline='') as stream:
        concepts=list(csv.DictReader(stream))
    concept_links=base.core.read_jsonl(SOURCE/'preview_concept_links.jsonl')
    factual_preview=base.core.read_jsonl(SOURCE/'preview_factual_relations.jsonl')
    semantic=base.core.read_jsonl(SOURCE/'preview_semantic_links.jsonl')
    facts=[r for bucket in factual_preview for r in bucket['original_relations']]
    require((len(rows),len(merges),len(concepts),len(concept_links),len(facts),len(semantic))==(905,69,32,118,1307,492),'Incorrect approved input counts')
    require(read_json('validation.json')['all_1307_facts_exactly_restorable'],'2F.2 validation missing')
    print('АРТЕФАКТЫ ПРОЧИТАНЫ: MAPPING 905, MERGE 69, CONCEPTS 32, FACTS 1307, SIMILAR_TO 492.',flush=True)
    banner('НАЧИНАЮ ОБРАБОТКУ 905 СУЩНОСТЕЙ')
    original={};approved={};row_by_id={}
    for row in progress(rows,total=905,desc='ОБРАБОТКА СУЩНОСТЕЙ'):
        eid=row['canonical_entity_id'];e=row['original_canonical_entity']
        require(eid not in original and e['canonical_entity_id']==eid,'Duplicate or wrong mapping ID')
        live=source_nodes[SOURCE_NS+':'+eid]['properties']
        require(json.loads(live['source_json'])==e,'Approved source entity differs from live graph')
        require(live['confidence']>=.46 and live['ner_entity_confidence_strategy']=='max_member','Wrong baseline NER configuration')
        original[eid]=e;approved[eid]=row['proposed_final_id'];row_by_id[eid]=row
    require(all(target in original and approved[target]==target for target in approved.values()),'Non-idempotent approved mapping')
    print('ОБРАБОТАНО 905/905 СУЩНОСТЕЙ; ИСХОДНЫЕ ТИПЫ И PROVENANCE СОХРАНЕНЫ.',flush=True)
    banner('НАЧИНАЮ ПРИМЕНЕНИЕ 69 УТВЕРЖДЁННЫХ MERGE-ГРУПП')
    mapping={eid:eid for eid in original};used=set();merge_by_target={}
    for group in progress(merges,total=69,desc='MERGE-ГРУППЫ'):
        ids=group['member_canonical_ids'];target=group['representative_id']
        require(target in ids and not used&set(ids),'Overlapping approved groups or invalid representative')
        require(len({original[eid]['document_id'] for eid in ids})==1,'Merge mixes source documents')
        for eid in ids:
            require(approved[eid]==target and row_by_id[eid]['merge_group_id']==group['merge_group_id'],'Merge differs from approved mapping')
            mapping[eid]=target
        used.update(ids);merge_by_target[target]=group
    require(mapping==approved and len(used)==163 and len(set(mapping.values()))==811,'Mapping counts differ')
    print('ПРИМЕНЕНО 69/69 УТВЕРЖДЁННЫХ ГРУПП; НОВЫЕ ГРУППЫ НЕ СОЗДАВАЛИСЬ.',flush=True)
    grouped=defaultdict(list)
    for eid,target in mapping.items():grouped[target].append(eid)
    signature=dict(stage='2F.3',schema_version='er2f3-lossless-v1',namespace=TARGET_NS,source_namespace=SOURCE_NS,
                   ner_threshold=.46,ner_confidence_strategy='max_member',embedding_model='intfloat/e5-small-v2',semantic_threshold=.95,
                   keep_all_similar_to=True,keep_parallel_relationships=True,keep_self_loops=True,
                   source_snapshot_sha256=source_digest,
                   inputs={p.relative_to(ROOT).as_posix():hashes[p] for p in [SOURCE/name for name in INPUT_NAMES]})
    fingerprint=base.core.digest(signature)
    def marked(properties,identity):
        return dict(properties,key=key(identity),namespace=TARGET_NS,fingerprint=fingerprint,
                    source_namespace=SOURCE_NS,reconstruction_stage='2F.3')
    nodes={};edges={};source_members={}
    for k,row in source_nodes.items():
        if 'RELEXDocument' in row['labels']:
            doc=row['properties']['document_id']
            nodes[key(doc)]=dict(label='RELEXDocument',properties=marked(dict(row['properties'],source_key=k),doc))
    banner('НАЧИНАЮ ФОРМИРОВАНИЕ 811 ИТОГОВЫХ КОНТЕКСТНЫХ СУЩНОСТЕЙ')
    for target,ids in progress(sorted(grouped.items()),total=811,desc='ИТОГОВЫЕ СУЩНОСТИ'):
        ids=sorted(ids);representative=original[target]
        members=[m['source_entity'] for eid in ids for m in original[eid]['members']]
        require(len({m['entity_id'] for m in members})==len(members),'Duplicate source entity in merged node')
        best=min(members,key=lambda m:(-m['confidence'],m['entity_id']))
        own=next(m for m in representative['members'] if m['source_entity']['entity_id']==target)['source_entity']
        snapshots=[source_nodes[SOURCE_NS+':'+eid] for eid in ids]
        source=dict(schema_version='er2f3-context-v1',representative_id=target,source_canonical_nodes=snapshots,
                    approved_merge=merge_by_target.get(target),source_mapping=[dict(canonical_entity_id=eid,proposed_final_id=target) for eid in ids])
        p=dict(entity_id=target,canonical_entity_id=target,name=representative['surface_text'],surface_text=representative['surface_text'],
               document_id=representative['document_id'],source_canonical_ids=ids,
               entity_types=sorted({t for eid in ids for t in original[eid]['entity_types']}),
               source_original_type_sets_json=dumps({eid:original[eid]['entity_types'] for eid in ids}),
               aliases=sorted({original[eid]['surface_text'] for eid in ids}|{m['surface_text'] for m in members}),
               source_entity_ids=sorted(m['entity_id'] for m in members),
               source_mention_ids=sorted({mid for eid in ids for mid in original[eid]['source_mention_ids']}),
               source_offsets_json=dumps([o for eid in ids for o in original[eid]['source_offsets']]),
               source_confidences_json=dumps([dict(entity_id=m['entity_id'],confidence=m['confidence']) for m in members]),
               source_canonical_confidences_json=dumps({eid:source_nodes[SOURCE_NS+':'+eid]['properties']['confidence'] for eid in ids}),
               confidence=best['confidence'],max_member_confidence=best['confidence'],confidence_source_entity_id=best['entity_id'],
               representative_confidence=own['confidence'],representative_entity_id=target,span_source_entity_id=target,start=own['start'],end=own['end'],
               ner_entity_threshold=.46,ner_entity_confidence_strategy='max_member',source_json=dumps(source),
               merge_group_id=merge_by_target[target]['merge_group_id'] if target in merge_by_target else None,
               representative_context_json=dumps(representative['representative_context']))
        nodes[key(target)]=dict(label='RELEXEntity',properties=marked(p,target))
        source_members[target]=members
    require(sum(len(v) for v in source_members.values())==993,'Source-entity coverage lost')
    print('СФОРМИРОВАНО 811 СУЩНОСТЕЙ; 163 ID В 69 ГРУППАХ, УСТРАНЕНО 94 ПОВТОРА.',flush=True)
    banner('НАЧИНАЮ СОЗДАНИЕ 32 ГЛОБАЛЬНЫХ КОНЦЕПТОВ')
    for raw in progress(concepts,total=32,desc='ГЛОБАЛЬНЫЕ КОНЦЕПТЫ'):
        cid=raw['concept_id'];ids=json.loads(raw['source_canonical_ids'])
        p=base.props(raw)
        p.update(concept_id=cid,name=raw['label'],label=raw['label'],entity_types=['CONCEPT'],
                 contextual_entity_count=int(raw['contextual_entity_count']),
                 source_document_ids=json.loads(raw['document_ids']),source_canonical_ids=ids,
                 contextual_target_ids=json.loads(raw['contextual_target_ids']),source_entity_ids=sorted({sid for eid in ids for sid in original[eid]['source_entity_ids']}),
                 source_mention_ids=sorted({mid for eid in ids for mid in original[eid]['source_mention_ids']}),
                 source_contexts_json=dumps({eid:[m['context'] for m in original[eid]['members']] for eid in ids}),
                 source_file=(SOURCE/'concept_candidates.csv').relative_to(ROOT).as_posix(),fact_inheritance_allowed=False)
        require('document_id' not in p and key(cid) not in nodes,'Concept has a document owner or duplicate ID')
        nodes[key(cid)]=dict(label='RELEXConcept',properties=marked(p,cid))
    print('СОЗДАНО РОВНО 32 КОНЦЕПТА; ДОКУМЕНТНЫХ MEMBERSHIP У КОНЦЕПТОВ НЕТ.',flush=True)
    banner('НАЧИНАЮ СОЗДАНИЕ 118 КОНЦЕПТУАЛЬНЫХ ОТНОШЕНИЙ')
    for raw in progress(concept_links,total=118,desc='HAS_SHARED_CONCEPT'):
        require(all(mapping[eid]==raw['head_id'] for eid in raw['source_canonical_ids']),'Concept link differs from mapping')
        p=base.props(raw)
        p.update(source_entity_ids=sorted({sid for eid in raw['source_canonical_ids'] for sid in original[eid]['source_entity_ids']}),
                 source_mention_ids=sorted({mid for eid in raw['source_canonical_ids'] for mid in original[eid]['source_mention_ids']}))
        p=marked(p,raw['concept_link_id'])
        edges[p['key']]=dict(type='HAS_SHARED_CONCEPT',head=key(raw['head_id']),tail=key(raw['tail_id']),properties=p)
    print('СОЗДАНО РОВНО 118 HAS_SHARED_CONCEPT.',flush=True)
    banner('НАЧИНАЮ ОБРАБОТКУ 1307 ФАКТИЧЕСКИХ ОТНОШЕНИЙ')
    for raw in progress(facts,total=1307,desc='ФАКТИЧЕСКИЕ СВЯЗИ'):
        rid=raw['canonical_triple_id'];source=source_edges[SOURCE_NS+':'+rid]
        require(source['type']=='RELEX_REL' and json.loads(source['properties']['source_json'])==raw,'Source fact differs from approved provenance')
        a,b=mapping[raw['head_id']],mapping[raw['tail_id']]
        p=dict(source['properties'],source_relationship_json=dumps(source),original_head_id=raw['head_id'],original_tail_id=raw['tail_id'],
               head_id=a,tail_id=b,original_is_self_loop=raw['is_self_loop'],is_self_loop=a==b,
               self_loop_created_by_er2f3=a==b and raw['head_id']!=raw['tail_id'])
        p=marked(p,rid)
        require(p['key'] not in edges,'Factual ID collision')
        edges[p['key']]=dict(type='RELEX_REL',head=key(a),tail=key(b),properties=p)
    print('СОХРАНЕНО 1307/1307 RELEX_REL; ПАРАЛЛЕЛЬНЫЕ ОТНОШЕНИЯ И ПЕТЛИ НЕ УДАЛЯЛИСЬ.',flush=True)
    banner('НАЧИНАЮ ОБРАБОТКУ ВСЕХ 492 СЕМАНТИЧЕСКИХ СВЯЗЕЙ')
    for preview in progress(semantic,total=492,desc='SIMILAR_TO'):
        raw=preview['original_link'];rid=raw['semantic_link_id'];source=source_edges[SOURCE_NS+':'+rid]
        require(source['type']=='SIMILAR_TO' and json.loads(source['properties']['source_json'])==raw,'Source similarity differs from approved provenance')
        a,b=mapping[raw['left_canonical_id']],mapping[raw['right_canonical_id']]
        require(a==preview['head_id'] and b==preview['tail_id'],'Semantic endpoints differ from approved preview')
        p=dict(source['properties'],source_relationship_json=dumps(source),original_left_canonical_id=raw['left_canonical_id'],
               original_right_canonical_id=raw['right_canonical_id'],left_canonical_id=a,right_canonical_id=b,
               original_head_id=raw['left_canonical_id'],original_tail_id=raw['right_canonical_id'],
               is_self_loop=a==b,resolution_decision_2f2=preview['decision'],preview_status_2f2=preview['status'])
        p=marked(p,rid)
        require(p['key'] not in edges and p['threshold']==.95,'Semantic ID collision or threshold changed')
        edges[p['key']]=dict(type='SIMILAR_TO',head=key(a),tail=key(b),properties=p)
    print('СОХРАНЕНО 492/492 SIMILAR_TO, ВКЛЮЧАЯ REJECT, REVIEW И SELF-LOOPS.',flush=True)
    banner('НАЧИНАЮ ФОРМИРОВАНИЕ 811 ДОКУМЕНТНЫХ MEMBERSHIP ИЗ 905 ИСХОДНЫХ')
    old_mentions={r['tail'][len(SOURCE_NS)+1:]:r for r in source_edges.values() if r['type']=='RELEX_MENTIONS'}
    for target,ids in progress(sorted(grouped.items()),total=811,desc='RELEX_MENTIONS'):
        n=nodes[key(target)]['properties'];original_memberships=[old_mentions[eid] for eid in sorted(ids)]
        require(all(r['head']==SOURCE_NS+':'+n['document_id'] for r in original_memberships),'Membership mixes documents')
        rid='mention:'+target
        p=marked(dict(document_id=n['document_id'],source_canonical_ids=sorted(ids),source_membership_ids=[r['key'] for r in original_memberships],
                      source_membership_count=len(original_memberships),source_memberships_json=dumps(original_memberships),
                      source_entity_ids=n['source_entity_ids'],source_mention_ids=n['source_mention_ids'],source_offsets_json=n['source_offsets_json']),rid)
        edges[p['key']]=dict(type='RELEX_MENTIONS',head=key(n['document_id']),tail=key(target),properties=p)
    print('СОЗДАНО 811 MEMBERSHIP; ПОЛНЫЕ 905 ИСХОДНЫХ ЗАПИСЕЙ СОХРАНЕНЫ.',flush=True)
    require(len(nodes)==873 and len(edges)==2728,'Physical payload counts mismatch')
    require(Counter(r['label'] for r in nodes.values())==Counter(RELEXDocument=30,RELEXEntity=811,RELEXConcept=32),'Node label counts mismatch')
    require(Counter(r['type'] for r in edges.values())==Counter(RELEX_REL=1307,SIMILAR_TO=492,HAS_SHARED_CONCEPT=118,RELEX_MENTIONS=811),'Relationship type counts mismatch')
    for r in edges.values():
        require(r['head'] in nodes and r['tail'] in nodes,'Missing endpoint')
        if r['type']=='RELEX_REL':require(nodes[r['head']]['label']==nodes[r['tail']]['label']=='RELEXEntity','Concept inherited fact')
    calculated,only_loops=metrics(nodes,edges)
    require((calculated['isolated'],calculated['weak_components'],calculated['largest_component_vertices'],calculated['largest_component_documents'])==(172,223,369,17),'Computed graph differs from 2F.2 all-links view')
    require(sum(len(r['properties']['source_triple_ids']) for r in edges.values() if r['type']=='RELEX_REL')==1490,'Raw source triples lost')
    write_json('manifest.json',dict(**signature,fingerprint=fingerprint,counts=EXPECTED,connectivity=calculated,
                                  merge_groups=69,merged_canonical_ids=163,removed_vertices=94,
                                  source_entity_ids=993,source_mention_ids=1101,original_memberships=905,
                                  source_code_sha256=base.sha(Path(__file__)),tqdm_version=__import__('tqdm').__version__))
    with (OUT/'canonical_mapping.jsonl').open('w',encoding='utf-8',newline='\n') as stream:
        for eid,target in sorted(mapping.items()):stream.write(dumps(dict(source_canonical_id=eid,target_canonical_id=target,source_key=SOURCE_NS+':'+eid,target_key=key(target),document_id=original[eid]['document_id']))+'\n')
    write_json('self_loop_only_vertices.json',[dict(key=n,name=nodes[n]['properties']['name'],document_id=nodes[n]['properties']['document_id']) for n in only_loops])
    print('ЛОКАЛЬНЫЙ ГРАФ ПРОВЕРЕН: '+dumps(dict(nodes=873,relationships=2728,connectivity=calculated)),flush=True)
    return nodes,edges,mapping,fingerprint,calculated,signature


def assert_namespace(tx,fingerprint):
    for query in ['MATCH (n {namespace:$ns}) RETURN DISTINCT n.fingerprint AS fingerprint',
                  'MATCH ()-[r]->() WHERE r.namespace=$ns RETURN DISTINCT r.fingerprint AS fingerprint']:
        require(all(r['fingerprint']==fingerprint for r in tx.run(query,ns=TARGET_NS)),
                'Target namespace belongs to another snapshot; refusing overwrite')


def import_once(session,nodes,edges,fingerprint,pass_number):
    global LAST_BATCH
    banner(f'НАЧИНАЮ ИМПОРТ GLOBAL KG В NEO4J — ПРОХОД {pass_number}')
    def preflight(tx):
        assert_namespace(tx,fingerprint)
        return tx.run(statements()[0]).single().data()
    existing=session.execute_read(preflight)
    if existing['total_nodes'] or existing['total_relationships']:
        require(existing==dict(namespace=TARGET_NS,**EXPECTED),'Existing target snapshot is incomplete; refusing partial rewrite')
        print('СНИМОК С ЭТИМ FINGERPRINT СУЩЕСТВУЕТ. ПОЛНОЕ ЧТЕНИЕ ПЕРЕД ИДЕМПОТЕНТНЫМ NO-OP.',flush=True)
        verify(session,nodes,edges)
        print('ИДЕМПОТЕНТНЫЙ ПРОХОД ПОДТВЕРЖДЁН: СОЗДАНО 0 УЗЛОВ, 0 ОТНОШЕНИЙ; ЗАПИСЕЙ НЕТ.',flush=True)
        return dict(nodes=0,relationships=0)
    def transaction(tx):
        global LAST_BATCH
        assert_namespace(tx,fingerprint)
        counters=dict(nodes=0,relationships=0)
        def consume(query,**params):
            result=tx.run(query,namespace=TARGET_NS,**params)
            records=result.data()
            summary=result.consume()
            counters['nodes']+=summary.counters.nodes_created
            counters['relationships']+=summary.counters.relationships_created
            return records
        with progress(total=len(nodes),desc='ИМПОРТ УЗЛОВ') as bar:
            documents=[r for r in nodes.values() if r['label']=='RELEXDocument']
            for i,row in enumerate(documents,1):
                consume(base.core.DOCUMENT_QUERY,key=row['properties']['key'],properties=row['properties'])
                LAST_BATCH=f'PASS {pass_number}, RELEXDocument, {i}/{len(documents)}';bar.update(1)
            for label in ('RELEXEntity','RELEXConcept'):
                data=[r for r in nodes.values() if r['label']==label]
                for start in range(0,len(data),BATCH):
                    batch=data[start:start+BATCH]
                    consume(f'UNWIND $rows AS row MERGE (n:{label} {{key:row.properties.key,namespace:$namespace}}) SET n += row.properties',rows=batch)
                    LAST_BATCH=f'PASS {pass_number}, {label}, {start+1}..{start+len(batch)}/{len(data)}';bar.update(len(batch))
        with progress(total=len(edges),desc='ИМПОРТ ОТНОШЕНИЙ') as bar:
            for kind in REL_TYPES:
                data=[r for r in edges.values() if r['type']==kind]
                for start in range(0,len(data),BATCH):
                    batch=data[start:start+BATCH]
                    if kind=='RELEX_REL':
                        records=consume(base.core.EDGE_QUERY,rows=[dict(head_key=r['head'],tail_key=r['tail'],properties=r['properties']) for r in batch])
                        require(records[0]['n']==len(batch),'Incomplete factual batch')
                    else:
                        hlabel='RELEXDocument' if kind=='RELEX_MENTIONS' else 'RELEXEntity'
                        tlabel='RELEXConcept' if kind=='HAS_SHARED_CONCEPT' else 'RELEXEntity'
                        query=(f'UNWIND $rows AS row MATCH (h:{hlabel} {{key:row.head,namespace:$namespace}}) '
                               f'MATCH (t:{tlabel} {{key:row.tail,namespace:$namespace}}) '
                               f'MERGE (h)-[r:{kind} {{key:row.properties.key,namespace:$namespace}}]->(t) SET r += row.properties RETURN count(r) AS n')
                        records=consume(query,rows=batch)
                        require(records[0]['n']==len(batch),'Incomplete relationship batch '+kind)
                    LAST_BATCH=f'PASS {pass_number}, {kind}, {start+1}..{start+len(batch)}/{len(data)}';bar.update(len(batch))
        counts=tx.run(statements()[0]).single().data()
        require(counts==dict(namespace=TARGET_NS,**EXPECTED),'Atomic count verification failed: '+dumps(counts))
        return counters
    result=session.execute_write(transaction)
    print('ТРАНЗАКЦИЯ ПОДТВЕРЖДЕНА: '+dumps(result),flush=True)
    return result


def verify(session,nodes,edges):
    banner('НАЧИНАЮ ПРОВЕРКУ ПОЛНОТЫ ИМПОРТИРОВАННОГО ГРАФА')
    actual_nodes={};actual_edges={};digest=hashlib.sha256()
    result=session.run('MATCH (n {namespace:$ns}) RETURN n.key AS key,labels(n) AS labels,properties(n) AS properties ORDER BY key',ns=TARGET_NS)
    for row in progress(result,total=873,desc='ПРОВЕРКА УЗЛОВ'):
        row=row.data();wanted=nodes.get(row['key'])
        require(wanted is not None and row['labels']==[wanted['label']],'Unexpected node or label')
        require(row['properties']=={k:v for k,v in wanted['properties'].items() if v is not None},'Node properties/provenance mismatch '+row['key'])
        actual_nodes[row['key']]=dict(label=wanted['label'],properties=row['properties'])
        digest.update((dumps(row)+'\n').encode())
    require(not result.consume().counters.contains_updates,'Verification updated nodes')
    result=session.run('MATCH (h)-[r]->(t) WHERE r.namespace=$ns RETURN r.key AS key,type(r) AS type,h.key AS head,t.key AS tail,properties(r) AS properties ORDER BY key',ns=TARGET_NS)
    for row in progress(result,total=2728,desc='ПРОВЕРКА ОТНОШЕНИЙ'):
        row=row.data();wanted=edges.get(row['key'])
        require(wanted is not None and all(row[k]==wanted[k] for k in ('type','head','tail')),'Relationship identity/type/endpoints mismatch')
        require(row['properties']=={k:v for k,v in wanted['properties'].items() if v is not None},'Relationship provenance mismatch '+row['key'])
        actual_edges[row['key']]=dict(type=row['type'],head=row['head'],tail=row['tail'],properties=row['properties'])
        digest.update((dumps(row)+'\n').encode())
    require(not result.consume().counters.contains_updates,'Verification updated relationships')
    require(set(actual_nodes)==set(nodes) and set(actual_edges)==set(edges),'Incomplete live graph coverage')
    result=metrics(actual_nodes,actual_edges)[0]
    print('ВСЕ 873 УЗЛА И 2728 ОТНОШЕНИЙ ПРОЧИТАНЫ И СВЕРЕНЫ.',flush=True)
    return result,digest.hexdigest(),actual_nodes,actual_edges


def verify_restoration(source_nodes,source_edges,actual_nodes,actual_edges):
    banner('НАЧИНАЮ ПРОВЕРКУ ВОССТАНОВИМОСТИ ИСХОДНЫХ ID И PROVENANCE')
    restored_nodes={};restored_edges={}
    for row in progress(actual_nodes.values(),total=873,desc='ВОССТАНОВЛЕНИЕ SOURCE NODES'):
        p=row['properties']
        if row['label']=='RELEXEntity':
            decoded=json.loads(p['source_json'])
            for original in decoded['source_canonical_nodes']:
                require(original['key'] not in restored_nodes,'Source entity represented twice')
                restored_nodes[original['key']]=original
            raw_types=sorted({t for source in decoded['source_canonical_nodes'] for t in source['properties']['entity_types']})
            require(raw_types==p['entity_types'],'Proposed primary type replaced source types')
        elif row['label']=='RELEXDocument':
            source=source_nodes[p['source_key']]
            require(p['source_json']==source['properties']['source_json'],'Document source JSON changed')
    raw_ids=set();source_native=set();source_mentions=set();canonical_ids=set()
    for restored in restored_nodes.values():
        e=json.loads(restored['properties']['source_json'])
        canonical_ids.add(e['canonical_entity_id']);source_native.update(e['source_entity_ids']);source_mentions.update(e['source_mention_ids'])
    for row in progress(actual_edges.values(),total=2728,desc='ВОССТАНОВЛЕНИЕ SOURCE EDGES'):
        p=row['properties']
        if row['type'] in ('RELEX_REL','SIMILAR_TO'):
            source=json.loads(p['source_relationship_json']);require(source['key'] not in restored_edges,'Source edge duplicated')
            restored_edges[source['key']]=source
            if row['type']=='RELEX_REL':raw_ids.update(p['source_triple_ids'])
        elif row['type']=='RELEX_MENTIONS':
            for source in json.loads(p['source_memberships_json']):
                require(source['key'] not in restored_edges,'Source membership duplicated');restored_edges[source['key']]=source
        else:
            require(not p['factual'] and not p['fact_inheritance_allowed'],'Concept edge inherited fact')
    require(restored_nodes=={k:r for k,r in source_nodes.items() if 'RELEXEntity' in r['labels']},'Cannot reconstruct all 905 entity snapshots')
    require(restored_edges==source_edges,'Cannot reconstruct all source relationships and memberships')
    require((len(canonical_ids),len(source_native),len(source_mentions),len(raw_ids))==(905,993,1101,1490),'Source ID coverage lost')
    print('ВОССТАНОВЛЕНЫ: 905 CANONICAL ID, 993 ENTITY ID, 1101 MENTION ID, 1490 RAW TRIPLE ID И ВСЕ 905 MEMBERSHIP.',flush=True)
    return dict(canonical_ids=905,source_entity_ids=993,source_mention_ids=1101,raw_triple_ids=1490,
                original_memberships=905,original_factual_relations=1307,original_similar_to=492,
                full_source_node_properties_exact=True,full_source_edge_properties_exact=True,types_not_replaced=True)


def statements():
    text='\n'.join(line for line in (OUT/'graph_stats.cypher').read_text(encoding='utf-8').splitlines() if not line.lstrip().startswith('//'))
    return [q.strip() for q in text.split(';') if q.strip()]


def write_queries():
    prefix=f"CYPHER 25\nWITH '{TARGET_NS}' AS ns\n"
    queries=[]
    queries.append(prefix+'''CALL (ns) { MATCH (n:RELEXDocument {namespace:ns}) RETURN count(n) AS documents }
CALL (ns) { MATCH (n:RELEXEntity {namespace:ns}) RETURN count(n) AS entities }
CALL (ns) { MATCH (n:RELEXConcept {namespace:ns}) RETURN count(n) AS concepts }
CALL (ns) { MATCH ()-[r:RELEX_REL {namespace:ns}]->() RETURN count(r) AS factual_relations }
CALL (ns) { MATCH ()-[r:SIMILAR_TO {namespace:ns}]->() RETURN count(r) AS semantic_links }
CALL (ns) { MATCH ()-[r:HAS_SHARED_CONCEPT {namespace:ns}]->() RETURN count(r) AS concept_links }
CALL (ns) { MATCH ()-[r:RELEX_MENTIONS {namespace:ns}]->() RETURN count(r) AS document_mentions }
RETURN ns AS namespace,documents,entities,concepts,entities+concepts AS kg_vertices,
       documents+entities+concepts AS total_nodes,factual_relations,semantic_links,concept_links,document_mentions,
       factual_relations+semantic_links+concept_links AS kg_edges,
       factual_relations+semantic_links+concept_links+document_mentions AS total_relationships''')
    queries.append(prefix+'''MATCH (a:RELEXEntity {namespace:ns})-[r:SIMILAR_TO {namespace:ns}]->(b:RELEXEntity {namespace:ns})
RETURN count(r) AS semantic_links,count(DISTINCT r.semantic_link_id) AS distinct_original_ids,
sum(CASE WHEN a.document_id<>b.document_id THEN 1 ELSE 0 END) AS cross_document,
sum(CASE WHEN a=b THEN 1 ELSE 0 END) AS self_loops,min(r.similarity) AS minimum_cosine,max(r.similarity) AS maximum_cosine''')
    queries.append(prefix+'''MATCH (a:RELEXEntity {namespace:ns})-[r:RELEX_REL {namespace:ns}]->(b:RELEXEntity {namespace:ns})
WITH count(r) AS factual_relations,count(DISTINCT r.canonical_triple_id) AS distinct_original_ids,
sum(CASE WHEN a=b THEN 1 ELSE 0 END) AS self_loops,
sum(CASE WHEN a.document_id<>b.document_id THEN 1 ELSE 0 END) AS cross_document,
reduce(all_ids=[], ids IN collect(r.source_triple_ids) | all_ids+ids) AS source_ids
UNWIND source_ids AS source_id
RETURN factual_relations,distinct_original_ids,self_loops,cross_document,count(DISTINCT source_id) AS raw_source_triple_ids''')
    queries.append(prefix+'''MATCH (e {namespace:ns}) WHERE e:RELEXEntity OR e:RELEXConcept
WITH ns,e,
 EXISTS { MATCH (e)-[r:RELEX_REL|SIMILAR_TO|HAS_SHARED_CONCEPT]-(o {namespace:ns}) WHERE r.namespace=ns } AS touched,
 EXISTS { MATCH (e)-[r:RELEX_REL|SIMILAR_TO|HAS_SHARED_CONCEPT]-(o {namespace:ns}) WHERE r.namespace=ns AND o<>e } AS external
RETURN count(e) AS kg_vertices,sum(CASE WHEN NOT touched THEN 1 ELSE 0 END) AS isolated,
sum(CASE WHEN touched AND NOT external THEN 1 ELSE 0 END) AS self_loop_only_vertices,
sum(CASE WHEN NOT external THEN 1 ELSE 0 END) AS without_external_edges''')
    queries.append(prefix+'''MATCH (e {namespace:ns}) WHERE e:RELEXEntity OR e:RELEXConcept
WITH ns,e ORDER BY e.key
WITH ns,collect(e) AS entities
UNWIND range(0,size(entities)-1) AS vertex
WITH ns,entities,vertex,entities[vertex] AS e
CALL (e,ns,entities) {
 OPTIONAL MATCH (e)-[r:RELEX_REL|SIMILAR_TO|HAS_SHARED_CONCEPT]-(other {namespace:ns}) WHERE r.namespace=ns
 WITH entities,collect(DISTINCT other) AS neighbors
 RETURN [i IN range(0,size(entities)-1) WHERE entities[i] IN neighbors] AS neighbor_indices
}
WITH entities,vertex,neighbor_indices ORDER BY vertex
WITH entities,collect(neighbor_indices) AS adjacency
WITH entities,reduce(labels=range(0,size(entities)-1), round IN range(1,size(entities)-1) |
 [i IN range(0,size(entities)-1) | reduce(best=labels[i], neighbor IN adjacency[i] |
 CASE WHEN labels[neighbor]<best THEN labels[neighbor] ELSE best END)]) AS labels
UNWIND range(0,size(entities)-1) AS vertex
WITH labels[vertex] AS component_id,entities[vertex].document_id AS document_id
WITH component_id,count(*) AS component_size,count(DISTINCT document_id) AS document_count
ORDER BY component_size DESC,component_id
RETURN count(*) AS weak_components,sum(component_size) AS kg_vertices,
max(component_size) AS largest_component_vertices,head(collect(document_count)) AS largest_component_documents''')
    queries.append(prefix+'''MATCH (a:RELEXEntity {namespace:ns})-[r:RELEX_REL {namespace:ns}]->(b:RELEXEntity {namespace:ns})
WITH a.key AS head,r.relation_type AS relation_type,b.key AS tail,count(r) AS records
RETURN count(*) AS unique_structures,sum(records) AS factual_records,
sum(CASE WHEN records>1 THEN 1 ELSE 0 END) AS parallel_groups,sum(records-1) AS parallel_record_excess''')
    queries.append(prefix+'''MATCH (c:RELEXConcept {namespace:ns})
OPTIONAL MATCH (e:RELEXEntity {namespace:ns})-[r:HAS_SHARED_CONCEPT {namespace:ns}]->(c)
RETURN c.concept_id AS concept_id,c.name AS concept,count(r) AS contextual_links,count(DISTINCT e.document_id) AS source_documents
ORDER BY contextual_links DESC,concept_id''')
    queries.append(prefix+'''MATCH (d:RELEXDocument {namespace:ns})-[r:RELEX_MENTIONS {namespace:ns}]->(e:RELEXEntity {namespace:ns})
RETURN count(r) AS memberships,sum(r.source_membership_count) AS restored_original_memberships,
sum(CASE WHEN d.document_id<>e.document_id THEN 1 ELSE 0 END) AS mismatched_documents''')
    queries.append(prefix+'''MATCH (c:RELEXConcept {namespace:ns})
RETURN count(c) AS concepts,
sum(CASE WHEN c.document_id IS NOT NULL THEN 1 ELSE 0 END) AS concepts_with_document_owner,
sum(CASE WHEN EXISTS { MATCH (c)-[r:RELEX_REL]-() WHERE r.namespace=ns } THEN 1 ELSE 0 END) AS concepts_with_factual_edges,
sum(CASE WHEN EXISTS { MATCH ()-[r:RELEX_MENTIONS]->(c) WHERE r.namespace=ns } THEN 1 ELSE 0 END) AS concepts_with_memberships''')
    queries.append(prefix+'''MATCH (e:RELEXEntity {namespace:ns})-[r:HAS_SHARED_CONCEPT {namespace:ns}]->(c:RELEXConcept {namespace:ns,name:'Patients'})
RETURN e.canonical_entity_id AS contextual_id,e.name AS name,e.document_id AS document_id,
e.source_canonical_ids AS original_canonical_ids,e.entity_types AS original_type_union,r.source_canonical_ids AS concept_evidence_ids
ORDER BY document_id,contextual_id''')
    (OUT/'graph_stats.cypher').write_text('// Stage 2F.3: read-only queries; run statements separately. No APOC/GDS.\n\n'+';\n\n'.join(queries)+';\n',encoding='utf-8',newline='\n')
    full=f"""// All 873 nodes and all 2728 relationships, including documents, concepts and isolated entities.
// Set Neo4j Browser Graph visualization node limit >= 1000.
WITH '{TARGET_NS}' AS ns
MATCH (n {{namespace:ns}})
OPTIONAL MATCH (n)-[r]->(m {{namespace:ns}})
WHERE r.namespace=ns
RETURN n,r,m;
"""
    (OUT/'show_full_graph.cypher').write_text(full,encoding='utf-8',newline='\n')


def verify_queries(session,calculated):
    banner('НАЧИНАЮ НЕЗАВИСИМУЮ ПРОВЕРКУ ГРАФА ЧЕРЕЗ LIVE CYPHER')
    from neo4j import Query
    results=[]
    for i,query in enumerate(progress(statements(),total=10,desc='ПРОВЕРКИ CYPHER'),1):
        result=session.run(Query(query,timeout=90));data=result.data();summary=result.consume()
        require(not summary.counters.contains_updates and data,'Read-only Cypher check failed')
        require(not any(s.is_notification and 'deprecat' in s.status_description.lower() for s in summary.gql_status_objects),'Deprecated Cypher syntax')
        row=data[0]
        if i==1:require(row==dict(namespace=TARGET_NS,**EXPECTED),'Counts differ in Cypher')
        elif i==2:require(row['semantic_links']==row['distinct_original_ids']==492 and row['cross_document']==121 and row['self_loops']==72,'Semantic loss or loop loss')
        elif i==3:require(row==dict(factual_relations=1307,distinct_original_ids=1307,self_loops=39,cross_document=0,raw_source_triple_ids=1490),'Factual provenance counts differ')
        elif i==4:require(row=={k:calculated[k] for k in ['kg_vertices','isolated','self_loop_only_vertices','without_external_edges']},'Isolation differs')
        elif i==5:require(row=={k:calculated[k] for k in ['weak_components','kg_vertices','largest_component_vertices','largest_component_documents']},'Live WCC differ')
        elif i==6:require(row==dict(unique_structures=1076,factual_records=1307,parallel_groups=164,parallel_record_excess=231),'Parallel facts collapsed')
        elif i==7:require(len(data)==32 and sum(r['contextual_links'] for r in data)==118,'Concept links differ')
        elif i==8:require(row==dict(memberships=811,restored_original_memberships=905,mismatched_documents=0),'Source memberships lost')
        elif i==9:require(row==dict(concepts=32,concepts_with_document_owner=0,concepts_with_factual_edges=0,concepts_with_memberships=0),'Concept layer violation')
        elif i==10:require(len(data)==16 and len({r['document_id'] for r in data})==5,'Patients layer differs from approved input')
        results.append(dict(query=i,rows=len(data),result=data))
    result=session.run((OUT/'show_full_graph.cypher').read_text(encoding='utf-8'))
    visible_nodes=set();visible_edges=set()
    for row in result:
        visible_nodes.add(row['n']['key'])
        if row['m'] is not None:visible_nodes.add(row['m']['key'])
        if row['r'] is not None:visible_edges.add(row['r']['key'])
    require(len(visible_nodes)==873 and len(visible_edges)==2728,'Full visualization query omits elements')
    return results,dict(nodes=len(visible_nodes),relationships=len(visible_edges))


def report(manifest,verification=None):
    c=manifest['connectivity'];success=verification is not None
    lines=['# Stage 2F.3 — Global KG Reconstruction & Neo4j Import','',
           '**Фактический импорт выполнен и проверен.**' if success else '**Подготовлен локальный payload; успешный импорт ещё не подтверждён.**','',
           f'Target: `{TARGET_NS}`. Source: `{SOURCE_NS}`. Дата: 12 ноября 2026, Europe/Moscow.','',
           'Применён только утверждённый mapping 2F.2: 69 групп /163 canonical IDs, устранены 94 повтора. Новые решения ER не выполнялись. NER 0.46/max_member, E5-small/name cosine 0.95 сохранены.','',
           '| Метрика | 2F.1 | 2F.3 |','|---|---:|---:|','| Документы | 30 | 30 |','| Контекстные сущности | 905 | 811 |',
           '| RELEXConcept | 0 | 32 |','| KG-вершины без документов | 905 | 843 |','| Физические узлы | 935 | 873 |',
           '| RELEX_REL | 1307 | 1307 |','| SIMILAR_TO | 492 | 492 |','| HAS_SHARED_CONCEPT | 0 | 118 |',
           '| RELEX_MENTIONS | 905 | 811 |','| Физические отношения | 2704 | 2728 |','| Изолированные KG-вершины (петля = связь) | 175 | 172 |',
           '| Слабосвязные компоненты KG | 230 | 223 |','| Крупнейшая компонента: вершины / документы | 378 / 17 | 369 / 17 |',
           f"| Вершины только с self-loops, без внешних KG-связей | — | {c['self_loop_only_vertices']} |",
           f"| Все вершины без внешних KG-связей | — | {c['without_external_edges']} |",'',
           '## Сохранность и представление','',
           '- Все 905 исходных canonical IDs однозначно отображены в 811 существующих representatives. Документы внутри групп не смешиваются. `canonical_mapping.jsonl` сохраняет это отображение.',
           '- У entity `source_json.source_canonical_nodes` содержит полные снимки всех исходных entity-узлов, включая их original source_json, offsets, contexts, scores, types и provenance. Восстановимы 993 source entity IDs и 1101 typed mention ID. `entity_types` — объединение исходных наборов; proposed_primary_type из 2F.2 не применяется.',
           '- Entity confidence — максимум сохранённых native member confidences; все индивидуальные source scores сохранены отдельно. Start/end относятся только к representative/span_source_entity_id. Новые NER scores не рассчитываются.',
           '- **Все 1307 RELEX_REL физически сохранены раздельно**, ID не пересчитаны. Их 1490 raw source triple IDs, исходные confidence/evidence/source_json сохранены. Endpoints переназначены по mapping; original_head_id/original_tail_id и source_relationship_json сохраняют исходное состояние.',
           '- **Все 492 SIMILAR_TO сохранены**, включая 37 REJECT и 70 REVIEW из 2F.2. Решения хранятся как диагностические свойства resolution_decision_2f2/preview_status_2f2 и не фильтруют граф. Не используется display-view из 2F.2 с сокращённым набором рёбер.',
           '- У SIMILAR_TO сохраняется semantic_link_id, точный исходный cosine и original endpoints; current left_canonical_id/right_canonical_id переназначены. Имена/типы/contexts и source IDs evidence на ребре относятся к исходным endpoints; полное исходное состояние есть в source_relationship_json. Прямые междокументные SIMILAR_TO: 121, внутридокументные: 371.',
           '- Parallel edges сохраняют разные relationship keys. Физических factual edges 1307, структурных (head,type,tail) 1076; 164 parallel groups, 231 дополнительных записей. Factual self-loops: 39 (12 прежних +27 вследствие mapping), semantic self-loops: 72. Они не удаляются и не требуют нового содержательного аудита этого этапа.',
           '- У 811 RELEX_MENTIONS поле source_memberships_json хранит полные 905 прежних membership records, включая их keys, endpoints и свойства. Концепты не имеют single document_id и не получают MENTIONS или RELEX_REL.',
           '- 32 concept IDs и ровно 118 HAS_SHARED_CONCEPT взяты из готовых файлов. Source IDs, исходные типы, контекстное evidence и provenance сохранены. Это слой абстракции, не наследование фактов.','',
           '## Связность и изоляция','',
           'KG состоит из RELEXEntity+RELEXConcept и RELEX_REL+SIMILAR_TO+HAS_SHARED_CONCEPT, без документов и RELEX_MENTIONS. Для слабых компонент направления игнорируются, параллельные записи сохраняются. Петля даёт ненулевую степень, поэтому вершина только с петлями не входит в 172 изолированные. Её считают отдельно в self_loop_only_vertices.json. Число документов в компоненте учитывает только contextual entities: concept не добавляет фиктивный документ.',
           '',f"Только self-loops: **{c['self_loop_only_vertices']}**; без каких-либо внешних связей: **{c['without_external_edges']} = 172 + {c['self_loop_only_vertices']}**. Рост связности или уменьшение числа вершин не интерпретируется как доказательство улучшения качества Graph-RAG.",'',
           '## Примеры концептов и Patients','',
           'Концепты включают Patients, Convergence, Genes, Eyes, Intraocular pressure, Energy consumption и другие 32 утверждённых класса. Общего Results нет. Концепты не получают факты исходных когорт/результатов.','',
           '`concept:2f2:patients` имеет 16 contextual links в пяти документах; их evidence охватывает 20 прежних contextual IDs, включая qualified patient forms. Когорта 21 ESRD/TKA-пациент объединяет только утверждённые N398/N399/N400; подгруппы other 20, six with complications, nineteen improved outcomes и Most patients остаются раздельными.',
           '',
           'Другие документные контексты: 10 CRSwNP patients, 40 GDD/PK patients, 16 esophageal-cancer/RT patients и общие популяции rHuEPO recipients/подгруппа с антителами. В rHuEPO-работе не домысливается собственная enrolled clinical cohort. Concept Patients — общая категория, не один коллектив; ни одно RELEX_REL не перенесено на неё. Полные source scopes и исходные типы остаются при contextual nodes.','',
           '## Импорт и воспроизведение','',
           'Импорт использует существующие connect_neo4j, DOCUMENT_QUERY, EDGE_QUERY и namespace-scoped MERGE patterns предыдущих импортёров. Создана только новая область TARGET; новые глобальные schema constraints не добавлялись. Ключи включают namespace и стабильный source/concept ID. Другой fingerprint в TARGET вызывает отказ до записи. Первичное создание узлов/рёбер выполняется одной транзакцией с проверкой counts до commit. Если TARGET уже содержит полный снимок с тем же fingerprint, импортёр полностью сверяет все свойства, ID и endpoints и выполняет no-op без повторных SET. Это обеспечивает идемпотентность без расходования transaction memory на повторную запись больших provenance-строк.',
           '',
           'Настоящая библиотека tqdm показывает обработку 905 сущностей, 69 групп, 811 итоговых сущностей, 32 концептов, 118 class links, 1307 factual edges, 492 SIMILAR_TO и 811 memberships; импорт и read-back показывают реальные counts. Batch-progress обновляется после успешного server response. Неопределённые по проценту операции полного hash-снимка сопровождаются CAPS-сообщениями. Искусственных задержек нет. При ошибке выводятся stage, последний успешно обработанный batch и исключение; success-banner не печатается при prepare-only/ошибке.',
           '',
           '```powershell','.venv/Scripts/python.exe scripts/reconstruct_global_kg_2f3.py','```','',
           'Нужны существующие NEO4J_* либо --neo4j-env <existing-file>. --prepare-only читает baseline и подготавливает файлы без записи в Neo4j. Credentials не записываются в артефакты. Зависимости neo4j/tqdm уже объявлены в extra relex; модели не загружаются.','',
           'Полный граф: выполнить show_full_graph.cypher; Node visualization limit ≥1000. Запрос возвращает все 873 узла и 2728 отношений, включая documents, concepts и KG-isolates с их служебными mentions. graph_stats.cypher содержит 10 самостоятельных read-only проверок, в том числе live WCC без GDS/APOC.','']
    if success:
        lines += ['## Подтверждённые результаты проверок','',
                  f"Время: {verification['verified_at_moscow']}. База: `{verification['database']}`.",'',
                  f"- Первый проход: `{dumps(verification['first_import'])}`; повторный проход: `{dumps(verification['repeated_import'])}`.",
                  '- После обоих проходов все 873 nodes и 2728 relationships полностью прочитаны; exact keys/labels/types/endpoints/properties совпадают с payload. SHA-256 целевого графа одинаков после двух проходов.',
                  '- Из прочитанных свойств строго восстановлены все 905 source entity nodes и все 2704 source relationships, включая 905 original memberships; полные source property dictionaries совпали с baseline.',
                  '- Все 10 live Cypher-проверок прошли; независимый Cypher WCC подтвердил 843/172/223/369/17. Проверены parallel edges, loops, raw triples и отсутствие factual links у concepts.',
                  '- Проверка с чужим fingerprint была отклонена в read-only guard без записи.',
                  '- Полный внешний снимок узлов/отношений вне TARGET и schema metadata одинаковы до/после. Предыдущие namespaces и файлы не изменены.',
                  '- Модели, новый ER, cosine, очистка/дедупликация отношений и этап 2F.4 не запускались.','']
        if (OUT/'recovery_notes.json').exists():
            lines += ['## Восстановление после лимита памяти повторной записи','',
                      'Первый фактический импорт создал 873 узла и 2728 отношений и прошёл полное чтение/проверку provenance. При попытке второго полного write-прохода с теми же свойствами Neo4j отклонила транзакцию по dbms.memory.transaction.total.max (716.8 MiB). Драйвер исчерпал retry; операция зафиксирована в initial_attempt_failure.json с этапом и последним обработанным batch.',
                      '',
                      'Идемпотентный путь исправлен: повторный запуск проверяет fingerprint, counts и точное содержимое всего TARGET, затем выполняет no-op. Настройки памяти, пароль, старые namespaces и утверждённая структура не менялись. Два завершившихся прохода подтверждены полным read-back и одинаковым target SHA-256. Подробности первоначального создания и исправления — recovery_notes.json; first_import/repeated_import в verification.json относятся к финальному успешному запуску, поэтому оба равны 0/0.',
                      '']
    lines+=['Артефакты: report.md, manifest.json, verification.json (при успешном импорте), graph_stats.cypher, show_full_graph.cypher, canonical_mapping.jsonl, self_loop_only_vertices.json.','']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')


def main():
    global IMPORT_COMMITTED
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--neo4j-env',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    banner('НАЧИНАЮ ФИКСАЦИЮ SHA-256 ИСХОДНЫХ ДАННЫХ И ПРЕДЫДУЩИХ АРТЕФАКТОВ')
    protected={p for p in ER.rglob('*') if p.is_file() and not p.is_relative_to(OUT)}
    protected.update([Path(__file__),ROOT/'scripts/build_relex_graphs.py',ROOT/'scripts/import_global_semantic_graph.py'])
    hashes={p:base.sha(p) for p in progress(sorted(protected),desc='SHA-256 ВХОДОВ')}
    banner('НАЧИНАЮ ПОДКЛЮЧЕНИЕ К СУЩЕСТВУЮЩЕЙ NEO4J')
    driver,database=base.core.connect_neo4j(args.neo4j_env)
    try:
        with driver.session(database=database) as session:
            source_nodes,source_edges,source_digest=source_snapshot(session)
            nodes,edges,mapping,fingerprint,calculated,signature=build(source_nodes,source_edges,source_digest,hashes)
            verify_restoration(source_nodes,source_edges,nodes,edges)
            write_queries()
            manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
            report(manifest)
            if args.prepare_only:
                banner('ЛОКАЛЬНАЯ ПОДГОТОВКА ЗАВЕРШЕНА; ИМПОРТ В NEO4J НЕ ВЫПОЛНЯЛСЯ')
                return
            banner('НАЧИНАЮ ПОЛНЫЙ СНИМОК СУЩЕСТВУЮЩИХ NAMESPACES — ПРОЦЕНТ НЕ ОПРЕДЕЛЁН')
            outside_before=base.outside_snapshot(session)
            write_json('outside_snapshot_before.json',outside_before)
            schema_query='SHOW CONSTRAINTS YIELD name,type,labelsOrTypes,properties RETURN name,type,labelsOrTypes,properties ORDER BY name'
            schema_before=session.run(schema_query).data()
            print('ВНЕШНИЙ СНИМОК ГОТОВ: '+dumps({k:v for k,v in outside_before.items() if k!='sha256'}),flush=True)
            banner('ВНИМАНИЕ! НАЧИНАЮ ФАКТИЧЕСКИЙ ИМПОРТ В NEO4J\nTARGET: '+TARGET_NS+'\nИСХОДНЫЕ NAMESPACES НЕ ИЗМЕНЯЮТСЯ')
            first=import_once(session,nodes,edges,fingerprint,1);IMPORT_COMMITTED=True
            actual,digest1,live_nodes,live_edges=verify(session,nodes,edges)
            require(actual==calculated,'Actual connectivity differs from constructed payload')
            restoration=verify_restoration(source_nodes,source_edges,live_nodes,live_edges)
            second=import_once(session,nodes,edges,fingerprint,2)
            require(second==dict(nodes=0,relationships=0),'Repeated import created duplicates')
            actual2,digest2,_,_=verify(session,nodes,edges)
            require(actual2==actual and digest1==digest2,'Repeated import changed data')
            banner('НАЧИНАЮ ПРОВЕРКУ ОТКАЗА ПРИ ЧУЖОМ FINGERPRINT')
            try:session.execute_read(lambda tx:assert_namespace(tx,'invalid-fingerprint-read-only-test'))
            except ValueError as exc:require('another snapshot' in str(exc),'Unexpected fingerprint error')
            else:raise ValueError('Fingerprint guard accepted another snapshot')
            checks,visualization=verify_queries(session,calculated)
            banner('НАЧИНАЮ ПОВТОРНЫЙ ПОЛНЫЙ СНИМОК ВНЕ TARGET — ПРОЦЕНТ НЕ ОПРЕДЕЛЁН')
            outside_after=base.outside_snapshot(session)
            schema_after=session.run(schema_query).data()
            require(outside_after==outside_before and schema_after==schema_before,'Existing namespaces or schema changed')
            banner('НАЧИНАЮ ПРОВЕРКУ НЕИЗМЕННОСТИ ПРЕДЫДУЩИХ ФАЙЛОВ')
            for p,h in progress(hashes.items(),total=len(hashes),desc='ПРОВЕРКА SHA-256'):
                require(base.sha(p)==h,'Protected file changed: '+str(p))
            verification=dict(namespace=TARGET_NS,database=database,verified_at_moscow=datetime.now(timezone(timedelta(hours=3))).isoformat(),
                              first_import=first,repeated_import=second,counts=EXPECTED,connectivity=actual,
                              target_snapshot_after_first_import=digest1,target_snapshot_after_second_import=digest2,
                              provenance_restoration=restoration,cypher_checks=checks,visualization_query_coverage=visualization,
                              invalid_fingerprint_refused=True,outside_snapshot_before=outside_before,outside_snapshot_after=outside_after,
                              existing_namespaces_unchanged=True,schema_unchanged=True,protected_files_unchanged=True,
                              protected_sha256={p.relative_to(ROOT).as_posix():h for p,h in hashes.items()},
                              models_launched=False,new_entity_resolution_performed=False,all_similar_to_preserved=True,
                              self_loops_preserved=True,parallel_edges_preserved=True,stage_2f4_started=False)
            write_json('verification.json',verification)
            report(manifest,verification)
    finally:driver.close()
    banner('ЭТАП 2F.3 ЗАВЕРШЁН\nGLOBAL KG УСПЕШНО ИМПОРТИРОВАН И ПРОВЕРЕН')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        error=dict(stage=STAGE,last_successfully_processed_batch=LAST_BATCH,exception_type=type(exc).__name__,
                   message=str(exc),import_transaction_committed=IMPORT_COMMITTED)
        print('\nОШИБКА ЭТАПА 2F.3: '+dumps(error),file=sys.stderr,flush=True)
        if OUT.exists():write_json('failure.json',error)
        raise SystemExit(1)
