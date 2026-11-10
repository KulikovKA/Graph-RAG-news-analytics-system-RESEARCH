"""Two local ReLEx-base workers: fixed-span cross-document RE and separate KG.

Run: .venv/Scripts/python.exe scripts/add_cross_document_relex.py
Uses existing NEO4J_* credentials, or --neo4j-env with a valid existing env file.
No downloads, embeddings, entity merging, or writes to the source namespace.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from importlib.metadata import version
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import import_global_semantic_graph as base

core, ROOT = base.core, base.ROOT
SOURCE_NS = base.NS
NS = 'relex_crossdoc_e5_small_t095'
OUT = base.ER / 'cross_document_re'
RESULTS = OUT / 'cross_document_relations.jsonl'
CONFIG = ROOT / 'reports/re_v2/relex_comparison/threshold035/e3_base/experiment_config.json'
ONTOLOGY = ROOT / 'contracts/ner_v2/ontology.v2.json'
MODEL_PATH = ROOT / 'artifacts/models/gliner-relex-base-v1.0'
MODEL_ID, THRESHOLD, WORKERS = 'knowledgator/gliner-relex-base-v1.0', .35, 2
_model = _cfg = _entities = _documents = None

VIEW_QUERY = f"""CYPHER 25
MATCH (a:RELEXEntity {{namespace:'{NS}'}})
      -[r:RELEX_REL|SIMILAR_TO|CROSS_DOC_RE_CANDIDATE]->
      (b:RELEXEntity {{namespace:'{NS}'}})
WHERE r.namespace='{NS}'
RETURN a,r,b
ORDER BY CASE WHEN type(r)='CROSS_DOC_RE_CANDIDATE' THEN 0 ELSE 1 END
LIMIT 300;"""

STATS_QUERY = f"""CYPHER 25
WITH '{NS}' AS ns
CALL (ns) {{ MATCH (d:RELEXDocument {{namespace:ns}}) RETURN count(d) AS documents }}
CALL (ns) {{
  MATCH (e:RELEXEntity {{namespace:ns}})
  RETURN count(e) AS entities,
    sum(CASE WHEN NOT EXISTS {{
      MATCH (e)-[r:RELEX_REL|SIMILAR_TO|CROSS_DOC_RE_CANDIDATE]-(other:RELEXEntity {{namespace:ns}})
      WHERE r.namespace=ns
    }} THEN 1 ELSE 0 END) AS isolated
}}
CALL (ns) {{ MATCH ()-[r:RELEX_REL {{namespace:ns}}]->() RETURN count(r) AS factual_relations }}
CALL (ns) {{ MATCH ()-[r:SIMILAR_TO {{namespace:ns}}]->() RETURN count(r) AS similar_to }}
CALL (ns) {{
  MATCH (a:RELEXEntity {{namespace:ns}})-[r:CROSS_DOC_RE_CANDIDATE {{namespace:ns}}]->(b:RELEXEntity {{namespace:ns}})
  RETURN count(r) AS cross_doc_candidates, min(r.confidence) AS min_confidence,
    max(r.confidence) AS max_confidence,
    sum(CASE WHEN r.suspicious_type_hint THEN 1 ELSE 0 END) AS type_warnings,
    sum(CASE WHEN r.absence_of_direct_evidence THEN 1 ELSE 0 END) AS without_direct_evidence,
    collect(DISTINCT a.document_id)+collect(DISTINCT b.document_id) AS linked_docs
}}
RETURN ns AS namespace, documents, entities, factual_relations, similar_to,
  cross_doc_candidates, isolated, 100.0*isolated/entities AS isolated_pct,
  min_confidence, max_confidence, type_warnings, without_direct_evidence,
  size(reduce(unique=[], d IN linked_docs | CASE WHEN d IN unique THEN unique ELSE unique+[d] END)) AS linked_documents;
"""


def checked(ok, message):
    base.require(ok, message)


def digest(value):
    return core.digest(value)


def initialize(cfg, entities, documents):
    global _model, _cfg, _entities, _documents
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                      OMP_NUM_THREADS=str(cfg['threads']), MKL_NUM_THREADS=str(cfg['threads']))
    import torch
    from gliner import GLiNER
    torch.set_num_threads(cfg['threads'])
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    _cfg, _entities, _documents = cfg, entities, documents
    _model = GLiNER.from_pretrained(str(MODEL_PATH), local_files_only=True).to('cpu').eval()


def profile(eid, radius):
    e = _entities[eid]
    context = e['representative_context']
    source = next(m['source_entity'] for m in e['members']
                  if m['source_entity']['entity_id']==context['source_entity_id'])
    doc = _documents[e['document_id']]
    start, end = source['start'], source['end']
    low, high = max(context['start'], start-radius), min(context['end'], end+radius)
    checked(doc['text'][context['start']:context['end']]==context['text'], 'Changed original context')
    checked(doc['text'][start:end]==source['surface_text'], 'Changed original entity offsets')
    return dict(canonical_entity_id=eid, document_id=e['document_id'], name=e['text_name'],
                entity_types=e['entity_types'], source_entity_ids=e['source_entity_ids'],
                source_mention_ids=e['source_mention_ids'], source_offsets=e['source_offsets'],
                selected_source_entity_id=source['entity_id'], mention_start=start, mention_end=end,
                mention_text=source['surface_text'], source_context=context,
                used_fragment=dict(document_id=e['document_id'], start=low, end=high,
                                   text=doc['text'][low:high], revision_id=doc['revision_id']),
                source_url=doc['source_url'], quality_flags=e['quality_flags'])


def joint_input(link, radius):
    endpoints = [profile(link[side+'_canonical_id'],radius) for side in ('left','right')]
    text, members = '', []
    for endpoint in endpoints:
        fragment = endpoint['used_fragment']
        text += f"[DOCUMENT {endpoint['document_id']}]\n"
        offset = len(text)
        start = offset+endpoint['mention_start']-fragment['start']
        end = offset+endpoint['mention_end']-fragment['start']
        text += fragment['text']+f"\n[END DOCUMENT {endpoint['document_id']}]\n\n"
        members.append(dict(entity_id=endpoint['canonical_entity_id'], start=start,end=end,
                            surface_text=endpoint['mention_text'],entity_types=endpoint['entity_types']))
    return endpoints,text,members


def infer(link):
    # Cropping changes only the amount of source context, never an entity span.
    for radius in (120,90,60,40,20,0):
        endpoints,text,members = joint_input(link,radius)
        tokens, starts, ends = _model.prepare_inputs([text])
        for member in members:
            checked(text[member['start']:member['end']]==member['surface_text'], 'Joint-text offset mismatch')
            checked(member['start'] in starts[0] and member['end'] in ends[0], 'Span not aligned with ReLEx word boundaries')
        processor = _model.data_processor
        prompted,_ = processor.prepare_inputs(tokens,_cfg['entity_labels'],relations=list(core.PREDICATES.values()))
        token_count = len(processor.transformer_tokenizer(prompted,is_split_into_words=True,
                                                        truncation=False,padding=False)['input_ids'][0])
        if token_count<=_cfg['token_limit']:
            break
    else:
        raise ValueError('Two complete entity spans and prompts do not fit token budget')
    window = dict(window_id='joint:'+digest([link['semantic_link_id'],text]),
                  document_id='synthetic:'+link['semantic_link_id'],start=0,end=len(text),text=text)
    for member in members: member['document_id']=window['document_id']
    raw,audit = core.predict_window(_model,window,members,_cfg['entity_labels'],THRESHOLD,_cfg['token_limit'])
    checked(audit['pairs']==2 and audit['spans']==2 and audit['tokens']==token_count, 'Adapter/prompt truncation error')
    lookup = {(m['start'],m['end']):(m,e) for m,e in zip(members,endpoints)}
    reverse = {label:code for code,label in core.PREDICATES.items()}
    relations = []
    for prediction in raw:
        pair = []
        for side in ('head','tail'):
            point = prediction[side]
            member,endpoint = lookup[point['start'],point['end']]
            checked(point['text']==member['surface_text'], 'Prediction changed a supplied span')
            pair.append(endpoint)
        head,tail = pair
        checked(head['document_id']!=tail['document_id'], 'Prediction is not cross-document')
        code = reverse[prediction['relation']]
        hints = core.TYPE_HINTS.get(code)
        warning = bool(hints and any(expected is not None and not expected.intersection(e['entity_types'])
                                    for expected,e in zip(hints,pair)))
        relations.append(dict(candidate_id='xre:'+digest([head['canonical_entity_id'],code,tail['canonical_entity_id']]),
            head_id=head['canonical_entity_id'],tail_id=tail['canonical_entity_id'],head_name=head['name'],tail_name=tail['name'],
            head_document_id=head['document_id'],tail_document_id=tail['document_id'],
            head_types=head['entity_types'],tail_types=tail['entity_types'],relation_type=code,confidence=prediction['score'],
            direction='left_to_right' if head is endpoints[0] else 'right_to_left',model=MODEL_ID,threshold=THRESHOLD,
            source_semantic_link_id=link['semantic_link_id'],source_fragments=[e['used_fragment'] for e in pair],
            model_input_id=window['window_id'],suspicious_type_hint=warning,absence_of_direct_evidence=True,
            evidence=[],evidence_status='no_shared_source_passage',experimental=True))
    relations.sort(key=lambda r:r['candidate_id'])
    checked(len({r['candidate_id'] for r in relations})==len(relations), 'Repeated directed typed prediction')
    return dict(record_type='pair_result',pair_id=link['semantic_link_id'],left=endpoints[0],right=endpoints[1],
                model_input=dict(window,synthetic=True,is_source_evidence=False),audit=audit,
                context_radius=radius,relations=relations,status='predicted' if relations else 'no_predictions')


def infer_all(links, cfg, entities, documents):
    from tqdm import tqdm
    patients = [r for r in links if all(entities[r[s+'_canonical_id']]['text_name'].casefold()=='patients' for s in ('left','right'))]
    partial = [r for r in links if any('original_partial_token' in entities[r[s+'_canonical_id']]['quality_flags'] for s in ('left','right'))]
    order = patients[:2]+partial+sorted(links,key=lambda r:float(r['similarity']))[:2]+links
    smoke = list({r['semantic_link_id']:r for r in order}.values())[:10]
    checked(len(smoke)==10 and patients and any(r['semantic_link_id']==patients[0]['semantic_link_id'] for r in smoke), 'patients smoke coverage missing')
    def collect(executor, tasks, title, initial=0):
        futures = [executor.submit(infer,task) for task in tasks]
        return {r['pair_id']:r for future in tqdm(as_completed(futures),total=len(futures)+initial,initial=initial,desc=title,unit='pair')
                for r in [future.result()]}
    print(f'Starting {WORKERS} local ReLEx-base workers, CPU, {cfg["threads"]} threads each.',flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn'),initializer=initialize,
                             initargs=(cfg,entities,documents)) as executor:
        first = collect(executor,smoke,'Smoke 10')
        second = collect(executor,smoke,'Repeat smoke')
        checked(digest(first)==digest(second),'Smoke predictions are not reproducible')
        print('Smoke passed: exact offsets, 2 directed pairs, no truncation, identical repeat; patients included.',flush=True)
        first.update(collect(executor,[r for r in links if r['semantic_link_id'] not in first],'Cross-document RE',initial=len(first)))
    return [first[r['semantic_link_id']] for r in links],[r['semantic_link_id'] for r in smoke]


def read_graph(session, namespace):
    nodes = session.run('MATCH (n {namespace:$ns}) RETURN n.key AS key,labels(n) AS labels,properties(n) AS properties',ns=namespace).data()
    edges = session.run('MATCH (a)-[r]->(b) WHERE r.namespace=$ns RETURN r.key AS key,type(r) AS type,a.key AS head,b.key AS tail,properties(r) AS properties',ns=namespace).data()
    return sorted(nodes,key=lambda r:r['key']),sorted(edges,key=lambda r:r['key'])


def outside(session):
    h = hashlib.sha256()
    for query in ("MATCH (n) WHERE coalesce(n.namespace,'')<>$ns RETURN elementId(n) AS id,labels(n) AS labels,properties(n) AS p ORDER BY id",
                  "MATCH (a)-[r]->(b) WHERE coalesce(r.namespace,'')<>$ns RETURN elementId(r) AS id,type(r) AS type,elementId(a) AS head,elementId(b) AS tail,properties(r) AS p ORDER BY id"):
        for row in session.run(query,ns=NS):
            data=row.data()
            if 'labels' in data: data['labels'].sort()
            h.update((base.dumps(data)+'\n').encode())
    return h.hexdigest()


def clone_payload(source, pairs, fingerprint):
    remap = lambda k:NS+k[len(SOURCE_NS):]
    nodes,edges = {},{}
    for collection,destination in zip(source,(nodes,edges)):
        for row in collection:
            checked(row['key'].startswith(SOURCE_NS+':'),'Unexpected source key')
            properties = dict(row['properties'],namespace=NS,key=remap(row['key']),fingerprint=fingerprint,
                              cloned_from_namespace=SOURCE_NS,cloned_from_key=row['key'],
                              cloned_from_fingerprint=row['properties']['fingerprint'])
            if 'labels' in row:
                checked(len(row['labels'])==1 and row['labels'][0] in ('RELEXDocument','RELEXEntity'),'Unknown source label')
                destination[properties['key']] = dict(label=row['labels'][0],properties=properties)
            else:
                destination[properties['key']] = dict(type=row['type'],head=remap(row['head']),tail=remap(row['tail']),properties=properties)
    for pair in pairs:
        for relation in pair['relations']:
            properties = dict(base.props(relation),key=NS+':'+relation['candidate_id'],namespace=NS,fingerprint=fingerprint,
                              model_input_json=base.dumps(pair['model_input']),source_pair_file=RESULTS.relative_to(ROOT).as_posix())
            checked(properties['key'] not in edges,'Duplicate cross-document candidate')
            edges[properties['key']] = dict(type='CROSS_DOC_RE_CANDIDATE',head=NS+':'+relation['head_id'],
                                            tail=NS+':'+relation['tail_id'],properties=properties)
    return nodes,edges


def import_once(session, nodes, edges, fingerprint):
    def write(tx):
        for query in ('MATCH (n {namespace:$ns}) RETURN DISTINCT n.fingerprint AS f',
                      'MATCH ()-[r]->() WHERE r.namespace=$ns RETURN DISTINCT r.fingerprint AS f'):
            checked(all(r['f']==fingerprint for r in tx.run(query,ns=NS)),'Namespace contains a different experiment')
        created = dict(nodes=0,relationships=0)
        def run(query,rows):
            for start in range(0,len(rows),100):
                summary=tx.run(query,rows=rows[start:start+100],ns=NS).consume()
                created['nodes']+=summary.counters.nodes_created;created['relationships']+=summary.counters.relationships_created
        for label in ('RELEXDocument','RELEXEntity'):
            run(f'UNWIND $rows AS row MERGE (n:{label} {{key:row.properties.key,namespace:$ns}}) SET n += row.properties',
                [r for r in nodes.values() if r['label']==label])
        for kind in ('RELEX_MENTIONS','RELEX_REL','SIMILAR_TO','CROSS_DOC_RE_CANDIDATE'):
            label='RELEXDocument' if kind=='RELEX_MENTIONS' else 'RELEXEntity'
            run(f'UNWIND $rows AS row MATCH (a:{label} {{key:row.head,namespace:$ns}}) '
                'MATCH (b:RELEXEntity {key:row.tail,namespace:$ns}) '
                f'MERGE (a)-[r:{kind} {{key:row.properties.key,namespace:$ns}}]->(b) SET r += row.properties',
                [r for r in edges.values() if r['type']==kind])
        checked(tx.run('MATCH (n {namespace:$ns}) RETURN count(n) AS n',ns=NS).single()['n']==len(nodes),'Atomic node count mismatch')
        checked(tx.run('MATCH ()-[r]->() WHERE r.namespace=$ns RETURN count(r) AS n',ns=NS).single()['n']==len(edges),'Atomic edge count mismatch')
        return created
    return session.execute_write(write)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--neo4j-env',type=Path)
    parser.add_argument('--recompute',action='store_true',help='Repeat inference and require identical saved results')
    args=parser.parse_args()
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    ontology=json.loads(ONTOLOGY.read_text(encoding='utf-8'))
    checked(cfg['relation_threshold']==THRESHOLD and cfg['variant']=='base' and cfg['predicates']==core.PREDICATES,
            'Original ReLEx protocol differs')
    checked(set(ontology['relation_types'])==set(core.PREDICATES),'Ontology labels differ')
    files=[MODEL_PATH/name for name in cfg['models']['base']]
    file_hashes={path:base.sha(path) for path in files}
    checked(all(file_hashes[path]==cfg['models']['base'][path.name] for path in files),'Local checkpoint differs from relex_base_t035')
    documents,entities,relations,semantic,expected=base.prepare()
    documents={d['document_id']:d for d in documents};entities={e['canonical_entity_id']:e for e in entities}
    links=sorted([r for r in semantic if r['scope']=='cross_document'],key=lambda r:r['semantic_link_id'])
    checked(len(links)==194,'Expected 194 cross-document pairs')
    protected={path:base.sha(path) for path in base.ER.rglob('*') if path.is_file() and not path.is_relative_to(OUT)}
    protected.update({path:base.sha(path) for path in (CONFIG,ONTOLOGY,base.DOCUMENTS,Path(core.__file__))})
    driver,database=core.connect_neo4j(args.neo4j_env)
    try:
        with driver.session(database=database) as session:
            source=read_graph(session,SOURCE_NS)
            checked(len(source[0])==1152 and Counter(r['type'] for r in source[1])==
                    Counter(RELEX_MENTIONS=1122,RELEX_REL=1760,SIMILAR_TO=685),'Source Global KG counts differ')
            source_fps={n['properties']['fingerprint'] for n in source[0]}
            checked(len(source_fps)==1,'Mixed source fingerprints')
            original_nodes,original_edges=base.payload(list(documents.values()),list(entities.values()),relations,semantic,next(iter(source_fps)))
            base.verify(session,original_nodes,original_edges,expected)
            signature=dict(stage='2E.1-v1',model=MODEL_ID,threshold=THRESHOLD,workers=WORKERS,threads=cfg['threads'],
                runner_sha256=base.sha(Path(__file__)),
                token_limit=cfg['token_limit'],source_graph=digest(source),adapter_sha256=protected[Path(core.__file__)],
                inputs={str(path.relative_to(ROOT)):protected[path] for path in (CONFIG,ONTOLOGY,base.ENTITIES,base.RELATIONS,base.LINKS,base.DOCUMENTS)},
                checkpoint={path.name:sha for path,sha in file_hashes.items()},
                runtime={name:version(name) for name in ('torch','gliner','transformers')})
            fingerprint=digest(signature)
            cached=core.read_jsonl(RESULTS) if RESULTS.exists() else None
            if cached:
                checked(cached[0]['fingerprint']==fingerprint and cached[0]['results_sha256']==digest(cached[1:]),'Saved experiment fingerprint/checksum differs')
            if cached and not args.recompute:
                metadata,pairs=cached[0],cached[1:]
                print('Using validated saved pair results; no model inference.',flush=True)
            else:
                pairs,smoke=infer_all(links,dict(cfg,fingerprint=fingerprint),entities,documents)
                metadata=dict(record_type='metadata',fingerprint=fingerprint,signature=signature,
                              smoke_pair_ids=smoke,smoke_repeat_identical=True,results_sha256=digest(pairs))
                if cached: checked(digest(pairs)==digest(cached[1:]),'Recomputed results differ from cache')
            checked(len(pairs)==194 and [p['pair_id'] for p in pairs]==[r['semantic_link_id'] for r in links],'Pair coverage mismatch')
            predictions=[r for pair in pairs for r in pair['relations']]
            checked(len(predictions)==len({r['candidate_id'] for r in predictions}),'Duplicate directed typed predictions')
            checked(all(r['confidence']>THRESHOLD and r['relation_type'] in core.PREDICATES and
                        r['head_document_id']!=r['tail_document_id'] for r in predictions),'Invalid saved cross-document relation')
            # Keep inference results even if a later database operation fails.
            if not cached:
                OUT.mkdir(parents=True,exist_ok=True)
                temporary=RESULTS.with_suffix('.jsonl.tmp')
                temporary.write_text(''.join(base.dumps(r)+'\n' for r in [metadata]+pairs),encoding='utf-8')
                temporary.replace(RESULTS)
            checked(digest(read_graph(session,SOURCE_NS))==signature['source_graph'],'Source KG changed during inference')
            outside_before=outside(session)
            nodes,edges=clone_payload(source,pairs,fingerprint)
            first=import_once(session,nodes,edges,fingerprint)
            second=import_once(session,nodes,edges,fingerprint)
            checked(second==dict(nodes=0,relationships=0),'Repeated import created duplicates')
            actual_nodes,actual_edges=read_graph(session,NS)
            checked(len(actual_nodes)==len(nodes) and len(actual_edges)==len(edges),'Read-back coverage mismatch')
            for row in actual_nodes:
                checked(row['properties']==nodes[row['key']]['properties'],'Cloned node properties differ')
            for row in actual_edges:
                checked(all(row[k]==edges[row['key']][k] for k in ('type','head','tail','properties')),'Imported relationship differs')
            kg_nodes={r['key']:r['properties']['document_id'] for r in actual_nodes if r['labels']==['RELEXEntity']}
            kg_edges=[(r['head'],r['tail']) for r in actual_edges if r['type']!='RELEX_MENTIONS']
            checked(base.connectivity(kg_nodes,kg_edges)==expected,'Unexpected Global KG connectivity change')
            # Every predicted pair already had SIMILAR_TO, so no full-KG component can be joined.
            semantic_pairs={frozenset((r['left_canonical_id'],r['right_canonical_id'])) for r in links}
            checked(all(frozenset((r['head_id'],r['tail_id'])) in semantic_pairs for r in predictions),'Candidate not traceable to SIMILAR_TO')
            checked(outside(session)==outside_before,'Data outside target namespace changed')
            query_stats=session.run(STATS_QUERY.rstrip().rstrip(';')).single().data()
            checked(query_stats['cross_doc_candidates']==len(predictions) and query_stats['isolated']==200,'Cypher statistics mismatch')
            session.run(VIEW_QUERY.rstrip().rstrip(';')).consume()
        checked(all(base.sha(path)==sha for path,sha in {**protected,**file_hashes}.items()),'Source files or weights changed')
    finally:
        driver.close()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'graph_stats.cypher').write_text(VIEW_QUERY+'\n\n'+STATS_QUERY,encoding='utf-8')
    summary=dict(namespace=NS,processed_pairs=len(pairs),positive_pairs=sum(bool(p['relations']) for p in pairs),
        unique_cross_doc_candidates=len(predictions),linked_documents=len({r[s+'_document_id'] for r in predictions for s in ('head','tail')}),
        full_kg_component_bridges=0,original_factual_component_bridges=len(predictions),
        suspicious_type_hints=sum(r['suspicious_type_hint'] for r in predictions),without_direct_evidence=len(predictions),
        relation_types=dict(Counter(r['relation_type'] for r in predictions)),connectivity=expected,
        import_created=first,repeated_import_created=second)
    scores=[r['confidence'] for r in predictions]
    if scores: summary['confidence']=dict(min=min(scores),median=statistics.median(scores),mean=statistics.mean(scores),max=max(scores))
    summary['confidence_bins']={f'{low:g}..{high:g}':sum(low<=s<high or high==1 and s==1 for s in scores)
                                for low,high in ((.35,.5),(.5,.7),(.7,.9),(.9,1))}
    examples=sorted(predictions,key=lambda r:(-r['confidence'],r['candidate_id']))[:20]
    lines=['# Cross-document ReLEx experiment','', '```json',json.dumps(summary,ensure_ascii=False,indent=2),'```','',
        'Две параллельные копии исходного ReLEx-base, неизменённый fixed-span адаптер и 20 labels. Сохранены все предсказания строго >0.35 в обоих направлениях. Smoke 10 (включая patients) повторён с идентичным результатом. JSONL: одна metadata-строка и 194 pair_result со списками relations; пустой список — полноценный отрицательный результат.','',
        'Это экспериментальные кандидаты CROSS_DOC_RE_CANDIDATE, не подтверждённые факты. У всех отсутствует общий исходный passage: два source fragments сохранены отдельно, синтетический вход НЕ evidence. Типовые подсказки — диагностика, не фильтр.','',
        'Новые рёбра не соединяют разные компоненты полного Global KG: их концы уже связаны SIMILAR_TO. В исходном фактическом подграфе без SIMILAR_TO концы находятся в разных документных компонентах. Сохранены 30 документов, 1122 сущности, 1760 RELEX_REL, 685 SIMILAR_TO и 1122 служебных RELEX_MENTIONS.','',
        '## 20 примеров','', '| Head / document | Relation / confidence | Tail / document | Head source fragment | Tail source fragment |',
        '|---|---|---|---|---|']
    for r in examples:
        values=[r['head_name']+' / '+r['head_document_id'],r['relation_type']+f" / {r['confidence']:.6f}",
                r['tail_name']+' / '+r['tail_document_id']]+[f['text'] for f in r['source_fragments']]
        lines.append('| '+' | '.join(s.replace('|','\\|').replace('\n',' ') for s in values)+' |')
    if len(examples)<20:
        for pair in [p for p in pairs if not p['relations']][:20-len(examples)]:
            a,b=pair['left'],pair['right']
            values=[a['name']+' / '+a['document_id'],'NONE above 0.35',b['name']+' / '+b['document_id'],a['used_fragment']['text'],b['used_fragment']['text']]
            lines.append('| '+' | '.join(s.replace('|','\\|').replace('\n',' ') for s in values)+' |')
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('Results:',OUT)


if __name__=='__main__':
    mp.freeze_support()
    main()
