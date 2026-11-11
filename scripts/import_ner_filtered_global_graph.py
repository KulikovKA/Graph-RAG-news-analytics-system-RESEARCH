"""Build the fixed max-member NER >= .46 Global KG in its own namespace.

Reads the original 2E sources; no inference, merging, embedding or edge scoring.
Use --prepare-only for local artifacts without accessing Neo4j.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding='utf-8')
import import_global_semantic_graph as base

ROOT=base.ROOT
ER=base.ER
OUT=ER/'ner_threshold_analysis/global_kg_t046_max'
NS='relex_sem_e5_small_name_t095_ner_t046_max'
BASELINE_NS='relex_sem_e5_small_name_t095'
NER_THRESHOLD=.46
STRATEGY='max_member'
ANALYSIS=ER/'ner_threshold_analysis/threshold_045_050'
base.NS=NS
base.OUT=OUT
require=base.require
dumps=base.dumps


def csv_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def write_json(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')


def write_csv(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k:dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)


def prepare():
    protected={p for p in ER.rglob('*') if p.is_file() and not p.is_relative_to(OUT)}
    protected.update([base.DOCUMENTS,ROOT/'scripts/build_relex_graphs.py',ROOT/'scripts/import_global_semantic_graph.py',Path(__file__)])
    hashes={p:base.sha(p) for p in sorted(protected)}
    documents,entities,relations,links,_=base.prepare()
    source_scores={e['canonical_entity_id']:max(m['source_entity']['confidence'] for m in e['members']) for e in entities}
    ids={eid for eid,score in source_scores.items() if score>=NER_THRESHOLD}
    kept_entities=[e for e in entities if e['canonical_entity_id'] in ids]
    kept_relations=[r for r in relations if r['head_id'] in ids and r['tail_id'] in ids]
    kept_links=[r for r in links if r['left_canonical_id'] in ids and r['right_canonical_id'] in ids]
    prior=next(r for r in csv_rows(ANALYSIS/'comparison.csv') if r['strategy']==STRATEGY and float(r['threshold'])==NER_THRESHOLD)
    previous=next(r for r in base.core.read_jsonl(ANALYSIS/'virtual_subgraphs.jsonl') if r['strategy']==STRATEGY and r['threshold']==NER_THRESHOLD)
    require(ids==set(previous['entity_ids']),'Vertex set differs from 0.46 analysis')
    require({r['canonical_triple_id'] for r in kept_relations}==set(previous['relex_rel_ids']),'Factual edge set differs')
    require({r['semantic_link_id'] for r in kept_links}==set(previous['similar_to_ids']),'Semantic edge set differs')
    expected=base.connectivity({e['canonical_entity_id']:e['document_id'] for e in kept_entities},
        [(r['head_id'],r['tail_id']) for r in kept_relations]+[(r['left_canonical_id'],r['right_canonical_id']) for r in kept_links])
    require(expected==dict(entities=905,isolated=175,weak_components=230,largest_component_vertices=378,largest_component_documents=17),'Unexpected induced graph')
    require(len(kept_relations)==int(prior['relex_rel'])==1307 and len(kept_links)==int(prior['similar_to'])==492,'Unexpected edge counts')
    require(sum(r['scope']=='cross_document' for r in kept_links)==121,'Unexpected cross-document links')
    require(sum(r['head_id']==r['tail_id'] for r in kept_relations)==12,'Unexpected loops')
    signature=dict(stage='NER-filtered-Global-KG-v1',namespace=NS,baseline_namespace=BASELINE_NS,
                   ner_entity_threshold=NER_THRESHOLD,ner_entity_confidence_strategy=STRATEGY,
                   semantic_model=base.MODEL,semantic_mode=base.MODE,semantic_threshold=base.THRESHOLD,
                   inputs={p.relative_to(ROOT).as_posix():hashes[p] for p in [base.DOCUMENTS,base.ENTITIES,base.RELATIONS,base.LINKS,ANALYSIS/'comparison.csv',ANALYSIS/'virtual_subgraphs.jsonl']})
    fingerprint=base.core.digest(signature)
    # Reuse the original lossless payload schema. Filter by vertex membership only.
    all_nodes,all_edges=base.payload(documents,entities,relations,links,fingerprint)
    node_rows={key:row for key,row in all_nodes.items() if row['label']=='RELEXDocument' or row['properties']['entity_id'] in ids}
    edge_rows={key:row for key,row in all_edges.items() if row['head'] in node_rows and row['tail'] in node_rows}
    for row in node_rows.values():
        row['properties'].update(baseline_namespace=BASELINE_NS,ner_entity_threshold=NER_THRESHOLD,ner_entity_confidence_strategy=STRATEGY)
    inventory=[]
    for e in kept_entities:
        eid=e['canonical_entity_id']
        rep=next(m['source_entity'] for m in e['members'] if m['source_entity']['entity_id']==eid)
        best=sorted((m['source_entity'] for m in e['members']),key=lambda m:(-m['confidence'],m['entity_id']))[0]
        props=node_rows[base.key(eid)]['properties']
        props.update(confidence=best['confidence'],confidence_source_entity_id=best['entity_id'],
                     max_member_confidence=best['confidence'],representative_confidence=rep['confidence'],
                     representative_entity_id=eid,span_source_entity_id=eid)
        inventory.append(dict(canonical_entity_id=eid,document_id=e['document_id'],name=e['surface_text'],
                              entity_types=e['entity_types'],confidence=best['confidence'],representative_confidence=rep['confidence'],
                              confidence_source_entity_id=best['entity_id'],source_entity_ids=e['source_entity_ids']))
    require(len(node_rows)==935 and len(edge_rows)==2704,'Filtered payload counts mismatch')
    require(Counter(r['type'] for r in edge_rows.values())==Counter(RELEX_MENTIONS=905,RELEX_REL=1307,SIMILAR_TO=492),'Filtered relation types mismatch')
    require(all(r['properties']['namespace']==NS for r in list(node_rows.values())+list(edge_rows.values())),'Wrong namespace')
    require(all(base.sha(p)==h for p,h in hashes.items()),'Protected input changed during preparation')
    OUT.mkdir(parents=True,exist_ok=True)
    write_csv('entities.csv',inventory)
    write_csv('edges.csv',[dict(edge_id=r['properties']['key'][len(NS)+1:],type=r['type'],
                              head_id=r['head'][len(NS)+1:],tail_id=r['tail'][len(NS)+1:],
                              relation_type=r['properties'].get('relation_type',''),
                              source_confidence_values=r['properties'].get('source_confidence_values',[]),
                              similarity=r['properties'].get('similarity'),scope=r['properties'].get('scope',''),
                              source_document=r['properties'].get('document_id',r['properties'].get('left_document_id','')),
                              target_document=r['properties'].get('right_document_id','')) for r in edge_rows.values()])
    write_csv('documents.csv',[dict(document_id=d['document_id'],title=d['title'],discipline=d['discipline']) for d in documents])
    write_json('membership.json',previous)
    write_json('manifest.json',dict(**signature,fingerprint=fingerprint,expected_connectivity=expected,
                                  counts=dict(documents=30,entities=905,removed_entities=217,relex_rel=1307,similar_to=492,
                                              cross_document_links=121,within_document_links=371,mentions=905,total_nodes=935,total_relationships=2704)))
    # Existing, previously verified read-only Cypher, scoped to the new namespace.
    queries=(ER/'semantic_resolution/global_kg_import/graph_stats.cypher').read_text(encoding='utf-8').replace(BASELINE_NS,NS)
    (OUT/'graph_stats.cypher').write_text(queries,encoding='utf-8')
    full=f"""// All 935 nodes and all 2704 relationships, including isolates and documents.
// Set Neo4j Browser Graph visualization node limit >= 1000 before running.
MATCH (n {{namespace: '{NS}'}})
OPTIONAL MATCH (n)-[r]->(m {{namespace: '{NS}'}})
WHERE r.namespace = '{NS}'
RETURN n, r, m;
"""
    (OUT/'show_full_graph.cypher').write_text(full,encoding='utf-8')
    return node_rows,edge_rows,expected,fingerprint,hashes


def import_once(session,nodes,edges,fingerprint):
    def transaction(tx):
        for query in ['MATCH (n {namespace:$ns}) RETURN DISTINCT n.fingerprint AS fingerprint',
                      'MATCH ()-[r]->() WHERE r.namespace=$ns RETURN DISTINCT r.fingerprint AS fingerprint']:
            require(all(r['fingerprint']==fingerprint for r in tx.run(query,ns=NS)),
                    'Target namespace contains another snapshot; refusing overwrite')
        created=dict(nodes=0,relationships=0)
        def run(query,rows):
            summary=tx.run(query,namespace=NS,rows=rows).consume()
            created['nodes']+=summary.counters.nodes_created
            created['relationships']+=summary.counters.relationships_created
        for label in ['RELEXDocument','RELEXEntity']:
            rows=[r for r in nodes.values() if r['label']==label]
            for start in range(0,len(rows),100):
                run(f'UNWIND $rows AS row MERGE (n:{label} {{key:row.properties.key,namespace:$namespace}}) SET n += row.properties',rows[start:start+100])
        for kind in ['RELEX_MENTIONS','RELEX_REL','SIMILAR_TO']:
            rows=[r for r in edges.values() if r['type']==kind]
            label='RELEXDocument' if kind=='RELEX_MENTIONS' else 'RELEXEntity'
            query=(f'UNWIND $rows AS row MATCH (h:{label} {{key:row.head,namespace:$namespace}}) '
                   'MATCH (t:RELEXEntity {key:row.tail,namespace:$namespace}) '
                   f'MERGE (h)-[r:{kind} {{key:row.properties.key,namespace:$namespace}}]->(t) SET r += row.properties')
            for start in range(0,len(rows),100):
                run(query,rows[start:start+100])
        actual=tx.run(base.statements()[0]).single().data()
        require(actual==dict(namespace=NS,documents=30,entities=905,total_nodes=935,factual_relations=1307,semantic_links=492,
                             document_mentions=905,kg_edges=1799,total_relationships=2704),f'Atomic count check failed: {actual}')
        low=tx.run('MATCH (e:RELEXEntity {namespace:$ns}) WHERE e.confidence < $threshold OR e.confidence <> e.max_member_confidence RETURN count(e) AS invalid',
                   ns=NS,threshold=NER_THRESHOLD).single()['invalid']
        require(low==0,'Invalid NER confidence in payload')
        return created
    return session.execute_write(transaction)


def write_report(fingerprint,expected,verified=None):
    state='Фактически импортирован и проверен в Neo4j' if verified else 'Локальные артефакты подготовлены; импорт в Neo4j ещё не выполнен'
    lines=['# Global KG: NER ≥ 0.46, max member confidence','',state+'.','',
           f'Namespace: `{NS}`. Исходный namespace `{BASELINE_NS}` сохранён.','',
           '| Показатель | Значение |','|---|---:|','| Документы | 30 |','| Сущности | 905 |','| Удалено сущностей относительно baseline | 217 |',
           '| RELEX_REL | 1307 |','| SIMILAR_TO | 492 |','| Междокументные / внутридокументные SIMILAR_TO | 121 / 371 |',
           '| RELEX_MENTIONS | 905 |','| Все физические узлы / отношения | 935 / 2704 |',
           '| Изолированные entity-вершины | 175 |','| Слабосвязные компоненты entity KG | 230 |',
           '| Крупнейшая компонента: вершины / документы | 378 / 17 |','| Исходные фактические петли | 12 |','',
           'Фильтрация: максимум confidence всех members существующей canonical entity ≥ 0.46. Исходный cutoff NER 0.35, ReLEx 0.35, E5-small/name cosine threshold 0.95 сохранены. Ребро сохраняется только по наличию обоих концов; его confidence, similarity, тип и provenance не пересчитываются. Все 30 документов сохранены. Связность считается без документов и RELEX_MENTIONS.','',
           'Entity `confidence` и `max_member_confidence` — максимум по members, `confidence_source_entity_id` указывает его источник. `representative_confidence` — исходный confidence канонического представителя. `start/end`, `span_source_entity_id` и `representative_entity_id` по-прежнему относятся к представителю; score другого member не подменяет offsets. Полный исходный canonical объект и все member scores сохранены в source_json и source_confidences_json.','',
           'Полный граф в Browser (включая документы, междокументные связи и изолированные сущности):','',
           '```cypher',f"MATCH (n {{namespace: '{NS}'}})",f"OPTIONAL MATCH (n)-[r]->(m {{namespace: '{NS}'}})",f"WHERE r.namespace = '{NS}'",'RETURN n, r, m;','```','',
           'OPTIONAL MATCH сохраняет строки вершин без исходящих рёбер. Начальный MATCH включает все документы и все сущности, поэтому изолированные entity-вершины тоже видны; они могут иметь служебную связь от документа. Возвращаются RELEX_MENTIONS, RELEX_REL и SIMILAR_TO. SIMILAR_TO хранится одним физическим ребром на пару, без удвоения. Для полного отображения установите Graph visualization node limit ≥ 1000 (или 1500); полный результат — 935 вершин, 2704 отношения.','',
           'Файлы: `show_full_graph.cypher`, `graph_stats.cypher`, `manifest.json`, `membership.json`, `entities.csv`, `documents.csv`, `edges.csv`. CSV — компактное представление; исходные полные свойства восстанавливаются неизменённым импортёром из JSONL/CSV предыдущих этапов.','',
           'Повторный импорт: `.venv/Scripts/python.exe scripts/import_ner_filtered_global_graph.py` с существующими NEO4J_* либо `--neo4j-env <existing-file>`. `--prepare-only` создаёт локальные артефакты без доступа к Neo4j. Другой fingerprint в целевом namespace приводит к отказу; удалений, смены пароля и изменения конфигурации сервера нет.','',
           'Модели NER/ReLEx/embeddings/LLM, Entity Merge и пересчёт cosine не выполнялись.','',f'Fingerprint: `{fingerprint}`.','']
    if verified:
        lines += ['Проверки фактического импорта:','',
                  '- Прочитаны все узлы и отношения с проверкой ID, типов, направлений, всех ожидаемых свойств и provenance; связность вычислена по прочитанному графу.',
                  '- Второй MERGE-проход не создал узлов или отношений, полное повторное чтение совпало.',
                  '- Все 9 read-only запросов graph_stats.cypher выполнены; WCC подтверждены независимым live Cypher, без GDS/APOC.',
                  '- Полный запрос визуализации выполнен: подтверждено покрытие всех 935 узлов и всех 2704 отношений.',
                  '- SHA-256 полного снимка узлов и отношений вне нового namespace совпал до и после; предыдущие графы не изменены.',
                  '- SHA-256 входных файлов и предыдущих отчётов не изменились.',
                  '',f"Время проверки (Europe/Moscow): {verified['verified_at_moscow']}. Подробности: `verification.json`.",'']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--neo4j-env',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    nodes,edges,expected,fingerprint,hashes=prepare()
    write_report(fingerprint,expected)
    print('Prepared: '+dumps(dict(namespace=NS,documents=30,entities=905,relex_rel=1307,similar_to=492,cross_document=121)),flush=True)
    if args.prepare_only:
        return
    driver,database=base.core.connect_neo4j(args.neo4j_env)
    try:
        with driver.session(database=database) as session:
            version=session.run('CALL dbms.components() YIELD name,versions,edition RETURN name,versions,edition').data()
            outside_before=base.outside_snapshot(session)
            first=import_once(session,nodes,edges,fingerprint)
            actual=base.verify(session,nodes,edges,expected)
            print('Imported and full read-back verified: '+dumps(actual),flush=True)
            second=import_once(session,nodes,edges,fingerprint)
            require(second==dict(nodes=0,relationships=0),'Repeated import created duplicates')
            require(base.verify(session,nodes,edges,expected)==actual,'Repeated import changed graph')
            checks=[]
            from neo4j import Query
            for i,query in enumerate(base.statements(),1):
                result=session.run(Query(query,timeout=90))
                data=result.data()
                summary=result.consume()
                notifications=[s for s in summary.gql_status_objects if s.is_notification]
                require(not any('deprecat' in s.status_description.lower() for s in notifications),'Deprecated Cypher syntax')
                require(data,'Empty graph_stats query '+str(i))
                if i==2:
                    require(data[0]['cross_document']==121 and data[0]['within_document']==371,'Scope mismatch')
                if i==4:
                    require(data[0]['isolated']==175,'Isolation mismatch')
                if i==5:
                    require(data[0]==dict(weak_components=230,entities=905,largest_component_vertices=378,largest_component_documents=17),'Live components mismatch')
                checks.append(dict(query=i,rows=len(data),notifications=len(notifications)))
                print(f'graph_stats {i}/9 passed',flush=True)
            visualization=(OUT/'show_full_graph.cypher').read_text(encoding='utf-8')
            result=session.run(visualization)
            visible_nodes,visible_edges=set(),set()
            for row in result:
                visible_nodes.add(row['n']['key'])
                if row['m'] is not None:
                    visible_nodes.add(row['m']['key'])
                if row['r'] is not None:
                    visible_edges.add(row['r']['key'])
            require(visible_nodes==set(nodes) and visible_edges==set(edges),'Visualization query omits graph elements')
            outside_after=base.outside_snapshot(session)
            require(outside_before==outside_after,'Graph outside target namespace changed')
        require(all(base.sha(p)==h for p,h in hashes.items()),'Protected local artifact changed')
    finally:
        driver.close()
    verification=dict(namespace=NS,database=database,verified_at_moscow=datetime.now(timezone(timedelta(hours=3))).isoformat(),
                      first_import=first,repeated_import=second,connectivity=actual,cypher_checks=checks,
                      full_visualization_query_coverage=dict(nodes=len(visible_nodes),relationships=len(visible_edges)),
                      outside_snapshot_before=outside_before,outside_snapshot_after=outside_after,
                      outside_namespace_unchanged=True,local_protected_files_unchanged=True,
                      source_sha256={p.relative_to(ROOT).as_posix():h for p,h in hashes.items()},neo4j_version=version,
                      models_launched=False,entity_merge_performed=False,cosine_recomputed=False)
    write_json('verification.json',verification)
    write_report(fingerprint,expected,verification)
    print(dumps(dict(namespace=NS,first_import=first,repeated_import=second,connectivity=actual,verified=True)),flush=True)


if __name__=='__main__':
    main()
