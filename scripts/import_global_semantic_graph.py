"""Stage 2E: import the fixed E5/name/0.95 snapshot into its own Neo4j namespace.

Run with existing NEO4J_* credentials:
    .venv/Scripts/python.exe scripts/import_global_semantic_graph.py
Only existing JSONL/CSV are read. No models, graph merging or source graph writes.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding='utf-8')
import build_relex_graphs as core

ROOT = core.ROOT
ER = ROOT / 'reports/entity_resolution/relex_base_t035'
OUT = ER / 'semantic_resolution/global_kg_import'
ENTITIES = ER / 'semantic_resolution/input/canonical_entities.jsonl'
RELATIONS = ER / 'relation_remapping/canonical_relations.jsonl'
LINKS = ER / 'semantic_resolution/global_linking_preview/semantic_links.csv'
DOCUMENTS = core.SOURCE / 'input_abstracts.jsonl'
NS, MODEL, MODE, THRESHOLD = 'relex_sem_e5_small_name_t095', 'intfloat/e5-small-v2', 'text_name', .95


def require(ok, message):
    if not ok:
        raise ValueError(message)


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def key(identity):
    # The old namespace validator deliberately allows only extraction namespaces.
    # Keep it unchanged; this importer is locked to the single new namespace above.
    return NS + ':' + identity


def props(source):
    """Neo4j property types are flat; lossless source_json retains every field."""
    result = {'source_json': dumps(source)}
    for name, value in source.items():
        if isinstance(value, dict) or isinstance(value, list) and any(isinstance(v,(dict,list)) for v in value):
            result[name+'_json'] = dumps(value)
        else:
            result[name] = value
    return result


def connectivity(nodes, edges):
    adjacency = {n:set() for n in nodes}
    touched = set()
    for a,b in edges:
        require(a in nodes and b in nodes, 'Unknown connectivity endpoint')
        touched.update((a,b)); adjacency[a].add(b); adjacency[b].add(a)
    seen, groups = set(), []
    for seed in sorted(nodes):
        if seed in seen:
            continue
        stack, group = [seed], []
        seen.add(seed)
        while stack:
            current = stack.pop(); group.append(current)
            for other in adjacency[current]-seen:
                seen.add(other); stack.append(other)
        groups.append(group)
    groups.sort(key=lambda g:(-len(g),min(g)))
    return dict(entities=len(nodes), isolated=len(nodes)-len(touched), weak_components=len(groups),
                largest_component_vertices=len(groups[0]),
                largest_component_documents=len({nodes[n] for n in groups[0]}))


def prepare():
    documents, entities, relations = map(core.read_jsonl,(DOCUMENTS,ENTITIES,RELATIONS))
    docs = {d['document_id']:d for d in documents}
    nodes = {e['canonical_entity_id']:e['document_id'] for e in entities}
    require(len(documents)==len(docs)==30 and len(entities)==len(nodes)==1122, 'Incomplete documents/entities')
    require(set(nodes.values())==set(docs), 'Document coverage mismatch')
    require(len({i for e in entities for i in e['source_entity_ids']})==1211 and
            len({i for e in entities for i in e['source_mention_ids']})==1327, 'Incomplete original ID coverage')
    require(len(relations)==len({r['canonical_triple_id'] for r in relations})==1760, 'Incomplete relations')
    require(sum(r['source_relation_count'] for r in relations)==1971 and
            len({i for r in relations for i in r['source_triple_ids']})==1971, 'Incomplete factual provenance')
    require(all(nodes[r['head_id']]==nodes[r['tail_id']]==r['document_id'] for r in relations), 'Cross-document factual edge')
    require(sum(r['head_id']==r['tail_id'] for r in relations)==13, 'Unexpected factual loops')
    with LINKS.open(encoding='utf-8-sig',newline='') as stream:
        all_links = list(csv.DictReader(stream))
    links = [r for r in all_links if (r['model'],r['mode'],float(r['threshold']))==(MODEL,MODE,THRESHOLD)]
    require(len(all_links)==3028 and len(links)==len({r['semantic_link_id'] for r in links})==685, 'Incomplete 2D links')
    entities_by_id = {e['canonical_entity_id']:e for e in entities}
    entity_hash = sha(ENTITIES)
    for r in links:
        a,b = r['left_canonical_id'],r['right_canonical_id']
        require(a in nodes and b in nodes and a<b and r['directed']=='False', 'Invalid undirected pair')
        require(THRESHOLD<=float(r['similarity'])<=1 and r['relation_type']=='SIMILAR_TO', 'Invalid semantic score/type')
        require(r['left_document_id']==nodes[a] and r['right_document_id']==nodes[b], 'Semantic document mismatch')
        require(r['scope']==('cross_document' if nodes[a]!=nodes[b] else 'within_document'), 'Semantic scope mismatch')
        require(r['source_input_sha256']==entity_hash, '2D input hash mismatch')
        for side,eid in [('left',a),('right',b)]:
            e = entities_by_id[eid]
            require(r[side+'_name']==e['text_name'] and json.loads(r[side+'_types'])==e['entity_types'] and
                    json.loads(r[side+'_source_entity_ids'])==e['source_entity_ids'], 'Changed semantic entity metadata')
    cross = sum(r['scope']=='cross_document' for r in links)
    require(cross==194 and len(links)-cross==491, 'Unexpected semantic scopes')
    expected = connectivity(nodes,[(r['head_id'],r['tail_id']) for r in relations]+
                            [(r['left_canonical_id'],r['right_canonical_id']) for r in links])
    require(expected==dict(entities=1122,isolated=200,weak_components=258,largest_component_vertices=580,
                           largest_component_documents=23), 'Source snapshot differs from 2D')
    return documents,entities,relations,links,expected


def payload(documents, entities, relations, links, fingerprint):
    def marked(source, identity):
        return dict(props(source),key=key(identity),namespace=NS,fingerprint=fingerprint,
                    source_namespace='relex_base_t035')
    node_rows, edge_rows = {}, {}
    for d in documents:
        properties = marked(d,d['document_id'])
        node_rows[properties['key']] = dict(label='RELEXDocument',properties=properties)
    for e in entities:
        eid = e['canonical_entity_id']
        original = next(m['source_entity'] for m in e['members'] if m['source_entity']['entity_id']==eid)
        properties = marked(e,eid)
        properties.update(entity_id=eid,name=e['text_name'],start=original['start'],end=original['end'],
                          confidence=original['confidence'],confidence_source_entity_id=eid,
                          source_confidences_json=dumps([dict(entity_id=m['source_entity']['entity_id'],
                                                             confidence=m['source_entity']['confidence']) for m in e['members']]))
        node_rows[properties['key']] = dict(label='RELEXEntity',properties=properties)
        edge_id = 'mention:'+eid
        edge_rows[key(edge_id)] = dict(type='RELEX_MENTIONS',head=key(e['document_id']),tail=key(eid),
            properties=dict(key=key(edge_id),namespace=NS,fingerprint=fingerprint,document_id=e['document_id'],
                            source_entity_ids=e['source_entity_ids'],source_mention_ids=e['source_mention_ids'],
                            source_offsets_json=dumps(e['source_offsets'])))
    for r in relations:
        properties = marked(r,r['canonical_triple_id'])
        properties.update(triple_id=r['canonical_triple_id'],source_stage='1C',
                          source_confidence_values=[x['confidence'] for x in r['source_confidences']])
        edge_rows[properties['key']] = dict(type='RELEX_REL',head=key(r['head_id']),tail=key(r['tail_id']),properties=properties)
    for r in links:
        typed = dict(r)
        for name in ('threshold','similarity','margin_above_threshold'):
            typed[name] = float(typed[name])
        for name in ('directed','type_set_conflict','types_disjoint','same_normalized_name','damaged_name',
                     'partial_token','baseline_same_component','individually_reviewed_2c'):
            typed[name] = typed[name]=='True'
        for side in ('left','right'):
            for suffix in ('types','quality_flags','source_entity_ids','source_mention_ids','context'):
                typed[side+'_'+suffix] = json.loads(typed[side+'_'+suffix])
        properties = marked(typed,r['semantic_link_id'])
        properties.update(source_json=dumps(r),undirected=True,source_stage='2D',
                          document_ids=[r['left_document_id'],r['right_document_id']],
                          cross_document=r['scope']=='cross_document',source_file=LINKS.relative_to(ROOT).as_posix())
        edge_rows[properties['key']] = dict(type='SIMILAR_TO',head=key(r['left_canonical_id']),
                                           tail=key(r['right_canonical_id']),properties=properties)
    require(len(node_rows)==1152 and len(edge_rows)==3567, 'Payload IDs collide')
    return node_rows,edge_rows


def outside_snapshot(session):
    digest, counts = hashlib.sha256(), {}
    queries = dict(nodes="MATCH (n) WHERE coalesce(n.namespace,'')<>$ns RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS properties ORDER BY id",
                   relationships="MATCH (a)-[r]->(b) WHERE coalesce(r.namespace,'')<>$ns RETURN elementId(r) AS id, type(r) AS type, elementId(a) AS head, elementId(b) AS tail, properties(r) AS properties ORDER BY id")
    for kind,query in queries.items():
        count = 0
        for row in session.run(query,ns=NS):
            data = row.data()
            if 'labels' in data: data['labels'].sort()
            digest.update((kind+dumps(data)+'\n').encode()); count+=1
        counts[kind] = count
    return dict(**counts,sha256=digest.hexdigest())


def statements():
    text = '\n'.join(line for line in (OUT/'graph_stats.cypher').read_text(encoding='utf-8').splitlines()
                     if not line.lstrip().startswith('//'))
    return [q.strip() for q in text.split(';') if q.strip()]


def import_once(session, node_rows, edge_rows, fingerprint):
    def transaction(tx):
        for query in ("MATCH (n {namespace:$ns}) RETURN DISTINCT n.fingerprint AS fingerprint",
                      "MATCH ()-[r]->() WHERE r.namespace=$ns RETURN DISTINCT r.fingerprint AS fingerprint"):
            require(all(r['fingerprint']==fingerprint for r in tx.run(query,ns=NS)),
                    'Target namespace belongs to another snapshot; refusing overwrite')
        created = dict(nodes=0,relationships=0)
        def run(query, **params):
            result = tx.run(query,namespace=NS,**params)
            summary = result.consume()
            created['nodes'] += summary.counters.nodes_created
            created['relationships'] += summary.counters.relationships_created
        for row in node_rows.values():
            if row['label']=='RELEXDocument':
                run(core.DOCUMENT_QUERY,key=row['properties']['key'],properties=row['properties'])
        entities = [r for r in node_rows.values() if r['label']=='RELEXEntity']
        for start in range(0,len(entities),100):
            run('UNWIND $rows AS row MERGE (e:RELEXEntity {key:row.properties.key,namespace:$namespace}) SET e += row.properties',
                rows=entities[start:start+100])
        for rel_type in ('RELEX_MENTIONS','RELEX_REL','SIMILAR_TO'):
            rows = [r for r in edge_rows.values() if r['type']==rel_type]
            # rel_type is from this fixed tuple, never from user data.
            head_label = 'RELEXDocument' if rel_type=='RELEX_MENTIONS' else 'RELEXEntity'
            query = (f'UNWIND $rows AS row MATCH (h:{head_label} {{key:row.head,namespace:$namespace}}) '
                     'MATCH (t:RELEXEntity {key:row.tail,namespace:$namespace}) '
                     f'MERGE (h)-[r:{rel_type} {{key:row.properties.key,namespace:$namespace}}]->(t) '
                     'SET r += row.properties')
            for start in range(0,len(rows),100): run(query,rows=rows[start:start+100])
        actual = tx.run(statements()[0]).single().data()
        require(actual==dict(namespace=NS,documents=30,entities=1122,total_nodes=1152,factual_relations=1760,
                             semantic_links=685,document_mentions=1122,kg_edges=2445,total_relationships=3567),
                f'Atomic count check failed: {actual}')
        return created
    return session.execute_write(transaction)


def verify(session, node_rows, edge_rows, expected):
    actual_nodes = session.run('MATCH (n {namespace:$ns}) RETURN n.key AS key,labels(n) AS labels,properties(n) AS properties',ns=NS).data()
    require(len(actual_nodes)==len(node_rows) and {r['key'] for r in actual_nodes}==set(node_rows), 'Neo4j node coverage mismatch')
    for r in actual_nodes:
        source = node_rows[r['key']]
        require(source['label'] in r['labels'] and all(r['properties'].get(k)==v for k,v in source['properties'].items() if v is not None),
                'Neo4j node metadata/provenance mismatch')
    actual_edges = session.run('MATCH (h)-[r]->(t) WHERE r.namespace=$ns RETURN r.key AS key,type(r) AS type,h.key AS head,t.key AS tail,properties(r) AS properties,h.namespace AS head_namespace,t.namespace AS tail_namespace',ns=NS).data()
    require(len(actual_edges)==len(edge_rows) and {r['key'] for r in actual_edges}==set(edge_rows), 'Neo4j relationship coverage mismatch')
    graph_edges = []
    for r in actual_edges:
        source = edge_rows[r['key']]
        require(r['head_namespace']==r['tail_namespace']==NS and
                all(r[k]==source[k] for k in ('type','head','tail')) and
                all(r['properties'].get(k)==v for k,v in source['properties'].items() if v is not None),
                'Neo4j relationship direction/type/metadata/provenance mismatch')
        if r['type'] in ('RELEX_REL','SIMILAR_TO'): graph_edges.append((r['head'],r['tail']))
    graph_nodes = {r['key']:r['properties']['document_id'] for r in actual_nodes if 'RELEXEntity' in r['labels']}
    require(connectivity(graph_nodes,graph_edges)==expected,'Neo4j connectivity differs from 2D')
    return connectivity(graph_nodes,graph_edges)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--neo4j-env',type=Path)
    args = parser.parse_args()
    protected = {path for path in ER.rglob('*') if path.is_file() and not path.is_relative_to(OUT)}
    protected.update([DOCUMENTS,ROOT/'scripts/build_relex_graphs.py'])
    hashes = {path:sha(path) for path in sorted(protected)}
    documents,entities,relations,links,expected = prepare()
    signature = dict(stage='2E-v1',namespace=NS,model=MODEL,mode=MODE,threshold=THRESHOLD,
                     inputs={path.relative_to(ROOT).as_posix():hashes[path] for path in (DOCUMENTS,ENTITIES,RELATIONS,LINKS)})
    fingerprint = core.digest(signature)
    node_rows,edge_rows = payload(documents,entities,relations,links,fingerprint)
    print('Input validated: 30 documents, 1122 entities, 1760 facts, 685 semantic links (194 cross / 491 within).',flush=True)
    driver,database = core.connect_neo4j(args.neo4j_env)
    try:
        with driver.session(database=database) as session:
            version = session.run('CALL dbms.components() YIELD name,versions,edition RETURN name,versions,edition').data()
            outside_before = outside_snapshot(session)
            print('Importing only namespace '+NS,flush=True)
            first = import_once(session,node_rows,edge_rows,fingerprint)
            actual = verify(session,node_rows,edge_rows,expected)
            print('First import and lossless read-back verified: '+dumps(actual),flush=True)
            second = import_once(session,node_rows,edge_rows,fingerprint)
            require(second==dict(nodes=0,relationships=0),'Repeated import created duplicates')
            require(verify(session,node_rows,edge_rows,expected)==actual,'Repeated import changed data')
            checks = []
            from neo4j import Query
            for i,query in enumerate(statements(),1):
                result = session.run(Query(query,timeout=90))
                data = result.data()
                summary = result.consume()
                notifications = [s for s in summary.gql_status_objects if s.is_notification]
                require(not any('deprecat' in s.status_description.lower() for s in notifications),'Deprecated Cypher syntax')
                require(data,'Empty graph_stats query '+str(i))
                if i==2: require(data[0]['cross_document']==194 and data[0]['within_document']==491,'Cypher scope mismatch')
                if i==4: require(data[0]['isolated']==200,'Cypher isolation mismatch')
                if i==5: require(data[0]==dict(weak_components=258,entities=1122,largest_component_vertices=580,largest_component_documents=23),'Live Cypher components mismatch')
                checks.append(dict(query=i,rows=len(data),notifications=len(notifications)))
                print('graph_stats query '+str(i)+' passed, rows='+str(len(data)),flush=True)
            outside_after = outside_snapshot(session)
            require(outside_before==outside_after,'Data outside target namespace changed')
        require(all(sha(path)==digest for path,digest in hashes.items()),'Source artifact changed')
    finally:
        driver.close()
    lines = ['# Этап 2E — фактический импорт Global KG','',
             f'Namespace: `{NS}`. База: `{database}`. Конфигурация: `{MODEL}`, `{MODE}`, threshold `{THRESHOLD}`. Фактическая проверка UTC: `{datetime.now(timezone.utc).isoformat()}`.','',
             '| Показатель | Фактически в Neo4j |','|---|---:|',
             '| Документы RELEXDocument | 30 |','| Канонические сущности RELEXEntity | 1122 |',
             '| Исходные фактические RELEX_REL | 1760 |','| SIMILAR_TO | 685 |',
             '| Междокументные / внутридокументные SIMILAR_TO | 194 / 491 |',
             '| Служебные RELEX_MENTIONS | 1122 |','| Всего физических узлов / отношений | 1152 / 3567 |',
             '| Рёбра KG без служебных связей | 2445 |','| Изолированные сущности | 200 |',
             '| Доля изолированных | 17.83% |','| Слабосвязные компоненты | 258 |',
             '| Крупнейшая компонента: вершины / документы | 580 / 23 |','',
             'Все показатели связности совпали с 2D. Они рассчитаны по сущностям и рёбрам RELEX_REL/SIMILAR_TO, без документов и RELEX_MENTIONS. Исходные 13 фактических петель сохранены. Новых петель и фактических междокументных отношений нет.','',
             'SIMILAR_TO хранится одним физическим отношением на неупорядоченную пару, в стабильном порядке canonical IDs, с directed=false и undirected=true. Для обхода используется неориентированный шаблон. Два встречных ребра не создаются. На ребре сохранены точный cosine, модель, режим, порог, scope, оба документа, исходные ID, флаги и решения 2C. Эти решения касаются тождества и не доказывают корректность SIMILAR_TO.','',
             'Схема следует scripts/build_relex_graphs.py: RELEXDocument/RELEXEntity, namespace-prefixed key, фактические типы в relation_type у RELEX_REL, принадлежность документа через RELEX_MENTIONS. Старый валидатор namespace и старый код импорта не изменены. Полные исходные объекты сохранены как source_json; вложенные структуры дополнительно хранятся в *_json. Все исходные confidence, offsets, evidence, source triple/entity/mention IDs и provenance доступны без усреднения. Плоские confidence/start/end сущности относятся только к исходному canonical ID; полные списки упоминаний сохранены.','',
             '## Проверки','',f'- Проход MERGE перед финальной проверкой добавил: `{dumps(first)}`. Это счётчики текущего запуска, а не размер namespace; при уже выполненном импорте ожидается 0.',
             f'- Повторный импорт создал: `{dumps(second)}`; полное чтение свойств подтвердило отсутствие изменений и дублей.',
             '- Все 30 документов, 1122 сущности, 1760 фактических и 685 семантических отношений прочитаны из Neo4j и сопоставлены с подготовленным payload, включая направления, типы и все свойства/provenance.',
             '- Связность по фактически прочитанным узлам/рёбрам подтверждена независимым живым Cypher-запросом: распространение минимального индекса вершины по соседям за V−1 раундов, без GDS/APOC и кэшированных компонент. Это локальный расчёт внутри read-only запроса, не изменение вершин/типов.',
             f'- Все 9 запросов graph_stats.cypher выполнены: `{dumps(checks)}`.',
             f'- Данные вне целевого namespace не изменились: `{dumps(outside_before)}` (сравнивались ID, labels, endpoints, типы и все свойства узлов/рёбер до и после).',
             '- SHA-256 входных файлов и прошлых артефактов совпали до и после. Никакие NER/RE/embedding/LLM-модели не запускались. Пароли и конфигурация сервера не менялись и не записывались в отчёт.','',
             '## Cypher и повторный импорт','',
             'graph_stats.cypher содержит 9 самостоятельных запросов для Neo4j 2026.08.1 / Cypher 25: общие количества, scope семантических связей, междокументные примеры, изоляция и степень, точные слабые компоненты, охват документов, топ документов, топ сущностей и реальные междокументные пути. Используется CALL (variables), без устаревшего importing WITH. Запрос компонент выполняет точное распространение минимальных индексов по спискам смежности, без перечисления путей и ограничений на диаметр. После V−1 раундов метка совпадает у всех и только у вершин одной компоненты. Данные читаются непосредственно из БД.','',
             'Синтаксис: [CALL scope](https://neo4j.com/docs/cypher-manual/current/subqueries/call-subquery/index.html), [reduce](https://neo4j.com/docs/cypher-manual/current/functions/list/).','',
             'Повторный импорт: `.venv/Scripts/python.exe scripts/import_global_semantic_graph.py` с действующими NEO4J_* в окружении, либо `--neo4j-env <existing-file>`. Учётные данные не встроены в скрипт. MERGE ограничен единственным namespace; другой fingerprint в нём приводит к отказу без перезаписи. Все записи одного прохода выполняются одной транзакцией с проверкой количеств перед commit.','',
             f'Fingerprint: `{fingerprint}`. Neo4j: `{dumps(version)}`.','',
             'Качество Graph-RAG и достоверность новых семантических переходов не оценивались. Импорт не означает подтверждения эквивалентности сущностей. Этап 2E завершён.','',
             'SHA-256 входов и защищённых артефактов:','']
    lines += [f'- `{path.relative_to(ROOT).as_posix()}`: `{digest}`' for path,digest in hashes.items()]
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(dumps(dict(namespace=NS,first_import=first,repeated_import=second,connectivity=actual,cypher_checks=checks)),flush=True)


if __name__=='__main__':
    main()
