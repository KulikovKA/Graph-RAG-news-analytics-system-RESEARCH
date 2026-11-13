"""Import the frozen Stage 2G.2 Hybrid C result. Default: read-only dry run."""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import tempfile

import build_dynamic_concept_graph as graph

ROOT = graph.ROOT
SOURCE = graph.TARGET_NS
TARGET = 'relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v2_sem_name_t0830'
REPORT = graph.base.ER / 'semantic_concept_sweep_extended'
require, dumps = graph.require, graph.dumps
EXPECTED = dict(documents=30, entities=811, concepts=36, kg_vertices=847, total_nodes=877,
                factual_relations=1307, semantic_links=492, concept_links=138, document_mentions=811,
                kg_edges=1937, total_relationships=2748)


def load_delta():
    manifest = json.loads((REPORT / 'manifest.json').read_text(encoding='utf-8'))
    original = json.loads((graph.OUT / 'verification.json').read_text(encoding='utf-8'))
    original_manifest = json.loads((graph.OUT / 'manifest.json').read_text(encoding='utf-8'))
    require(manifest['source_namespace'] == SOURCE and manifest['source_sha256'] == original['target_sha256'] and manifest['source_fingerprint'] == original_manifest['fingerprint'], 'Experiment baseline metadata mismatch')
    for name in ('new_memberships.jsonl', 'validator_only_decisions.jsonl', 'semantic_decisions.jsonl.gz', 'quality_audit.csv'):
        require(graph.base.sha(REPORT / name) == manifest['artifact_files_sha256'][name], 'Artifact hash mismatch: ' + name)
    require(manifest['embedding']['model']['model'] == 'intfloat/e5-small-v2', 'Wrong model')
    rows = graph.base.core.read_jsonl(REPORT / 'new_memberships.jsonl')
    delta = {}
    for row in graph.progress(rows, total=len(rows), desc='ЧТЕНИЕ MEMBERSHIPS'):
        if (row.get('embedding_mode'), row.get('threshold'), row.get('status')) == ('text_name', '0.830', 'ACCEPT'):
            pair = (row['source_entity_id'], row['target_concept_id'])
            if pair in delta:
                require(all(delta[pair][k] == row[k] for k in ('document_id', 'target_concept_name')), 'Conflicting duplicate membership')
                if (row.get('cosine_similarity') or 0, row['decision_id']) <= (delta[pair].get('cosine_similarity') or 0, delta[pair]['decision_id']):
                    continue
            delta[pair] = row
    require(len(delta) == 8, 'Expected eight unique accepted memberships')
    with (REPORT / 'quality_audit.csv').open(encoding='utf-8-sig', newline='') as stream:
        reviewed = {(r['source_entity_id'], r['target_concept_id']) for r in csv.DictReader(stream) if r['audit_verdict'] == 'SUSPICIOUS'}
    require(reviewed == {p for p, r in delta.items() if r['target_concept_name'].casefold() == 'cell'} and len(reviewed) == 2, 'Cell review audit mismatch')
    baseline = graph.base.core.read_jsonl(REPORT / 'validator_only_decisions.jsonl')
    require({(r['source_entity_id'], r['target_concept_id']) for r in baseline if r['status'] == 'ACCEPT'} <= set(delta), 'Validator-only improvements missing')
    decisions = {}
    with gzip.open(REPORT / 'semantic_decisions.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in graph.progress(stream, total=manifest['decisions'], desc='ЧТЕНИЕ РЕШЕНИЙ'):
            row = json.loads(line)
            if row.get('decision_id') in {r['decision_id'] for r in delta.values()}:
                decisions[row['decision_id']] = row
    for row in delta.values():
        if row.get('origin') == 'validator_only':
            match = next((r for r in baseline if r['decision_id'] == row['decision_id']), None)
        else:
            match = decisions.get(row['decision_id'])
        require(match is not None and match['status'] == 'ACCEPT' and match['source_entity_id'] == row['source_entity_id'] and match['target_concept_id'] == row['target_concept_id'], 'Missing accepted decision evidence')
        row['decision_evidence'] = match
    return manifest, delta


def build(nodes, edges, manifest, delta):
    require(graph.snapshot_hash(nodes, edges) == manifest['source_sha256'], 'Baseline SHA-256 mismatch')
    require(all(r['properties']['fingerprint'] == manifest['source_fingerprint'] for r in [*nodes.values(), *edges.values()]), 'Baseline fingerprint mismatch')
    require(graph.counts(nodes, edges) == dict(documents=30, entities=811, concepts=34, kg_vertices=845, total_nodes=875, factual_relations=1307, semantic_links=492, concept_links=130, document_mentions=811, kg_edges=1929, total_relationships=2740), 'Baseline counts mismatch')
    signature = dict(stage='2G.3', source_sha256=manifest['source_sha256'], source_fingerprint=manifest['source_fingerprint'], target=TARGET, mode='text_name', threshold='0.830', resolver='Hybrid C', delta_sha256=graph.digest(sorted(delta)))
    fingerprint = graph.digest(signature)
    def key(k):
        require(k.startswith(SOURCE + ':'), 'Cross-namespace key')
        return TARGET + k[len(SOURCE):]
    def properties(p):
        return dict(p, key=key(p['key']), namespace=TARGET, fingerprint=fingerprint)
    new_nodes = {}
    for k, row in graph.progress(nodes.items(), total=len(nodes), desc='ПОДГОТОВКА УЗЛОВ'):
        new_nodes[key(k)] = dict(label=row['label'], properties=properties(row['properties']))
    new_edges = {}
    for k, row in graph.progress(edges.items(), total=len(edges), desc='ПОДГОТОВКА СВЯЗЕЙ'):
        require(row['head'] in nodes and row['tail'] in nodes, 'Baseline cross-namespace edge')
        new_edges[key(k)] = dict(type=row['type'], head=key(row['head']), tail=key(row['tail']), properties=properties(row['properties']))
    for (eid, cid), row in graph.progress(sorted(delta.items()), total=len(delta), desc='ДОБАВЛЕНИЕ ACCEPT'):
        entity_key, concept_key = TARGET + ':' + eid, TARGET + ':' + cid
        require(entity_key in new_nodes and new_nodes[entity_key]['properties']['document_id'] == row['document_id'], 'Entity/document mismatch')
        if concept_key not in new_nodes:
            p = dict(key=concept_key, namespace=TARGET, fingerprint=fingerprint, concept_id=cid, name=row['target_concept_name'], label=row['target_concept_name'], entity_types=['CONCEPT'], source_namespace=SOURCE, fact_inheritance_allowed=False)
            new_nodes[concept_key] = dict(label='RELEXConcept', properties=p)
        require(new_nodes[concept_key]['label'] == 'RELEXConcept' and new_nodes[concept_key]['properties']['name'].casefold() == row['target_concept_name'].casefold(), 'Concept mismatch')
        rid = 'semantic-membership:' + graph.digest([eid, cid])[:24]
        edge_key = TARGET + ':' + rid
        require(edge_key not in new_edges, 'Membership ID collision')
        suspicious = row['target_concept_name'].casefold() == 'cell'
        p = dict(key=edge_key, namespace=TARGET, fingerprint=fingerprint, head_id=eid, tail_id=cid, document_id=row['document_id'], relation_type='HAS_SHARED_CONCEPT', factual=False, identity=False, fact_inheritance_allowed=False, requires_review=suspicious, review_status='SUSPICIOUS' if suspicious else 'ACCEPT', resolver_version=row['resolver_version'], embedding_mode='text_name', threshold='0.830', reason=row['reason'], decision_id=row['decision_id'], evidence_json=dumps(row['decision_evidence']))
        if row.get('cosine_similarity') is not None:
            p['cosine_similarity'] = row['cosine_similarity']
        new_edges[edge_key] = dict(type='HAS_SHARED_CONCEPT', head=entity_key, tail=concept_key, properties=p)
    require(graph.counts(new_nodes, new_edges) == EXPECTED, 'Payload counts mismatch')
    require(sum(r['properties'].get('requires_review') is True for r in new_edges.values()) == 2, 'Cell review flags mismatch')
    pairs = [(r['head'], r['tail']) for r in new_edges.values() if r['type'] == 'HAS_SHARED_CONCEPT']
    require(len(pairs) == len(set(pairs)), 'Duplicate memberships')
    names = [graph.class_form(r['properties']['name']) for r in new_nodes.values() if r['label'] == 'RELEXConcept']
    require(len(names) == len(set(names)), 'Duplicate concepts')
    for edge in new_edges.values():
        require(edge['head'] in new_nodes and edge['tail'] in new_nodes, 'Cross-namespace relationship')
        if edge['type'] == 'RELEX_REL':
            require(new_nodes[edge['head']]['properties']['document_id'] == new_nodes[edge['tail']]['properties']['document_id'], 'Cross-document fact')
        if edge['type'] == 'HAS_SHARED_CONCEPT':
            require(all(edge['properties'].get(k) is False for k in ('factual', 'identity', 'fact_inheritance_allowed')), 'Unsafe concept membership')
    return new_nodes, new_edges, fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--apply', action='store_true')
    parser.add_argument('--neo4j-env', type=Path)
    args = parser.parse_args()
    graph.banner('ЧТЕНИЕ ГОТОВЫХ РЕШЕНИЙ STAGE 2G.2; МОДЕЛИ НЕ ЗАПУСКАЮТСЯ')
    manifest, delta = load_delta()
    driver, database = graph.base.core.connect_neo4j(args.neo4j_env)
    try:
        with driver.session(database=database) as session:
            graph.banner('ЧТЕНИЕ И ПРОВЕРКА BASELINE NEO4J')
            nodes, edges = graph.read_snapshot(session, SOURCE, (875, 2740))
            payload_nodes, payload_edges, fingerprint = build(nodes, edges, manifest, delta)
            print('ПРОВЕРЕННЫЙ PAYLOAD:', dumps(EXPECTED), flush=True)
            if not args.apply:
                print('DRY-RUN: ЗАПИСЬ НЕ ВЫПОЛНЯЛАСЬ.', flush=True)
                return
            with tempfile.TemporaryDirectory(prefix='stage_2g3_') as directory:
                graph.kg.TARGET_NS = TARGET
                graph.kg.OUT = Path(directory)
                graph.kg.EXPECTED = EXPECTED
                graph.kg.write_queries()
                graph.base.NS = TARGET
                graph.banner('СНИМОК ДРУГИХ NAMESPACES ПЕРЕД ИМПОРТОМ')
                outside_before = graph.base.outside_snapshot(session)
                first = graph.kg.import_once(session, payload_nodes, payload_edges, fingerprint, 1)
                metrics, target_hash, _, _ = graph.kg.verify(session, payload_nodes, payload_edges)
                second = graph.kg.import_once(session, payload_nodes, payload_edges, fingerprint, 2)
                require(second == dict(nodes=0, relationships=0), 'Idempotence failure')
                source_nodes, source_edges = graph.read_snapshot(session, SOURCE, (875, 2740))
                require(graph.snapshot_hash(source_nodes, source_edges) == manifest['source_sha256'], 'Source namespace changed')
                require(graph.base.outside_snapshot(session) == outside_before, 'Other namespaces changed')
                cross = session.run("MATCH (a)-[r]->(b) WHERE (a.namespace=$ns OR b.namespace=$ns OR r.namespace=$ns) AND (coalesce(a.namespace,'')<>$ns OR coalesce(b.namespace,'')<>$ns OR coalesce(r.namespace,'')<>$ns) RETURN count(r) AS n", ns=TARGET).single()['n']
                require(cross == 0, 'Cross-namespace relationship')
                print('ИМПОРТ ПРОВЕРЕН:', dumps(dict(first=first, repeated=second, target_sha256=target_hash, metrics=metrics, **EXPECTED)), flush=True)
    finally:
        driver.close()


if __name__ == '__main__':
    main()
