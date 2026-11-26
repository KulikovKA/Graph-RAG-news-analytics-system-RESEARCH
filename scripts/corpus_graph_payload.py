"""Corpus-wide resolution, semantic layers and isolated Neo4j publication."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from itertools import combinations

import build_relex_graphs as relex
from cgraph.concepts import Concept, ConceptResolver, Entity, class_form
from cgraph.ner_v2.io import validate_span
from cgraph.semantic_concepts import HybridConceptResolver, cosine_matrix, reference_records
from exact_matching_audit import audit
from import_global_semantic_graph import dumps, props, require
from prepare_semantic_entity_input import member_record
from reconstruct_global_kg_2f3 import LABELS, REL_TYPES, metrics
from sweep_semantic_concepts import binary_cutoff, compact_proof, evaluate
from tqdm import tqdm

NONFACTUAL = dict(factual=False, identity=False, fact_inheritance_allowed=False)
EDGE_LABELS = {
    "RELEX_MENTIONS": ("RELEXDocument", "RELEXEntity"),
    "RELEX_REL": ("RELEXEntity", "RELEXEntity"),
    "SIMILAR_TO": ("RELEXEntity", "RELEXEntity"),
    "HAS_SHARED_CONCEPT": ("RELEXEntity", "RELEXConcept"),
}
READ_NODES = "MATCH (n {namespace:$ns}) RETURN n.key AS key,labels(n) AS labels,properties(n) AS properties ORDER BY key"
READ_EDGES = ("MATCH (h)-[r]->(t) WHERE r.namespace=$ns RETURN r.key AS key,type(r) AS type,"
              "h.key AS head,t.key AS tail,properties(r) AS properties ORDER BY key")


def resolve_entities(documents, ner, re, threshold):
    sources = [e for doc in documents for e in re[doc]["entities"]]
    relations = [r for doc in documents for r in re[doc]["relations"]]
    groups, _ = audit(sources, relations, documents)
    by_id = {e["entity_id"]: e for e in sources}
    kept = {eid for eid, e in by_id.items() if e["confidence"] >= threshold}
    mapping = {eid: eid for eid in sorted(kept)}
    memberships = {eid: member_record(by_id[eid], documents[by_id[eid]["document_id"]], None) for eid in mapping}
    keys = defaultdict(dict)
    for eid in mapping:
        e = by_id[eid]
        keys[e["document_id"]][e["start"], e["end"]] = e
    supported = defaultdict(list)
    for doc, result in ner.items():
        definitions = result["abbreviations"]
        expansions = defaultdict(set)
        for d in definitions:
            expansions[d["short_form"]].add(d["long_form"])
        for d in definitions:
            left = keys[doc].get((d["long_start"], d["long_end"]))
            right = keys[doc].get((d["short_start"], d["short_end"]))
            if not left or not right or len(expansions[d["short_form"]]) != 1:
                continue
            ids = (left["entity_id"], right["entity_id"])
            if (left["entity_types"] == right["entity_types"] and len(left["entity_types"]) == 1
                    and not any(memberships[i]["quality_flags"] for i in ids)):
                supported[ids].append(d)
    incident = Counter(eid for pair in supported for eid in pair)
    merges = []
    for (left, right), evidence in sorted(supported.items()):
        # Только пары с прямым подтверждением и без конкурирующего определения.
        if incident[left] == incident[right] == 1:
            mapping[right] = left
            merges.append(dict(representative=left, source_ids=[left, right], evidence=evidence))
    final = defaultdict(list)
    for eid, target in mapping.items():
        final[target].append(memberships[eid])
    return dict(pre_entities=len(sources), kept_before_er=len(kept), exact_groups=sum(
        g["record_type"] == "exact_group" for g in groups), merge_groups=merges,
        mapping=mapping, entities=dict(final), relations=[r for r in relations
         if r["head_id"] in kept and r["tail_id"] in kept], unique_source_relations=len(relations))


def semantic_links(resolution, vectors, threshold):
    import numpy as np

    by_id = {m["source_entity"]["entity_id"]: m for members in resolution["entities"].values() for m in members}
    ids = sorted(by_id)
    if not ids:
        return []
    embeddings = np.asarray([vectors[by_id[i]["text_name"]] for i in ids], dtype=np.float32)
    mapping = resolution["mapping"]
    links = []
    for start in range(0, len(ids), 256):
        scores = (embeddings[start:start+256] @ embeddings.T).astype(np.float64)
        for row, column in zip(*np.nonzero(scores >= binary_cutoff(threshold))):
            i, j = start + int(row), int(column)
            if i >= j:
                continue
            a, b = ids[i], ids[j]
            left, right = by_id[a]["source_entity"], by_id[b]["source_entity"]
            links.append(dict(link_id="sl:" + relex.digest([a, b]), head_id=mapping[a], tail_id=mapping[b],
                              source_head_id=a, source_tail_id=b, left_document_id=left["document_id"],
                              right_document_id=right["document_id"], left_context=by_id[a]["context"],
                              right_context=by_id[b]["context"], similarity=float(scores[row, column]),
                              threshold=threshold, mode="text_name", model="intfloat/e5-small-v2", **NONFACTUAL))
    return links


def profiles(documents, resolution):
    entities = []
    for eid, members in sorted(resolution["entities"].items()):
        rep = next(m["source_entity"] for m in members if m["source_entity"]["entity_id"] == eid)
        doc = documents[rep["document_id"]]
        aliases, evidence = set(), []
        for member in members:
            e, ctx = member["source_entity"], member["context"]
            aliases.update(e["normalized_forms"])
            for mention in e["source_mentions"]:
                for d in mention["abbreviation_candidates"]:
                    if class_form(e["surface_text"]) in {class_form(d["long_form"]), class_form(d["short_form"])}:
                        aliases.update((d["long_form"], d["short_form"]))
            evidence.append(dict(document_id=doc["document_id"], revision_id=doc["revision_id"],
                                 source_url=doc["source_url"], source_entity_id=e["entity_id"],
                                 source_mention_ids=member["source_mention_ids"], start=ctx["start"],
                                 end=ctx["end"], text=ctx["text"], mention_start=e["start"], mention_end=e["end"],
                                 mention_text=e["surface_text"]))
        entities.append(Entity(eid, doc["document_id"], rep["surface_text"], tuple(rep["entity_types"]),
                               tuple(sorted(aliases)), tuple(evidence)))
    return entities


def lexical_concepts(entities, links):
    by = {e.entity_id: e for e in entities}
    similar = [(r["head_id"], r["tail_id"]) for r in links]
    result = ConceptResolver([], []).resolve_batch(
        entities, similar, progress=lambda rows: tqdm(rows, desc="Shared Concepts", unit="entity"))
    catalog = []
    for row in result.concepts:
        ids = row["contextual_target_ids"]
        aliases = {name for eid in ids for name in (by[eid].name, *by[eid].aliases)}
        # Только имена класса образуют его лексический ключ. Прочие aliases
        # остаются в профиле участников и в provenance.
        forms = {class_form(a) for a in row["aliases"]} | {class_form(row["name"])}
        catalog.append(Concept(row["concept_id"], row["name"],
                               tuple(sorted(a for a in aliases if class_form(a) in forms)), tuple(ids)))
    existing = {(m["entity_id"], m["concept_id"]) for m in result.memberships}
    return result, catalog, existing


def hybrid_candidates(entities, catalog, existing, links, vectors, threshold, top_entities, top_concepts):
    import numpy as np

    if not entities:
        return []
    by = {e.entity_id: e for e in entities}
    doc_ids = np.array([e.document_id for e in entities])
    entity_vectors = np.asarray([vectors[e.name] for e in entities], dtype=np.float32)
    ee = cosine_matrix(entity_vectors, entity_vectors)
    cutoff, preserve = binary_cutoff(threshold), binary_cutoff("0.950")
    selected = set()
    for i in range(len(entities)):
        valid = np.flatnonzero((doc_ids != doc_ids[i]) & (ee[i] >= cutoff))
        ranked = valid[np.argsort(-ee[i, valid], kind="stable")][:top_entities]
        selected.update((min(i, int(j)), max(i, int(j))) for j in ranked)
        selected.update((i, int(j)) for j in valid if j > i and ee[i, j] >= preserve)
    mode = "text_name"
    similar = {tuple(sorted((r["head_id"], r["tail_id"]))) for r in links}
    candidates = []
    for i, j in sorted(selected):
        a, b = entities[i], entities[j]
        candidates.append(dict(candidate_id="ee:" + relex.digest([mode, a.entity_id, b.entity_id])[:24],
                               kind="entity_entity", source_entity_id=a.entity_id, reference_entity_id=b.entity_id,
                               existing_similar_to=(a.entity_id, b.entity_id) in similar,
                               cosine_similarity=float(ee[i, j])))
    if catalog:
        references = {(r["concept_id"], r["excluded_document_id"]): r for r in reference_records(catalog, by)}
        ec = cosine_matrix(entity_vectors, np.asarray([vectors[c.name] for c in catalog], dtype=np.float32))
        for i, entity in enumerate(entities):
            valid = [j for j in np.flatnonzero(ec[i] >= cutoff)
                     if (entity.entity_id, catalog[j].concept_id) not in existing]
            ranked = sorted(valid, key=lambda j: (-ec[i, j], catalog[j].concept_id))
            for rank, j in enumerate(ranked):
                if rank >= top_concepts and ec[i, j] < preserve:
                    break
                concept = catalog[j]
                cid = concept.concept_id
                ref = references.get((cid, entity.document_id), references[cid, None])
                candidates.append(dict(candidate_id="ec:" + relex.digest([mode, entity.entity_id, cid])[:24],
                                       kind="entity_concept", source_entity_id=entity.entity_id, target_concept_id=cid,
                                       reference_id=ref["reference_id"], cosine_similarity=float(ec[i, j])))
    return sorted((dict(row, embedding_mode=mode, qualifying_thresholds=[threshold]) for row in candidates),
                  key=lambda row: (row["kind"], row["candidate_id"]))


def hybrid_concepts(entities, concepts, catalog, existing, candidates, threshold):
    by = {e.entity_id: e for e in entities}
    cat = {c.concept_id: c for c in catalog}
    resolver = HybridConceptResolver(catalog, entities, existing)
    universe = {(r["entity_id"], r["concept_id"]) for r in concepts.decisions}
    extra = {}
    for eid, cid in sorted(universe - existing):
        if cid not in cat:
            continue
        c, entity = cat[cid], by[eid]
        references = sorted((by[i] for i in c.member_ids if i != eid),
                            key=lambda ref: (ref.document_id == entity.document_id, ref.entity_id))
        for ref in references:
            status, reason, evidence = resolver.validate_semantic(entity, c, ref)
            if status == "ACCEPT":
                extra[eid, cid] = dict(decision_id="validator-only:" + relex.digest([eid, cid])[:24],
                                       source_entity_id=eid, target_concept_id=cid, target_concept_name=c.name,
                                       document_id=entity.document_id, status=status, reason=reason,
                                       reference_entity_id=ref.entity_id, embedding_mode="lexical",
                                       evidence=compact_proof(evidence))
                break
    decisions, _ = evaluate(candidates, entities, catalog, existing, compact=True, memoize=True)
    accepted = {}
    for row in sorted((r for r in decisions if r["status"] == "ACCEPT" and r.get("adds_membership", True)),
                      key=lambda r: (-r["cosine_similarity"], r["decision_id"])):
        accepted.setdefault((row["source_entity_id"], row["target_concept_id"]), row)
    support = defaultdict(set)
    for (eid, cid), row in accepted.items():
        support[cid].add(row["document_id"])
    new_nodes = {class_form(c.name): c.concept_id for c in catalog}
    rows = dict(extra)
    rows.update({(eid, cid): row for (eid, cid), row in accepted.items()
                 if cid in cat or len(support[cid]) >= 2})
    for (eid, cid), row in sorted(rows.items()):
        name = row["target_concept_name"]
        normalized = class_form(name)
        if normalized in new_nodes and new_nodes[normalized] != cid:
            raise ValueError("Hybrid concept duplicates a lexical class")
        if normalized not in new_nodes:
            concepts.concepts.append(dict(concept_id=cid, name=name, normalized_name=normalized, **NONFACTUAL))
            new_nodes[normalized] = cid
        concepts.memberships.append(dict(membership_id="hybrid:" + relex.digest([eid, cid]), entity_id=eid,
                                         concept_id=cid, document_id=row["document_id"], evidence=row,
                                         requires_review=normalized == "cell", threshold=threshold, **NONFACTUAL))
    return dict(concepts=concepts.concepts, memberships=concepts.memberships,
                lexical_decisions=len(concepts.decisions), hybrid_candidates=len(candidates),
                hybrid_memberships=len(rows))


def build_payload(namespace, fingerprint, documents, resolution, links, concepts):
    nodes, edges = {}, {}
    key = lambda i: namespace + ":" + i

    def properties(identity, data):
        return {k: v for k, v in props(data).items() if v is not None} | dict(
            key=key(identity), namespace=namespace, fingerprint=fingerprint)

    def node(identity, label, data):
        require(key(identity) not in nodes, "Duplicate graph node")
        nodes[key(identity)] = dict(label=label, properties=properties(identity, data))

    def edge(identity, kind, head, tail, data):
        require(key(identity) not in edges, "Duplicate graph relation")
        require(key(head) in nodes and key(tail) in nodes, "Missing graph endpoint")
        edges[key(identity)] = dict(type=kind, head=key(head), tail=key(tail), properties=properties(identity, data))

    for did, doc in documents.items():
        node(did, "RELEXDocument", doc)
    source_entities = {}
    for eid, members in resolution["entities"].items():
        rep = next(m for m in members if m["source_entity"]["entity_id"] == eid)
        e = rep["source_entity"]
        did = e["document_id"]
        doc = documents[did]
        offsets = []
        for m in members:
            original, ctx = m["source_entity"], m["context"]
            require(original["document_id"] == did == ctx["document_id"], "ER crossed a document boundary")
            validate_span(doc["text"], original["start"], original["end"], original["surface_text"])
            validate_span(doc["text"], ctx["start"], ctx["end"], ctx["text"])
            source_entities[original["entity_id"]] = original
            offsets.append({k: original[k] for k in ("entity_id", "start", "end", "surface_text")})
        node(eid, "RELEXEntity", dict(entity_id=eid, document_id=did, name=e["surface_text"],
             entity_types=e["entity_types"], aliases=sorted({a for m in members for a in
             m["source_entity"]["normalized_forms"]}), source_offsets=offsets, members=members,
             max_member_confidence=max(m["source_entity"]["confidence"] for m in members),
             representative_context=rep["context"], revision_id=doc["revision_id"]))
        edge("mention:" + eid, "RELEX_MENTIONS", did, eid,
             dict(document_id=did, revision_id=doc["revision_id"], source_offsets=offsets))
    for concept in concepts["concepts"]:
        node(concept["concept_id"], "RELEXConcept", concept | NONFACTUAL)
    mapping = resolution["mapping"]
    require(set(mapping) == set(source_entities), "ER provenance coverage mismatch")
    for row in resolution["relations"]:
        h, t, doc = source_entities[row["head_id"]], source_entities[row["tail_id"]], documents[row["document_id"]]
        require(h["document_id"] == t["document_id"] == doc["document_id"], "Cross-document fact")
        for window in row["evidence"]:
            validate_span(doc["text"], window["start"], window["end"], window["text"])
            for original in (h, t):
                require(original["entity_id"] in window["entity_ids"], "Fact source missing from window")
        head, tail = mapping[row["head_id"]], mapping[row["tail_id"]]
        edge(row["triple_id"], "RELEX_REL", head, tail,
             dict(row, head_id=head, tail_id=tail, source_relation=row, revision_id=doc["revision_id"], factual=True))
    for row in links:
        for side in ("left_context", "right_context"):
            ctx = row[side]
            validate_span(documents[ctx["document_id"]]["text"], ctx["start"], ctx["end"], ctx["text"])
        edge(row["link_id"], "SIMILAR_TO", row["head_id"], row["tail_id"], row)
    pairs = set()
    for row in concepts["memberships"]:
        eid, cid = row["entity_id"], row["concept_id"]
        require((eid, cid) not in pairs, "Duplicate concept membership")
        require(nodes[key(eid)]["properties"]["document_id"] == row["document_id"], "Invalid concept owner")
        pairs.add((eid, cid))
        edge(row["membership_id"], "HAS_SHARED_CONCEPT", eid, cid,
             row | dict(head_id=eid, tail_id=cid) | NONFACTUAL)
    return nodes, edges


def graph_statistics(nodes, edges, resolution, ner, re):
    kind = Counter(row["type"] for row in edges.values())
    kg_edges = len(edges) - kind["RELEX_MENTIONS"]
    connected = metrics(nodes, edges)[0] if kg_edges or resolution["entities"] else dict(
        kg_vertices=0, weak_components=0, largest_component_vertices=0, isolated=0, self_loop_only_vertices=0)
    documents = defaultdict(set)
    cross_semantic = []
    for row in edges.values():
        p = row["properties"]
        if row["type"] == "HAS_SHARED_CONCEPT":
            documents[row["tail"]].add(p["document_id"])
        elif row["type"] == "SIMILAR_TO" and p["left_document_id"] != p["right_document_id"]:
            cross_semantic.append(tuple(sorted((p["left_document_id"], p["right_document_id"]))))
    concept_pairs = {pair for ids in documents.values() for pair in combinations(sorted(ids), 2)}
    return dict(documents=len(re), ner_mentions=sum(len(r["mentions"]) for r in ner.values()),
                raw_relations=sum(r["raw_relation_count"] for r in re.values()),
                unique_source_relations=resolution["unique_source_relations"],
                unique_graph_triples=len({(r["head"], r["properties"]["relation_type"], r["tail"])
                                           for r in edges.values() if r["type"] == "RELEX_REL"}),
                factual_edges=kind["RELEX_REL"], entities_before_resolution=resolution["pre_entities"],
                entities_after_filter_before_er=resolution["kept_before_er"],
                entities_after_resolution=len(resolution["entities"]), er_merge_groups=len(resolution["merge_groups"]),
                semantic_links=kind["SIMILAR_TO"], shared_concepts=len(documents), memberships=kind["HAS_SHARED_CONCEPT"],
                cross_document_semantic_links=len(cross_semantic), cross_document_concept_pairs=len(concept_pairs),
                linked_document_pairs=len(set(cross_semantic) | concept_pairs),
                total_nodes=len(nodes), total_relationships=len(edges), kg_edges=kg_edges,
                mean_degree=2*kg_edges/connected["kg_vertices"] if connected["kg_vertices"] else 0, **connected)


def outside_snapshot(tx, namespace):
    snapshot = hashlib.sha256()
    counts = {}
    queries = {
        "nodes": "MATCH (n) WHERE coalesce(n.namespace,'')<>$ns RETURN elementId(n) AS id, "
                 "labels(n) AS labels, properties(n) AS properties ORDER BY id",
        "relationships": "MATCH (a)-[r]->(b) WHERE coalesce(r.namespace,'')<>$ns "
                         "RETURN elementId(r) AS id,type(r) AS type,elementId(a) AS head,"
                         "elementId(b) AS tail,properties(r) AS properties ORDER BY id",
    }
    for kind, query in queries.items():
        counts[kind] = 0
        for record in tx.run(query, ns=namespace):
            row = record.data()
            if "labels" in row:
                row["labels"].sort()
            snapshot.update((kind + dumps(row) + "\n").encode("utf-8"))
            counts[kind] += 1
    return counts | dict(sha256=snapshot.hexdigest())


def verify_neo4j(tx, namespace, nodes, edges, *, complete=True):
    snapshot = hashlib.sha256()
    keys = []
    for query, wanted in ((READ_NODES, nodes), (READ_EDGES, edges)):
        seen = set()
        for record in tx.run(query, ns=namespace):
            row = record.data()
            key = row.pop("key")
            expected = wanted.get(key)
            require(expected is not None and key not in seen, "Unexpected or duplicate Neo4j key")
            if "labels" in row:
                labels = row.pop("labels")
                require(len(labels) == 1, "Unexpected Neo4j labels")
                row["label"] = labels[0]
            require(row == expected, "Neo4j property, provenance or endpoint mismatch: " + key)
            seen.add(key)
            snapshot.update((dumps(row) + "\n").encode("utf-8"))
        if complete:
            require(seen == wanted.keys(), "Incomplete Neo4j read-back")
        keys.append(seen)
    cross = tx.run(
        "MATCH (a)-[r]->(b) WHERE (a.namespace=$ns OR b.namespace=$ns OR r.namespace=$ns) "
        "AND (coalesce(a.namespace,'')<>$ns OR coalesce(b.namespace,'')<>$ns OR coalesce(r.namespace,'')<>$ns) "
        "RETURN count(r) AS n", ns=namespace).single(strict=True)["n"]
    require(cross == 0, "Cross-namespace relationships")
    return dict(nodes=len(keys[0]), relationships=len(keys[1]), cross_namespace=cross,
                sha256=snapshot.hexdigest()), keys


def import_waves(rows, workers, *, relationships, batch_size=100):
    """Пакеты одной волны рёбер не блокируют общие узлы."""
    if not relationships or workers == 1:
        width = workers * batch_size
        for begin in range(0, len(rows), width):
            yield [rows[start:start+batch_size] for start in range(begin, min(begin + width, len(rows)), batch_size)]
        return

    pending = rows
    while pending:
        batches = [[] for _ in range(workers)]
        owners = {}
        postponed = []
        assigned = 0
        for index, row in enumerate(pending):
            if assigned == workers * batch_size:
                postponed.extend(pending[index:])
                break
            keys = row["head"], row["tail"]
            active = {owners[key] for key in keys if key in owners}
            if len(active) > 1:
                postponed.append(row)
                continue
            slot = next(iter(active)) if active else min(range(workers), key=lambda i: len(batches[i]))
            if len(batches[slot]) == batch_size:
                postponed.append(row)
                continue
            batches[slot].append(row)
            owners.update((key, slot) for key in keys)
            assigned += 1
        yield [batch for batch in batches if batch]
        pending = postponed


def import_neo4j(driver, database, namespace, nodes, edges, *, workers=1):
    require(workers > 0, "Neo4j import requires at least one worker")

    def write_batch(tx, query, batch):
        result = tx.run(query, rows=batch, ns=namespace)
        require(result.single(strict=True)["n"] == len(batch), "Neo4j batch count mismatch")
        counters = result.consume().counters
        return counters.nodes_created, counters.relationships_created

    def import_batch(query, batch):
        with driver.session(database=database) as worker_session:
            return worker_session.execute_write(write_batch, query, batch)

    with driver.session(database=database) as session, ThreadPoolExecutor(max_workers=workers) as pool:
        before = session.execute_read(outside_snapshot, namespace)
        _, present = session.execute_read(lambda tx: verify_neo4j(tx, namespace, nodes, edges, complete=False))
        created_nodes, created_edges = 0, 0
        for kinds, payload, field, verified_keys in ((LABELS, nodes, "label", present[0]),
                                                     (REL_TYPES, edges, "type", present[1])):
            desc = f"Neo4j {field} ({workers} workers)" if workers > 1 else f"Neo4j {field}"
            with tqdm(total=len(payload), initial=len(verified_keys), desc=desc, unit="object") as bar:
                for kind in kinds:
                    rows = [r for key, r in payload.items() if r[field] == kind and key not in verified_keys]
                    if field == "label":
                        query = (f"UNWIND $rows AS row MERGE (n:{kind} {{key:row.properties.key,"
                                 "namespace:$ns}) ON CREATE SET n += row.properties RETURN count(n) AS n")
                    else:
                        head, tail = EDGE_LABELS[kind]
                        query = (f"UNWIND $rows AS row MATCH (a:{head} {{key:row.head, namespace:$ns}}),"
                                 f"(b:{tail} {{key:row.tail, namespace:$ns}}) "
                                 "WHERE a.fingerprint=row.properties.fingerprint AND b.fingerprint=row.properties.fingerprint "
                                 f"MERGE (a)-[r:{kind} {{key:row.properties.key,namespace:$ns}}]->(b) "
                                 "ON CREATE SET r += row.properties RETURN count(r) AS n")
                    for wave in import_waves(rows, workers, relationships=field == "type"):
                        if workers == 1:
                            batch, = wave
                            new_nodes, new_edges = session.execute_write(write_batch, query, batch)
                            created_nodes += new_nodes
                            created_edges += new_edges
                            bar.update(len(batch))
                            continue
                        jobs = {pool.submit(import_batch, query, batch): len(batch) for batch in wave}
                        for job in as_completed(jobs):
                            new_nodes, new_edges = job.result()
                            created_nodes += new_nodes
                            created_edges += new_edges
                            bar.update(jobs[job])
        require(created_nodes == len(nodes)-len(present[0]) and created_edges == len(edges)-len(present[1]),
                "Neo4j creation count mismatch")
        verified, _ = session.execute_read(verify_neo4j, namespace, nodes, edges)
        outside = session.execute_read(outside_snapshot, namespace)
        require(outside == before, "Another namespace changed during import")
        first = dict(created_nodes=created_nodes, created_relationships=created_edges, verified=verified, outside=outside)
        repeated, reread = session.execute_read(lambda tx: verify_neo4j(tx, namespace, nodes, edges, complete=False))
        require(reread[0] == nodes.keys() and reread[1] == edges.keys() and verified == repeated,
                "Idempotence failure: missing or changed objects on second pass")
        second = dict(created_nodes=0, created_relationships=0, verified=repeated,
                      outside=session.execute_read(outside_snapshot, namespace))
        require(second["outside"] == before, "Other namespaces differ on repeat")
    return dict(status="PASS", first=first, repeated=second)
