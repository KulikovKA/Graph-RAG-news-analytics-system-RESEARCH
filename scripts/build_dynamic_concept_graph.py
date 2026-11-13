"""Stage 2G: additive concept experiment on exactly the frozen 2F.3 corpus.

--prepare-only never writes to Neo4j. --baseline-cache permits fully offline
preparation, but its full digest must equal the approved live 2F.3 snapshot.
Without --prepare-only, use existing NEO4J_* or --neo4j-env <existing-file>.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import csv
from datetime import datetime, timedelta, timezone
import hashlib
from importlib.metadata import version
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgraph.concepts import Concept, ConceptResolver, Entity, POLICY, VERSION, class_form, digest
import reconstruct_global_kg_2f3 as kg

base = kg.base
SOURCE_NS = "relex_sem_e5_small_name_t095_ner_t046_er2f3"
TARGET_NS = SOURCE_NS + "_concept_v1"
OUT = base.ER / "dynamic_concept_graph"
APPROVED = base.ER / "global_kg_2f3"
READ_NODES = "MATCH (n {namespace:$ns}) RETURN n.key AS key,labels(n) AS labels,properties(n) AS properties ORDER BY key"
READ_EDGES = "MATCH (h)-[r]->(t) WHERE r.namespace=$ns RETURN r.key AS key,type(r) AS type,h.key AS head,t.key AS tail,properties(r) AS properties ORDER BY key"
SCHEMA_QUERY = "SHOW CONSTRAINTS YIELD name,type,labelsOrTypes,properties RETURN name,type,labelsOrTypes,properties ORDER BY name"
INDEX_QUERY = "SHOW INDEXES YIELD name,type,entityType,labelsOrTypes,properties,owningConstraint RETURN name,type,entityType,labelsOrTypes,properties,owningConstraint ORDER BY name"
require, dumps, banner, progress = base.require, base.dumps, kg.banner, kg.progress


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                            encoding="utf-8", newline="\n")


def jsonl(name, rows):
    base.core.write_jsonl(OUT / name, rows)


def snapshot_hash(nodes, edges):
    h = hashlib.sha256()
    for k, r in sorted(nodes.items()):
        h.update((dumps(dict(key=k, labels=[r["label"]], properties=r["properties"])) + "\n").encode())
    for k, r in sorted(edges.items()):
        h.update((dumps(dict(key=k, **r)) + "\n").encode())
    return h.hexdigest()


def unpack_snapshot(raw):
    nodes, edges = {}, {}
    for r in raw["nodes"]:
        require(r["key"] not in nodes and len(r["labels"]) == 1, "Duplicate source node or labels")
        nodes[r["key"]] = dict(label=r["labels"][0], properties=r["properties"])
    for r in raw["relationships"]:
        require(r["key"] not in edges, "Duplicate source edge")
        edges[r["key"]] = {k: v for k, v in r.items() if k != "key"}
    return nodes, edges


def read_snapshot(session, namespace, totals):
    raw = {}
    for name, query, total in zip(("nodes", "relationships"), (READ_NODES, READ_EDGES), totals):
        result = session.run(query, ns=namespace)
        raw[name] = [r.data() for r in progress(result, total=total, desc="ЧТЕНИЕ " + name.upper())]
        require(not result.consume().counters.contains_updates, "Snapshot read performed updates")
    return unpack_snapshot(raw)


def validate_baseline(nodes, edges):
    expected = json.loads((APPROVED / "verification.json").read_text(encoding="utf-8"))["target_snapshot_after_second_import"]
    require(snapshot_hash(nodes, edges) == expected, "Full baseline hash differs from approved 2F.3")
    require(Counter(r["label"] for r in nodes.values()) == Counter(RELEXDocument=30, RELEXEntity=811, RELEXConcept=32), "Baseline node counts")
    require(Counter(r["type"] for r in edges.values()) == Counter(RELEX_REL=1307, SIMILAR_TO=492, HAS_SHARED_CONCEPT=118, RELEX_MENTIONS=811), "Baseline edge counts")
    require(all(r["head"] in nodes and r["tail"] in nodes for r in edges.values()), "Baseline foreign endpoint")
    return expected


def profiles(nodes, edges):
    """All 993 original source spans and contexts retained after approved merges."""
    docs = {r["properties"]["document_id"]: r["properties"] for r in nodes.values() if r["label"] == "RELEXDocument"}
    entities = []
    for row in progress(nodes.values(), total=len(nodes), desc="ПРОВЕРКА ИСХОДНЫХ КОНТЕКСТОВ"):
        if row["label"] != "RELEXEntity":
            continue
        p = row["properties"]; doc = docs[p["document_id"]]
        evidence = []; aliases = set(p["aliases"])
        decoded = json.loads(p["source_json"])
        for source in decoded["source_canonical_nodes"]:
            original = json.loads(source["properties"]["source_json"])
            for member in original["members"]:
                e, ctx = member["source_entity"], member["context"]
                require(e["document_id"] == ctx["document_id"] == doc["document_id"], "Context crosses documents")
                require(doc["text"][e["start"]:e["end"]] == e["surface_text"], "Changed source mention")
                require(doc["text"][ctx["start"]:ctx["end"]] == ctx["text"], "Changed source context")
                require(all(m["revision_id"] == doc["revision_id"] for m in e["source_mentions"]), "Source revision mismatch")
                aliases.update(e.get("normalized_forms", []))
                for mention in e["source_mentions"]:
                    for a in mention.get("abbreviation_candidates", []):
                        if class_form(e["surface_text"]) not in {class_form(a["long_form"]), class_form(a["short_form"])}:
                            continue
                        if (doc["text"][a["long_start"]:a["long_end"]] == a["long_form"] and
                            doc["text"][a["short_start"]:a["short_end"]] == a["short_form"]):
                            aliases.update((a["long_form"], a["short_form"]))
                evidence.append(dict(document_id=doc["document_id"], revision_id=doc["revision_id"],
                    source_url=doc["source_url"], source_canonical_id=original["canonical_entity_id"],
                    source_entity_id=e["entity_id"], source_mention_ids=[m["mention_id"] for m in e["source_mentions"]],
                    start=ctx["start"], end=ctx["end"], text=ctx["text"],
                    mention_start=e["start"], mention_end=e["end"], mention_text=e["surface_text"]))
        entities.append(Entity(p["entity_id"], p["document_id"], p["name"], tuple(p["entity_types"]), tuple(sorted(aliases)), tuple(evidence)))
    by = {e.entity_id: e for e in entities}
    links = {(nodes[r["head"]]["properties"]["entity_id"], nodes[r["tail"]]["properties"]["concept_id"])
             for r in edges.values() if r["type"] == "HAS_SHARED_CONCEPT"}
    members = defaultdict(set)
    for eid, cid in links:
        members[cid].add(eid)
    catalog = []
    for r in nodes.values():
        if r["label"] != "RELEXConcept":
            continue
        p = r["properties"]; cid = p["concept_id"]
        # Only explicit names/aliases of confirmed members; never NER categories.
        aliases = {a for eid in members[cid] for a in (by[eid].name, *by[eid].aliases)}
        catalog.append(Concept(cid, p["name"], tuple(sorted(aliases)), tuple(sorted(members[cid]))))
    similar = [(nodes[r["head"]]["properties"]["entity_id"], nodes[r["tail"]]["properties"]["entity_id"])
               for r in edges.values() if r["type"] == "SIMILAR_TO"]
    require(len(entities) == 811 and len(docs) == 30, "Frozen corpus mismatch")
    return entities, catalog, links, similar, docs


def copy_baseline(nodes, edges, fingerprint):
    def key(k):
        require(k.startswith(SOURCE_NS + ":"), "Non-source key")
        return TARGET_NS + k[len(SOURCE_NS):]
    def properties(p):
        return dict(p, key=key(p["key"]), namespace=TARGET_NS, fingerprint=fingerprint,
                    stage_2g_baseline_key=p["key"], stage_2g_baseline_fingerprint=p["fingerprint"])
    return ({key(k): dict(label=r["label"], properties=properties(r["properties"])) for k, r in nodes.items()},
            {key(k): dict(type=r["type"], head=key(r["head"]), tail=key(r["tail"]), properties=properties(r["properties"])) for k, r in edges.items()})


def extend(nodes, edges, resolution, fingerprint):
    key = lambda identity: TARGET_NS + ":" + identity
    for c in resolution.concepts:
        cid = c["concept_id"]
        require(key(cid) not in nodes, "New concept ID collision")
        evidence = [m["evidence"] for m in resolution.memberships if m["concept_id"] == cid]
        p = base.props(dict(c, source_provenance=evidence))
        p.update(key=key(cid), namespace=TARGET_NS, fingerprint=fingerprint, entity_types=["CONCEPT"],
                 source_namespace=SOURCE_NS, concept_stage="2G")
        nodes[key(cid)] = dict(label="RELEXConcept", properties=p)
    for m in resolution.memberships:
        rid = m["membership_id"]
        require(key(rid) not in edges, "New membership ID collision")
        p = base.props(m)
        p.update(key=key(rid), namespace=TARGET_NS, fingerprint=fingerprint, source_namespace=SOURCE_NS,
                 head_id=m["entity_id"], tail_id=m["concept_id"], relation_type="HAS_SHARED_CONCEPT")
        edges[key(rid)] = dict(type="HAS_SHARED_CONCEPT", head=key(m["entity_id"]), tail=key(m["concept_id"]), properties=p)


def validate_extension(source_nodes, source_edges, nodes, edges):
    require(TARGET_NS != SOURCE_NS, "Source overwrite forbidden")
    for source, target in ((source_nodes, nodes), (source_edges, edges)):
        for k, row in source.items():
            target_key = TARGET_NS + k[len(SOURCE_NS):]
            require(target_key in target, "Missing baseline record")
            actual = target[target_key]
            for field in ("label", "type", "head", "tail"):
                if field in row:
                    want = TARGET_NS + row[field][len(SOURCE_NS):] if field in ("head", "tail") else row[field]
                    require(actual[field] == want, "Baseline labels/type/endpoints changed")
            for name, value in row["properties"].items():
                if name not in ("key", "namespace", "fingerprint"):
                    require(actual["properties"][name] == value, "Baseline property/provenance changed: " + name)
    require({k for k, r in nodes.items() if r["label"] == "RELEXDocument"} ==
            {TARGET_NS + k[len(SOURCE_NS):] for k, r in source_nodes.items() if r["label"] == "RELEXDocument"}, "Corpus changed")
    require(sum(r["label"] == "RELEXEntity" for r in nodes.values()) == 811, "Entity set changed")
    concept_names = [class_form(r["properties"]["name"]) for r in nodes.values() if r["label"] == "RELEXConcept"]
    require(len(concept_names) == len(set(concept_names)), "Duplicate concept names")
    memberships = set()
    for k, r in edges.items():
        require(r["head"] in nodes and r["tail"] in nodes, "Foreign namespace endpoint")
        require(k.startswith(TARGET_NS + ":") and r["properties"]["namespace"] == TARGET_NS, "Foreign edge ownership")
        if r["type"] == "RELEX_REL":
            require(nodes[r["head"]]["label"] == nodes[r["tail"]]["label"] == "RELEXEntity", "Concept inherited fact")
            require(nodes[r["head"]]["properties"]["document_id"] == nodes[r["tail"]]["properties"]["document_id"], "Cross-document fact")
        if r["type"] == "HAS_SHARED_CONCEPT":
            require(nodes[r["head"]]["label"] == "RELEXEntity" and nodes[r["tail"]]["label"] == "RELEXConcept", "Invalid concept endpoints")
            require((r["head"], r["tail"]) not in memberships, "Duplicate concept membership")
            memberships.add((r["head"], r["tail"]))
            require(all(r["properties"].get(f) is False for f in ("factual", "identity", "fact_inheritance_allowed")), "Unsafe concept semantics")
    for kind, count in (("RELEX_REL", 1307), ("SIMILAR_TO", 492), ("RELEX_MENTIONS", 811)):
        require(sum(r["type"] == kind for r in edges.values()) == count, "Fixed layer changed")
    return dict(baseline_records_exact_modulo_namespace_metadata=True, documents_unchanged=True,
                entity_mapping_unchanged=True, all_facts_and_provenance_preserved=True,
                all_similar_to_preserved=True, original_concepts_and_memberships_preserved=True,
                duplicate_concepts=0, duplicate_memberships=0, cross_namespace_relationships=0)


def counts(nodes, edges):
    n, e = Counter(r["label"] for r in nodes.values()), Counter(r["type"] for r in edges.values())
    return dict(documents=n["RELEXDocument"], entities=n["RELEXEntity"], concepts=n["RELEXConcept"],
                kg_vertices=n["RELEXEntity"] + n["RELEXConcept"], total_nodes=len(nodes),
                factual_relations=e["RELEX_REL"], semantic_links=e["SIMILAR_TO"], concept_links=e["HAS_SHARED_CONCEPT"],
                document_mentions=e["RELEX_MENTIONS"], kg_edges=len(edges) - e["RELEX_MENTIONS"], total_relationships=len(edges))


def concept_connections(nodes, edges):
    members = defaultdict(list)
    for k, r in edges.items():
        if r["type"] == "HAS_SHARED_CONCEPT":
            members[r["tail"]].append((r["head"], k))
    doc_pairs, bridges = set(), set()
    for c, rows in sorted(members.items()):
        docs = sorted({nodes[e]["properties"]["document_id"] for e, _ in rows})
        for a, b in combinations(docs, 2):
            doc_pairs.add((a, b)); bridges.add((c.split(":concept:", 1)[-1], a, b))
    return members, doc_pairs, bridges


def factual_route(seed, nodes, edges):
    """Deterministic BFS over arbitrary-length undirected intra-document facts.

    Return a maximal-distance shortest route from the bridge entity. Retain each
    original direction/predicate; traversal direction is explicitly recorded.
    """
    adjacency = defaultdict(list)
    for k, r in edges.items():
        if r["type"] == "RELEX_REL" and r["head"] != r["tail"]:
            adjacency[r["head"]].append((r["tail"], k))
            adjacency[r["tail"]].append((r["head"], k))
    queue = deque([seed]); paths = {seed: []}
    while queue:
        current = queue.popleft()
        for other, key in sorted(adjacency[current]):
            if other not in paths:
                paths[other] = paths[current] + [dict(edge_key=key, from_key=current, to_key=other,
                    head=edges[key]["head"], tail=edges[key]["tail"],
                    head_name=nodes[edges[key]["head"]]["properties"].get("name", edges[key]["head"]),
                    tail_name=nodes[edges[key]["tail"]]["properties"].get("name", edges[key]["tail"]),
                    predicate=edges[key]["properties"]["relation_type"],
                    document_id=edges[key]["properties"]["document_id"],
                    source_triple_ids=edges[key]["properties"]["source_triple_ids"])]
                queue.append(other)
    return min(paths.values(), key=lambda p: (-len(p), dumps(p)))


def reachable_document_pairs(nodes, edges):
    """Document pairs sharing any KG component, excluding service mentions."""
    adjacency = {k: set() for k, r in nodes.items() if r["label"] != "RELEXDocument"}
    for r in edges.values():
        if r["type"] != "RELEX_MENTIONS":
            adjacency[r["head"]].add(r["tail"]); adjacency[r["tail"]].add(r["head"])
    seen, pairs = set(), set()
    for seed in sorted(adjacency):
        if seed in seen:
            continue
        todo = [seed]; seen.add(seed); docs = set()
        while todo:
            current = todo.pop()
            doc = nodes[current]["properties"].get("document_id")
            if doc:
                docs.add(doc)
            for other in adjacency[current] - seen:
                seen.add(other); todo.append(other)
        pairs.update(combinations(sorted(docs), 2))
    return pairs


def compare(source_nodes, source_edges, nodes, edges, resolution):
    old, old_pairs, old_bridges = concept_connections(source_nodes, source_edges)
    new, new_pairs, new_bridges = concept_connections(nodes, edges)
    examples = []; pair_counts = Counter(); new_ids = {m["membership_id"] for m in resolution.memberships}
    # Exhaustive entity--concept--entity bridge count; not an unbounded count
    # of arbitrary walks (cycles would make that quantity infinite).
    bridge_counts = []
    for graph_nodes, members in ((source_nodes, old), (nodes, new)):
        total = 0
        for c, rows in members.items():
            docs = Counter(graph_nodes[e]["properties"]["document_id"] for e, _ in rows)
            total += sum(docs[a] * docs[b] for a, b in combinations(sorted(docs), 2))
        bridge_counts.append(total)
    for c, rows in sorted(new.items()):
        for (a, al), (b, bl) in combinations(sorted(rows), 2):
            da, db = nodes[a]["properties"]["document_id"], nodes[b]["properties"]["document_id"]
            if da == db or not any(edges[k]["properties"].get("membership_id") in new_ids for k in (al, bl)):
                continue
            pair_counts[tuple(sorted((da, db)))] += 1
            left, right = factual_route(a, nodes, edges), factual_route(b, nodes, edges)
            if left and right and len(examples) < 10:
                examples.append(dict(document_a=da, document_b=db, concept_id=nodes[c]["properties"]["concept_id"],
                    document_a_title=nodes[TARGET_NS + ":" + da]["properties"]["title"],
                    document_b_title=nodes[TARGET_NS + ":" + db]["properties"]["title"],
                    concept_name=nodes[c]["properties"]["name"], bridge_entity_a=a, bridge_entity_b=b,
                    membership_a=al, membership_b=bl, factual_route_a=left, factual_route_b=right,
                    semantic_only=True, new_factual_dependency=False))
    # Prefer distinct document pairs among the saved examples.
    examples.sort(key=lambda p: (p["document_a"], p["document_b"], p["concept_id"]))
    hubs = []
    for c, rows in sorted(new.items()):
        ds = sorted({nodes[e]["properties"]["document_id"] for e, _ in rows})
        hubs.append(dict(concept_id=nodes[c]["properties"]["concept_id"], name=nodes[c]["properties"]["name"],
                         entities=len(rows), document_ids=ds, documents=len(ds),
                         added_memberships=sum(edges[k]["properties"].get("membership_id") in new_ids for _, k in rows)))
    result = dict(baseline_counts=counts(source_nodes, source_edges), target_counts=counts(nodes, edges),
        baseline_connectivity=kg.metrics(source_nodes, source_edges)[0], target_connectivity=kg.metrics(nodes, edges)[0],
        baseline_multidocument_concepts=sum(len({source_nodes[e]["properties"]["document_id"] for e, _ in rows}) > 1 for rows in old.values()),
        target_multidocument_concepts=sum(h["documents"] > 1 for h in hubs),
        new_concepts=len(resolution.concepts), new_memberships=len(resolution.memberships),
        newly_directly_concept_connected_document_pairs=sorted(new_pairs - old_pairs),
        newly_reachable_document_pairs=sorted(reachable_document_pairs(nodes, edges) - reachable_document_pairs(source_nodes, source_edges)),
        new_document_pair_concept_bridges=len(new_bridges - old_bridges),
        entity_concept_entity_paths_before=bridge_counts[0], entity_concept_entity_paths_after=bridge_counts[1],
        new_entity_concept_entity_paths=bridge_counts[1] - bridge_counts[0],
        changed_document_pairs=[dict(documents=list(p), added_paths=n) for p, n in sorted(pair_counts.items())],
        decisions=dict(Counter(d["status"] for d in resolution.decisions)),
        connectivity_is_not_quality_evidence=True)
    return result, examples, hubs


def inventory(nodes, edges):
    return dict(snapshot_sha256=snapshot_hash(nodes, edges),
                nodes={k: digest(r) for k, r in sorted(nodes.items())},
                relationships={k: digest(r) for k, r in sorted(edges.items())})


def configure_importer(expected):
    # Reuse Stage 2F.3's atomic importer, strict read-back, fingerprint guard and
    # no-op idempotence. Only its namespace, output and expected counts change.
    kg.TARGET_NS, kg.OUT, kg.EXPECTED = TARGET_NS, OUT, expected
    base.NS = TARGET_NS
    kg.write_queries()
    stats = OUT / "graph_stats.cypher"
    stats.write_text(stats.read_text(encoding="utf-8").replace("Stage 2F.3", "Stage 2G (reused 2F.3 checks)"), encoding="utf-8", newline="\n")
    text = (OUT / "show_full_graph.cypher").read_text(encoding="utf-8")
    text = text.replace("873", str(expected["total_nodes"])).replace("2728", str(expected["total_relationships"]))
    (OUT / "show_full_graph.cypher").write_text(text, encoding="utf-8", newline="\n")
    (OUT / "show_new_memberships.cypher").write_text(f"""// Only newly accepted memberships; no implied factual relationship.
MATCH (e:RELEXEntity {{namespace:'{TARGET_NS}'}})-[r:HAS_SHARED_CONCEPT {{namespace:'{TARGET_NS}'}}]->(c:RELEXConcept {{namespace:'{TARGET_NS}'}})
WHERE r.resolver_version='{VERSION}'
RETURN e,r,c;
""", encoding="utf-8", newline="\n")
    (OUT / "new_concept_bridges.cypher").write_text(f"""// Semantic entity--concept--entity bridges involving a new membership.
MATCH (a:RELEXEntity {{namespace:'{TARGET_NS}'}})-[r1:HAS_SHARED_CONCEPT {{namespace:'{TARGET_NS}'}}]->(c:RELEXConcept {{namespace:'{TARGET_NS}'}})<-[r2:HAS_SHARED_CONCEPT {{namespace:'{TARGET_NS}'}}]-(b:RELEXEntity {{namespace:'{TARGET_NS}'}})
WHERE a.document_id<b.document_id AND
 (coalesce(r1.resolver_version,'')='{VERSION}' OR coalesce(r2.resolver_version,'')='{VERSION}')
RETURN a.document_id AS document_a,b.document_id AS document_b,a,r1,c,r2,b;
""", encoding="utf-8", newline="\n")


def live_checks(session, comparison):
    cross = session.run("MATCH (a)-[r]->(b) WHERE (a.namespace=$ns OR b.namespace=$ns OR r.namespace=$ns) "
        "AND (coalesce(a.namespace,'')<>$ns OR coalesce(b.namespace,'')<>$ns OR coalesce(r.namespace,'')<>$ns) RETURN count(r) AS n", ns=TARGET_NS).single()["n"]
    require(cross == 0, "Cross-namespace relationships detected")
    # Stage 2F.3 Cypher WCC is independent of its Python connectivity helper.
    wcc = session.run(kg.statements()[4]).single().data()
    bridges = session.run((OUT / "new_concept_bridges.cypher").read_text(encoding="utf-8")).data()
    require(len(bridges) == comparison["new_entity_concept_entity_paths"], "Independent concept path count mismatch")
    return dict(cross_namespace_relationships=cross, independent_cypher_wcc=wcc,
                independent_cypher_new_concept_paths=len(bridges))


def write_report(result, paths, hubs, imported):
    rows = [("Документы", "documents"), ("Контекстные сущности", "entities"), ("Концепты", "concepts"),
            ("RELEX_REL", "factual_relations"), ("SIMILAR_TO", "semantic_links"),
            ("HAS_SHARED_CONCEPT", "concept_links"), ("RELEX_MENTIONS", "document_mentions")]
    lines = ["# Stage 2G — Dynamic Global Concept Linking", "",
             "Новый namespace импортирован и проверен двумя проходами." if imported else "Локальная подготовка проверена; импорт ещё не подтверждён.", "",
             f"Baseline: `{SOURCE_NS}`. Target: `{TARGET_NS}`.", "",
             "Корпус строго тот же: 30 документов Stage 2F.3. NER/RE, обучение и расширение корпуса не запускались.", "",
             "| Метрика | Stage 2F.3 | Новый граф |", "|---|---:|---:|"]
    lines += [f"| {title} | {result['baseline_counts'][k]} | {result['target_counts'][k]} |" for title, k in rows]
    lines += [f"| Концепты в нескольких документах | {result['baseline_multidocument_concepts']} | {result['target_multidocument_concepts']} |"]
    lines += [f"| {title} | {result['baseline_connectivity'][k]} | {result['target_connectivity'][k]} |"
              for title, k in (("Weakly connected components", "weak_components"), ("Изолированные KG-вершины", "isolated"), ("Крупнейшая компонента", "largest_component_vertices"))]
    lines += ["", f"Добавлено концептов: **{result['new_concepts']}**, memberships: **{result['new_memberships']}**.",
              f"Решения: `{dumps(result['decisions'])}`. По уточнению пользователя только ACCEPT/REJECT; недостаточные доказательства дают REJECT без ребра.", "",
              "## Правила и границы доказательства", "",
              "Кандидаты: исходные имена, нормализация NFC, ограниченная регулярная морфология, явно определённые в исходном тексте аббревиатуры, каталог 32 концептов, пересечения токенов имён и все 492 SIMILAR_TO. Новые модели и embeddings не вычисляются. Совпадение темы, имени, типа или cosine само по себе недостаточно.", "",
              "ACCEPT требует явно названного класса/алиаса, проверенных source spans, контекстных признаков общего смысла и совместимой абстракции. Словари смысловых признаков — явные эвристики, а не независимая научная верификация. Новые концепты требуют поддержанных memberships минимум в двух документах. RESULT, общие Study/Results и неоднозначные категории отклоняются. Рёбра не переносят факты: factual=false, identity=false, fact_inheritance_allowed=false.", "",
              "## Междокументные пути", "",
              f"Пар entity→concept←entity было {result['entity_concept_entity_paths_before']}, стало {result['entity_concept_entity_paths_after']}; добавлено {result['new_entity_concept_entity_paths']}. Это конечный подсчёт простых мостов, не всех возможных маршрутов/циклических обходов. Новых комбинаций (concept, document A, document B): {result['new_document_pair_concept_bridges']}.", "",
              "Пары, впервые получившие прямой мост через общий концепт (могли быть связаны косвенно и раньше):"]
    lines += [f"- `{a}` ↔ `{b}`" for a, b in result["newly_directly_concept_connected_document_pairs"]] or ["- Нет."]
    lines += ["", "Полные документы, source URLs, revision IDs, offsets и текстовые evidence: `document_lock.json`, `candidate_decisions.jsonl`, `memberships.jsonl`. Все новые пути с выбранными фактическими маршрутами: `path_examples.jsonl`.", ""]
    lines += [f"Новых пар документов, ранее вообще недостижимых через KG, — {len(result['newly_reachable_document_pairs'])}. Прямой новый concept-мост и первая достижимость в полном KG — разные метрики.", ""]
    for i, p in enumerate(paths[:5], 1):
        lines += [f"### Пример {i}: {p['document_a']} ↔ {p['document_b']} через {p['concept_name']}", "",
                  f"A: {p['document_a_title']}. B: {p['document_b_title']}.", "",
                  f"`{p['bridge_entity_a']}` → HAS_SHARED_CONCEPT → `{p['concept_id']}` ← HAS_SHARED_CONCEPT ← `{p['bridge_entity_b']}`.", ""]
        for side in ("a", "b"):
            lines.append(f"Фактический маршрут в `{p['document_' + side]}` (длина {len(p['factual_route_' + side])}; обход может идти против исходного направления):")
            lines.append("")
            for e in p["factual_route_" + side]:
                lines.append(f"- {e['head_name']} —{e['predicate']}→ {e['tail_name']}; IDs `{e['head'].split(TARGET_NS + ':')[-1]}` → `{e['tail'].split(TARGET_NS + ':')[-1]}`; edge `{e['edge_key'].split(TARGET_NS + ':')[-1]}`; raw triples `{', '.join(e['source_triple_ids'])}`.")
            lines.append("")
        lines += ["Это семантический маршрут между исследованиями; новая фактическая зависимость не установлена.", ""]
    lines += ["## Ошибки и hubs", "",
              "Потенциальные ошибки: регулярное приведение числа, многозначность имён, грубые словари смыслов, шум исходного NER/RE. ACCEPT — воспроизводимое решение алгоритма, не gold precision. Непроверенные варианты остаются в REJECT с evidence; исходные 118 memberships сохраняются независимо от новой политики.", "",
              "Крупнейшие концепты (полный аудит: `concept_audit.json`):", "",
              "| Концепт | Сущности | Документы | Новые memberships |", "|---|---:|---:|---:|"]
    lines += [f"| {h['name']} | {h['entities']} | {h['documents']} | {h['added_memberships']} |" for h in sorted(hubs, key=lambda h: (-h["documents"], -h["entities"], h["concept_id"]))[:10]]
    lines += ["", "Роста связности недостаточно для вывода об улучшении качества KG. Широкие Patients/Genes сохраняют только категориальную общность; по ним нельзя наследовать состав когорт, измерения или результаты. Новый общий Study/Results не создаётся.", "",
              "## Сохранность и повторный запуск", "",
              "Каждый исходный ID, label/type, endpoint и property проверен точно; только namespace/key/fingerprint заменены для отдельной копии. Исходный fingerprint/key сохранены дополнительно. Полный SHA-256 baseline сверяется с verification Stage 2F.3. Manifest фиксирует исходные файлы, корпус, код, параметры и hashes. Снимки инвентаря до/после включают per-record hashes. Importer, atomic transaction, fingerprint guard, exact read-back и no-op повторного запуска взяты из Stage 2F.3. Глобальные constraints не создаются и не меняются.", "",
              "Воспроизведение: `.venv/Scripts/python.exe scripts/build_dynamic_concept_graph.py --neo4j-env <existing-file>`; подготовка без записи: добавить `--prepare-only`. Offline: `--prepare-only --baseline-cache <verified-snapshot.json>`.", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--baseline-cache", type=Path)
    parser.add_argument("--neo4j-env", type=Path)
    args = parser.parse_args()
    require(not args.baseline_cache or args.prepare_only, "Cached baseline allowed only for offline preparation")
    OUT.mkdir(parents=True, exist_ok=True)
    banner("ФИКСИРУЮ НЕИЗМЕНЯЕМЫЕ АРТЕФАКТЫ И КОРПУС STAGE 2F.3")
    protected = sorted(p for p in base.ER.rglob("*") if p.is_file() and not p.is_relative_to(OUT))
    protected.append(base.DOCUMENTS)
    protected_hashes = {str(p.relative_to(ROOT).as_posix()): base.sha(p) for p in progress(protected, desc="HASH ИСХОДНЫХ ФАЙЛОВ")}
    driver = None
    try:
        if args.baseline_cache:
            source_nodes, source_edges = unpack_snapshot(json.loads(args.baseline_cache.read_text(encoding="utf-8")))
        else:
            driver, database = base.core.connect_neo4j(args.neo4j_env)
            with driver.session(database=database) as session:
                banner("ЧИТАЮ ПОЛНЫЙ ИСХОДНЫЙ СНИМОК 2F.3 — БЕЗ ЗАПИСИ")
                source_nodes, source_edges = read_snapshot(session, SOURCE_NS, (873, 2728))
        baseline_digest = validate_baseline(source_nodes, source_edges)
        save("baseline_snapshot_before.json", inventory(source_nodes, source_edges))
        banner("ПРОВЕРЯЮ 811 СУЩНОСТЕЙ И ИСХОДНЫЕ КОНТЕКСТЫ; NER/RE НЕ ЗАПУСКАЮТСЯ")
        entities, catalog, links, similar, docs = profiles(source_nodes, source_edges)
        doc_lock = [dict(document_id=d, revision_id=p["revision_id"], text_sha256=p["text_sha256"],
                         title=p["title"], source_url=p["source_url"]) for d, p in sorted(docs.items())]
        save("document_lock.json", doc_lock)
        banner("ВЫПОЛНЯЮ DYNAMIC CONCEPT RESOLUTION НА ТОМ ЖЕ КОРПУСЕ")
        resolver = ConceptResolver(catalog, entities, links)
        resolution = resolver.resolve_batch(entities, similar, progress=lambda rows: progress(rows, total=811, desc="CONCEPT RESOLUTION"))
        code_files = [Path(__file__), ROOT / "src/cgraph/concepts.py", ROOT / "scripts/reconstruct_global_kg_2f3.py", ROOT / "scripts/import_global_semantic_graph.py", ROOT / "scripts/build_relex_graphs.py"]
        code_hashes = {p.relative_to(ROOT).as_posix(): base.sha(p) for p in code_files}
        signature = dict(stage="2G", namespace=TARGET_NS, source_namespace=SOURCE_NS,
                         baseline_sha256=baseline_digest, document_lock_sha256=digest(doc_lock),
                         policy=POLICY, code_sha256=code_hashes,
                         decisions_sha256=digest(resolution.decisions),
                         new_concepts_sha256=digest(resolution.concepts), memberships_sha256=digest(resolution.memberships))
        fingerprint = digest(signature)
        nodes, edges = copy_baseline(source_nodes, source_edges, fingerprint)
        extend(nodes, edges, resolution, fingerprint)
        validation = validate_extension(source_nodes, source_edges, nodes, edges)
        # Reordering the entire corpus must not change proposals or decisions.
        repeated = resolver.resolve_batch(reversed(entities), reversed(similar))
        require(resolution == repeated, "Non-deterministic resolution")
        validation["resolution_repeat_and_input_order_invariant"] = True
        jsonl("candidate_decisions.jsonl", resolution.decisions)
        for status in ("ACCEPT", "REJECT"):
            fields = ["status", "entity_id", "entity_name", "document_id", "concept_id", "concept_name", "reason", "decision_id"]
            with (OUT / (status.lower() + ".csv")).open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
                writer.writeheader()
                writer.writerows(r for r in resolution.decisions if r["status"] == status)
        jsonl("concept_nodes.jsonl", resolution.concepts)
        jsonl("memberships.jsonl", resolution.memberships)
        result, examples, hubs = compare(source_nodes, source_edges, nodes, edges, resolution)
        save("comparison.json", result); save("concept_audit.json", hubs)
        jsonl("path_examples.jsonl", examples)
        save("validation.json", validation)
        save("target_snapshot_prepared.json", inventory(nodes, edges))
        configure_importer(counts(nodes, edges))
        manifest = dict(**signature, fingerprint=fingerprint, source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                        prepared_at=datetime.now(timezone(timedelta(hours=3))).isoformat(), timezone="Europe/Moscow",
                        protected_files_sha256=protected_hashes, corpus_expansion=False, ner_re_inference=False,
                        models_loaded=False, runtime={p: version(p) for p in ("neo4j", "tqdm")},
                        comparison=result, import_completed=False)
        save("manifest.json", manifest)
        write_report(result, examples, hubs, False)
        require(all(base.sha(ROOT / p) == h for p, h in protected_hashes.items()), "Protected file changed during preparation")
        if args.prepare_only:
            banner("ЛОКАЛЬНАЯ ПОДГОТОВКА И ВАЛИДАЦИЯ ЗАВЕРШЕНЫ; ЗАПИСЕЙ В NEO4J НЕТ")
            print(dumps(result), flush=True)
            return
        with driver.session(database=database) as session:
            banner("СНИМОК ВСЕХ ЧУЖИХ NAMESPACES И СХЕМЫ ПЕРЕД ИМПОРТОМ — ПРОЦЕНТ НЕ ОПРЕДЕЛЁН")
            outside_before = base.outside_snapshot(session)
            schema_before = dict(constraints=session.run(SCHEMA_QUERY).data(), indexes=session.run(INDEX_QUERY).data())
            save("outside_snapshot_before.json", dict(snapshot=outside_before, schema=schema_before))
            # Refresh the source immediately before allowing the first write.
            fresh_nodes, fresh_edges = read_snapshot(session, SOURCE_NS, (873, 2728))
            validate_baseline(fresh_nodes, fresh_edges)
            banner("НАЧИНАЮ ИМПОРТ ТОЛЬКО В НОВЫЙ NAMESPACE " + TARGET_NS.upper())
            first = kg.import_once(session, nodes, edges, fingerprint, 1)
            calculated, first_hash, live_nodes, live_edges = kg.verify(session, nodes, edges)
            validate_extension(source_nodes, source_edges, live_nodes, live_edges)
            live = live_checks(session, result)
            require(live["independent_cypher_wcc"] == {k: calculated[k] for k in live["independent_cypher_wcc"]}, "Independent WCC mismatch")
            save("target_snapshot_after_first_import.json", inventory(live_nodes, live_edges))
            second = kg.import_once(session, nodes, edges, fingerprint, 2)
            _, second_hash, live_nodes, live_edges = kg.verify(session, nodes, edges)
            require(first_hash == second_hash == snapshot_hash(nodes, edges), "Idempotence hash mismatch")
            require(second == dict(nodes=0, relationships=0), "Repeated import created records")
            try:
                session.execute_read(lambda tx: kg.assert_namespace(tx, "invalid-fingerprint-control"))
            except ValueError as exc:
                require("another snapshot" in str(exc), "Unexpected fingerprint guard failure")
            else:
                raise ValueError("Foreign fingerprint was not refused")
            after_nodes, after_edges = read_snapshot(session, SOURCE_NS, (873, 2728))
            require(validate_baseline(after_nodes, after_edges) == baseline_digest, "Source namespace modified")
            save("baseline_snapshot_after.json", inventory(after_nodes, after_edges))
            save("target_snapshot_after_second_import.json", inventory(live_nodes, live_edges))
            banner("ПРОВЕРЯЮ ПОЛНЫЙ ВНЕШНИЙ СНИМОК И СХЕМУ ПОСЛЕ ИМПОРТА — ПРОЦЕНТ НЕ ОПРЕДЕЛЁН")
            outside_after = base.outside_snapshot(session)
            schema_after = dict(constraints=session.run(SCHEMA_QUERY).data(), indexes=session.run(INDEX_QUERY).data())
            require(outside_before == outside_after and schema_before == schema_after, "External namespace or schema changed")
            require(all(base.sha(ROOT / p) == h for p, h in protected_hashes.items()), "Prior artifact modified")
            save("outside_snapshot_after.json", dict(snapshot=outside_after, schema=schema_after))
            verification = dict(first_import=first, repeated_import=second, target_sha256=second_hash,
                baseline_sha256_before=baseline_digest, baseline_sha256_after=validate_baseline(after_nodes, after_edges),
                exact_target_readback=True, target_snapshots_identical=True, outside_snapshots_identical=True,
                constraints_and_indexes_unchanged=True, invalid_fingerprint_refused=True,
                prior_files_unchanged=True, **{**live, **validation})
            # Both checks report the same zero; keep a single serialized field.
            save("verification.json", verification)
            manifest.update(import_completed=True, database=database, verification_sha256=digest(verification))
            save("manifest.json", manifest)
            write_report(result, examples, hubs, True)
            banner("STAGE 2G ЗАВЕРШЁН: ИМПОРТ, СОХРАННОСТЬ И ИДЕМПОТЕНТНОСТЬ ПОДТВЕРЖДЕНЫ")
    finally:
        if driver is not None:
            driver.close()


if __name__ == "__main__":
    main()
