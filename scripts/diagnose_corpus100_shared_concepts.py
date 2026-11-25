"""Read-only graph audit and cached replay for Stage 3A.5.11."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
import csv
import gzip
from itertools import combinations
import json
import os
from pathlib import Path
from statistics import mean, median
from unittest.mock import patch

import numpy as np
from neo4j import READ_ACCESS
from tqdm import tqdm

import build_corpus_graph as runner
import build_dynamic_concept_graph as historical
import corpus_graph_payload as current
from cgraph import concepts
from cgraph.semantic_concepts import HybridConceptResolver, passes

ROOT = runner.ROOT
WORK = runner.DEFAULT_OUTPUT
ER = runner.ER_HISTORY
HISTORY = ER / "semantic_concept_sweep_extended"
CACHE = ROOT / ".cache/semantic_concept_sweep"
OUT = ROOT / "artifacts/shared_concepts_diagnostics_3a5_11"
NS30 = historical.SOURCE_NS
NS30C = "relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v2_sem_name_t0830"
NS100 = runner.NAMESPACE
LIMIT = 10_000
THRESHOLD = "0.830"
VIEW = """WITH $ns AS ns
MATCH (n {namespace:ns})
OPTIONAL MATCH (n)-[r]->(m {namespace:ns})
WHERE r.namespace=ns
RETURN n.key AS a, labels(n)[0] AS al, r.key AS key, type(r) AS kind, m.key AS b, labels(m)[0] AS bl"""


def load_rows(path):
    source = gzip.open if path.suffix == ".gz" else open
    with source(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def checkpoint(path):
    row = runner.read(path)
    checksum = row.pop("checksum")
    assert runner.signature(row) == checksum and row["status"] == "complete", path
    return row


def save_intermediate(path,value):
    if path.exists():
        assert runner.read(path)==json.loads(json.dumps(value,allow_nan=False)), path
    else:
        runner.atomic_json(path,value)


def graph_metrics(nodes, edges):
    import networkx as nx
    from scipy.sparse.csgraph import shortest_path

    by = {k: row["properties"].get("document_id") for k, row in nodes.items() if row["label"] == "RELEXEntity"}
    kg = sorted(k for k, row in nodes.items() if row["label"] != "RELEXDocument")
    all_edges = [(r["head"], r["tail"]) for r in edges.values() if r["type"] != "RELEX_MENTIONS"]
    simple = nx.Graph()
    simple.add_nodes_from(kg)
    simple.add_edges_from(sorted({tuple(sorted(pair)) for pair in all_edges if pair[0] != pair[1]}))
    components = list(nx.connected_components(simple))
    largest = simple.subgraph(max(components, key=len))
    paths = shortest_path(nx.to_scipy_sparse_array(largest, format="csr"), directed=False, unweighted=True)
    incident = {key for pair in all_edges for key in pair}
    communities = nx.community.louvain_communities(simple, seed=42, resolution=1)
    return dict(entities=len(by), kg_vertices=len(kg), kg_edges=len(all_edges), simple_pairs=simple.number_of_edges(),
                density=nx.density(simple), average_degree=2*len(all_edges)/len(kg), components=len(components),
                largest=len(largest), largest_share=len(largest)/len(kg), isolates=len(by.keys()-incident),
                isolate_share=len(by.keys()-incident)/len(by),
                cross_document=sum(a in by and b in by and by[a] != by[b] for a, b in all_edges),
                mean_path=float(paths.sum()/(len(largest)*(len(largest)-1))),
                modularity=nx.community.modularity(simple, communities, resolution=1),
                loops=sum(a == b for a, b in all_edges))


def inventory(nodes, edges, shown_edges=()):
    members = defaultdict(list)
    for key, row in edges.items():
        if row["type"] != "HAS_SHARED_CONCEPT":
            continue
        entity = nodes[row["head"]]["properties"]
        members[row["tail"]].append(dict(entity_id=entity["entity_id"], name=entity["name"],
            doc=entity["document_id"], types=entity["entity_types"], shown=key in shown_edges,
            review=row["properties"].get("requires_review", False)))
    output = []
    for key, row in nodes.items():
        if row["label"] == "RELEXConcept":
            p = row["properties"]
            output.append(dict(id=p["concept_id"], name=p["name"], members=sorted(
                members[key], key=lambda x:(x["doc"], x["name"], x["entity_id"]))))
    sizes = [len(r["members"]) for r in output]
    doc_counts = [len({m["doc"] for m in r["members"]}) for r in output]
    entities = {m["entity_id"] for r in output for m in r["members"]}
    types = Counter(code for r in output for code in {t for m in r["members"] for t in m["types"]})
    return dict(concepts=sorted(output,key=lambda r:r["name"].casefold()), entities=len(entities),
                entity_share=len(entities)/sum(r["label"]=="RELEXEntity" for r in nodes.values()),
                documents=len({m["doc"] for r in output for m in r["members"]}),
                multidoc=sum(n>=2 for n in doc_counts), document_sizes=dict(Counter(doc_counts)),
                sizes=dict(Counter(sizes)), minimum=min(sizes), maximum=max(sizes), mean=mean(sizes), median=median(sizes),
                types=types)


def frozen_inputs():
    docs = {d["document_id"]:d for d in runner.read_documents(runner.CORPUS)}
    result = runner.read(WORK / "results.json")
    read = lambda stage: checkpoint(WORK/"checkpoints"/stage/"corpus.json")
    resolved, semantic, shared = (read(name) for name in ("entity_resolution","semantic","concepts"))
    assert all(r["config_fingerprint"] == result["config_fingerprint"] for r in (resolved,semantic,shared))
    resolution, links, concepts_result = (r["results"] for r in (resolved,semantic,shared))
    assert runner.signature([result["config_fingerprint"],resolution,links,concepts_result])==result["graph_fingerprint"]
    old_docs = runner.relex.read_jsonl(runner.relex.SOURCE/"input_abstracts.jsonl")
    overlap = {key:len({r[key] for r in old_docs if r.get(key)} & {r[key] for r in docs.values() if r.get(key)})
               for key in ("document_id","doi","revision_id","text_sha256")}
    return docs, result, resolution, links, concepts_result, overlap


def snapshot(session, namespace):
    nodes, edges = {}, {}
    for row in session.run(current.READ_NODES, ns=namespace):
        assert row["key"] not in nodes and len(row["labels"]) == 1
        nodes[row["key"]] = dict(label=row["labels"][0],properties=row["properties"])
    for row in session.run(current.READ_EDGES, ns=namespace):
        assert row["key"] not in edges and row["head"] in nodes and row["tail"] in nodes
        edges[row["key"]] = {key:row[key] for key in ("type","head","tail","properties")}
    return nodes,edges


def graph_audit(session, inputs):
    docs, results, resolved, links, shared, overlap = inputs
    nodes, edges = snapshot(session,NS100)
    expected_nodes, expected_edges = current.build_payload(NS100,results["graph_fingerprint"],docs,resolved,links,shared)
    assert nodes==expected_nodes and edges==expected_edges
    cross = session.run("""MATCH (a)-[r]->(b)
       WHERE (a.namespace=$ns OR b.namespace=$ns OR r.namespace=$ns)
         AND (coalesce(a.namespace,'')<>$ns OR coalesce(b.namespace,'')<>$ns OR coalesce(r.namespace,'')<>$ns)
       RETURN count(r) AS n""",ns=NS100).single()["n"]
    view = [r.data() for r in session.run(VIEW+" LIMIT $limit",ns=NS100,limit=LIMIT)]
    shown = {r["a"]:r["al"] for r in view}|{r["b"]:r["bl"] for r in view if r["b"]}
    shown_ids = {r["key"] for r in view if r["key"]}
    prefix_nodes = {k:nodes[k] for k in shown}
    prefix_edges = {k:edges[k] for k in shown_ids}
    full_size = session.run(VIEW.replace(
        "RETURN n.key AS a, labels(n)[0] AS al, r.key AS key, type(r) AS kind, m.key AS b, labels(m)[0] AS bl",
        "RETURN count(*) AS n"),ns=NS100).single()["n"]
    assert (Counter(shown.values()), Counter(r["type"] for r in prefix_edges.values())) == (
        dict(RELEXDocument=100,RELEXEntity=4506,RELEXConcept=19),
        dict(RELEX_MENTIONS=4506,RELEX_REL=1441,SIMILAR_TO=3860,HAS_SHARED_CONCEPT=40))
    audits = {}
    for name,n,e in [("corpus100",nodes,edges),("prefix",prefix_nodes,prefix_edges),
                      ("stage2f3",*snapshot(session,NS30)),("stage2g3",*snapshot(session,NS30C))]:
        audits[name]=dict(counts=historical.counts(n,e),metrics=graph_metrics(n,e),
                          concepts=inventory(n,e,shown_ids),fingerprints=sorted({r["properties"]["fingerprint"] for r in n.values()}))
    audits["view"]=dict(rows=len(view),null_relationships=len(view)-len(prefix_edges),full_rows=full_size,cross_namespace=cross)
    audits["overlap"]=overlap
    audits["shown_concepts"]=sorted(k for k,v in shown.items() if v=="RELEXConcept")
    audits["relations_by_document"]=relation_audit(docs,edges)
    return audits


def relation_audit(docs, edges):
    kept = Counter(r["properties"]["document_id"] for r in edges.values() if r["type"]=="RELEX_REL")
    er = Counter(row["properties"]["document_id"] for row in edges.values() if row["type"]=="RELEX_MENTIONS")
    rows = []
    for did,d in tqdm(docs.items(),desc="RE checkpoints"):
        ner, re = (checkpoint(WORK/"checkpoints"/stage/(did+".json")) for stage in ("ner","re"))
        assert all(x["document_id"]==did and x["revision_id"]==d["revision_id"] and
                   x["text_sha256"]==d["text_sha256"] for x in (ner,re))
        result = re["results"]
        assert result["raw_relation_count"]==sum(len(w["predictions"]) for w in result["raw_windows"])
        rows.append(dict(document_id=did,topic=d["topic"],ner_mentions=len(ner["results"]["mentions"]),
            pre_er_entities=len(result["entities"]),kg_entities=er[did],raw_relations=result["raw_relation_count"],
            unique_extracted=len(result["relations"]),physical_relations=kept[did]))
    return rows


def selected_score(row):
    return row["embedding_mode"]=="text_name" and passes(row["cosine_similarity"],THRESHOLD)


def summarize(candidates,decisions,proposals):
    ids = {r["candidate_id"] for r in candidates}
    accepted = {r["candidate_id"] for r in decisions if r["status"]=="ACCEPT"}
    rejected = {r["candidate_id"] for r in decisions if r["status"]=="REJECT"} - accepted
    unique = {}
    support = defaultdict(set)
    for r in decisions:
        if r["status"]!="ACCEPT" or not r.get("adds_membership",True):
            continue
        key=(r["source_entity_id"],r["target_concept_id"])
        unique[key]=r
        support[r["target_concept_id"]].add(r["document_id"])
    return dict(candidates=len(ids), by_kind=Counter(r["kind"] for r in candidates), accepted=len(accepted),
                rejected=len(rejected), unevaluated=len(ids-accepted-rejected),
                already_covered=sum(r.get("adds_membership") is False for r in decisions),
                rejection_reasons=Counter(r["reason"] for r in decisions if r["status"]=="REJECT"),
                decision_status=Counter(r["status"] for r in decisions), unique_memberships=len(unique),
                gate_survivors=sum(not r.get("new_concept") or len(support[cid])>=2 for (_,cid),r in unique.items()),
                proposal_status=Counter(r["status"] for r in proposals),
                proposal_reasons=Counter(r["reason"] for r in proposals if r["status"]=="REJECT"))


def historical_funnel():
    manifest = runner.read(HISTORY/"manifest.json")
    paths = {k:HISTORY/(k+".jsonl.gz") for k in ("semantic_candidates","semantic_decisions","new_concept_proposals")}
    for path in paths.values():
        assert runner.sha(path)==manifest["artifact_files_sha256"][path.name]
    rows = [[r for r in tqdm(load_rows(path),desc=path.stem) if selected_score(r)] for path in paths.values()]
    summary=summarize(*rows)
    # Исторические свёрнутые артефакты квалифицируются по точному cosine.
    assert summary["candidates"]==8955
    for file,key in [(HISTORY/"candidate_distribution.csv","distribution"),
                     (HISTORY/"threshold_comparison.csv","comparison")]:
        with file.open(encoding="utf-8-sig",newline="") as stream:
            summary[key]=[r for r in csv.DictReader(stream) if r["threshold"]=="0.830" and r["embedding_mode"]=="text_name"]
    decisions=list(load_rows(ER/"dynamic_concept_graph/candidate_decisions.jsonl"))
    summary["lexical_status"]=Counter(r["status"] for r in decisions)
    summary["lexical_reasons"]=Counter(r["reason"] for r in decisions if r["status"]=="REJECT")
    summary["validator_only_status"]=Counter(r["status"] for r in load_rows(HISTORY/"validator_only_decisions.jsonl"))
    return summary


def vector_cache(label):
    if label=="corpus100":
        out={}
        cfg=runner.read(WORK/"manifest.json")["configuration"]
        e5_fingerprint=runner.signature({"e5":cfg["model_files"]["e5"]})
        for path in sorted((WORK/"checkpoints/embeddings").glob("concept-*.json")):
            row=checkpoint(path)
            assert row["model_fingerprint"]==e5_fingerprint
            out[path.name]=row
        return out
    manifest=runner.read(CACHE/"embedding_manifest.json")
    path=CACHE/"text_name_vectors.npy"
    assert runner.sha(path)==manifest["cache_files_sha256"][path.name]
    assert manifest["cache_files_sha256"][path.name]==runner.read(HISTORY/"manifest.json")[
        "embedding"]["cache_files_sha256"][path.name]
    rows=runner.read(CACHE/"representations.json")
    return dict(zip((r["text_name"] for r in rows["entities"]+rows["concepts"]),np.load(path,allow_pickle=False),strict=True))


def corpus_vectors(entities, concept_names):
    names=list(dict.fromkeys([e.name for e in entities]+list(concept_names)))
    cache=vector_cache("corpus100")
    assert (len(names)+7)//8==len(cache)
    vectors={}
    for number in range(0,len(names),8):
        row=cache[f"concept-{number//8:05}.json"]
        block=names[number:number+8]
        assert row["upstream_sha256"]==runner.signature(block)
        vectors.update(zip(block,row["results"],strict=True))
    return vectors


def reproducible_examples(entities,candidates,decisions,docs):
    by={e.entity_id:e for e in entities}
    classified={r["candidate_id"] for r in decisions if r["status"]=="ACCEPT"}
    seen=set()
    rows=[]
    for r in sorted(decisions,key=lambda r:(-r["cosine_similarity"],r["candidate_id"])):
        if r["status"]!="REJECT" or r["candidate_id"] in classified:
            continue
        a,b=by.get(r["source_entity_id"]),by.get(r.get("reference_entity_id"))
        if not a or not b or a.document_id==b.document_id:
            continue
        fa,fb=concepts.class_form(a.name),concepts.class_form(b.name)
        if not fa or not fb or not (set(fa.split())&set(fb.split())):
            continue
        topic=docs[a.document_id]["topic"]
        key=(topic,tuple(sorted((fa,fb))))
        if key in seen:
            continue
        seen.add(key)
        rows.append(dict(a=a.name,b=b.name,left=a.document_id,right=b.document_id,topic=topic,
            score=r["cosine_similarity"],reason=r["reason"],a_types=a.entity_types,b_types=b.entity_types,
            candidate_id=r["candidate_id"],kind=candidates[r["candidate_id"]]["kind"]))
    topics=defaultdict(list)
    for r in rows:
        topics[r["topic"]].append(r)
    amounts={topic: (6 if any(ord(c)>127 for c in topic) and "NLP" not in topic else 3) for topic in topics}
    result=[r for topic,rs in topics.items() for r in rs[:amounts[topic]]]
    ids={r["candidate_id"] for r in result}
    return (result+[r for r in rows if r["candidate_id"] not in ids])[:20]


def replay(label, entities, links, docs, expected, output, additional_vectors=None):
    entities=sorted(entities,key=lambda e:e.entity_id)
    proposed={}
    ordinary=concepts.assess
    def observe(entity,concept,reference):
        if not concept.existing:
            proposed.setdefault(concept.concept_id,len(concept.member_ids))
        return ordinary(entity,concept,reference)
    with patch.object(concepts,"assess",observe):
        lexical,catalog,existing=current.lexical_concepts(entities,links)
    summary=dict(lexical_status=Counter(r["status"] for r in lexical.decisions),
                 lexical_reasons=Counter(r["reason"] for r in lexical.decisions if r["status"]=="REJECT"),
                 proposed_sizes=Counter(proposed.values()), lexical_concepts=len(catalog), lexical_memberships=len(existing))
    names=list(dict.fromkeys([e.name for e in entities]+[c.name for c in sorted(catalog,key=lambda c:c.concept_id)]))
    if label=="corpus100":
        vectors=corpus_vectors(entities,(c.name for c in sorted(catalog,key=lambda c:c.concept_id)))
        borrowed=[]
    else:
        vectors=vector_cache(label)
        borrowed=sorted((set(names)-vectors.keys()) & (additional_vectors or {}).keys())
        if borrowed:
            model=runner.read(CACHE/"embedding_manifest.json")["model"]
            historical_weight=next(row["sha256"] for row in model["files"]
                                   if row["path"].endswith("model.safetensors"))
            current_weight=runner.read(WORK/"manifest.json")["configuration"]["model_files"]["e5"]["model.safetensors"]
            assert historical_weight==current_weight
        vectors.update((name,additional_vectors[name]) for name in borrowed)
    missing=sorted(set(names)-vectors.keys())
    summary["missing_vectors"]=missing
    summary["reused_from_corpus100"]=borrowed
    save_intermediate(output.with_name(label+"_lexical.json"),summary)
    if missing:
        return summary
    scores=[]
    original_cos=current.cosine_matrix
    def cosine(left,right):
        result=original_cos(left,right)
        scores.append(result)
        return result
    with patch.object(current,"cosine_matrix",cosine):
        candidates=current.hybrid_candidates(entities,catalog,existing,links,vectors,THRESHOLD,20,10)
    document_ids=np.array([e.document_id for e in entities])
    left,right=np.triu_indices(len(entities),1)
    cross=document_ids[left]!=document_ids[right]
    ee=scores[0][left[cross],right[cross]]
    possible_ec=len(entities)*len(catalog)-len(existing)
    cutoff=current.binary_cutoff(THRESHOLD)
    ec=sum(scores[1][i,j]>=cutoff for i,e in enumerate(entities)
           for j,c in enumerate(catalog) if (e.entity_id,c.concept_id) not in existing) if catalog else 0
    summary.update(all_ee=len(left),cross_ee=len(ee),eligible_ec=possible_ec,
                   threshold_ee=int(np.count_nonzero(ee>=cutoff)),threshold_ec=int(ec),
                   ee_quantiles=dict(zip(("p10","p50","p90","p99"),np.quantile(ee,[.1,.5,.9,.99]).tolist())))
    del scores,left,right,ee,cross
    original_eval=current.evaluate
    def evaluate(*args,**kwargs):
        decisions,proposals=original_eval(*args,**kwargs)
        summary["semantic"]=summarize(candidates,decisions,proposals)
        if label=="corpus100":
            summary["examples"]=reproducible_examples(entities,{r["candidate_id"]:r for r in candidates},decisions,docs)
        save_intermediate(output.with_name(label+"_semantic.json"),summary)
        return decisions,proposals
    with patch.object(current,"evaluate",evaluate):
        final=current.hybrid_concepts(entities,lexical,catalog,existing,candidates,THRESHOLD)
    final_pairs={(r["entity_id"],r["concept_id"]) for r in final["memberships"]}
    if expected is not None:
        assert final==expected, "Replay differs from frozen Corpus100 concept checkpoint"
    summary.update(final_concepts=len(final["concepts"]),memberships=len(final_pairs), hybrid=final["hybrid_memberships"],
                   reproduced_checkpoint=expected is not None,final_sizes=Counter(Counter(cid for _,cid in final_pairs).values()),
                   order_policy="entities sorted by ID, stable top-k, lexical forms sorted")
    # Сохраняем компактный результат контроля с теми же единицами учёта.
    summary["final_rows"]={k:final[k] for k in ("concepts","memberships")}
    return summary


def cohesion(members, vectors):
    names=[r["name"] for r in members]
    if len(names)<2 or any(name not in vectors for name in names):
        return None
    vectors=np.asarray([vectors[name] for name in names],dtype=np.float32)
    scores=current.cosine_matrix(vectors,vectors)[np.triu_indices(len(names),1)]
    return dict(pairs=len(scores),mean=float(scores.mean()),minimum=float(scores.min()))


def control_graph(summary,nodes,edges):
    if summary["missing_vectors"]:
        return summary
    nodes={k:r for k,r in nodes.items() if r["label"]!="RELEXConcept"}
    edges={k:r for k,r in edges.items() if r["type"]!="HAS_SHARED_CONCEPT"}
    key=lambda value:NS30+":"+value
    for row in summary["final_rows"]["concepts"]:
        nodes[key(row["concept_id"])]=dict(label="RELEXConcept",properties={"concept_id":row["concept_id"]})
    for row in summary["final_rows"]["memberships"]:
        edges[key(row["membership_id"])]=dict(type="HAS_SHARED_CONCEPT",head=key(row["entity_id"]),
                                               tail=key(row["concept_id"]),properties={})
    summary["counts"]=historical.counts(nodes,edges)
    summary["metrics"]=graph_metrics(nodes,edges)
    return summary


def validator_only(entities, links, shared):
    by={e.entity_id:e for e in entities}
    memberships={(r["entity_id"],r["concept_id"]) for r in shared["memberships"]
                 if r["membership_id"].startswith("membership:2g:")}
    proposals=[r for r in shared["concepts"] if r["concept_id"].startswith("concept:2g:")]
    aliases=defaultdict(set)
    for e in entities:
        for form in e.forms:
            aliases[form].add(e.entity_id)
    neighbors=defaultdict(set)
    for row in links:
        a,b=row["head_id"],row["tail_id"]
        neighbors[a].add(b)
        neighbors[b].add(a)
    resolver=HybridConceptResolver([],entities,memberships)
    decisions,reasons=Counter(),Counter()
    accepted=set()
    for row in tqdm(proposals,desc="Validator-only candidates"):
        cid=row["concept_id"]
        proposal=concepts.Concept(cid,row["name"],tuple(row["aliases"]),existing=False)
        proposal_ids=set().union(*(aliases[f] for f in proposal.forms))
        reference_ids=row["contextual_target_ids"]
        forms=proposal.forms
        confirmed={name for eid in reference_ids for name in (by[eid].name,*by[eid].aliases)
                   if concepts.class_form(name) in forms}
        target=concepts.Concept(cid,row["name"],tuple(sorted(confirmed)),tuple(reference_ids))
        name_tokens=concepts.tokens(target.name)-concepts.GENERIC
        for e in entities:
            eid=e.entity_id
            if (eid,cid) in memberships:
                continue
            if not (concepts.lexical_match(e,proposal) or neighbors[eid] & (proposal_ids-{eid})
                    or concepts.tokens(e.name) & name_tokens):
                continue
            refs=sorted((by[i] for i in reference_ids if i!=eid),
                        key=lambda ref:(ref.document_id==e.document_id,ref.entity_id))
            checks=[]
            for ref in refs:
                status,reason,_=resolver.validate_semantic(e,target,ref)
                checks.append((status,reason))
                if status=="ACCEPT":
                    accepted.add((eid,cid))
                    break
            status,reason=checks[-1]
            decisions[status]+=1
            if status=="REJECT":
                reasons[checks[0][1]]+=1
    final_lexical={(m["entity_id"],m["concept_id"]) for m in shared["memberships"]
                   if m["evidence"].get("embedding_mode")=="lexical"}
    assert final_lexical<=accepted
    return dict(decisions=decisions,first_rejection_reason=reasons,final_lexical_only=len(final_lexical))


def details(session, graphs, docs, entities, vectors, links, shared):
    grouped=defaultdict(list)
    terms={"lignocellulosic biomass","enzymatic hydrolysis","microalgae","lipid production",
           "anaerobic digestion","photovoltaic","pv power forecasting","weather condition"}
    for e in entities:
        form=concepts.class_form(e.name)
        if form in terms:
            grouped[docs[e.document_id]["topic"],form].append(e)
    term_audits=[]
    for (topic,name),rows in sorted(grouped.items()):
        ids={e.document_id for e in rows}
        if len(ids)<2 or topic not in {"Энергетика","Биотехнологии"}:
            continue
        c=concepts.Concept("audit:"+name,name,existing=False)
        checks=[concepts.assess(a,c,b) for a,b in combinations(rows,2) if a.document_id!=b.document_id]
        term_audits.append(dict(topic=topic,name=name,documents=len(ids),decision=Counter(x[0] for x in checks),
                               reasons=Counter(x[1] for x in checks)))
    historical_vectors=vector_cache("control30")
    all_cohesion={}
    for name in ("corpus100","prefix","stage2f3","stage2g3"):
        selected=vectors if name in {"corpus100","prefix"} else historical_vectors
        all_cohesion[name]={r["id"]:cohesion(r["members"],selected) for r in graphs[name]["concepts"]["concepts"]}
    duplicates={}
    historical_edges=None
    for namespace in (NS30,NS30C,NS100):
        _,edge=snapshot(session,namespace)
        if namespace==NS30:
            historical_edges=edge
        rels=[(r["head"],r["properties"]["relation_type"],r["tail"])
              for r in edge.values() if r["type"]=="RELEX_REL"]
        duplicates[namespace]=dict(key_duplicates=0,fact_duplicate_triples=len(rels)-len(set(rels)),
            loops=Counter(r["type"] for r in edge.values() if r["head"]==r["tail"]))
    base=_historical_relations(historical_edges)
    topics=defaultdict(set)
    for r in graphs["corpus100"]["concepts"]["concepts"]:
        for m in r["members"]:
            topics[docs[m["doc"]]["topic"]].add(m["doc"])
    topic_docs=Counter(d["topic"] for d in docs.values())
    return dict(term_audits=term_audits,cohesion=all_cohesion,duplicates=duplicates,historical_relations_by_document=base,
                validator_only=validator_only(entities,links,shared),
                topic_coverage={topic:[len(topics[topic]),n] for topic,n in topic_docs.items()})


def _historical_relations(edges):
    root=ROOT/"reports/re_v2/relex_comparison/threshold035/e3_base"
    source_mentions=Counter(r["document_id"] for r in load_rows(runner.relex.SOURCE/"E3/mentions_035.jsonl"))
    kept=Counter(r["properties"]["document_id"] for r in edges.values() if r["type"]=="RELEX_REL")
    entities=Counter(r["properties"]["document_id"] for r in edges.values() if r["type"]=="RELEX_MENTIONS")
    raw={r["document_id"]:r["raw_relation_count"] for r in load_rows(root/"predictions.jsonl")}
    with (root/"per_document.csv").open(encoding="utf-8-sig",newline="") as stream:
        return [dict(document_id=r["document_id"],topic="",ner_mentions=source_mentions[r["document_id"]],
                     pre_er_entities=int(r["entities"]),kg_entities=entities[r["document_id"]],
                     raw_relations=raw[r["document_id"]],unique_extracted=int(r["relations"]),
                     physical_relations=kept[r["document_id"]]) for r in csv.DictReader(stream)]


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--neo4j-env",type=Path)
    parser.add_argument("--output",type=Path,default=OUT)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--phase",choices=("graphs","historical","control30_cached","corpus100","details","order30"))
    args=parser.parse_args()
    os.environ["PYTHONUTF8"]="1"
    args.output.mkdir(parents=True,exist_ok=True)
    inputs=frozen_inputs()
    docs,results,resolved,links,shared,_=inputs
    input_manifest=dict(graph_fingerprint=results["graph_fingerprint"],config_fingerprint=results["config_fingerprint"],
                        historical_manifest_sha256=runner.sha(HISTORY/"manifest.json"),
                        historical_embedding_manifest_sha256=runner.sha(CACHE/"embedding_manifest.json"))
    save_intermediate(args.output/"input_manifest.json",input_manifest)
    driver,db=runner.relex.connect_neo4j(args.neo4j_env)
    with driver,driver.session(database=db,default_access_mode=READ_ACCESS) as session:
        src_nodes,src_edges=snapshot(session,NS30)
        historical_input=historical.profiles(src_nodes,src_edges)
        entities=current.profiles(docs,resolved)
        def cached_vectors():
            lexical=sorted((r for r in shared["concepts"] if r["concept_id"].startswith("concept:2g:")),
                           key=lambda r:r["concept_id"])
            return corpus_vectors(entities,(r["name"] for r in lexical))
        def order_control():
            profile_order=list(reversed(historical_input[0]))
            link_order=[{"head_id":a,"tail_id":b} for a,b in reversed(historical_input[3])]
            control=replay("control30_reversed",profile_order,link_order,None,None,
                           args.output/"order30.json",cached_vectors())
            expected=runner.read(args.output/"control30_cached.json")
            # Сравниваем результат после сортировки входного набора в production-порядке.
            assert runner.signature(control["final_rows"])==runner.signature(expected["final_rows"])
            return dict(status="PASS",entities=len(profile_order),concepts=control["final_concepts"],
                        memberships=control["memberships"])
        steps=[
            ("graphs",lambda:graph_audit(session,inputs)),
            ("historical",historical_funnel),
            ("control30_cached",lambda:control_graph(replay("control30_cached",historical_input[0],
                [{"head_id":a,"tail_id":b} for a,b in historical_input[3]],None,None,
                args.output/"control30_cached.json",cached_vectors()),src_nodes,src_edges)),
            ("corpus100",lambda:replay("corpus100",entities,links,docs,shared,args.output/"corpus100.json")),
            ("details",lambda:details(session,runner.read(args.output/"graphs.json"),docs,entities,
                                      cached_vectors(),links,shared)),
            ("order30",order_control),
        ]
        for name,calculate in steps:
            if args.phase and args.phase!=name:
                continue
            path=args.output/(name+".json")
            if path.exists():
                if not args.resume:
                    raise ValueError("Diagnostic output already exists: "+str(path))
                print("Restored diagnostic phase",name,flush=True)
                continue
            print("Diagnostic phase",name,flush=True)
            runner.atomic_json(path,calculate())


if __name__=="__main__":
    main()
