"""Независимая проверка опубликованных выдач от физических рёбер и первичных источников."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

from inputs import HERE, file_sha, inputs, read_json, require
from cgraph.retrieval_types import graph_path_id


def proof(pid, path, graph, physical):
    nodes, expected = [], []
    for edge in path["edges"]:
        row = physical[edge["key"]]
        props = row["properties"]
        h, t = row["head"]["key"], row["tail"]["key"]
        require(props["fingerprint"] == graph.fingerprint and edge["head_key"] == h and edge["tail_key"] == t
                and row["kind"] == edge["type"], "Physical edge differs")
        left, right = (t, h) if edge["traversed_reverse"] else (h, t)
        if not nodes:
            nodes.append(left)
        require(nodes[-1] == left and right not in nodes, "Disconnected or cyclic path")
        nodes.append(right)
        if edge["type"] != "RELEX_REL":
            require(edge["document_id"] is None and edge["relation_type"] is None, "Navigation became factual")
            continue
        doc_id = props["document_id"]
        doc = graph.documents[doc_id]
        require(doc_id == edge["document_id"] == row["head"]["document_id"] == row["tail"]["document_id"] and
                edge["relation_type"] == props["relation_type"] and
                sha256(doc["text"].encode()).hexdigest() == doc["text_sha256"], "Source mismatch")
        sources = json.loads(props["source_relations_json"])
        require(sorted(s["triple_id"] for s in sources) == sorted(props["source_triple_ids"]), "Triple mapping")
        for source in sources:
            require(source["document_id"] == doc_id and source["relation_type"] == edge["relation_type"],
                    "Source attribution mismatch")
            windows = []
            for window in source["evidence"]:
                start, end = window["start"], window["end"]
                require(window["document_id"] == doc_id and 0 <= start < end <= len(doc["text"])
                        and window["text"] == doc["text"][start:end], "Source quote differs")
                windows.append(dict(document_id=doc_id, start=start, end=end, window_id=window["window_id"],
                                    revision_id=doc["revision_id"], text_sha256=doc["text_sha256"]))
            require(bool(windows), "Missing citation")
            expected.append(dict(document_id=doc_id, source_triple_id=source["triple_id"], edge_key=edge["key"],
                                 windows=windows))
    docs = {graph.nodes[n]["properties"].get("document_id") for n in nodes} - {None}
    require(len(nodes) <= 6 and pid == graph_path_id(nodes[0], nodes, path["edges"]) and expected == path["facts"]
            and sorted({f["document_id"] for f in expected}) == path["factual_document_ids"] and
            sorted(docs) == path["document_ids"], "Published provenance differs")
    return sum(len(f["windows"]) for f in expected)


def metrics(paths, ref):
    required = set(ref["document_contributions"])
    docs = set()
    factual = set()
    hits, rr, gain = set(), 0.0, 0.0
    sources, windows, transitions = Counter(), Counter(), Counter()
    for rank, path in enumerate(paths, 1):
        docs.update(path["document_ids"])
        transitions.update(e["type"] for e in path["edges"])
        by_path = set()
        for fact in path["facts"]:
            doc = fact["document_id"]
            sources[(doc, fact["source_triple_id"])] += 1
            by_path.add(doc)
            for w in fact["windows"]:
                windows[(doc, w["revision_id"], w["text_sha256"], w["start"], w["end"])] += 1
        new = (by_path & required) - factual
        if new:
            rr = rr or 1/rank
            gain += len(new)/math.log2(rank+1)
        factual.update(by_path)
    for e in ref["evidence"]:
        if any(doc == e["document_id"] and rev == e["revision_id"] and start < e["end"] and end > e["start"]
               for doc, rev, _, start, end in windows):
            hits.add(e["evidence_id"])
    return {"FactualGoldDocRecall@5": len(required & factual)/len(required),
            "GoldDocRecall@5": len(required & docs)/len(required), "AllRequiredDocs@5": required <= docs,
            "EvidenceWindowOverlap@5": len(hits)/len(ref["evidence"]), "overlapping_evidence_ids": sorted(hits),
            "nDCG@5_factual_doc_first_hit": gain/len(required), "MRR@5_factual_doc_first_hit": rr,
            "unique_documents": len(docs), "unique_factual_documents": len(factual),
            "off_gold_documents": sorted(docs-required), "off_gold_sources": sorted(factual-required),
            "duplicate_source_triples": sources.total()-len(sources),
            "duplicate_evidence_windows": windows.total()-len(windows), "transition_counts": dict(transitions),
            "factual_gold_documents_in_top_5": sorted(factual & required)}


def verify(directory, replay=None):
    graph, _, _, refs, archived = inputs()
    physical = {r["properties"]["key"]: r for rows in graph.adjacency.values() for r in rows}
    numbers = {}
    for phase in tqdm(("diagnosis", "matrix", "lexical", "combined", "confirmation"), desc="Independent audit"):
        root = directory/phase
        manifest = read_json(root/"manifest.json")
        for name, signature in manifest["files"].items():
            require(file_sha(root/name) == signature, "Artifact changed")
        evidence = read_json(root/"evidence.json")
        windows = sum(proof(pid, path, graph, physical) for pid, path in evidence.items())
        rows = [json.loads(line) for line in (root/"questions.jsonl").read_bytes().splitlines()]
        seen, groups = set(), defaultdict(list)
        for r in rows:
            key = r["key"].removeprefix("PARAPHRASE/")
            require((r["key"], r["profile"]) not in seen and len(r["top"]) == len(set(r["top"])) <= 5, "Duplicates")
            seen.add((r["key"], r["profile"]))
            m = metrics([evidence[p] for p in r["top"]], refs[key])
            for k, v in m.items():
                saved = r["metrics"][k]
                require((sorted(saved) == v if isinstance(v, list) else saved == v), f"Metric differs {key} {k}")
            if r["key"] == key and r["profile"] in ("production-fifo-100-0", "k5-fifo-200-0"):
                archive = archived[key, "B" if r["profile"].startswith("k5") else "D"]
                require(r["top"] == [p["path_id"] for p in archive["top_5"]], "Archive ranks differ")
            group = r["key"].split("/")[0]
            for g in ([group, "approved"] if group in ("A", "B", "C") else [group]):
                groups[r["profile"], g].append((m, r, refs[key]))
        summary = read_json(root/"summary.json")
        require({(p,g) for p, gs in summary.items() for g in gs} == groups.keys(), "Split mismatch")
        for (profile, group), values in groups.items():
            s = summary[profile][group]
            total = sum(len(ref["document_contributions"]) for _, _, ref in values)
            hits = sum(len(m["factual_gold_documents_in_top_5"]) for m, _, _ in values)
            require((s["questions"], s["hits"], s["required"]) == (len(values), hits, total)
                    and s["FactualGoldDocRecall@5_micro"] == hits/total, "Micro aggregate differs")
            for name, field in (("FactualGoldDocRecall@5_macro", "FactualGoldDocRecall@5"),
                                ("AllRequiredDocs@5", "AllRequiredDocs@5"), ("EvidenceWindowOverlap@5", "EvidenceWindowOverlap@5"),
                                ("nDCG@5_factual_doc_first_hit", "nDCG@5_factual_doc_first_hit"),
                                ("MRR@5_factual_doc_first_hit", "MRR@5_factual_doc_first_hit")):
                require(math.isclose(s[name], np.mean([m[field] for m, _, _ in values])), "Macro differs")
            for p in (50,95):
                require(math.isclose(s[f"replay_p{p}"], np.percentile([sum(r["seconds"].values())
                        for _, r, _ in values], p)), "Latency percentile differs")
        if replay:
            repeat = [json.loads(line) for line in (replay/phase/"questions.jsonl").read_bytes().splitlines()]
            def stable(r):
                measured = {"seconds", "replay_seconds", "upstream_seconds", "clause_e5_seconds_upper_bound",
                            "peak_process_rss"}
                return {k: v for k, v in r.items() if k not in measured}
            require([stable(r) for r in rows] == [stable(r) for r in repeat] and
                    evidence == read_json(replay/phase/"evidence.json"), "Repeat differs")
            if (root/"losses.json").exists():
                require(read_json(root/"losses.json") == read_json(replay/phase/"losses.json"), "Loss trace differs")
        numbers[phase] = dict(rows=len(rows), paths=len(evidence), windows=windows, aggregates=len(groups))
    coverage = read_json(directory/"stage_coverage.json")
    totals = defaultdict(Counter)
    for r in coverage["rows"]:
        required = set(refs[r["key"]]["document_contributions"])
        hits = {k: set(v) for k, v in r["hits"].items()}
        require(all(v <= required for v in hits.values()), "Unexpected reference document")
        if "eligible" in hits:
            require(hits["top"] <= hits["preselection"] <= hits["eligible"] <= hits["raw_relex"] <= hits["raw_docs"]
                    and hits["accepted_seed_docs"] <= hits["uncapped_seed_docs"], "Stage subset violation")
        if r["profile"] in ("production-fifo-100-0", "k5-fifo-200-0"):
            base = archived[r["key"], "B" if r["profile"].startswith("k5") else "D"]
            require(hits["raw_docs"] == required.intersection(base["candidate_documents"]) and
                    hits["eligible"] == required.intersection(base["candidate_factual_documents"]) and
                    hits["top"] == required.intersection(base["top_factual_docs"]), "Pool oracle differs from archive")
        totals[r["profile"]].update({k: len(v) for k, v in hits.items()})
    require(dict(totals) == coverage["approved_hits"], "Stage aggregates differ")
    if replay:
        require(file_sha(directory/"final_losses.json") == file_sha(replay/"final_losses.json"),
                "Full loss audit differs")
    return numbers


if __name__ == "__main__":
    print(json.dumps(verify(*map(Path, sys.argv[1:])), indent=2))
