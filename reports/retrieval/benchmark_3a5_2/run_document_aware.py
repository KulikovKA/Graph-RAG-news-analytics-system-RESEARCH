"""Replay the frozen 3A.5.2 ranking from its compact archive, offline."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys

from validate_draft import HERE, documents, jsonl, require

sys.path.insert(0, str(HERE.parents[2] / "src"))
from cgraph.path_reranking import rerank_paths
from document_aware_reranking import (BASE_SCORE_MARGIN, DOCUMENT_FIRST_SLOTS, REPEATED_DOCUMENT_PENALTY,
                                     SECOND_PATH_WEIGHT, score_candidates, select_paths)


BASELINE = HERE / "block_c_retrieval.json"
BASELINE_SHA256 = "91045d2e50fd31809fd2fe2951f34ed2b7144b43d9ae8924a6725a51e43374fa"
BASELINE_COMMIT = "ab9c8f56b96e7cda6c0621f33dd6942bbd84d367"
GOLD = HERE / "questions_draft.jsonl"
PATH_FIELDS = ("path_id", "seed_key", "node_keys", "document_ids")


def check_fact(fact, corpus):
    doc = corpus[fact["document_id"]]
    sources = fact["source_relations"]
    require(fact["status"] == "valid" and not fact["reasons"] and bool(sources), "Invalid archived fact")
    ids = set()
    for source in sources:
        raw = source["source"]
        triple_id = source["source_triple_id"]
        require(source["status"] == "valid" and not source["reasons"] and bool(source["windows"])
                and triple_id == raw["triple_id"] and bool(triple_id) and triple_id not in ids
                and raw["document_id"] == fact["document_id"] and raw["relation_type"] == fact["relation_type"]
                and raw["head_id"] and raw["tail_id"], "Invalid archived source triple")
        ids.add(triple_id)
        require(raw["evidence"] == [w["source"] for w in source["windows"]], "Archived source mapping differs")
        for w in source["windows"]:
            require(w["status"] == "valid" and not w["reasons"] and bool(w["window_id"])
                    and all(w[k] == doc[k] for k in ("document_id", "revision_id", "text_sha256",
                                                      "title", "source_url", "doi"))
                    and all(w[k] == w["source"][k] for k in ("document_id", "window_id", "start", "end", "text"))
                    and type(w["start"]) is int and type(w["end"]) is int
                    and 0 <= w["start"] < w["end"] <= len(doc["text"])
                    and w["text"] == doc["text"][w["start"]:w["end"]], "Archived evidence differs from corpus")


def restore_question(saved, corpus, fingerprint):
    retrieval = {**saved["traversal"], "paths": []}
    namespace = retrieval["namespace"]
    require(retrieval["graph_fingerprint"] == fingerprint, "Archived retrieval fingerprint differs")
    for fact in saved["facts"].values():
        check_fact(fact, corpus)
    bundles = []
    for key, archived in saved["candidate_paths"].items():
        path = {field: archived[field] for field in PATH_FIELDS}
        require(path["path_id"] == key and not archived["evidence_reasons"], "Invalid archived path")
        path["edges"] = []
        bundle = {"path_id": key, "edges": [], "reasons": []}
        factual_documents = set()
        for flat in archived["edges"]:
            # Проекция для проверок реранкера. Исходные свойства рёбер в архиве отсутствуют.
            props = {"key": flat["key"], "namespace": namespace, "fingerprint": fingerprint}
            edge = {**flat, "properties": props}
            path["edges"].append(edge)
            if flat["type"] == "RELEX_REL":
                fact = saved["facts"][flat["key"]]
                require(all(fact[k] == flat[k] for k in ("head_key", "tail_key", "relation_type", "document_id")),
                        "Archived fact and path disagree")
                props.update(relation_type=flat["relation_type"], document_id=flat["document_id"],
                             head_id=flat["head_key"][len(namespace) + 1:],
                             tail_id=flat["tail_key"][len(namespace) + 1:])
                factual_documents.add(fact["document_id"])
                item = {"non_factual": False, "status": fact["status"], "source_relations": fact["source_relations"]}
            else:
                item = {"non_factual": True, "status": "non_factual", "source_relations": []}
            bundle["edges"].append({"edge": edge, **item})
        require(sorted(factual_documents) == archived["factual_documents"]
                and archived["evidence_status"] == ("valid" if factual_documents else "non_factual"),
                "Archived evidence summary differs")
        # Для зафиксированного SHA архив сохраняет результат upstream trust-проверки в evidence_reasons.
        bundle["navigation_valid"] = True
        bundle["documents"] = [{"document_id": d, "fingerprint": fingerprint} for d in sorted(factual_documents)]
        retrieval["paths"].append(path)
        bundles.append(bundle)
    return retrieval, bundles


def replay_baseline(saved, retrieval, bundles, seeds):
    replayed = rerank_paths(saved["question"], retrieval, bundles, seeds)
    top = [{k: v for k, v in p.items() if k not in ("path", "evidence")} for p in replayed["paths"]]
    require(top == saved["top_5"], f"{saved['question_id']}: ordered Top-5 or scores differ")
    require({k: v for k, v in replayed.items() if k != "paths"} == saved["reranking"],
            f"{saved['question_id']}: exclusions or diagnostics differ")
    return replayed


def path_output(item):
    factual = []
    for edge in item["evidence"]["edges"]:
        if edge["edge"]["type"] != "RELEX_REL":
            continue
        for source in edge["source_relations"]:
            factual.append({"document_id": edge["edge"]["document_id"], "edge_key": edge["edge"]["key"],
                            "source_triple_id": source["source_triple_id"],
                            "windows": [{k: w[k] for k in ("document_id", "window_id", "start", "end", "text",
                                                          "revision_id", "text_sha256", "source_url")}
                                        for w in source["windows"]]})
    return {**{k: item[k] for k in ("path_id", "rank", "base_score", "score", "score_breakdown")},
            **{k: item[k] for k in ("selection_phase", "primary_document", "new_factual_documents",
                                    "new_source_triples") if k in item},
            "document_ids": item["path"]["document_ids"],
            "factual_document_ids": sorted({f["document_id"] for f in factual}), "facts": factual}


def measure(paths, gold):
    required = set(gold["document_contributions"])
    present = {d for p in paths for d in p["document_ids"]}
    factual = {d for p in paths for d in p["factual_document_ids"]}
    facts = [f for p in paths for f in p["facts"]]
    windows = [w for fact in facts for w in fact["windows"]]
    overlaps = [e["evidence_id"] for e in gold["evidence"] if any(
        w["document_id"] == e["document_id"] and w["revision_id"] == e["revision_id"]
        and max(w["start"], e["start"]) < min(w["end"], e["end"]) for w in windows)]
    sources = Counter((f["document_id"], f["source_triple_id"]) for f in facts)
    return {"gold_documents_in_top_5": sorted(present & required),
            "missing_gold_documents": sorted(required - present),
            "factual_gold_documents_in_top_5": sorted(factual & required),
            "GoldDocRecall@5": len(present & required) / len(required),
            "FactualGoldDocRecall@5": len(factual & required) / len(required),
            "AllRequiredDocs@5": required <= present,
            "EvidenceWindowOverlap@5": len(overlaps) / len(gold["evidence"]),
            "overlapping_evidence_ids": overlaps,
            "unique_factual_documents": len(factual),
            "duplicate_source_triples": sum(n - 1 for n in sources.values()),
            "repeated_sources": [{"document_id": d, "source_triple_id": i, "occurrences": n}
                                 for (d, i), n in sorted(sources.items()) if n > 1],
            "SemanticEvidenceCoverage@5": "NOT_EVALUATED"}


def summarize(questions, policy):
    metrics = [q["policies"][policy]["metrics"] for q in questions]
    gold_total = sum(len(q["gold_documents"]) for q in questions)
    evidence_total = sum(q["gold_evidence_count"] for q in questions)
    gold_hits = sum(len(m["gold_documents_in_top_5"]) for m in metrics)
    factual_hits = sum(len(m["factual_gold_documents_in_top_5"]) for m in metrics)
    overlaps = sum(len(m["overlapping_evidence_ids"]) for m in metrics)
    return {"gold_doc_hits": gold_hits, "gold_doc_total": gold_total,
            "GoldDocRecall@5_micro": gold_hits / gold_total,
            "GoldDocRecall@5_macro": mean(m["GoldDocRecall@5"] for m in metrics),
            "factual_gold_doc_hits": factual_hits,
            "FactualGoldDocRecall@5_micro": factual_hits / gold_total,
            "FactualGoldDocRecall@5_macro": mean(m["FactualGoldDocRecall@5"] for m in metrics),
            "all_required_docs_questions": sum(m["AllRequiredDocs@5"] for m in metrics),
            "AllRequiredDocs@5": mean(m["AllRequiredDocs@5"] for m in metrics),
            "overlapping_evidence_occurrences": overlaps, "evidence_total": evidence_total,
            "EvidenceWindowOverlap@5_micro": overlaps / evidence_total,
            "EvidenceWindowOverlap@5_macro": mean(m["EvidenceWindowOverlap@5"] for m in metrics),
            "mean_unique_factual_documents": mean(m["unique_factual_documents"] for m in metrics),
            "total_duplicate_source_triples": sum(m["duplicate_source_triples"] for m in metrics),
            "SemanticEvidenceCoverage@5": "NOT_EVALUATED"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New experiment JSON. Omit for baseline replay only.")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("--output already exists")
    raw = BASELINE.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == BASELINE_SHA256, "Unexpected baseline SHA-256")
    archived = json.loads(raw)
    require(archived["status"] == "COMPLETE" and len(archived["questions"]) == 10
            and sum(len(q["candidate_paths"]) for q in archived["questions"]) == 2000, "Unexpected baseline scope")
    fingerprint = archived["preflight"]["graph_fingerprint"]
    require(fingerprint == archived["postflight"]["graph_fingerprint"], "Baseline fingerprint changed")
    corpus = documents()
    inputs = []
    for saved in archived["questions"]:
        retrieval, bundles = restore_question(saved, corpus, fingerprint)
        seeds = saved["entity_linking"]["traversal_seeds"]
        replayed = replay_baseline(saved, retrieval, bundles, seeds)
        inputs.append((saved, retrieval, bundles, seeds, replayed))
        print(saved["question_id"], "EXACT", len(replayed["paths"]), flush=True)
    if not args.output:
        return

    questions = []
    for saved, retrieval, bundles, seeds, replayed in inputs:
        candidates = score_candidates(saved["question"], retrieval, bundles, seeds)
        require(len(candidates) == replayed["ranking_diagnostics"]["eligible_paths"], "Eligibility changed")
        by_id = {c["path_id"]: c for c in candidates}
        for path in replayed["paths"]:
            require(by_id[path["path_id"]]["base_score"] == path["base_score"], "Base score changed")
        policies = {"A": {"top_5": [path_output(p) for p in replayed["paths"]]}}
        for policy in ("B", "C"):
            selected, profiles = select_paths(candidates, policy)
            policies[policy] = {"top_5": [path_output(p) for p in selected]}
            if policy == "C":
                policies[policy]["document_profiles"] = profiles
        best_by_doc = {}
        for c in sorted(candidates, key=lambda c: (-c["base_score"], c["path_id"])):
            for doc in sorted(c["features"]["documents"]):
                best_by_doc.setdefault(doc, {"base_score": c["base_score"], "path_id": c["path_id"],
                                             "local_query_coverage": c["document_coverage"][doc]})
        questions.append({"question_id": saved["question_id"], "candidate_count": len(retrieval["paths"]),
                          "eligible_paths": len(candidates), "baseline_replay": "EXACT",
                          "candidate_documents": sorted({d for p in retrieval["paths"] for d in p["document_ids"]}),
                          "candidate_factual_documents": sorted(best_by_doc),
                          "best_base_by_factual_document": best_by_doc, "policies": policies})

    # Gold загружается только после отбора всех путей для A/B/C.
    gold_sha256 = hashlib.sha256(GOLD.read_bytes()).hexdigest()
    require(gold_sha256 == archived["benchmark_input"]["sha256"], "Gold differs from baseline")
    gold_rows = {r["question_id"]: r for r in jsonl(GOLD) if r["category"] == "C"}
    for question, (saved, *_rest) in zip(questions, inputs, strict=True):
        gold = gold_rows[question["question_id"]]
        required = set(gold["document_contributions"])
        question.update(gold_documents=sorted(required), gold_evidence_count=len(gold["evidence"]),
                        gold_docs_available=sorted(required & set(question["candidate_documents"])),
                        missing_from_candidates=sorted(required - set(question["candidate_documents"])))
        for output in question["policies"].values():
            output["metrics"] = measure(output["top_5"], gold)
        for metric in ("GoldDocRecall@5", "AllRequiredDocs@5", "EvidenceWindowOverlap@5"):
            require(question["policies"]["A"]["metrics"][metric] == saved["metrics"][metric],
                    f"Replayed A metric differs: {question['question_id']}: {metric}")
    report = {"stage": "3A.5.3", "baseline_commit": BASELINE_COMMIT, "baseline_sha256": BASELINE_SHA256,
              "gold_sha256": gold_sha256, "namespace": archived["namespace"], "graph_fingerprint": fingerprint,
              "candidate_count": 2000, "baseline_replay": {"status": "EXACT", "matched_ranks": 50,
                  "checked": ["rank", "path_id", "base_score", "score", "score_breakdown", "metrics", "exclusions",
                              "ranking_diagnostics"],
                  "navigation": "Archived live validation from the SHA-pinned baseline",
                  "unavailable_raw_fields": ["full edge properties and individual fingerprints",
                                             "full evidence document properties and individual fingerprints",
                                             "raw navigation trust flags"]},
              "parameters": {"min_relevance": "positive question-term coverage", "base_score_margin": BASE_SCORE_MARGIN,
                             "B_repeated_document_penalty": REPEATED_DOCUMENT_PENALTY,
                             "C_second_independent_path_weight": SECOND_PATH_WEIGHT,
                             "C_document_first_slots": DOCUMENT_FIRST_SLOTS},
              "candidate_ceiling": {"gold_doc_hits": sum(len(q["gold_docs_available"]) for q in questions),
                                    "all_required_docs_questions": sum(not q["missing_from_candidates"] for q in questions)},
              "summary": {policy: summarize(questions, policy) for policy in ("A", "B", "C")},
              "questions": questions}
    with args.output.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(report, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write("\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
