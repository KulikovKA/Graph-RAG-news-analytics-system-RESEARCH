"""Offline scorer; exact source spans, directed edges, multiset one-to-one credit."""
from collections import Counter
import json


def prf(gold, predicted):
    gold, predicted = Counter(gold), Counter(predicted)
    tp = sum((gold & predicted).values())
    ng, np = sum(gold.values()), sum(predicted.values())
    return dict(tp=tp, fp=np - tp, fn=ng - tp, predicted=np, gold=ng,
                precision=tp / np if np else None, recall=tp / ng if ng else None,
                f1=2 * tp / (ng + np) if ng + np else None)


def entity_key(chunk, m, typed=True):
    return (chunk, m["start"], m["end"]) + ((m["kind"],) if typed else ())


def qualifier_key(assertion, prediction=False):
    fields = [(q["qualifier_type"], json.dumps(q["raw_value"], sort_keys=True), q.get("unit"), q.get("raw_text"))
              for q in assertion["qualifiers"]]
    return (assertion.get("polarity", "UNKNOWN"), assertion.get("modality", "UNKNOWN"),
            tuple(sorted(fields)), "UNSUPPORTED" if prediction and assertion.get("qualifier_status") == "UNKNOWN" else "COMPLETE")


def document_metrics(prediction, gold):
    gm = {m["mention_id"]: m for m in gold["mentions"]}
    pm = {m["mention_id"]: dict(m, chunk_id=c["chunk_id"]) for c in prediction["chunks"] for m in c["entities"]}
    gtyped = [entity_key(m["chunk_id"], m) for m in gm.values()]
    ptyped = [entity_key(m["chunk_id"], m) for m in pm.values()]
    gspan = [entity_key(m["chunk_id"], m, False) for m in gm.values()]
    pspan = [entity_key(m["chunk_id"], m, False) for m in pm.values()]

    def edge(a, entities):
        h, t = entities[a["head_mention_id"]], entities[a["tail_mention_id"]]
        return entity_key(h["chunk_id"], h), entity_key(t["chunk_id"], t), a["relation_type"]

    ga = [a for a in gold["assertions"] if a["support_status"] == "SUPPORTED"]
    pa = [a for c in prediction["chunks"] for a in c["assertions"]]
    raw = [a for c in prediction["chunks"] for a in c["raw_proposals"]]
    pairs = [p for c in prediction["chunks"] for p in c["candidate_pairs"]]
    candidate_keys = {(entity_key(pm[p["head_mention_id"]]["chunk_id"], pm[p["head_mention_id"]]),
                       entity_key(pm[p["tail_mention_id"]]["chunk_id"], pm[p["tail_mention_id"]])) for p in pairs}
    goldedges = [edge(a, gm) for a in ga]
    conditional_gold = [e for e in goldedges if e[:2] in candidate_keys]
    gold_entities_set = set(gtyped)
    conditional_pred = [edge(a, pm) for a in pa if edge(a, pm)[0] in gold_entities_set and edge(a, pm)[1] in gold_entities_set]
    metrics = dict(entity_span=prf(gspan, pspan), entity_typed=prf(gtyped, ptyped),
                   relation_raw_end_to_end=prf(goldedges, [edge(a, pm) for a in raw]),
                   relation_end_to_end=prf(goldedges, [edge(a, pm) for a in pa]),
                   relation_given_candidate_endpoints=prf(conditional_gold, conditional_pred),
                   qualified_relation=prf([edge(a, gm) + qualifier_key(a) for a in ga],
                                          [edge(a, pm) + qualifier_key(a, True) for a in pa]),
                   candidate_pair_recall=dict(covered=len(conditional_gold), gold=len(goldedges),
                                              recall=len(conditional_gold) / len(goldedges) if goldedges else None),
                   abstention=dict(abstained=sum(p["decision_status"] == "ABSTAIN" for p in pairs), candidate_pairs=len(pairs)),
                   accepted_assertions=len(pa), raw_proposals=len(raw),
                   unsupported_proposals=sum(bool(a.get("reason", "") and a["reason"].startswith("UNSUPPORTED_CAPABILITY")) for a in raw))
    return metrics


def evaluate(predictions, labels):
    if len(predictions) != len(labels) or {p["document_code"] for p in predictions} != set(labels):
        raise ValueError("Missing/duplicate prediction documents")
    results = {}
    for pred in predictions:
        g = labels[pred["document_code"]]
        for key in ("document_id", "revision_id", "split"):
            if pred[key] != g[key]:
                raise ValueError("Evaluation source mismatch")
        expected = {c["chunk_id"]: c["text_sha256"] for c in g["source_chunks"]}
        if {c["chunk_id"]: c["source_text_sha256"] for c in pred["chunks"]} != expected:
            raise ValueError("Evaluation chunk/hash mismatch")
        results[pred["document_code"]] = document_metrics(pred, g)
    micro, macro = {}, {}
    names = ["entity_span", "entity_typed", "relation_raw_end_to_end", "relation_end_to_end", "relation_given_candidate_endpoints", "qualified_relation"]
    for name in names:
        counts = {k: sum(m[name][k] for m in results.values()) for k in ("tp", "fp", "fn", "gold", "predicted")}
        tp, ng, np = counts["tp"], counts["gold"], counts["predicted"]
        counts.update(precision=tp / np if np else None, recall=tp / ng if ng else None, f1=2 * tp / (ng + np) if ng + np else None)
        micro[name] = counts
        vals = [m[name]["f1"] for m in results.values() if m[name]["f1"] is not None]
        macro[name] = dict(f1=sum(vals) / len(vals) if vals else None, documents=len(vals))
    covered = sum(m["candidate_pair_recall"]["covered"] for m in results.values())
    ng = sum(m["candidate_pair_recall"]["gold"] for m in results.values())
    abstained = sum(m["abstention"]["abstained"] for m in results.values())
    pairs = sum(m["abstention"]["candidate_pairs"] for m in results.values())
    micro["candidate_pair_recall"] = dict(covered=covered, gold=ng, recall=covered / ng if ng else None)
    micro["abstention"] = dict(abstained=abstained, candidate_pairs=pairs, rate=abstained / pairs if pairs else None)
    return dict(micro=micro, macro_document=macro, documents=results,
                matching_policy="Exact Unicode spans+kind; directed predicate; multiset one-to-one; no relaxed/coreference substitution",
                qualifier_policy="UNKNOWN qualifier completeness never receives qualified TP; gold qualifiers remain in denominator",
                annotation_limitation="Single owner review, AI-origin annotations; no independent secondary review",
                semantic_support_rate=None, semantic_support_status="NOT_MEASURED: requires independent review of predictions")
