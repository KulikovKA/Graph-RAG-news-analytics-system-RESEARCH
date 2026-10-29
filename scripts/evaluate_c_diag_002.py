"""Offline Gold evaluation AFTER all frozen variants have saved predictions."""
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgraph.common import file_sha256, fingerprint, read_json, write_json
from cgraph.evaluation import entity_key, evaluate, prf
from cgraph.inputs import load_inputs, load_ontology
from cgraph.labels import load_labels
from cgraph.validation import validate_prediction
from cgraph.zeroshot_evaluation import loss_audit


def typed_coverage(predictions, labels, ontology):
    result = {}
    for kind in ontology["entities"]:
        gold_t, pred_t, gold_s, pred_s = [], [], [], []
        for pred in predictions:
            code = pred["document_code"]
            for m in labels[code]["mentions"]:
                if m["kind"] == kind:
                    gold_t.append((code,) + entity_key(m["chunk_id"], m))
                    gold_s.append((code,) + entity_key(m["chunk_id"], m, False))
            for c in pred["chunks"]:
                for m in c["entities"]:
                    pred_s.append((code,) + entity_key(c["chunk_id"], m, False))
                    if m["kind"] == kind:
                        pred_t.append((code,) + entity_key(c["chunk_id"], m))
        result[kind] = dict(typed=prf(gold_t, pred_t),
                            exact_boundary_covered=sum((Counter(gold_s) & Counter(pred_s)).values()),
                            gold=len(gold_t))
    return result


def examples_and_fp(predictions, labels):
    examples, fp_categories = {}, Counter()
    def keep(category, example):
        if category not in examples:
            examples[category] = example
    for pred in predictions:
        code = pred["document_code"]
        gold = labels[code]
        gm = {m["mention_id"]: m for m in gold["mentions"]}
        pm = {m["mention_id"]: dict(m, chunk_id=c["chunk_id"]) for c in pred["chunks"] for m in c["entities"]}
        chunks = {c["chunk_id"]: c for c in pred["chunks"]}
        def edge(a, mentions):
            h, t = mentions[a["head_mention_id"]], mentions[a["tail_mention_id"]]
            return entity_key(h["chunk_id"], h), entity_key(t["chunk_id"], t), a["relation_type"]
        ga = [a for a in gold["assertions"] if a["support_status"] == "SUPPORTED"]
        ge = Counter(edge(a, gm) for a in ga)
        remaining = ge.copy()
        typed = {entity_key(m["chunk_id"], m) for m in gm.values()}
        spans = {key[:3] for key in typed}
        for c in pred["chunks"]:
            for a in c["assertions"]:
                e = edge(a, pm)
                correct = remaining[e] > 0
                if correct:
                    remaining[e] -= 1
                else:
                    if e[0] in typed and e[1] in typed:
                        category = "exact_gold_endpoints_wrong_or_unannotated_relation"
                    elif e[0][:3] in spans and e[1][:3] in spans:
                        category = "gold_boundaries_wrong_entity_kind"
                    else:
                        category = "at_least_one_non_gold_boundary"
                    fp_categories[category] += 1
                h, t = pm[a["head_mention_id"]], pm[a["tail_mention_id"]]
                keep("relation_TP" if correct else "relation_FP", dict(document=code, chunk_id=c["chunk_id"],
                     head=h["surface_text"], head_kind=h["kind"], tail=t["surface_text"], tail_kind=t["kind"],
                     relation=a["relation_type"], score=a["score"], evidence=a["evidence"][0]["text"]))
        pk = {entity_key(m["chunk_id"], m) for m in pm.values()}
        candidate_keys = {(entity_key(pm[p["head_mention_id"]]["chunk_id"], pm[p["head_mention_id"]]),
                           entity_key(pm[p["tail_mention_id"]]["chunk_id"], pm[p["tail_mention_id"]]))
                          for c in pred["chunks"] for p in c["candidate_pairs"]}
        raw_keys = {edge(a, pm) for c in pred["chunks"] for a in c["raw_proposals"]}
        for a in ga:
            e = edge(a, gm)
            if remaining[e] <= 0:
                continue
            h, t = gm[a["head_mention_id"]], gm[a["tail_mention_id"]]
            if e[0] not in pk or e[1] not in pk:
                stage = "NER"
            elif e[:2] not in candidate_keys:
                stage = "CONTEXT"
            elif e not in raw_keys:
                stage = "RE_OR_CAPABILITY"
            else:
                stage = "GATES"
            c = chunks[h["chunk_id"]]
            start, end = max(0, h["start"] - 70), min(len(c["source_text"]), max(h["end"], t["end"] if t["chunk_id"] == h["chunk_id"] else h["end"]) + 70)
            keep("relation_FN_" + stage, dict(document=code, chunk_id=h["chunk_id"], head=h["surface_text"],
                 head_kind=h["kind"], tail=t["surface_text"], tail_kind=t["kind"], relation=a["relation_type"],
                 stage=stage, evidence=c["source_text"][start:end]))
    return examples, dict(fp_categories)


def main():
    cfg = read_json("config/c_diag_002.json")
    variants = cfg["diagnostic_protocol"]["variants"]
    # Check every saved run before the first Gold read.
    inventories = {v: read_json(f"artifacts/c_diag_002/{v}/run.json") for v in variants}
    docs, ontology = load_inputs(cfg), load_ontology(cfg)
    predictions = {}
    for v, run in inventories.items():
        directory = Path("artifacts/c_diag_002") / v
        audit = read_json(directory / "access_audit.json")
        if run["config_sha256"] != fingerprint(cfg) or not audit["completed"] or audit["run_sha256"] != file_sha256(directory / "run.json"):
            raise ValueError("Run protocol/audit mismatch")
        if run["training_performed"] or run["gold_relations_in_inference"] or run["gold_in_inference"] != (v == "gold"):
            raise ValueError("Invalid inference regime")
        for path, expected in run["source_code_files_sha256"].items():
            if file_sha256(path) != expected:
                raise ValueError(f"Inference source changed: {path}")
        predictions[v] = []
        for doc in docs:
            code = doc["document_code"]
            path = directory / f"{code}.json"
            if file_sha256(path) != run["prediction_files_sha256"][code]:
                raise ValueError("Immutable prediction changed")
            p = read_json(path)
            if p["variant"] != v or p["config_sha256"] != fingerprint(cfg):
                raise ValueError("Mixed predictions")
            validate_prediction(p)
            predictions[v].append(p)
    labels = load_labels(cfg, docs)
    out = Path("reports/c_diag_002")
    summary = {}
    for v in variants:
        preds, run = predictions[v], inventories[v]
        result = evaluate(preds, labels)
        audit = loss_audit(preds, labels)
        zero = prf([], [])
        for relation in ontology["relations"]:
            audit["per_relation"].setdefault(relation, dict(raw=zero.copy(), final=zero.copy()))
        examples, fp_categories = examples_and_fp(preds, labels)
        counts = Counter()
        timings = Counter()
        reasons = Counter()
        for pred in preds:
            for c in pred["chunks"]:
                counts.update({k: len(c[k]) for k in ("entities", "candidate_pairs", "raw_proposals", "assertions", "skipped_re_contexts")})
                timings.update(c["stage_ms"])
                for pair in c["candidate_pairs"]:
                    reasons.update(pair["reasons"])
        unsupported_gold = Counter(a["relation_type"] for g in labels.values() for a in g["assertions"]
                                   if a["support_status"] == "SUPPORTED" and cfg["zero_shot_re"]["capabilities"][a["relation_type"]]["status"] == "UNSUPPORTED")
        result.update(architecture_id="C-DIAG-002", variant=v, mode=run["mode"], exploratory=True,
                      gold_entities_diagnostic_only=v == "gold", counts=dict(counts), stage_ms=dict(timings),
                      loss_audit=audit, entity_per_kind=typed_coverage(preds, labels, ontology),
                      abstention_or_rejection_pair_reasons=dict(reasons), final_fp_categories=fp_categories,
                      unsupported_relations=[r for r, cap in cfg["zero_shot_re"]["capabilities"].items() if cap["status"] == "UNSUPPORTED"],
                      unsupported_gold_relations=dict(unsupported_gold), selected_examples=examples,
                      performance={k: run[k] for k in ("wall_ms", "cpu_ms", "peak_ram_bytes", "ner_load_ms", "re_load_ms", "ner_calls", "ner_encoder_calls", "re_calls", "re_encoder_calls")},
                      run_sha256=file_sha256(Path("artifacts/c_diag_002") / v / "run.json"))
        write_json(out / f"{v}.json", result)
        write_json(out / f"{v}_run.json", run)
        write_json(out / f"{v}_access_audit.json", read_json(Path("artifacts/c_diag_002") / v / "access_audit.json"))
        summary[v] = dict(micro=result["micro"], counts=result["counts"], performance=result["performance"],
                          funnel=audit["correct_assertions_remaining"], final_fp_categories=fp_categories)
        print(v, result["micro"]["entity_typed"], result["micro"]["relation_end_to_end"], flush=True)
    # Same-run GLiNER control must reproduce the original baseline exactly.
    baseline = read_json("reports/c_restart_001/dev.json")["micro"]
    for key in ("entity_span", "entity_typed", "relation_raw_end_to_end", "relation_end_to_end", "qualified_relation", "candidate_pair_recall"):
        if summary["control"]["micro"][key] != baseline[key]:
            raise ValueError(f"Control differs from C-RESTART-001: {key}")
    write_json(out / "comparison.json", dict(variants=summary, original_control_reproduced=True,
               protocol_sha256=file_sha256("reports/C_DIAG_002_PROTOCOL.md"), config_sha256=fingerprint(cfg),
               evaluation_script_sha256=file_sha256(__file__), inference_completed_before_gold_scoring=True))


if __name__ == "__main__":
    main()
