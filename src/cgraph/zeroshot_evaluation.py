"""Offline-only DEV scorer and stage-loss audit; never called by extraction."""
from collections import Counter
from pathlib import Path

from .common import file_sha256, fingerprint, read_json, write_json
from .evaluation import entity_key, prf


def loss_audit(predictions, labels):
    totals, per_doc, per_label = Counter(), {}, {}
    rejected, rejected_correct = Counter(), Counter()
    order = ["gold", "untyped_endpoints", "typed_endpoints", "candidate_pairs", "model_served",
             "re_capability", "raw_re", "domain_range", "scope", "evidence", "final"]
    for pred in predictions:
        gold = labels[pred["document_code"]]
        gm = {m["mention_id"]: m for m in gold["mentions"]}
        pm = {m["mention_id"]: dict(m, chunk_id=c["chunk_id"]) for c in pred["chunks"] for m in c["entities"]}

        def edge(a, mentions):
            h, t = mentions[a["head_mention_id"]], mentions[a["tail_mention_id"]]
            return entity_key(h["chunk_id"], h), entity_key(t["chunk_id"], t), a["relation_type"]

        ga = [a for a in gold["assertions"] if a["support_status"] == "SUPPORTED"]
        ge = Counter(edge(a, gm) for a in ga)
        spans = {(e[0], e[1], e[2]) for e in (entity_key(m["chunk_id"], m) for m in pm.values())}
        typed = {entity_key(m["chunk_id"], m) for m in pm.values()}
        pairs = [p for c in pred["chunks"] for p in c["candidate_pairs"]]
        pk = {(entity_key(pm[p["head_mention_id"]]["chunk_id"], pm[p["head_mention_id"]]),
               entity_key(pm[p["tail_mention_id"]]["chunk_id"], pm[p["tail_mention_id"]])) for p in pairs}
        served = {(entity_key(pm[p["head_mention_id"]]["chunk_id"], pm[p["head_mention_id"]]),
                   entity_key(pm[p["tail_mention_id"]]["chunk_id"], pm[p["tail_mention_id"]])) for p in pairs if p["model_served"]}
        raw = [a for c in pred["chunks"] for a in c["raw_proposals"]]
        final = [a for c in pred["chunks"] for a in c["assertions"]]
        stages = {"gold": ge,
                  "untyped_endpoints": Counter({e: n for e, n in ge.items() if e[0][:3] in spans and e[1][:3] in spans}),
                  "typed_endpoints": Counter({e: n for e, n in ge.items() if e[0] in typed and e[1] in typed}),
                  "candidate_pairs": Counter({e: n for e, n in ge.items() if e[:2] in pk}),
                  "model_served": Counter({e: n for e, n in ge.items() if e[:2] in served}),
                  "re_capability": Counter({e: n for e, n in ge.items() if e[:2] in served and e[2] != "MEETS_REQUIREMENT"}),
                  "raw_re": ge & Counter(edge(a, pm) for a in raw),
                  "domain_range": ge & Counter(edge(a, pm) for a in raw if a["domain_range_pass"]),
                  "scope": ge & Counter(edge(a, pm) for a in raw if a["scope_pass"]),
                  "evidence": ge & Counter(edge(a, pm) for a in raw if a["evidence_pass"]),
                  "final": ge & Counter(edge(a, pm) for a in final)}
        counts = {name: sum(stages[name].values()) for name in order}
        if any(counts[b] > counts[a] for a, b in zip(order, order[1:])):
            raise ValueError("Non-monotone loss funnel")
        per_doc[pred["document_code"]] = counts
        totals.update(counts)
        for a in raw:
            if a["reason"]:
                rejected[a["reason"]] += 1
                if edge(a, pm) in ge:
                    rejected_correct[a["reason"]] += 1
        for label in {e[2] for e in ge} | {a["relation_type"] for a in raw}:
            row = per_label.setdefault(label, {name: Counter() for name in ("raw", "final")})
            gold_keys = list((ge & Counter({e: n for e, n in ge.items() if e[2] == label})).elements())
            for name, proposed in (("raw", raw), ("final", final)):
                metrics = prf(gold_keys, [edge(a, pm) for a in proposed if a["relation_type"] == label])
                row[name].update({k: metrics[k] for k in ("tp", "fp", "fn", "gold", "predicted")})
    for row in per_label.values():
        for name, counts in row.items():
            tp, ng, np = counts["tp"], counts["gold"], counts["predicted"]
            row[name] = dict(counts, precision=tp / np if np else None, recall=tp / ng if ng else None,
                             f1=2 * tp / (ng + np) if ng + np else None)
    return dict(order=order, correct_assertions_remaining=dict(totals),
                correct_assertions_lost={b: totals[a] - totals[b] for a, b in zip(order, order[1:])},
                documents=per_doc, per_relation=per_label, rejected_raw_proposals=dict(rejected),
                rejected_exact_gold_raw_proposals=dict(rejected_correct),
                note="Counts match exact typed directed edges; qualified completeness is audited by the separate strict metric")


def run_evaluation(cfg, args):
    from .cli import run_evaluation as historical_evaluate
    from .inputs import load_inputs
    from .labels import load_labels

    directory = Path(args.predictions)
    run = read_json(directory / "run.json")
    if (run.get("architecture_id") != "C-RESTART-001" or run.get("training_performed") is not False
            or run["relation_model"].get("train_documents") != []
            or run["config_sha256"] != fingerprint(cfg)):
        raise ValueError("Run is not the frozen zero-shot experiment")
    docs = load_inputs(cfg, args.subset)
    predictions = []
    for d in docs:
        code = d["document_code"]
        path = directory / f"{code}.json"
        if file_sha256(path) != run["prediction_files_sha256"][code]:
            raise ValueError("Prediction bytes differ from immutable run inventory")
        prediction = read_json(path)
        if prediction["architecture_id"] != "C-RESTART-001" or prediction["config_sha256"] != fingerprint(cfg):
            raise ValueError("Mixed prediction architecture/config")
        predictions.append(prediction)
    # Reuse the frozen exact multiset evaluator and schema/evidence checks.
    historical_evaluate(cfg, args)
    result = read_json(args.output)
    result.update(architecture_id="C-RESTART-001", training_performed=False,
                  independent_of_training_documents=True,
                  inferential_status="Exploratory DEV; zero-shot relative to this dataset, not an untouched confirmatory test",
                  loss_audit=loss_audit(predictions, load_labels(cfg, docs)))
    result["performance"].update({k: run[k] for k in ("re_encoder_forward_calls", "model_api_calls", "ner_load_ms", "re_load_ms")})
    write_json(args.output, result)
    return result["micro"]
