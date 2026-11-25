"""Corpus100 diagnostics, review samples and published aggregates."""
from ast import literal_eval
from collections import Counter
import csv
from itertools import product
import json
import os
import time

import numpy as np
import requests
from tqdm import tqdm

from build_corpus_graph import atomic_json, read, sha, signature
from build_relex_graphs import digest
from contextual_shared_concepts import PREFIX, RULES, SEED, THRESHOLDS, WINDOWS, fold, guard_reasons, pair_guard_codes

ANCHOR = (.85, .85)
REVIEW_WINDOWS = ("S0", "S2", "W16", "W32", "W128")


def atomic_csv(path, rows):
    def cell(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value

    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: cell(value) for key, value in row.items()} for row in rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def pair_position(i, j, left, right):
    i, j = sorted((i, j))
    lo, hi = left.searchsorted([i, i+1])
    part = right[lo:hi]
    pos = int(part.searchsorted(j))
    if pos >= len(part) or part[pos] != j:
        raise ValueError("Missing cross-document entity pair")
    return int(lo+pos)


def pair_evidence(entity):
    keys = ("entity_id", "name", "document_id", "topic", "entity_types", "evidence")
    return {k: entity[k] for k in keys}


def map_diagnostic(rows, entities, shared, left, right):
    lexical = [c for c in shared["concepts"] if c["concept_id"].startswith("concept:2g:")]
    result = []
    for row in rows:
        a_types, b_types = literal_eval(row["a_types"]), literal_eval(row["b_types"])
        a = [i for i, e in enumerate(entities) if (e["name"], e["document_id"], e["entity_types"])
             == (row["a"], row["left"], a_types)]
        b = [i for i, e in enumerate(entities) if (e["name"], e["document_id"], e["entity_types"])
             == (row["b"], row["right"], b_types)]
        matches = []
        for i, j in product(a, b):
            aid, bid = entities[i]["entity_id"], entities[j]["entity_id"]
            if row["kind"] == "entity_entity":
                cid = "ee:" + digest(["text_name", *sorted((aid, bid))])[:24]
                if cid == row["candidate_id"]:
                    matches.append((i, j, None))
            else:
                for concept in lexical:
                    cid = concept["concept_id"]
                    if ("ec:" + digest(["text_name", aid, cid])[:24] == row["candidate_id"]
                            and bid in concept["contextual_target_ids"]):
                        matches.append((i, j, concept))
        if not matches:
            raise ValueError(f"Unmapped diagnostic {row['candidate_id']}")
        # Исторический evaluate выбирал первый ID среди отклонённых междокументных референтов.
        i, j, concept = min(matches, key=lambda m: entities[m[1]]["entity_id"])
        if concept:
            reference = min(e["entity_id"] for e in entities
                            if e["document_id"] != entities[i]["document_id"]
                            and e["entity_id"] in concept["contextual_target_ids"])
            if entities[j]["entity_id"] != reference:
                raise ValueError(f"Historical reference order differs for {row['candidate_id']}")
        result.append((pair_position(i, j, left, right), i, j, row, concept, len(matches)))
    return result


def read_scores(output, label):
    return np.load(output/f"{label}_scores.npy", mmap_mode="r", allow_pickle=False)


def pair_results(positions, name_scores, output, windows):
    contexts = {w: read_scores(output, w)[positions].tolist() for w in windows}
    return name_scores[positions].tolist(), contexts


def packed_candidates(entities, left, right, name_scores, output, checks):
    name_domain = name_scores >= THRESHOLDS[0]
    indices = np.flatnonzero(name_domain).astype(np.int32)
    guard = pair_guard_codes(entities, left, right, name_domain)
    files = {}
    for window, check in tqdm(checks.items(), desc="Export candidate pairs", unit="window"):
        context = read_scores(output, window)
        eligible = context[indices] >= THRESHOLDS[0]
        pairs, rules = indices[eligible], guard[eligible]
        expected = check["rows"][0]
        if len(pairs) != expected["pure_pairs"] or np.count_nonzero(rules == 0) != expected["guarded_pairs"]:
            raise ValueError(f"Candidate export differs from the completed sweep: {window}")
        path = output/f"{window}_candidates.npz"
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as stream:
            np.savez(stream, cross_pair_index=pairs, guard_mask=rules)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        files[path.name] = sha(path)
    return dict(files_sha256=files, index="Zero-based upper triangle in entity ID order with same-document pairs removed",
                columns=["cross_pair_index", "guard_mask"], mask_bits={r: 1 << i for i, r in enumerate(RULES)},
                minimum_thresholds=[THRESHOLDS[0], THRESHOLDS[0]], absent="REJECT on every grid point")


def diagnostic_rows(mapped, entities, name_scores, output, windows):
    positions = np.array([row[0] for row in mapped])
    names, contexts = pair_results(positions, name_scores, output, windows)
    rows = []
    for n, (_, i, j, historical, concept, matches) in enumerate(mapped):
        a, b = entities[i], entities[j]
        reasons = guard_reasons(a, b)
        row = dict(index=int(historical["n"]), old_candidate_id=historical["candidate_id"],
                   old_kind=historical["kind"], old_cosine=float(historical["score"]),
                   old_reason=historical["reason"],
                   old_target_concept_id=concept["concept_id"] if concept else "",
                   old_target_concept_name=concept["name"] if concept else "", matching_ref_count=matches,
                   new_kind="entity_entity", name_cosine=names[n], guard_reasons=reasons,
                   a_expansions=a["expansions"], b_expansions=b["expansions"],
                   **{f"{side}_{key}": value for side, entity in (("a", a), ("b", b))
                      for key, value in pair_evidence(entity).items()})
        for window in windows:
            context = contexts[window][n]
            row[f"{window}_cosine"] = context
            row[f"{window}_pure"] = {f"{tn:.2f}/{tc:.2f}": "PASS" if names[n] >= tn and context >= tc else "REJECT"
                                    for tn, tc in product(THRESHOLDS, repeat=2)}
            row[f"{window}_guarded"] = {key: "REJECT" if reasons else value
                                       for key, value in row[f"{window}_pure"].items()}
        rows.append(row)
    return rows


def review_sample(data, entities, left, right, mapped, name_scores, output, windows):
    rng = np.random.default_rng(SEED)
    by_id = {e["entity_id"]: i for i, e in enumerate(entities)}
    by_name = np.array([fold(e["name"]) for e in entities], dtype=object)
    literal_names = np.array([e["name"] for e in entities], dtype=object)
    topics = np.array([e["topic"] for e in entities], dtype=object)
    strata = {}

    def add(positions, label):
        for p in positions:
            strata.setdefault(int(p), set()).add(label)

    def draw(mask, label, size=8):
        positions = np.flatnonzero(mask)
        add(rng.choice(positions, min(size, len(positions)), replace=False), label)

    add((m[0] for m in mapped), "hybrid_rejected_diagnostic")
    accepted = set()
    for membership in data["shared"]["memberships"]:
        evidence = membership["evidence"]
        i = by_id[membership["entity_id"]]
        j = by_id.get(evidence.get("reference_entity_id"))
        if j is not None and entities[i]["document_id"] != entities[j]["document_id"]:
            accepted.add(pair_position(i, j, left, right))
    candidates = sorted(accepted)
    add(rng.choice(candidates, min(16, len(candidates)), replace=False), "hybrid_accepted_membership_reference")

    exact = by_name[left] == by_name[right]
    draw(literal_names[left] == literal_names[right], "exact_literal_name", 16)
    cross_topic = topics[left] != topics[right]
    draw(exact, "exact_name", 16)
    short = read_scores(output, windows[0])
    draw(exact & (short < .85), "exact_name_low_context", 12)
    draw(exact & cross_topic, "same_name_different_topics", 12)
    draw((name_scores < .75) & (short < .75) & cross_topic, "negative_control_low_both", 16)
    for window in REVIEW_WINDOWS:
        if window not in windows:
            continue
        context = read_scores(output, window)
        draw((name_scores >= .90) & (context < .80), f"{window}_high_name_low_context")
        draw((name_scores < .80) & (context >= .90), f"{window}_low_name_high_context")
        draw(~exact & (name_scores >= .90) & (context >= .90), f"{window}_different_name_high_both")
    if "W128" in windows and "S0" in windows:
        wide = read_scores(output, "W128")
        draw((name_scores >= .80) & (short < .80) & (wide >= .90), "context_rise_at_W128", 24)
    anchor = read_scores(output, "W32" if "W32" in windows else windows[0])
    for topic in sorted(set(topics)):
        within = (topics[left] == topic) & (topics[right] == topic)
        draw(within & (name_scores >= ANCHOR[0]) & (anchor >= ANCHOR[1]), f"within_topic:{topic}", 12)

    positions = rng.permutation(sorted(strata))
    names, contexts = pair_results(positions, name_scores, output, windows)
    annotated, blind = [], []
    for n, position in enumerate(positions):
        i, j = left[position], right[position]
        a, b = entities[i], entities[j]
        if rng.integers(2):
            a, b = b, a
        visible = dict(review_id=f"pair-{n+1:03}", a=pair_evidence(a), b=pair_evidence(b))
        reasons = guard_reasons(a, b)
        selected = {w: contexts[w][n] for w in windows}
        blind.append(visible | dict(label="", notes=""))
        annotated.append(visible | dict(
            sampling_strata=sorted(strata[int(position)]), name_cosine=names[n],
            context_cosine=selected, guard_reasons=reasons,
            pass_anchor={w: dict(pure=names[n] >= ANCHOR[0] and score >= ANCHOR[1],
                                  guarded=not reasons and names[n] >= ANCHOR[0] and score >= ANCHOR[1])
                         for w, score in selected.items()}))
    atomic_csv(output/"MANUAL_REVIEW_SCORED.csv", annotated)
    atomic_csv(output/"MANUAL_REVIEW_BLIND.csv", blind)
    return dict(seed=SEED, size=len(blind), strata=dict(Counter(s for row in annotated for s in row["sampling_strata"])),
                strategy="Uniform without replacement per stratum, overlap deduplicated, seeded row and side order",
                review_windows=[w for w in REVIEW_WINDOWS if w in windows], anchor=list(ANCHOR))


def figures(root, checks):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm

    root.mkdir(parents=True, exist_ok=True)
    high = max(2, max(r["pure_pairs"] for check in checks.values() for r in check["rows"]))
    for window, check in tqdm(checks.items(), desc="Heatmaps", unit="window"):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), layout="constrained")
        for ax, key, title in zip(axes, ("pure_pairs", "guarded_pairs"), ("Two thresholds", "With guards")):
            values = np.array([r[key] for r in check["rows"]]).reshape(5, 5)
            pic = ax.imshow(values+1, cmap="Blues", norm=LogNorm(vmin=1, vmax=high+1))
            for (i, j), count in np.ndenumerate(values):
                color = "white" if pic.norm(count+1) >= .65 else "#152946"
                ax.text(j, i, f"{count:,}", color=color, ha="center", va="center", fontsize=8.5)
            labels = [f"{v:.2f}" for v in THRESHOLDS]
            ax.set(xticks=range(5), xticklabels=labels, yticks=range(5), yticklabels=labels,
                   xlabel="Context cosine threshold", ylabel="Name cosine threshold", title=title)
        fig.suptitle(f"Corpus100 · {window} · cross-document candidate pairs", fontsize=13)
        path = root/f"{window}.png"
        tmp = path.with_suffix(".tmp")
        fig.savefig(tmp, format="png", dpi=144)
        plt.close(fig)
        os.replace(tmp, path)


def observed_window_sizes(output, windows, model):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(model/"tokenizer.json"))
    result = {}
    for window in tqdm(windows, desc="Measure window lengths", unit="window"):
        contexts = read(output/f"{window}_windows.json")
        entries = [c for entity in contexts for c in entity["contexts"]]
        words = [len(c["text"].split()) for c in entries]
        tokens = [len(tokenizer.encode(PREFIX+c["input"]).ids) for c in entries]
        result[window] = dict(context_words_median=float(np.median(words)),
                              context_e5_tokens_median=float(np.median(tokens)),
                              context_e5_tokens_p95=float(np.percentile(tokens, 95)),
                              context_e5_tokens_max=max(tokens))
    return result


def track(uri, fingerprint, manifest, checks):
    if not uri:
        return dict(status="DISABLED")
    base = uri.rstrip("/") + "/api/2.0/mlflow/"

    def call(method, route, **kwargs):
        response = requests.request(method, base+route, timeout=(4, 12), **kwargs)
        response.raise_for_status()
        return response.json()

    try:
        exp = call("GET", "experiments/get-by-name", params={"experiment_name": "data_expansion"})
        exp_id = exp["experiment"]["experiment_id"]
        external = f"3a5_12-{fingerprint}"
        runs = call("POST", "runs/search", json=dict(
            experiment_ids=[exp_id], filter=f"tags.external_id = '{external}'", max_results=2)).get("runs", [])
        if len(runs) > 1:
            raise ValueError("Duplicate MLflow external_id")
        now = int(time.time()*1000)
        if runs:
            run = runs[0]
        else:
            run = call("POST", "runs/create", json=dict(experiment_id=exp_id, start_time=now,
                        run_name="Stage 3A.5.12 Corpus100 contextual Shared Concepts",
                        tags=[dict(key="external_id", value=external)]))["run"]
        run_id = run["info"]["run_id"]
        if run["info"]["status"] != "FINISHED":
            params = dict(model_revision=manifest["model_revision"], windows=str(WINDOWS),
                          threshold_name=str(THRESHOLDS), threshold_context=str(THRESHOLDS), seed=str(SEED),
                          entity_ids_sha256=manifest["entity_ids_sha256"])

            def log(values, step, parameters):
                call("POST", "runs/log-batch", json=dict(
                    run_id=run_id, params=[dict(key=k, value=v) for k, v in parameters.items()],
                    metrics=[dict(key=k, value=v, timestamp=now, step=step) for k, v in values.items()]))

            log(manifest["counts"] | {k: manifest["runtime"][k] for k in
                ("unique_context_inputs", "cache_hits", "inference_seconds", "aggregation_seconds",
                 "cosine_sweep_seconds", "peak_rss_bytes")} | {
                    "unique_name_inputs": manifest["name"]["unique_inputs"],
                    "reused_name_entity_ids": manifest["name"]["reused_entity_ids"],
                }, 0, params)
            keys = ("unique_inputs", "cache_hits", "truncated_windows", "inference_seconds",
                    "aggregation_seconds", "cosine_sweep_seconds", "peak_rss_bytes")
            log({f"{w}_{k}": c[k] for w, c in checks.items() for k in keys}, 0, {})
            for step, (w, row) in enumerate(((w, r) for w, c in checks.items() for r in c["rows"]), 1):
                values = {k: row[k] for k in ("threshold_name", "threshold_context", "passes_name", "passes_context",
                                              "pure_pairs", "guarded_pairs")}
                for variant in ("pure", "guarded"):
                    measured = row[variant]
                    values.update({f"{variant}_{k}": v for k, v in measured.items() if isinstance(v, (int, float))})
                    for topic_index, dim in enumerate(measured["topics"].values()):
                        values.update({f"{variant}_topic_{topic_index+1}_{k}": v for k, v in dim.items()})
                    for topic_index, count in enumerate(measured["cross_topics"].values()):
                        values[f"{variant}_cross_topic_{topic_index+1}"] = count
                log({f"{w}_{k}": v for k, v in values.items()}, step, {})
            call("POST", "runs/update", json=dict(run_id=run_id, status="FINISHED", end_time=int(time.time()*1000)))
        return dict(status="PASS", experiment_id=exp_id, run_id=run_id)
    except (requests.RequestException, ValueError, KeyError) as error:
        # Полный набор локальных результатов уже записан.
        response = getattr(error, "response", None)
        return dict(status="BLOCKED", error=type(error).__name__,
                    http_status=response.status_code if response is not None else None)


def publish(args, data, identity, name, entities, left, right, output, report, sampler, started):
    from benchmark_contextual_shared_concepts import checked_output

    fingerprint = signature(identity)
    checks = {}
    for window in tqdm(WINDOWS, desc="Prepare report", unit="window"):
        result = checked_output(output/f"{window}.json", fingerprint,
                                (f"{window}_windows.json", f"{window}_vectors.npy", f"{window}_scores.npy"))
        if result:
            checks[window] = result
    root = report if len(checks) == len(WINDOWS) and not args.smoke else output
    root.mkdir(parents=True, exist_ok=True)
    prior = read(root/"MANIFEST.json") if (root/"MANIFEST.json").exists() else {}
    if not args.smoke:
        from benchmark_contextual_shared_concepts import ROOT

        if any(sha(ROOT/path) != expected for path, expected in data["hashes"].items()):
            raise ValueError("Frozen inputs changed during the run")
    rows = [r for check in checks.values() for r in check["rows"]]
    atomic_csv(root/"THRESHOLD_SWEEP.csv", rows)
    window_rows = [dict(window="name", **name["distribution"])]
    sizes = observed_window_sizes(output, checks, args.model)
    for w, check in checks.items():
        window_rows.append(dict(window=w, **check["distribution"], **sizes[w], **{
            k: v for k, v in check.items() if k not in ("distribution", "fingerprint", "rows", "files")}))
    atomic_csv(root/"WINDOW_COMPARISON.csv", window_rows)
    figures(root/"figures", checks)
    manifest = dict(identity, fingerprint=fingerprint, windows_completed=list(checks), configurations=len(rows),
                    source_git=read(output/"git_source.json"),
                    report_code_sha256=sha(__file__),
                    counts=data["counts"] if not args.smoke else dict(entities=len(entities), cross_document_pairs=len(left)),
                    name=name, window_checkpoints={w: dict(files=c["files"], checksum=signature(c)) for w, c in checks.items()},
                    frozen_graph_fingerprint=data["baseline"]["graph_fingerprint"],
                    source_corpus_commit="750d03a801fd771879b956cdc32324404e72ce57",
                    frozen_inputs_unchanged="PASS" if not args.smoke else "NOT RUN",
                    mlflow_topic_keys=dict(topics=list(rows[0]["pure"]["topics"]),
                                           cross_topics=list(rows[0]["pure"]["cross_topics"])),
                    comparison_anchor=dict(name=ANCHOR[0], context=ANCHOR[1]),
                    runtime=dict(wall_seconds_this_invocation=time.perf_counter()-started,
                                 cumulative_window_seconds=sum(c["seconds"] for c in checks.values()),
                                 peak_rss_bytes=max(sampler.peak, *(c["peak_rss_bytes"] for c in checks.values())),
                                 name_cosine_seconds=name["seconds"], unique_context_inputs=sum(c["encoded"] for c in checks.values()),
                                 cache_hits=sum(c["cache_hits"] for c in checks.values()),
                                 **{k: sum(c[k] for c in checks.values()) for k in
                                    ("inference_seconds", "aggregation_seconds", "cosine_sweep_seconds")}),
                    mlflow=dict(status="NOT RUN"))
    if not args.smoke:
        name_scores = read_scores(output, "name")
        mapped = map_diagnostic(data["diagnostic"], entities, data["shared"], left, right)
        atomic_csv(root/"DIAGNOSTIC_CASES.csv",
                   diagnostic_rows(mapped, entities, name_scores, output, list(checks)))
        manifest["manual_review"] = review_sample(data, entities, left, right, mapped, name_scores, output, list(checks))
        baseline = data["baseline"]
        memberships = data["shared"]["memberships"]
        baseline_entities = {r["entity_id"] for r in memberships}
        baseline_docs = {r["document_id"] for r in memberships}
        baseline_topics = {t: {k: len({e[id_key] for e in entities if e["topic"] == t and e[id_key] in universe})
                              for k, id_key, universe in (("entities", "entity_id", baseline_entities),
                                                          ("documents", "document_id", baseline_docs))}
                           for t in sorted({e["topic"] for e in entities})}
        manifest["hybrid_c"] = dict(threshold=.83, **data["baseline_summary"], **baseline["counts"],
                                     unique_membership_entities=len(baseline_entities),
                                     unique_membership_documents=len(baseline_docs), topic_membership_coverage=baseline_topics)
        comparisons = [dict(method="Hybrid C", threshold_name=.83, unit="final memberships", concepts=len(data["shared"]["concepts"]),
                            count=len(memberships), entities=len(baseline_entities),
                            entity_share=len(baseline_entities)/len(entities), documents=len(baseline_docs),
                            document_share=len(baseline_docs)/len(data["documents"]), topics=baseline_topics)]
        semantic = data["baseline_summary"]
        comparisons.extend(dict(method="Hybrid C", threshold_name=.83, unit=unit, count=amount) for unit, amount in
                           (("entity/entity above cosine threshold", semantic["threshold_ee"]),
                            ("entity/concept above cosine threshold", semantic["threshold_ec"]),
                            ("entity/entity plus entity/concept after top-k and >=0.95 preservation",
                             baseline["counts"].get("hybrid_candidates", data["shared"]["hybrid_candidates"]))))
        for row in rows:
            if (row["threshold_name"], row["threshold_context"]) == ANCHOR:
                for variant in ("pure", "guarded"):
                    comparisons.append(dict(method=f"{row['window']} {variant}", threshold_name=ANCHOR[0],
                                            threshold_context=ANCHOR[1], unit="entity/entity candidate pairs",
                                            count=row[f"{variant}_pairs"], **row[variant]))
        atomic_csv(root/"COMPARISON.csv", comparisons)
        manifest["sample_files_sha256"] = {p.name: sha(p) for p in output.glob("MANUAL_REVIEW_*.csv")}
        manifest["candidate_exports"] = packed_candidates(entities, left, right, name_scores, output, checks)
    manifest["runtime"]["wall_seconds_this_invocation"] = time.perf_counter()-started
    manifest["runtime"]["peak_rss_bytes"] = max(manifest["runtime"]["peak_rss_bytes"], sampler.peak)
    if prior.get("fingerprint") == fingerprint and prior.get("windows_completed") == list(checks):
        prior_runtime = prior["runtime"]
        manifest["runtime"]["original_wall_seconds"] = prior_runtime.get(
            "original_wall_seconds", prior_runtime["wall_seconds_this_invocation"])
        manifest["runtime"]["original_peak_rss_bytes"] = prior_runtime.get(
            "original_peak_rss_bytes", prior_runtime["peak_rss_bytes"])
    results = [root/"THRESHOLD_SWEEP.csv", root/"WINDOW_COMPARISON.csv",
               *(root/"figures"/f"{w}.png" for w in checks)]
    if not args.smoke:
        results.extend(root/name for name in ("COMPARISON.csv", "DIAGNOSTIC_CASES.csv"))
    manifest["published_results_sha256"] = {p.relative_to(root).as_posix(): sha(p) for p in results}
    atomic_json(root/"MANIFEST.json", manifest)
    if root == report:
        manifest["mlflow"] = track(args.mlflow_uri, fingerprint, manifest, checks)
        atomic_json(root/"MANIFEST.json", manifest)
    print(json.dumps(dict(status="PASS", windows=list(checks), configurations=len(rows),
                          peak_rss_bytes=manifest["runtime"]["peak_rss_bytes"],
                          mlflow=manifest["mlflow"]), ensure_ascii=False), flush=True)
