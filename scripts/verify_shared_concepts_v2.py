"""Независимая сверка файлового dry-run с замороженными входами Corpus100."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import json

import numpy as np
from tqdm import tqdm

from build_corpus_graph import read, sha
from cgraph.shared_concepts_v2 import EE, digest
from contextual_shared_concepts import pair_guard_codes, pair_indices
from shared_concepts_v2_pipeline import (
    EXPECTED_EE, FROZEN_MANIFEST, PREVIOUS_RUN, frozen_corpus, load_checked, source_identity,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cosines(left, right):
    return np.einsum("...d,...d->...", left, right) / (np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1))


def verify_ee(entities, ee, frozen):
    manifest = read(FROZEN_MANIFEST)
    scores = []
    for label in ("name", "S1"):
        name = f"{label}_scores.npy"
        require(sha(frozen / name) == manifest["vector_and_score_artifacts"][label]["files"][name],
                f"Historical score checksum mismatch: {label}")
        scores.append(np.load(frozen / name, mmap_mode="r", allow_pickle=False))
    left, right = pair_indices([e.document_id for e in entities])
    mask = (scores[0] >= EE[0]) & (scores[1] >= EE[1])
    require(len(mask) == EXPECTED_EE["cross_document_pairs"] and mask.sum() == EXPECTED_EE["raw_pairs"],
            "Historical raw EE funnel differs")
    keep = pair_guard_codes([e.profile for e in entities], left, right, mask) == 0
    historical = {(entities[i].entity_id, entities[j].entity_id): (float(n), float(c))
                  for i, j, n, c in zip(left[mask][keep], right[mask][keep], scores[0][mask][keep],
                                        scores[1][mask][keep], strict=True)}
    pairs = {(a, b): (n, c) for a, b, n, c in ee["edges"]}
    require(len(pairs) == len(ee["edges"]) and historical.keys() == pairs.keys(), "Guarded EE edge set differs")
    error = max(abs(a - b) for key, actual in pairs.items() for a, b in zip(actual, historical[key], strict=True))
    require(error <= 1e-12, "EE cosine scores differ")
    require({key: ee["metrics"][key] for key in EXPECTED_EE} == EXPECTED_EE, "EE regression counters differ")
    return pairs, error


def verify_rounds(event, entities, ec, delta):
    original = set(event["source"])
    current = original
    final_reason = None
    for row in event["rounds"]:
        require(row["before"] == sorted(current) and final_reason is None, "Invalid EC round sequence")
        vectors = np.asarray([entities[key].vectors for key in sorted(current)])
        expected = cosines(vectors, vectors.sum(axis=0))
        np.testing.assert_allclose(expected, row["scores"], atol=1e-12, rtol=0)
        bad = {key for key, scores in zip(sorted(current), expected, strict=True) if np.any(scores < ec)}
        require(bad == set(row["bad"]), "EC exclusions differ")
        remaining = current - bad
        removed = len(original - remaining)
        require(row["cumulative_removed"] == removed and row["cumulative_fraction"] == removed / len(original),
                "Invalid cumulative exclusion count")
        if not bad:
            final_reason = "PASS"
        elif Fraction(removed, len(original)) > delta:
            final_reason = "EXCLUSION_BUDGET"
        elif len({entities[key].document_id for key in remaining}) < 2:
            final_reason = "MIN_MEMBERS_OR_DOCS"
        else:
            current = remaining
    expected_reason = "PASS" if event["reason"] == "NESTING" else event["reason"]
    require(final_reason == expected_reason and set(event["members"]) == current
            and set(event["excluded"]) == original - current, "EC outcome differs")


def verify_catalog(artifact, edges, entities, exported):
    ec = np.asarray([artifact["identity"]["config"]["ec"][name] for name in ("name", "context")])
    cc = np.asarray([artifact["identity"]["config"]["cc"][name] for name in ("name", "context")])
    delta = Fraction(artifact["identity"]["config"]["delta"])
    local = defaultdict(set)
    for a, b in edges:
        local[a].update((a, b))
        local[b].update((a, b))
    expected = defaultdict(list)
    for seed, group in sorted(local.items()):
        expected[tuple(sorted(group))].append(seed)
    active = {}
    config_hash = artifact["identity"]["config_hash"]
    namespace = artifact["identity"]["namespace"]
    counters = Counter()
    events = artifact["state"]["events"]
    for event in tqdm(events, desc="Audit EC and CC", unit="event"):
        operation = event["operation"]
        if operation not in {"initialize", "merge"}:
            continue
        counters.update([operation])
        verify_rounds(event, entities, ec, delta)
        members = frozenset(event["members"])
        if operation == "initialize":
            require(expected.pop(tuple(event["source"])) == event["seeds"], "Raw neighborhood or seed provenance differs")
            if event["reason"] == "PASS":
                key = "scv2:" + digest([namespace, config_hash, ["initialize", sorted(members)]])
                active[key] = (1, members)
            continue
        first, second = event["parents"]
        a, b = first[0], second[0]
        version_a, set_a = active[a]
        version_b, set_b = active[b]
        union = set_a | set_b
        require(version_a == first[1] and version_b == second[1] and not (set_a <= set_b or set_b <= set_a)
                and set(event["source"]) == union
                and Fraction(len(set_a & set_b), len(union)) == Fraction(event["jaccard"]),
                "Invalid CC versions, eligibility, union or Jaccard")
        vectors = [np.sum([entities[key].vectors for key in sorted(group)], axis=0) for group in (set_a, set_b)]
        pair_scores = cosines(*vectors)
        np.testing.assert_allclose(pair_scores, event["cc_scores"], atol=1e-12, rtol=0)
        require(np.all(pair_scores >= cc), "Attempted merge fails CC thresholds")
        if event["reason"] != "PASS":
            continue
        survivor, new_version = event["survivor"]
        affected = set(event["retired"]) | {survivor}
        old_sets = {key: group for key, (_, group) in active.items()}
        for key in affected & old_sets.keys():
            require(all(
                (not old_sets[key] < group or other not in affected and members < group)
                and (not group < old_sets[key] or other not in affected and group < members)
                for other, group in old_sets.items() if key != other), "Accepted merge breaks strict nesting")
        for key in event["retired"]:
            del active[key]
        active[survivor] = (new_version, members)
        counters.update(["accepted_merges"])
    require(not expected, "Missing raw neighborhoods")
    final_memberships = {}
    catalog = artifact["state"]["concepts"]
    require(len(catalog) == len({tuple(c["members"]) for c in catalog}) == len(active), "Duplicate or missing concepts")
    for row in tqdm(catalog, desc="Audit final memberships", unit="concept"):
        cid, members = row["concept_id"], row["members"]
        require(active[cid] == (row["version"], frozenset(members)), "Catalog differs from event replay")
        require(len(members) == row["member_count"] and digest(members) == row["member_fingerprint"],
                "Invalid member count or fingerprint")
        documents = sorted({entities[key].document_id for key in members})
        require(row["documents"] == documents and len(documents) >= 2, "Invalid concept document coverage")
        vectors = np.asarray([entities[key].vectors for key in members])
        sums = vectors.sum(axis=0)
        np.testing.assert_allclose(sums, row["sums"], atol=1e-12, rtol=0)
        scores = cosines(vectors, sums)
        np.testing.assert_allclose(scores, row["scores"], atol=1e-12, rtol=0)
        require(np.all(scores >= ec), "Final membership fails EC")
        best = members.index(row["representative"])
        require(entities[members[best]].name == row["name"] and scores[best, 0] >= max(scores[:, 0]) - 1e-12,
                "Incorrect concept representative")
        for key, score in zip(members, row["scores"], strict=True):
            final_memberships[key, cid] = (row["version"], score, row["events"])
    for row in exported:
        key, cid = row["entity_id"], row["concept_id"]
        concept_version, scores, lineage = final_memberships.pop((key, cid))
        require(row["relation"] == "HAS_SHARED_CONCEPT" and row["generation"] == artifact["generation"]
                and row["config_hash"] == config_hash and row["concept_version"] == concept_version
                and not any(row[name] for name in ("factual", "identity", "fact_inheritance_allowed"))
                and row["evidence"] == entities[key].profile["evidence"] and row["events"] == lineage
                and row["profile_fingerprint"] == entities[key].fingerprint
                and [row["scores"][name] for name in ("name", "context")] == scores, "Invalid membership export")
    require(not final_memberships, "Missing exported memberships")
    nested = sum(a < b or b < a for (_, a), (_, b) in combinations(active.values(), 2))
    require(nested == artifact["statistics"]["strict_inclusions"], "Incorrect final nesting count")
    return dict(counters, concepts=len(catalog), memberships=len(exported), strict_inclusions=nested)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("catalog_folder", type=Path)
    parser.add_argument("--frozen-run", type=Path, default=PREVIOUS_RUN)
    args = parser.parse_args()
    print("ПРЕДУПРЕЖДЕНИЕ: ПРОВЕРКА CORPUS100 МОЖЕТ ЗАНЯТЬ НЕСКОЛЬКО МИНУТ.", flush=True)
    folder = args.catalog_folder
    artifact = load_checked(folder / "catalog.json")
    require(artifact["identity"]["source"] == source_identity(), "Current source differs from run")
    entities, metadata = frozen_corpus(args.frozen_run)
    require(artifact["identity"]["input"] == metadata, "Frozen data differs from run")
    require(sha(folder / "ee.json") == artifact["ee_artifact_sha256"], "EE artifact checksum mismatch")
    require(sha(folder / "memberships.json") == artifact["memberships_sha256"], "Membership artifact checksum mismatch")
    pairs, error = verify_ee(entities, load_checked(folder / "ee.json"), args.frozen_run)
    metrics = verify_catalog(artifact, pairs, {e.entity_id: e for e in entities},
                             load_checked(folder / "memberships.json")["memberships"])
    print(json.dumps(dict(status="PASS", generation=artifact["generation"], ee_max_error=error,
                          historical_cases=len(artifact["historical_cases"]), **metrics), sort_keys=True))


if __name__ == "__main__":
    main()
