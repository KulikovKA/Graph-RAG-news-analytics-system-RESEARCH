"""Сборка и пополнение локального графа RESEARCH с утверждённым Shared Concepts V2."""

from argparse import ArgumentParser
import json
from pathlib import Path

from filelock import FileLock

import corpus_graph_payload as graph
from build_corpus_graph import sha, stamp
from cgraph.concepts import class_form
from cgraph.shared_concepts_v2 import DIMENSION, digest
from corpus_graph_v2_inputs import (
    APPROVED, BOOTSTRAP_EDGES, BOOTSTRAP_MEMBERS_SHA256, BOOTSTRAP_NODES, Inputs,
    approved_config, delta_identity, delta_inputs, frozen_inputs,
)
import corpus_graph_v2_neo4j as publication
from run_shared_concepts_v2 import limits, resource_options
from shared_concepts_v2_pipeline import (
    RESEARCH_CATALOG_NAMESPACE, RESEARCH_GRAPH_NAMESPACE as NAMESPACE, ROOT, load_checked,
    read_entity_bundle, run_catalog, write_checked, write_entity_bundle,
)

OUTPUT = ROOT / "artifacts" / f"{NAMESPACE}_no_similar_to"
CATALOG = "catalog"


def stored_inputs(folder, request):
    seal = load_checked(folder / "inputs.json")
    if seal["request"] != request or set(seal["files_sha256"]) != {"source.json", "entities.json"}:
        raise ValueError("Prepared inputs belong to another graph request")
    for name, checksum in seal["files_sha256"].items():
        if sha(folder / name) != checksum:
            raise ValueError(f"Prepared input checksum changed: {name}")
    return Inputs(load_checked(folder / "source.json"), read_entity_bundle(folder / "entities.json"), seal["metadata"])


def prepare_inputs(folder, request, prepare):
    if (folder / "inputs.json").exists():
        return stored_inputs(folder, request)
    inputs = prepare()
    write_checked(folder / "source.json", inputs.source)
    write_entity_bundle(folder / "entities.json", inputs.entities)
    write_checked(folder / "inputs.json", dict(
        request=request, metadata=inputs.metadata,
        files_sha256={name: sha(folder / name) for name in ("source.json", "entities.json")}))
    return inputs


def concept_projection(inputs, catalog, memberships):
    if catalog["generation"] != memberships["generation"] or catalog["status"] != "complete":
        raise ValueError("Catalog and memberships generation differs")
    by_id = {e.entity_id: e for e in inputs.entities}
    if ({key: e.fingerprint for key, e in by_id.items()} != catalog["fingerprints"]
            or by_id.keys() != inputs.source["resolution"]["entities"].keys()):
        raise ValueError("Catalog canonical IDs or vector provenance differ from ER")
    concepts, expected = [], {}
    events = catalog["state"]["events"]
    config_hash = catalog["identity"]["config_hash"]
    occupied, member_sets = set(), set()
    for row in catalog["state"]["concepts"]:
        cid = row["concept_id"]
        members = row["members"]
        count = len(members)
        documents = sorted({by_id[key].document_id for key in members})
        representative = row["representative"]
        if (cid in occupied or tuple(members) in member_sets or not cid.startswith("scv2:") or count != row["member_count"]
                or members != sorted(set(members)) or row["member_fingerprint"] != digest(members)
                or documents != row["documents"] or len(documents) < 2
                or representative not in members or by_id[representative].name != row["name"]
                or len(row["sums"]) != 2 or any(len(s) != DIMENSION for s in row["sums"])):
            raise ValueError(f"Inconsistent V2 concept: {cid}")
        occupied.add(cid)
        member_sets.add(tuple(members))
        name_sum, context_sum = row["sums"]
        parent_versions = [events[event]["parents"] for event in row["events"]
                           if events[event]["operation"] == "merge"]
        concepts.append(dict(
            concept_id=cid, name=row["name"], normalized_name=class_form(row["name"]),
            version=row["version"], member_ids=members, member_count=count, document_ids=documents,
            member_fingerprint=row["member_fingerprint"], representative_entity_id=representative,
            name_centroid=[value/count for value in name_sum], context_centroid=[value/count for value in context_sum],
            name_vector_sum=name_sum, context_vector_sum=context_sum, seed_ids=row["seeds"],
            lineage_parent_versions=parent_versions, event_ids=row["events"],
            generation=catalog["generation"], config_hash=config_hash,
        ))
        for key, score in zip(members, row["scores"], strict=True):
            expected[key, cid] = (row["version"], tuple(score), row["events"])
    rows = []
    for item in memberships["memberships"]:
        key, cid = item["entity_id"], item["concept_id"]
        variant = expected.pop((key, cid), None)
        evidence = by_id[key].profile["evidence"]
        score = item["scores"]["name"], item["scores"]["context"]
        if (variant != (item["concept_version"], score, item["events"])
                or item["relation"] != "HAS_SHARED_CONCEPT"
                or item["generation"] != catalog["generation"] or item["config_hash"] != config_hash
                or item["profile_fingerprint"] != by_id[key].fingerprint or item["evidence"] != evidence
                or any(item[flag] is not False for flag in ("factual", "identity", "fact_inheritance_allowed"))):
            raise ValueError(f"Invalid exported V2 membership: {key}, {cid}")
        rows.append(item | dict(membership_id="scv2-membership:" + digest([key, cid]),
                                document_id=by_id[key].document_id,
                                name_similarity=score[0], context_similarity=score[1],
                                representative_source="shared_concepts_v2"))
    if expected:
        raise ValueError("File catalog has memberships absent from its export")
    return dict(concepts=concepts, memberships=rows)


def assemble(inputs, catalog, memberships, catalog_sha256, *, parent=None, frozen=False):
    source = inputs.source
    if parent:
        anchor = parent["anchor"]
        fingerprint = parent["fingerprint"]
    else:
        anchor = NAMESPACE + ":" + min(source["documents"])
        fingerprint = digest([NAMESPACE, catalog["identity"]["config_hash"], inputs.metadata])
    projected = concept_projection(inputs, catalog, memberships)
    nodes, edges = graph.build_payload(NAMESPACE, fingerprint, source["documents"],
                                       source["resolution"], (), projected)
    nodes[anchor]["properties"].update(scv2_generation=catalog["generation"], scv2_catalog_sha256=catalog_sha256)
    expected = publication.count_payload(nodes, edges)
    if frozen:
        actual_members = digest(sorted(row["member_ids"] for row in projected["concepts"]))
        if (expected != dict(nodes=BOOTSTRAP_NODES, relationships=BOOTSTRAP_EDGES)
                or actual_members != BOOTSTRAP_MEMBERS_SHA256):
            raise ValueError(f"Corpus100 V2 differs from the approved study: counts={expected}, "
                             f"membership_sha256={actual_members}")
    return dict(nodes=nodes, edges=edges, expected=expected, anchor=anchor, fingerprint=fingerprint,
                generation=catalog["generation"], catalog_sha256=catalog_sha256)


def saved_payload(folder, expected=None):
    checksum = sha(folder / "payload.json")
    if expected is not None and checksum != expected["payload_sha256"]:
        raise ValueError("Graph payload checksum differs from CURRENT")
    payload = load_checked(folder / "payload.json")
    catalog_path = folder / CATALOG / "catalog.json"
    if (payload["catalog_sha256"] != sha(catalog_path)
            or expected is not None and payload["generation"] != expected["generation"]):
        raise ValueError("Graph payload and file catalog are inconsistent")
    catalog = load_checked(catalog_path)
    anchor = payload["nodes"][payload["anchor"]]["properties"]
    if (catalog["generation"] != payload["generation"] or anchor["scv2_generation"] != payload["generation"]
            or anchor["scv2_catalog_sha256"] != payload["catalog_sha256"]
            or sha(catalog_path.with_name("memberships.json")) != catalog["memberships_sha256"]
            or sha(catalog_path.with_name("ee.json")) != catalog["ee_artifact_sha256"]):
        raise ValueError("Graph payload is detached from its catalog, EE or membership export")
    return payload


def build_prepared(folder, inputs, config, options, parent_folder, parent_payload):
    output = folder / CATALOG
    base = load_checked(parent_folder / CATALOG / "catalog.json") if parent_folder else None
    if (folder / "payload.json").exists():
        return saved_payload(folder)
    run_catalog(inputs.entities, config, RESEARCH_CATALOG_NAMESPACE, output,
                metadata=inputs.metadata, base=base, regression=base is None, **options)
    catalog = load_checked(output / "catalog.json")
    if sha(output / "memberships.json") != catalog["memberships_sha256"]:
        raise ValueError("File catalog membership export checksum mismatch")
    payload = assemble(inputs, catalog, load_checked(output / "memberships.json"),
                       sha(output / "catalog.json"), parent=parent_payload, frozen=base is None)
    write_checked(folder / "payload.json", payload)
    return payload


def run_graph(driver, target, output, config, options, *, delta=None, model=None,
              workers=publication.BOOTSTRAP_WORKERS):
    """Файловый указатель меняется после точного подтверждения Neo4j."""
    output.mkdir(parents=True, exist_ok=True)
    current_path = output / "CURRENT.json"
    pending_path = output / "PENDING.json"
    input_plan = delta_identity(delta) if delta is not None else None
    request = digest(dict(target=target.record(), config=sha(APPROVED), delta=input_plan,
                          relationship_types=publication.V2_RELATIONSHIP_TYPES))
    with FileLock(output / ".lock", timeout=0):
        current = load_checked(current_path) if current_path.exists() else None
        pending = load_checked(pending_path) if pending_path.exists() else None
        if pending is not None and pending["request"] != request:
            raise ValueError("Resume the pending graph generation using its original command")
        if current and current["target"] != target.record():
            raise ValueError("The local catalog belongs to another Neo4j target")
        generations = output / "generations"
        if current and current["request"] == request:
            payload = saved_payload(generations / current["request"], current)
            with driver.session(database=target.database, default_access_mode=publication.READ_ACCESS) as session:
                verified = session.execute_read(publication.verify, payload)
            pending_path.unlink(missing_ok=True)
            return dict(status="PASS", mutations=0, **verified)
        if (current is None) != (delta is None):
            raise ValueError("Build Corpus100 first, then publish only new documents supplied by --delta")
        with driver.session(database=target.database, default_access_mode=publication.READ_ACCESS) as session:
            session.execute_read(publication.isolated)
        if current and input_plan["base_generation"] != current["generation"]:
            raise ValueError("Delta parent generation differs from CURRENT")
        parent_folder = generations / current["request"] if current else None
        parent_payload = saved_payload(parent_folder, current) if current else None
        folder = generations / request
        folder.mkdir(parents=True, exist_ok=True)
        if pending is None:
            write_checked(pending_path, dict(request=request, parent=current and current["generation"],
                                             target=target.record(), started_at=stamp()))
        try:
            if parent_folder:
                def prepare():
                    return delta_inputs(delta, input_plan, stored_inputs(parent_folder, current["request"]),
                                        load_checked(parent_folder / CATALOG / "catalog.json"),
                                        model, output / "vector_cache", options["threads"],
                                        options["max_seconds"], options["max_rss"])
            else:
                def prepare():
                    return frozen_inputs(config)

            inputs = prepare_inputs(folder, request, prepare)
            payload = build_prepared(folder, inputs, config, options, parent_folder, parent_payload)
            if parent_payload:
                result = publication.publish_append(driver, target.database, parent_payload, payload)
            else:
                result = publication.bootstrap(driver, target.database, payload, workers=workers)
            write_checked(folder / "verification.json", result | dict(verified_at=stamp()))
            write_checked(current_path, dict(request=request, generation=payload["generation"],
                                             payload_sha256=sha(folder / "payload.json"), target=target.record()))
            pending_path.unlink()
            return result
        except (Exception, KeyboardInterrupt) as exc:
            write_checked(folder / "last_error.json", dict(status="FAIL", reason=str(exc) or type(exc).__name__,
                                                           recorded_at=stamp()))
            raise


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--validate-frozen", action="store_true", help="Проверить только готовые входы и study reference")
    actions.add_argument("--apply", action="store_true", help="Построить или возобновить публикацию")
    parser.add_argument("--neo4j-uri")
    parser.add_argument("--database")
    parser.add_argument("--container")
    parser.add_argument("--neo4j-env", type=Path)
    parser.add_argument("--allow-shared-neo4j", action="store_true",
                        help="Разрешить только подтверждённую общую локальную базу neo4j")
    parser.add_argument("--workers", type=int, default=publication.BOOTSTRAP_WORKERS,
                        help="Параллельные сессии Neo4j при начальной загрузке (по умолчанию %(default)s)")
    parser.add_argument("--delta", type=Path, help="Подготовленные данные новых документов")
    parser.add_argument("--e5-model", type=Path, help="Локальный E5 для delta при отсутствии готовых V2 векторов")
    resource_options(parser)
    args = parser.parse_args(argv)
    config = approved_config()
    if args.validate_frozen:
        if any((args.neo4j_uri, args.database, args.container, args.neo4j_env, args.delta, args.e5_model,
                args.allow_shared_neo4j)):
            parser.error("--validate-frozen сверяет только исходные артефакты")
        print("ПРЕДУПРЕЖДЕНИЕ: СВЕРЯЮ ГОТОВЫЕ ЧЕКПОЙНТЫ CORPUS100.", flush=True)
        frozen_inputs(config)
        print(json.dumps(dict(status="PASS", checked="frozen_inputs_and_published_reference",
                              neo4j="NOT RUN", inference="NOT RUN", batch="NOT RUN"), ensure_ascii=False))
        return 0
    if not all((args.neo4j_uri, args.database, args.container)) or args.e5_model and args.delta is None:
        parser.error("--apply требует --neo4j-uri, --database и --container локальной цели")
    if args.workers < 1:
        parser.error("--workers должен быть положительным")
    target = publication.Target(args.neo4j_uri, args.database, args.container, args.allow_shared_neo4j)
    print("ПРЕДУПРЕЖДЕНИЕ: ПОСТРОЕНИЕ КАТАЛОГА И ПУБЛИКАЦИЯ В NEO4J МОГУТ ЗАНЯТЬ ПРОДОЛЖИТЕЛЬНОЕ ВРЕМЯ.",
          flush=True)
    with publication.connect(target, args.neo4j_env) as driver:
        result = run_graph(driver, target, OUTPUT, config, limits(args), delta=args.delta,
                           model=args.e5_model, workers=args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
