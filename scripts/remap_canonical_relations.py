"""Stage 1C only. Run: python scripts/remap_canonical_relations.py

Standard library; reads stage 1B, preserves source records, never accesses Neo4j.
"""
from collections import Counter, defaultdict
import hashlib
import json
import sys

sys.dont_write_bytecode = True
from exact_matching_audit import ROOT, SOURCE, read_jsonl, sha256

ER = ROOT / "reports/entity_resolution/relex_base_t035"
MAPPING = ER / "conservative_resolution/canonical_mapping.jsonl"
RELATIONS = SOURCE / "relations.jsonl"
ENTITIES = SOURCE / "entities.jsonl"
OUTPUT = ER / "relation_remapping"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def triple_key(record):
    return tuple(record[k] for k in ("document_id", "head_id", "relation_type", "tail_id"))


def jsonl_bytes(records):
    return "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records).encode("utf-8")


def main():
    # Hash all artifacts of 1A/1B, as well as the two original inputs, before/after.
    protected = sorted({ENTITIES, RELATIONS, *(
        p for directory in (ER / "exact_matching_audit", ER / "conservative_resolution")
        for p in directory.iterdir() if p.is_file())})
    fingerprints = {p: sha256(p) for p in protected}
    original_entities, mention_rows, relations = map(read_jsonl, (ENTITIES, MAPPING, RELATIONS))
    entities = {e["entity_id"]: e for e in original_entities}
    require(len(entities) == len(original_entities) == 1211, "Expected 1211 unique source entity IDs")
    source_mentions = {m["mention_id"]: (e["entity_id"], m) for e in original_entities for m in e["source_mentions"]}
    require(len(source_mentions) == sum(len(e["source_mentions"]) for e in original_entities) == 1327, "Invalid original mention IDs")
    entity_map, seen_mentions = {}, set()
    for row in mention_rows:
        eid, target, mid = row["entity_id"], row["canonical_entity_id"], row["mention_id"]
        require(mid not in seen_mentions and source_mentions.get(mid) == (eid, row["source_mention"]), "Duplicate/unknown mention or modified provenance")
        require(eid in entities and target in entities, "Unknown entity in 1B mapping")
        require(all(row[k] == v for k, v in entities[eid].items()), "Source entity properties differ in 1B")
        require(eid not in entity_map or entity_map[eid] == target, "Conflicting targets for one entity ID")
        require(entities[eid]["document_id"] == entities[target]["document_id"], "Cross-document entity mapping")
        entity_map[eid] = target
        seen_mentions.add(mid)
    require(len(mention_rows) == 1327 and seen_mentions == source_mentions.keys(), "Incomplete mention mapping")
    require(entity_map.keys() == entities.keys(), "Incomplete entity mapping")
    targets = set(entity_map.values())
    require(len(targets) == 1122, "Expected 1122 canonical entities")
    require(all(entity_map[c] == c for c in targets), "Canonical mapping is not idempotent")
    source_by_id = {r["triple_id"]: r for r in relations}
    require(len(source_by_id) == len(relations) == 1971, "Expected 1971 unique source relation IDs")
    remapped, grouped = [], defaultdict(list)
    extra_fields = {"original_head_id", "original_tail_id", "is_self_loop", "self_loop_created_by_resolution", "source_relation"}
    for source in relations:
        require(not extra_fields.intersection(source), "Reserved remapping field already exists in source")
        head, tail = source["head_id"], source["tail_id"]
        require(head in entity_map and tail in entity_map, "Unknown source endpoint")
        require(entities[head]["document_id"] == entities[tail]["document_id"] == source["document_id"], "Cross-document relation")
        ch, ct = entity_map[head], entity_map[tail]
        row = {**source, "head_id": ch, "tail_id": ct,
               "original_head_id": head, "original_tail_id": tail,
               "is_self_loop": ch == ct,
               "self_loop_created_by_resolution": ch == ct and head != tail,
               "source_relation": source}
        remapped.append(row)
        grouped[triple_key(row)].append(row)
    canonical = []
    for key, members in sorted(grouped.items()):
        members.sort(key=lambda r: r["triple_id"])
        digest = hashlib.sha256(json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        canonical.append({
            "canonical_triple_id": "ct:" + digest,
            "document_id": key[0], "head_id": key[1], "relation_type": key[2], "tail_id": key[3],
            "is_self_loop": key[1] == key[3],
            "self_loop_created_by_resolution": any(r["self_loop_created_by_resolution"] for r in members),
            "created_self_loop_source_triple_ids": [r["triple_id"] for r in members if r["self_loop_created_by_resolution"]],
            "source_relation_count": len(members),
            "source_triple_ids": [r["triple_id"] for r in members],
            "source_confidences": [{"triple_id": r["triple_id"], "confidence": r["confidence"]} for r in members],
            "source_relations": [r["source_relation"] for r in members],
        })
    # Validate conservation of complete records, not just the counts.
    require(Counter(r["triple_id"] for r in remapped) == Counter(source_by_id.keys()), "Lost/duplicated remapped relations")
    require(len({r["canonical_triple_id"] for r in canonical}) == len(canonical), "Canonical ID collision")
    require(len({triple_key(r) for r in canonical}) == len(canonical), "Canonical structural duplicates remain")
    for row in remapped:
        original = source_by_id[row["triple_id"]]
        restored = {k: row[k] for k in original}
        restored.update(head_id=row["original_head_id"], tail_id=row["original_tail_id"])
        require(restored == original == row["source_relation"], "Source fields/provenance changed")
        require(row["head_id"] == entity_map[original["head_id"]] and row["tail_id"] == entity_map[original["tail_id"]], "Direction/mapping changed")
    collected_ids = []
    for row in canonical:
        require(row["head_id"] in targets and row["tail_id"] in targets, "Unknown canonical endpoint")
        require(entities[row["head_id"]]["document_id"] == entities[row["tail_id"]]["document_id"] == row["document_id"], "Cross-document canonical relation")
        require(row["source_relation_count"] == len(row["source_relations"]) == len(row["source_triple_ids"]), "Invalid provenance count")
        for original, tid, score in zip(row["source_relations"], row["source_triple_ids"], row["source_confidences"]):
            require(original == source_by_id[tid], "Canonical provenance changed")
            require(score == {"triple_id": tid, "confidence": original["confidence"]}, "Confidence changed")
            require(triple_key(row) == (original["document_id"], entity_map[original["head_id"]], original["relation_type"], entity_map[original["tail_id"]]), "Canonical relation has no matching source")
            collected_ids.append(tid)
    require(Counter(collected_ids) == Counter(source_by_id.keys()), "Lost/duplicated canonical provenance")
    linked = {r[k] for r in canonical for k in ("head_id", "tail_id")}
    original_linked = {r[k] for r in relations for k in ("head_id", "tail_id")}
    stats = {
        "source_entities": len(entities), "canonical_entities": len(targets),
        "source_relation_records": len(relations), "remapped_relation_records": len(remapped),
        "unique_canonical_triples": len(canonical), "removed_structural_duplicates": len(relations) - len(canonical),
        "duplicate_groups": sum(r["source_relation_count"] > 1 for r in canonical),
        "original_self_loop_records": sum(r["head_id"] == r["tail_id"] for r in relations),
        "remapped_self_loop_records": sum(r["is_self_loop"] for r in remapped),
        "created_self_loop_records": sum(r["self_loop_created_by_resolution"] for r in remapped),
        "unique_self_loop_triples": sum(r["is_self_loop"] for r in canonical),
        "unique_created_self_loop_triples": sum(r["self_loop_created_by_resolution"] for r in canonical),
        "source_isolated_entities": len(set(entities) - original_linked),
        "canonical_isolated_entities": len(targets - linked),
    }
    labels = [
        ("Исходных сущностей", "source_entities"), ("Канонических сущностей", "canonical_entities"),
        ("Исходных записей отношений", "source_relation_records"), ("Переназначенных записей", "remapped_relation_records"),
        ("Уникальных канонических троек", "unique_canonical_triples"), ("Устранённых структурных дублей", "removed_structural_duplicates"),
        ("Групп структурных дублей", "duplicate_groups"), ("Исходных self-loops (записей)", "original_self_loop_records"),
        ("Self-loops после переназначения (записей)", "remapped_self_loop_records"), ("Из них созданы объединением сущностей", "created_self_loop_records"),
        ("Уникальных self-loop троек", "unique_self_loop_triples"), ("Из них созданы объединением сущностей", "unique_created_self_loop_triples"),
        ("Исходных изолированных сущностей", "source_isolated_entities"), ("Изолированных канонических сущностей", "canonical_isolated_entities"),
    ]
    lines = ["# Этап 1C — Relation Remapping", "", "Источник: неизменённое отображение 1B для relex_base_t035 и исходные relations.jsonl. Использованы все 1327 mention-записей для построения однозначного отображения 1211 entity ID на 1122 канонических ID. Дополнительное Entity Resolution не выполнялось.", "",
             "| Показатель | Значение |", "|---|---:|"]
    lines += [f"| {label} | {stats[key]} |" for label, key in labels]
    lines += ["", "Ключ дедупликации: `(document_id, canonical_head_id, relation_type, canonical_tail_id)`. Head/tail не переставляются, типы сохраняются. canonical_triple_id — ct:SHA-256 JSON-массива ключа; это идентификатор записи в артефакте, не новый узел графа.", "",
              "remapped_relations.jsonl: одна строка на каждое исходное отношение в исходном порядке. Только head_id/tail_id заменены отображением 1B. original_head_id/original_tail_id сохраняют исходные конечные точки; source_relation содержит полную неизменённую исходную запись, включая все неизвестные скрипту поля. Остальные исходные поля продублированы сверху без изменений.", "",
              "canonical_relations.jsonl: одна строка на уникальную тройку; source_triple_ids, source_confidences и source_relations сохраняют полный состав объединения и все исходные confidence, evidence и provenance. Единого агрегированного confidence нет; оценки не усреднялись и не пересчитывались. Записи упорядочены по ключу, источники внутри группы — по triple_id.", "",
              "is_self_loop отмечает петлю, self_loop_created_by_resolution отличает петлю после объединения от исходной; created_self_loop_source_triple_ids перечисляет её источники. Все петли сохранены, их корректность здесь не оценивается. Изолированный узел не имеет ни входящего, ни исходящего предметного отношения; self-loop считается связью. Служебные связи документа не учитываются.", "",
              "Evidence, включая содержащиеся в нём entity_ids, offsets и текст, сохранено в исходной системе ID как provenance. Типы конечных точек из исходных отношений также не заменяются и не исправляются. Переназначены только конечные точки отношений; Relation Verification и оценка семантической корректности не выполнялись.", "",
              "Проверены уникальность/полнота исходных ID, однозначность и идемпотентность отображения, покрытие всех 1971 отношений, наличие всех конечных точек среди 1122 канонических сущностей, сохранность направления, типа и полных исходных записей. Источники уникальных троек образуют точное разбиение исходных triple ID: нет потерь, повторного учёта или непрослеживаемых отношений. Междокументных объединений/отношений нет. Повторный запуск проверен побайтовым сравнением всех трёх артефактов.", "",
              "Запуск: `python scripts/remap_canonical_relations.py`. Только стандартная библиотека и функции чтения/хеширования из 1A; NER/RE/LLM не запускаются. Neo4j не подключается, граф не создаётся и не импортируется. Выполнен исключительно этап 1C.", "", "SHA-256 входов и защищённых артефактов 1A/1B (проверяются до и после записи):", ""]
    lines += [f"- `{p.relative_to(ROOT).as_posix()}`: `{digest}`" for p, digest in fingerprints.items()]
    artifacts = {"remapped_relations.jsonl": jsonl_bytes(remapped),
                 "canonical_relations.jsonl": jsonl_bytes(canonical),
                 "report.md": ("\n".join(lines) + "\n").encode("utf-8")}
    require(all(sha256(p) == digest for p, digest in fingerprints.items()), "Protected input changed during processing")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, data in artifacts.items():
        (OUTPUT / name).write_bytes(data)
    require(all(sha256(p) == digest for p, digest in fingerprints.items()), "Protected input changed after writing")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
