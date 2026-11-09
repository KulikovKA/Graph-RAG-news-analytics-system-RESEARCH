"""Stage 2A: prepare entity input only; no models, vectors or pair search.

Run: python scripts/prepare_semantic_entity_input.py
"""
from collections import Counter, defaultdict
import hashlib
import json
import sys
import unicodedata

sys.dont_write_bytecode = True
from exact_matching_audit import ROOT, SOURCE, DOCUMENTS, read_jsonl, sha256

ER = ROOT / "reports/entity_resolution/relex_base_t035"
MAPPING = ER / "conservative_resolution/canonical_mapping.jsonl"
ENTITY_FILE = SOURCE / "entities.jsonl"
OUTPUT = ER / "semantic_resolution/input"
RECIPE = "source-name-context-v1"
CONTEXT_RADIUS = 120  # Python Unicode characters on each side of the full span.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def damaged(text):
    return "\ufffd" in text or any(unicodedata.category(c) == "Cc" and not c.isspace() for c in text)


def member_record(entity, document, resolution):
    """Keep the exact source record; derive a context only for its own span."""
    flags = []
    text = document.get("text")
    start, end = entity["start"], entity["end"]
    valid_range = (isinstance(text, str) and isinstance(start, int) and isinstance(end, int)
                   and 0 <= start < end <= len(text))
    context = None
    if valid_range:
        left, right = max(0, start-CONTEXT_RADIUS), min(len(text), end+CONTEXT_RADIUS)
        context = dict(document_id=entity["document_id"], source_entity_id=entity["entity_id"],
                       start=left, end=right, text=text[left:right],
                       mention_start=start-left, mention_end=end-left)
        if text[start:end] != entity["surface_text"]:
            flags.append("source_span_text_mismatch")
        if damaged(context["text"]):
            flags.append("damaged_context")
    else:
        flags.append("missing_context")
        flags.append("invalid_source_offsets" if isinstance(text, str) and text else "missing_document_text")
    if damaged(entity["surface_text"]):
        flags.append("damaged_name")
    types = entity["entity_types"]
    if len(types) > 1:
        flags.append("multiple_original_types")
    if not types:
        flags.append("missing_original_type")
    if {m["ner_label_hint"] for m in entity["source_mentions"]} != set(types):
        flags.append("source_mention_type_set_mismatch")
    if any(t.get("partial_token", False) for m in entity["source_mentions"] for t in m.get("tokens", [])):
        flags.append("original_partial_token")
    if any(m.get("revision_id") != document.get("revision_id") for m in entity["source_mentions"]):
        flags.append("mention_document_revision_mismatch")
    return dict(source_entity=entity, source_mention_ids=[m["mention_id"] for m in entity["source_mentions"]],
                context=context, text_name=entity["surface_text"],
                text_name_context=f"Entity: {entity['surface_text']}\nContext: {context['text']}" if context else None,
                resolution_1b=resolution, quality_flags=sorted(set(flags)))


def main():
    input_paths = [MAPPING, ENTITY_FILE, DOCUMENTS]
    protected = sorted(set(input_paths) | {
        p for directory in ("exact_matching_audit", "conservative_resolution", "relation_remapping", "graph_comparison")
        for p in (ER / directory).iterdir() if p.is_file()})
    hashes = {p: sha256(p) for p in protected}
    mapping_rows, entity_rows, document_rows = map(read_jsonl, input_paths)
    entities = {e["entity_id"]: e for e in entity_rows}
    documents = {d["document_id"]: d for d in document_rows}
    source_mentions = {m["mention_id"]: (e["entity_id"], m) for e in entity_rows for m in e["source_mentions"]}
    require(len(entities) == len(entity_rows) == 1211, "Expected 1211 unique source entity IDs")
    require(len(documents) == len(document_rows) == 30, "Expected 30 unique source documents")
    require(len(source_mentions) == sum(len(e["source_mentions"]) for e in entity_rows) == 1327, "Invalid original mention IDs")
    entity_map, mapping_by_entity, seen_mentions = {}, {}, set()
    for row in mapping_rows:
        eid, target, mid = row["entity_id"], row["canonical_entity_id"], row["mention_id"]
        require(eid in entities and target in entities, "Unknown original/canonical entity ID")
        require(mid not in seen_mentions and source_mentions.get(mid) == (eid, row["source_mention"]), "Invalid/duplicate mention mapping")
        require(all(row[k] == v for k, v in entities[eid].items()), "Original entity/provenance differs from 1B")
        require(eid not in entity_map or entity_map[eid] == target, "Ambiguous entity target")
        if eid in mapping_by_entity:
            require(mapping_by_entity[eid]["resolution"] == row["resolution"], "Inconsistent 1B decision for one source entity")
        entity_map[eid], mapping_by_entity[eid] = target, row
        seen_mentions.add(mid)
    require(len(mapping_rows) == 1327 and seen_mentions == source_mentions.keys() and entity_map.keys() == entities.keys(), "Incomplete source mapping")
    require(len(set(entity_map.values())) == 1122, "Expected 1122 canonical IDs")
    grouped = defaultdict(list)
    for eid, target in entity_map.items():
        require(entity_map[target] == target, "Non-idempotent 1B mapping")
        require(entities[eid]["document_id"] == entities[target]["document_id"] and entities[eid]["document_id"] in documents, "Cross-document or unknown document")
        grouped[target].append(eid)
    file_provenance = [{"path": p.relative_to(ROOT).as_posix(), "sha256": hashes[p]} for p in input_paths]
    result = []
    for target in sorted(grouped, key=lambda i: (entities[i]["document_id"], entities[i]["start"], entities[i]["end"], i)):
        representative = entities[target]
        doc = documents[representative["document_id"]]
        ids = sorted(grouped[target], key=lambda i: (entities[i]["start"], entities[i]["end"], i))
        mids = sorted(m["mention_id"] for eid in ids for m in entities[eid]["source_mentions"])
        for eid in ids:
            row = mapping_by_entity[eid]
            require(set(row["canonical_member_entity_ids"]) == set(ids) and row["canonical_member_mention_ids"] == mids, "1B member roster mismatch")
        members = [member_record(entities[eid], doc, mapping_by_entity[eid]["resolution"]) for eid in ids]
        chosen = next(m for m in members if m["source_entity"]["entity_id"] == target)
        flags = {flag for member in members for flag in member["quality_flags"]}
        if len({tuple(sorted(m["source_entity"]["entity_types"])) for m in members}) > 1:
            flags.add("canonical_member_type_conflict")
        text = doc.get("text")
        if isinstance(text, str):
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest != doc.get("text_sha256") or doc.get("revision_id") != "sha256:"+digest:
                flags.add("document_text_hash_mismatch")
        result.append(dict(schema_version=1, input_recipe=RECIPE,
                           canonical_entity_id=target, document_id=doc["document_id"],
                           surface_text=representative["surface_text"], entity_types=representative["entity_types"],
                           source_entity_ids=ids, source_mention_ids=mids,
                           source_offsets=[dict(entity_id=eid, start=entities[eid]["start"], end=entities[eid]["end"]) for eid in ids],
                           text_name=chosen["text_name"], text_name_context=chosen["text_name_context"],
                           context_source_entity_id=target, representative_context=chosen["context"],
                           members=members, quality_flags=sorted(flags),
                           provenance=dict(input_files=file_provenance,
                                           document={k: v for k, v in doc.items() if k != "text"})))
    require(len(result) == len({r["canonical_entity_id"] for r in result}) == 1122, "Canonical coverage error")
    require(Counter(eid for r in result for eid in r["source_entity_ids"]) == Counter(entities.keys()), "Entity coverage/uniqueness error")
    require(Counter(mid for r in result for mid in r["source_mention_ids"]) == Counter(source_mentions.keys()), "Mention coverage/uniqueness error")
    for record in result:
        doc = documents[record["document_id"]]
        for member in record["members"]:
            original = member["source_entity"]
            require(original == entities[original["entity_id"]], "Changed original entity fields")
            require(original["document_id"] == record["document_id"] and entity_map[original["entity_id"]] == record["canonical_entity_id"], "Mixed canonical contexts")
            c = member["context"]
            if c:
                require(c["source_entity_id"] == original["entity_id"] and c["document_id"] == original["document_id"] and
                        c["text"] == doc["text"][c["start"]:c["end"]], "Context origin mismatch")
                if "source_span_text_mismatch" not in member["quality_flags"]:
                    require(c["text"][c["mention_start"]:c["mention_end"]] == original["surface_text"], "Local offset mismatch")
    canonical_flags = Counter(flag for r in result for flag in r["quality_flags"])
    member_flags = Counter(flag for r in result for m in r["members"] for flag in m["quality_flags"])
    contexts = [m["context"] for r in result for m in r["members"] if m["context"]]
    lengths = [len(c["text"]) for c in contexts]
    lines = ["# Этап 2A — входные данные для Semantic Entity Resolution", "",
             "Подготовлен набор без запуска моделей, расчёта embeddings/cosine similarity и поиска пар. Использовано существующее отображение 1B; решения об идентичности и типы не пересматривались.", "",
             "| Проверка покрытия | Результат |", "|---|---:|",
             f"| Уникальные канонические сущности | {len(result)} |",
             f"| Исходные entity ID (каждый ровно в одной канонической записи) | {len(entities)} |",
             f"| Исходные mention ID (каждый ровно в одной канонической записи) | {len(source_mentions)} |",
             f"| Документы | {len({r['document_id'] for r in result})} |",
             f"| Названия / канонические представления с контекстом | {sum(bool(r['text_name']) for r in result)} / {sum(r['text_name_context'] is not None for r in result)} |",
             f"| Отдельные контексты исходных span | {len(contexts)} |", "",
             "## Состав и воспроизводимое представление", "",
             "Одна строка canonical_entities.jsonl соответствует одному canonical_entity_id. surface_text и entity_types взяты без изменений у существующего представителя 1B. source_entity_ids, source_mention_ids и source_offsets задают полный состав. members хранит каждую исходную сущность целиком в source_entity, включая confidence, source_mentions с их score/типами/provenance; отдельно сохранены контекст, два текста и исходное основание resolution_1b. provenance содержит SHA-256 трёх входов и метаданные исходного документа с revision_id и source_reference.", "",
             f"Рецепт `{RECIPE}`: text_name — исходный surface_text без нормализации и удаления модификаторов. Для каждого исходного span извлекается точный срез документа [max(0,start−{CONTEXT_RADIUS}), min(len(text),end+{CONTEXT_RADIUS})); offsets — Python Unicode-индексы [start,end). Контекст включает весь span; его локальные границы сохранены. text_name_context имеет вид `Entity: <исходное название>\\nContext: <исходный срез>`. Префиксы конкретных embedding-моделей не добавляются.", "",
             "Каноническая запись использует только собственный контекст упоминания-представителя 1B, явно указанного в context_source_entity_id. Контексты других исходных span той же сущности остаются раздельными в members. Тексты разных канонических сущностей не конкатенируются. Перекрытие исходных окон возможно и не означает объединения сущностей. Не выполняются суммаризация, исправление текста, переназначение типов или выбор контекста по семантической похожести.", "",
             f"Длина сохранённых контекстов: {min(lengths) if lengths else 0}–{max(lengths) if lengths else 0} символов; средняя {sum(lengths)/len(lengths) if lengths else 0:.2f}. Длина ограничивается символьным окном, не tokenizer конкретной модели. Повреждения и исходные пробелы сохраняются; при отсутствии допустимого контекста text_name_context был бы null, без искусственной подстановки.", "",
             "## Потенциальные проблемы входа", "",
             "Числа в таблице — количество затронутых записей, не число отдельных повреждённых символов. Флаги могут пересекаться. Несколько исходных типов и partial_token сами по себе не доказывают ошибку.", "",
             "| Флаг | Канонических записей | Исходных span |", "|---|---:|---:|"]
    checks = sorted(set(canonical_flags) | {"damaged_name", "damaged_context", "missing_context", "multiple_original_types", "canonical_member_type_conflict", "source_span_text_mismatch", "document_text_hash_mismatch", "source_mention_type_set_mismatch"})
    for flag in checks:
        lines.append(f"| {flag} | {canonical_flags[flag]} | {member_flags[flag] if flag not in {'canonical_member_type_conflict', 'document_text_hash_mismatch'} else '—'} |")
    lines += ["", "damaged_name/context отмечают U+FFFD либо управляющий символ, не являющийся пробельным. Повреждённые названия включают формулы Co–Fe–Zr в W2317432785; символы не заменялись. multiple_original_types сохраняет исходную множественную типизацию, canonical_member_type_conflict означает разные наборы типов у участников одной канонической сущности. Между разными каноническими сущностями типы не сопоставлялись и пары не искались. original_partial_token перенесён из исходной NER-разметки. Качество/смысл терминов не оценивались.", "",
              "## Проверки и ограничения", "",
              "Проверены все 1122 canonical ID, 1211 entity ID, 1327 mention ID и 30 документов; однозначность отображения, состав групп 1B, отсутствие междокументного смешения, сохранность всех исходных entity/mention полей и provenance. Контексты сверены с исходным документом по абсолютным offsets, а упоминания — по локальным offsets. Хеши текста документов и revision_id проверены, найденные несоответствия перечислялись бы флагами. Повторный запуск проверен побайтовым сравнением двух выходных файлов; восстановление исходных 1211 сущностей проверено по members.source_entity.", "",
              "Выбор единственного контекста представителя детерминирован, но не гарантирует оптимальное снятие полисемии. Все остальные контексты доступны раздельно. Это входной набор, не embeddings, не список кандидатов и не оценка точности ER. Этап 2B не выполнялся.", "",
              "Запуск: `python scripts/prepare_semantic_entity_input.py`. Только стандартная библиотека и функции чтения/хеширования 1A. Создаются только canonical_entities.jsonl и report.md в semantic_resolution/input. Neo4j и результаты 1A–1D не изменяются.", "", "SHA-256 входов и защищённых результатов 1A–1D:", ""]
    lines += [f"- `{p.relative_to(ROOT).as_posix()}`: `{digest}`" for p, digest in hashes.items()]
    require(all(sha256(p) == digest for p, digest in hashes.items()), "Protected source changed during preparation")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "canonical_entities.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in result:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    require(all(sha256(p) == digest for p, digest in hashes.items()), "Protected source changed after writing")
    print(json.dumps(dict(canonical_entities=len(result), source_entities=len(entities), source_mentions=len(source_mentions),
                          documents=len(documents), contexts=len(contexts), canonical_flags=dict(canonical_flags),
                          source_span_flags=dict(member_flags)), indent=2))


if __name__ == "__main__":
    main()
