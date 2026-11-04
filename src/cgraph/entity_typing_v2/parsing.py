"""Recover complete JSON records without inventing semantic fields."""
import json
from collections import Counter

from jsonschema import Draft202012Validator

from .prompting import response_schema


def _complete_objects(text):
    """Scan only braces outside JSON strings; ignore incomplete enclosing objects."""
    decoder = json.JSONDecoder()
    quoted = escaped = False
    for index, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == "{":
            try:
                obj, _ = decoder.raw_decode(text, index)
            except ValueError:
                continue
            if isinstance(obj, dict) and "candidate_id" in obj:
                yield obj


def parse_response(content, expected_ids, labels):
    expected = set(expected_ids)
    errors = []
    schema = response_schema(labels)
    try:
        decoded = json.loads(content)
        root_errors = list(Draft202012Validator(schema).iter_errors(decoded))
        if root_errors:
            errors.append({"kind": "json_schema", "detail": "; ".join(e.message for e in root_errors)[:1500]})
        records = decoded.get("decisions", []) if isinstance(decoded, dict) else []
        if not isinstance(records, list):
            records = []
        if root_errors and not records:
            records = list(_complete_objects(content))
    except (ValueError, TypeError) as exc:
        errors.append({"kind": "invalid_json", "detail": str(exc)})
        records = list(_complete_objects(content)) if isinstance(content, str) else []
    ids = [r.get("candidate_id") for r in records if isinstance(r, dict) and isinstance(r.get("candidate_id"), str)]
    duplicates = {cid for cid, n in Counter(ids).items() if n > 1}
    if duplicates:
        errors.append({"kind": "duplicate_ids", "candidate_ids": sorted(duplicates)})
    valid = {}
    validator = Draft202012Validator(schema["properties"]["decisions"]["items"])
    for record in records:
        problems = list(validator.iter_errors(record))
        if problems:
            errors.append({"kind": "record_schema", "detail": "; ".join(p.message for p in problems)[:1500]})
            continue
        cid = record["candidate_id"]
        if cid not in expected:
            errors.append({"kind": "unexpected_id", "candidate_id": cid})
            continue
        if cid in duplicates:
            continue
        status, label = record["typing_status"], record["entity_type"]
        if ((status == "TYPED" and label not in labels) or (status != "TYPED" and label != status)):
            errors.append({"kind": "status_type_mismatch", "candidate_id": cid})
            continue
        # jsonschema rejects NaN poorly when used directly as a Python value.
        import math
        if not math.isfinite(record["confidence_estimate"]):
            errors.append({"kind": "nonfinite_confidence", "candidate_id": cid})
            continue
        valid[cid] = record
    missing = sorted(expected - valid.keys())
    if missing:
        errors.append({"kind": "missing_ids", "candidate_ids": missing})
    return valid, errors
