"""Conservative local scope; unsupported qualifiers do not erase a binary edge."""
import re

from .rules import evidence

CHEMICAL_NO = re.compile(r"\bNO(?:\s*x|[xₓ₂2])?\b")
NEGATION = re.compile(r"\b(?:not(?!\s+only\b)|never|without|failed\s+to|no)\b", re.I)
CLAUSE = re.compile(r";|\b(?:but|whereas|while)\b", re.I)
DIRECTION = re.compile(r"\b(increas\w*|decreas\w*|reduc\w*|unchanged)\b", re.I)
TEMPERATURE = re.compile(r"\bat\s+([−-]?\d+(?:\.\d+)?)\s*(°\s*C|K)\b")


def linguistic_negations(text):
    # Uppercase NO/NOx/NO₂ are chemical symbols, not English 'no'.
    masked = CHEMICAL_NO.sub(lambda m: " " * len(m.group()), text)
    return list(NEGATION.finditer(masked))


def qualify(doc, chunk, pair, relation, assertion_id, ontology):
    h, t = pair["head"], pair["tail"]
    start, end = pair["start"], pair["end"]
    # Restrict scope to the smallest punctuation clause containing BOTH endpoints.
    for match in CLAUSE.finditer(chunk["text"], start, end):
        if match.end() <= min(h["start"], t["start"]):
            start = match.end()
        elif match.start() >= max(h["end"], t["end"]):
            end = match.start()
            break
        else:
            return None, "UNSUPPORTED_CAPABILITY:CROSS_CLAUSE_SCOPE"
    text = chunk["text"][start:end]
    negations = linguistic_negations(text)
    if negations and (len(negations) > 1 or re.search(r"\b(?:and|or)\b", text, re.I)):
        return None, "UNSUPPORTED_CAPABILITY:NEGATION_ATTACHMENT"
    if negations:
        lo, hi = sorted((h, t), key=lambda m: m["start"])
        # Negative attribution outside the endpoint interval is ambiguous.
        if not (lo["end"] <= start + negations[0].start() < hi["start"]):
            return None, "UNSUPPORTED_CAPABILITY:NEGATION_ATTACHMENT"
    polarity = "NEGATIVE" if negations else "POSITIVE"
    if re.search(r"\b(?:may|might|could|potential\w*|would)\b", text, re.I):
        modality = "HYPOTHETICAL"
    elif re.search(r"\b(?:propos\w*|suggest\w*)\b", text, re.I):
        modality = "PROPOSED"
    elif re.search(r"\b(?:compared|versus|than)\b", text, re.I):
        modality = "COMPARED"
    else:
        modality = "ASSERTED"
    rule = ontology["relations"][relation]
    qualifiers, scope = [], [evidence(doc, chunk, start, end)]

    def add(qtype, raw_value, match, unit=None, group=0):
        a, b = start + match.start(group), start + match.end(group)
        span = evidence(doc, chunk, a, b)
        scope.append(span)
        qualifiers.append(dict(qualifier_type=qtype, raw_value=raw_value, normalized_value=None,
                               raw_text=span["text"], unit=unit, evidence_refs=[span["span_id"]],
                               applies_to_assertion_id=assertion_id))

    if relation == "CHANGES_PARAMETER":
        # Required direction is the physical change, never inferred desirability.
        left, right = sorted((h, t), key=lambda m: m["start"])
        hits = [m for m in DIRECTION.finditer(text)
                if left["end"] <= start + m.start() < right["start"]]
        if len(hits) != 1 or negations:
            return None, "UNSUPPORTED_CAPABILITY:REQUIRED_DIRECTION"
        word = hits[0].group().lower()
        value = "increase" if word.startswith("increas") else "unchanged" if word == "unchanged" else "decrease"
        add("direction", value, hits[0])
    # Attach a trailing explicit temperature only in a single-predicate clause.
    temps = list(TEMPERATURE.finditer(text))
    if (len(temps) == 1 and "temperature" in rule["optional_qualifiers"]
            and not re.search(r"\b(?:and|or|respectively)\b", text, re.I)
            and start + temps[0].start() >= max(h["end"], t["end"])):
        add("temperature", temps[0].group(1), temps[0], temps[0].group(2))
    if not set(rule["required_qualifiers"]) <= {q["qualifier_type"] for q in qualifiers}:
        return None, "UNSUPPORTED_CAPABILITY:REQUIRED_QUALIFIERS"
    if polarity not in rule["supported_polarities"] or modality not in rule["supported_modalities"]:
        return None, "UNSUPPORTED_CAPABILITY:POLARITY_OR_MODALITY"
    return dict(polarity=polarity, modality=modality, qualifiers=qualifiers,
                qualifier_status="UNKNOWN", assertion_scope="LOCAL_CLAUSE_PARTIAL",
                scope_evidence=scope, capability_note="Partial deterministic attachment; completeness unverified"), None
