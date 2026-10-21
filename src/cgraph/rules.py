import re
import unicodedata

from .common import sha256

# These cues are a precision screen after model inference, not an RE substitute.
CUES = {
    "HAS_PROPERTY": r"\b(?:has|have|having|with|exhibit\w*|show\w*|demonstrat\w*|due to|possess\w*)\b",
    "HAS_PERFORMANCE": r"\b(?:has|have|with|exhibit\w*|show\w*|demonstrat\w*|achiev\w*|display\w*)\b",
    "HAS_MORPHOLOGY": r"\b(?:has|have|with|form\w*|exhibit\w*|show\w*)\b",
    "HAS_COMPONENT": r"\b(?:contain\w*|consist\w*|compos\w*|compris\w*|incorporat\w*)\b",
    "USES_MATERIAL": r"\b(?:us\w*|bas\w*|built|made|fabricat\w*|compos\w*)\b",
    "FUNCTIONS_AS": r"\b(?:as|serv\w*|function\w*|act\w*)\b",
    "DETECTS": r"\b(?:detect\w*|sens\w*|respons\w*)\b",
    "USED_FOR": r"\b(?:for|appli\w*|us\w*)\b",
    "FABRICATED_BY": r"\b(?:fabricat\w*|produc\w*|prepar\w*|synthesi\w*|obtain\w*|built)\b",
    "IMPROVES": r"\b(?:improv\w*|enhanc\w*|better)\b",
    "ACTS_ON": r"\b(?:on|treat\w*|modif\w*|characteriz\w*|transform\w*)\b",
    "INTEGRATED_IN": r"\b(?:integrat\w*|incorporat\w*|weav\w*|woven|into)\b",
    "HAS_MECHANISM": r"\b(?:mechanism|due to|attribut\w*|explain\w*)\b",
    "MECHANISM_INVOLVES": r"\b(?:involv\w*|combin\w*|consist\w*)\b",
    "LOCATED_IN": r"\b(?:in|within|at|locat\w*)\b",
    "OPTIMIZES": r"\boptim\w*\b",
    "FACILITATES": r"\b(?:facilitat\w*|enabl\w*|promot\w*)\b",
    "CAUSES": r"\b(?:caus\w*|induc\w*|lead\w* to|result\w* in)\b",
    "EXPLAINS": r"\b(?:explain\w*|due to|attribut\w*|result\w* from|because)\b",
    "CHANGES_PARAMETER": r"\b(?:increas\w*|decreas\w*|reduc\w*|chang\w*|unchanged)\b",
    "DEPENDS_ON": r"\b(?:depend\w*|function of|var\w* with)\b",
    "HAS_MEMBER": r"\b(?:includ\w*|compris\w*|consist\w*)\b",
    "MEETS_REQUIREMENT": r"\b(?:meet\w*|satisf\w*|require\w*)\b"
}
NUMBER_UNIT = re.compile(r"(?<!\w)(?P<value>[−-]?\d+(?:\.\d+)?)\s*(?P<unit>°\s*C|°\s*F|K|ppm|ppb|%|nm|eV|days?|hours?|cycles?)(?!\w)", re.I)
NEGATION = re.compile(r"\b(?:not(?! only\b)|never|without|failed to|no)\b", re.I)


def evidence(doc, chunk, start, end):
    text = chunk["text"][start:end]
    return dict(span_id=f"{chunk['chunk_id']}:{start}:{end}", document_id=doc["document_id"],
                revision_id=doc["revision_id"], chunk_id=chunk["chunk_id"], start=start, end=end,
                text=text, source_text_sha256=chunk["text_sha256"])


def measurements(doc, chunk):
    result = []
    for m in NUMBER_UNIT.finditer(chunk["text"]):
        result.append(dict(raw_text=m.group(), raw_value=float(m["value"].replace("−", "-")), unit=m["unit"],
                           evidence=evidence(doc, chunk, m.start(), m.end()), attachment="UNKNOWN"))
    return result


def qualify(doc, chunk, pair, relation, assertion_id, ontology):
    h, t = pair["head"], pair["tail"]
    text = chunk["text"][pair["start"]:pair["end"]]
    left, right = sorted((h, t), key=lambda m: m["start"])
    between = chunk["text"][left["end"]:right["start"]]
    if left["end"] > right["start"]:
        return None, "UNSUPPORTED_CAPABILITY:OVERLAPPING_RELATION_ENDPOINTS"
    if not re.search(CUES[relation], between, re.I):
        return None, "NO_EXPLICIT_PREDICATE_BETWEEN_ENDPOINTS"
    negations = list(NEGATION.finditer(text))
    if negations and (len(negations) > 1 or re.search(r"\b(?:and|or|but|whereas|while)\b|;", text, re.I)):
        return None, "UNSUPPORTED_CAPABILITY:NEGATION_SCOPE"
    if negations and not NEGATION.search(between):
        return None, "UNSUPPORTED_CAPABILITY:NEGATION_ATTACHMENT"
    polarity = "NEGATIVE" if negations else "POSITIVE"
    if re.search(r"\b(?:compared|versus|than|respectively)\b", text, re.I):
        return None, "UNSUPPORTED_CAPABILITY:COMPARISON_OR_DISTRIBUTION"
    if re.search(r"\b(?:may|might|could|potential\w*|would)\b", text, re.I):
        modality = "HYPOTHETICAL"
    elif re.search(r"\b(?:propos\w*|suggest\w*)\b", text, re.I):
        modality = "PROPOSED"
    else:
        modality = "ASSERTED"
    qualifiers = []
    numeric = list(NUMBER_UNIT.finditer(text))
    # Parsing is broader than attachment. Reject unresolved quantities and conditions.
    if numeric or re.search(r"\b(?:at|under|during|when|if|temperature|humidity|concentration|room-temperature)\b", text, re.I):
        return None, "UNSUPPORTED_CAPABILITY:QUALIFIER_ATTACHMENT"
    required = ontology["relations"][relation]["required_qualifiers"]
    if required:
        return None, "UNSUPPORTED_CAPABILITY:REQUIRED_QUALIFIERS"
    return dict(polarity=polarity, modality=modality, qualifiers=qualifiers,
                qualifier_status="UNKNOWN", assertion_scope="UNKNOWN",
                scope_evidence=[evidence(doc, chunk, pair["start"], pair["end"])],
                capability_note="Simple sentence scope only; no full qualifier or coreference support"), None


def normalize_name(name):
    # Preserve chemical case, composition, subscripts, charge and punctuation.
    return " ".join(unicodedata.normalize("NFC", name).split())


def abbreviation_map(text):
    result = {}
    for match in re.finditer(r"(?P<long>[A-Za-z][A-Za-z -]{2,100})\s*\((?P<short>[A-Z][A-Za-z0-9-]{1,9})\)", text):
        short = match["short"]
        words = re.findall(r"[A-Za-z]+", match["long"])
        # Require an exact suffix acronym. No fuzzy alias equivalence.
        for n in range(2, min(len(words), len(short)) + 1):
            selected = words[-n:]
            if "".join(w[0] for w in selected) == short:
                long = " ".join(selected)
                result.setdefault(short, set()).add(long)
    return {k: next(iter(v)) for k, v in result.items() if len(v) == 1}


def link(doc, chunks):
    concepts, decisions = {}, []
    representative_spans = {}
    for chunk in chunks:
        aliases = abbreviation_map(chunk["source_text"])
        for m in chunk["entities"]:
            name = normalize_name(m["surface_text"])
            canonical = aliases.get(name, name)
            # Ambiguous device instances never merge merely because their names match.
            scope = m["mention_id"] if m["kind"] in {"DEVICE", "OTHER"} else doc["revision_id"]
            concept_id = sha256(f"{scope}|{m['kind']}|{canonical}".encode())[:24]
            m["concept_ref"] = concept_id
            if concept_id in concepts:
                decisions.append(dict(decision="SAME", rule="exact_case_sensitive_name_within_revision",
                                      mention_id=m["mention_id"], concept_ref=concept_id,
                                      evidence=m.get("source_spans", []) + representative_spans[concept_id]))
            concepts.setdefault(concept_id, dict(concept_id=concept_id, canonical_name=canonical, kind=m["kind"],
                                                 instance_scope=scope, normalization_version="c001-conservative-v1"))
            representative_spans.setdefault(concept_id, m.get("source_spans", []))
            if canonical != name:
                decisions.append(dict(decision="SAME", rule="explicit_local_exact_acronym", mention_id=m["mention_id"],
                                      concept_ref=concept_id, source_chunk_id=chunk["chunk_id"],
                                      evidence=[dict(span_id=f"{chunk['chunk_id']}:definition", document_id=doc["document_id"],
                                                     revision_id=doc["revision_id"], chunk_id=chunk["chunk_id"], start=0,
                                                     end=len(chunk["source_text"]), text=chunk["source_text"],
                                                     source_text_sha256=chunk["source_text_sha256"])]))
    return list(concepts.values()), decisions
