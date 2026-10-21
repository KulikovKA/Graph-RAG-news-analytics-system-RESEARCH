import re


def sentence_spans(text):
    """Conservative punctuation splitter; never rewrites source codepoints."""
    start = 0
    for match in re.finditer(r"[.!?](?:\s+|$)", text):
        end = match.start() + 1
        # Decimal dots have no following whitespace. Abbreviations remain a limitation.
        if text[start:end].strip():
            yield start, end
        start = match.end()
    if text[start:].strip():
        yield start, len(text)


def allowed_labels(head, tail, ontology):
    return [label for label, rule in ontology["relations"].items()
            if head["kind"] in rule["head_kinds"] and tail["kind"] in rule["tail_kinds"]]


def generate_pairs(chunk, entities, ontology, cfg):
    pairs = []
    for start, end in sentence_spans(chunk["text"]):
        local = [m for m in entities if start <= m["start"] < m["end"] <= end]
        for h in local:
            for t in local:
                if h["mention_id"] == t["mention_id"]:
                    continue
                gap = max(0, max(h["start"], t["start"]) - min(h["end"], t["end"]))
                labels = allowed_labels(h, t, ontology)
                if labels and gap <= cfg["relations"]["max_gap_characters"]:
                    pairs.append(dict(head=h, tail=t, allowed_labels=labels, start=start, end=end, gap=gap))
    pairs.sort(key=lambda p: (p["gap"], p["start"], p["head"]["start"], p["tail"]["start"], p["head"]["mention_id"], p["tail"]["mention_id"]))
    budget = cfg["relations"]["max_pairs_per_chunk"]
    return pairs[:budget], len(pairs) - min(len(pairs), budget)


def marked_text(chunk, pair):
    """Directional boundary events also support nested/overlapping mentions."""
    h, t = pair["head"], pair["tail"]
    events = {}
    for role, m in (("HEAD", h), ("TAIL", t)):
        events.setdefault(m["start"], []).append(f" [{role}_{m['kind']}] ")
        events.setdefault(m["end"], []).append(f" [END_{role}] ")
    parts = []
    for i in range(pair["start"], pair["end"]):
        parts.extend(sorted(events.get(i, [])))
        parts.append(chunk["text"][i])
    parts.extend(sorted(events.get(pair["end"], [])))
    direction = "HEAD_BEFORE_TAIL" if h["start"] < t["start"] else "TAIL_BEFORE_HEAD"
    return f"{direction} HEADTYPE_{h['kind']} TAILTYPE_{t['kind']} " + "".join(parts)
