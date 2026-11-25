"""Offset-preserving windows and pair computation for Stage 3A.5.12."""
from bisect import bisect_left, bisect_right
from collections import defaultdict
import math
import re
import unicodedata

import numpy as np
from tqdm import tqdm

WINDOWS = dict(S0=0, S1=1, S2=2, S3=3, W8=8, W16=16, W32=32, W64=64, W128=128)
THRESHOLDS = (.75, .80, .85, .90, .95)
PREFIX = "query: "
MAX_TOKENS = 512
SEED = 42
RULES = ("result_type", "incompatible_types", "conflicting_expansions")
FLEXIBLE_TYPES = {"TASK", "PROCESS", "ATTRIBUTE", "CONCEPT"}
FUNCTION_WORDS = {"a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"}
WORD = re.compile(r"\S+")
TERMINAL = re.compile(r"""[.!?]+[”’'"\])}]*(?=\s|$)""")
ABBREVIATION = re.compile(
    r"(?:\b(?:e\.g|i\.e|cf|vs|approx|al|figs?|eqs?|refs?|nos?|dr|prof|mr|mrs|ms|"
    r"vol|pp|supp|viz)\.|\b(?:[A-Za-z]\.){2,}|\b[A-Z]\.)$", re.IGNORECASE
)


def sentence_spans(text):
    """Сохраняет исходную пунктуацию и обходит точки в научных сокращениях."""
    start = 0
    spans = []
    for match in TERMINAL.finditer(text):
        end = match.end()
        following = re.search(r"\S", text[end:])
        if following and text[end + following.start()].islower():
            continue
        if match[0] == "." and following and ABBREVIATION.search(text[max(start, end - 24):end]):
            continue
        if text[start:end].strip():
            first = start + len(text[start:end]) - len(text[start:end].lstrip())
            spans.append((first, end))
        start = end
    if text[start:].strip():
        spans.append((start + len(text[start:]) - len(text[start:].lstrip()), len(text.rstrip())))
    return spans


class DocumentWindows:
    def __init__(self, text):
        self.text = text
        words = [(m.start(), m.end()) for m in WORD.finditer(text)]
        self.bounds = {"S": tuple(zip(*sentence_spans(text))), "W": tuple(zip(*words))}

    def window(self, name, start, end):
        if not 0 <= start < end <= len(self.text) or not self.text[start:end].strip():
            raise ValueError("Invalid source mention span")
        starts, ends = self.bounds[name[0]]
        radius = WINDOWS[name]
        first = bisect_right(ends, start)
        last = bisect_left(starts, end) - 1
        return min(start, starts[max(0, first - radius)]), max(end, ends[min(len(ends) - 1, last + radius)])


def fit_window(tokenizer, text, name, left, right, start, end):
    head = PREFIX + name + ". "
    encoded = tokenizer.encode(head + text[left:right])
    if len(encoded.ids) <= MAX_TOKENS:
        return left, right
    mention_left, mention_right = len(head) + start - left, len(head) + end - left
    before = [b - len(head) + left for a, b in encoded.offsets if len(head) < b <= mention_left]
    after = [a - len(head) + left for a, b in encoded.offsets if mention_right <= a < b]
    budget = MAX_TOKENS - (len(encoded.ids) - len(before) - len(after))
    if budget < 0 and len(tokenizer.encode(head + text[start:end]).ids) > MAX_TOKENS:
        raise ValueError("Name and intact source mention exceed E5 token limit")
    budget = max(0, budget)
    take_left = min(len(before), max(budget // 2, budget - len(after)))
    take_right = min(len(after), budget - take_left)
    while True:
        new_left = before[len(before) - take_left - 1] if take_left < len(before) else left
        new_right = after[take_right] if take_right < len(after) else right
        if len(tokenizer.encode(head + text[new_left:new_right]).ids) <= MAX_TOKENS:
            return new_left, new_right
        if take_left >= take_right and take_left:
            take_left -= 1
        elif take_right:
            take_right -= 1
        else:
            if len(tokenizer.encode(head + text[start:end]).ids) > MAX_TOKENS:
                raise ValueError("Name and intact source mention exceed E5 token limit")
            return start, end


def extract_windows(entities, documents, geometry, tokenizer, window):
    rows, truncated = [], 0
    for entity in tqdm(entities, desc=f"{window} · extract", unit="entity"):
        unique = defaultdict(list)
        for evidence in entity["evidence"]:
            doc_id = evidence["document_id"]
            text = documents[doc_id]["text"]
            start, end = evidence["mention_start"], evidence["mention_end"]
            left, right = geometry[doc_id].window(window, start, end)
            lo, hi = fit_window(tokenizer, text, entity["name"], left, right, start, end)
            truncated += (left, right) != (lo, hi)
            unique[text[lo:hi]].append(dict(
                document_id=doc_id, revision_id=evidence["revision_id"],
                source_url=evidence["source_url"], text_sha256=documents[doc_id]["text_sha256"],
                source_entity_id=evidence["source_entity_id"], source_mention_ids=evidence["source_mention_ids"],
                mention_start=start, mention_end=end, mention_text=evidence["mention_text"],
                window_start=left, window_end=right, input_start=lo, input_end=hi))
        if not unique or any(not fragment.strip() for fragment in unique):
            raise ValueError(f"Missing context for {entity['entity_id']}")
        rows.append(dict(entity_id=entity["entity_id"], contexts=[
            dict(text=fragment, input=entity["name"] + ". " + fragment, provenance=refs)
            for fragment, refs in unique.items()]))
    return rows, truncated


def normalized(vectors):
    values = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("Nonfinite or zero embedding")
    return values / norms


def mean_embeddings(vectors):
    values = np.asarray(vectors, dtype=np.float32)
    if not len(values) or not np.allclose(np.linalg.norm(values, axis=1), 1, rtol=0, atol=2e-6):
        raise ValueError("Expected individual L2-normalized embeddings")
    mean = values.mean(axis=0, dtype=np.float64)
    normalized(mean[None, :])
    return mean


def pair_indices(documents):
    size = len(documents)
    left, right = np.triu_indices(size, 1)
    docs = np.asarray(documents)
    allowed = docs[left] != docs[right]
    return left[allowed].astype(np.int32), right[allowed].astype(np.int32)


def cosine_scores(vectors, left, right, label, block=128):
    unit = normalized(vectors)
    _, equal = np.unique(vectors, axis=0, return_inverse=True)
    scores = np.empty(len(left), dtype=np.float64)
    for start in tqdm(range(0, len(unit), block), desc=f"{label} · cosine", unit="block"):
        end = min(start + block, len(unit))
        lo, hi = left.searchsorted([start, end])
        i, j = left[lo:hi], right[lo:hi]
        values = (unit[start:end] @ unit.T)[i - start, j]
        values[equal[i] == equal[j]] = 1.
        scores[lo:hi] = np.clip(values, -1, 1)
    return scores


def stats(scores):
    percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
    values = dict(zip((f"p{p}" for p in percentiles), np.percentile(scores, percentiles).tolist()))
    return dict(minimum=float(scores.min()), maximum=float(scores.max()), mean=float(scores.mean()),
                std=float(scores.std()), median=values["p50"], q1=values["p25"], q3=values["p75"], **values)


def fold(value):
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def guard_reasons(a, b):
    lt, rt = set(a["entity_types"]), set(b["entity_types"])
    reasons = []
    if "RESULT" in lt | rt:
        reasons.append(RULES[0])
    if not lt & rt and not lt | rt <= FLEXIBLE_TYPES:
        reasons.append(RULES[1])
    expansions_a, expansions_b = a["expansions"], b["expansions"]
    for short in expansions_a.keys() & expansions_b.keys():
        # Общие содержательные слова оставляют вариант раскрытия открытым.
        first = set(re.findall(r"\w+", " ".join(expansions_a[short]))) - FUNCTION_WORDS
        second = set(re.findall(r"\w+", " ".join(expansions_b[short]))) - FUNCTION_WORDS
        if first and second and first.isdisjoint(second):
            reasons.append(RULES[2])
            break
    return reasons


def pair_guard_codes(entities, left, right, mask):
    # Типы векторизуются по небольшому числу наблюдаемых комбинаций меток.
    categories = sorted({tuple(e["entity_types"]) for e in entities})
    lookup = {key: i for i, key in enumerate(categories)}
    types = np.array([lookup[tuple(e["entity_types"])] for e in entities])
    table = np.zeros((len(categories), len(categories)), dtype=np.uint8)
    for i, a in enumerate(categories):
        for j, b in enumerate(categories):
            reasons = guard_reasons(dict(entity_types=a, expansions={}), dict(entity_types=b, expansions={}))
            table[i, j] = sum(1 << RULES.index(rule) for rule in reasons)
    li, ri = left[mask], right[mask]
    codes = table[types[li], types[ri]]
    has = np.array([bool(e["expansions"]) for e in entities])
    for k in np.flatnonzero(has[li] & has[ri]):
        if RULES[2] in guard_reasons(entities[li[k]], entities[ri[k]]):
            codes[k] |= 1 << 2
    return codes


def filter_mask(name, context, name_threshold, context_threshold):
    return (name >= name_threshold) & (context >= context_threshold)


def pair_count(n):
    return math.comb(n, 2)
