"""Conservative linguistic forms. No identity decisions, stemming or source rewriting."""
import re
import unicodedata

from .io import validate_span

NORMALIZER_VERSION = "spacy-en-nfc-v1"


def safe_unicode(value):
    # NFKC/casefold would conflate subscripts, superscripts, case-sensitive symbols.
    return " ".join(unicodedata.normalize("NFC", value).split())


def _aligned_long_form(text, short, start, end):
    """Schwartz-Hearst character alignment, restricted to a bounded local definition."""
    chars = [c for c in short if c.isalnum()]
    if not 2 <= len(chars) <= 10 or not any(c.isalpha() for c in chars):
        return None
    words = list(re.finditer(r"\S+", text[start:end]))
    limit = min(len(chars) + 5, len(chars) * 2)
    if not words:
        return None
    start += words[max(0, len(words) - limit)].start()
    j = end - 1
    for index in range(len(chars) - 1, -1, -1):
        while j >= start:
            if (text[j].casefold() == chars[index].casefold() and
                    (index != 0 or j == start or not text[j - 1].isalnum())):
                break
            j -= 1
        if j < start:
            return None
        j -= 1
    begin = j + 1
    return begin, end


def detect_abbreviations(text):
    """Return definition occurrences with exact spans; ambiguous expansions coexist."""
    result = []
    for match in re.finditer(r"\(([^()\n]{2,120})\)", text):
        inner = match[1].strip()
        inner_start = match.start(1) + len(match[1]) - len(match[1].lstrip())
        boundary = max(text.rfind(c, 0, match.start()) for c in (".", ";", "!", "?", "\n")) + 1
        left_end = match.start()
        while left_end > boundary and text[left_end - 1].isspace():
            left_end -= 1
        short, short_start = inner, inner_start
        long_span = None
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9-]{1,14}", inner):
            long_span = _aligned_long_form(text, short, max(boundary, left_end - 250), left_end)
        if long_span is None:
            preceding = re.search(r"[A-Za-z][A-Za-z0-9-]{1,14}$", text[boundary:left_end])
            if preceding:
                short = preceding[0]
                short_start = boundary + preceding.start()
                long_span = _aligned_long_form(text, short, inner_start, inner_start + len(inner))
        if long_span and any(c.isupper() for c in short):
            a, b = long_span
            # A long form must actually be longer than the short form.
            if b - a <= len(short):
                continue
            result.append({"short_form": short, "long_form": text[a:b],
                           "short_start": short_start, "short_end": short_start + len(short),
                           "long_start": a, "long_end": b,
                           "scope": "document_candidate_only"})
    return sorted(result, key=lambda x: (x["short_start"], x["long_start"]))


class LinguisticNormalizer:
    def __init__(self):
        import spacy
        self.nlp = spacy.load("en_core_web_sm", exclude=["ner", "parser"])
        if "ner" in self.nlp.pipe_names:
            raise ValueError("spaCy NER must be disabled")

    def analyze(self, text):
        return self.nlp(text)

    def normalize(self, text, start, end, *, analyzed=None, abbreviations=None):
        surface = text[start:end]
        validate_span(text, start, end, surface)
        doc = self.analyze(text) if analyzed is None else analyzed
        if doc.text != text:
            raise ValueError("Linguistic analysis belongs to a different source")
        # Taggers can mistake the closing mark of 'quoted terms' for a plural possessive.
        quoted_marks = set()
        for quoted in re.finditer(r"(?<!\w)(?:'[^'\n]+'|‘[^’\n]+’|’[^’\n]+’)(?!\w)", text):
            quoted_marks.update((quoted.start(), quoted.end() - 1))
        pieces, tokens = [], []
        cursor = start
        for token in doc:
            a, b = token.idx, token.idx + len(token.text)
            if b <= start or a >= end:
                continue
            a, b = max(start, a), min(end, b)
            raw = text[a:b]
            complete = a == token.idx and b == token.idx + len(token.text)
            possessive = complete and token.tag_ == "POS" and token.idx not in quoted_marks
            # Preserve symbols, formulas, abbreviations, mixed case, proper names and punctuation.
            ordinary = complete and raw.isascii() and raw.isalpha() and raw.islower() and token.pos_ != "PROPN"
            lemma = token.lemma_ if ordinary and token.lemma_ else raw
            pieces.append(text[cursor:a])
            pieces.append("" if possessive else lemma)
            tokens.append({"text": raw, "start": a, "end": b, "lemma": lemma,
                           "pos": token.pos_, "tag": token.tag_, "morph": token.morph.to_dict(),
                           "partial_token": not complete, "possessive": possessive})
            cursor = b
        pieces.append(text[cursor:end])
        lemma = safe_unicode("".join(pieces))
        definitions = detect_abbreviations(text) if abbreviations is None else abbreviations
        candidates = [item for item in definitions if item["short_form"] == surface or
                      (item["long_start"] == start and item["long_end"] == end)]
        return {"surface_text": surface, "normalized_form": lemma or safe_unicode(surface),
                "lemma": lemma or safe_unicode(surface), "tokens": tokens,
                "abbreviation_candidates": candidates, "normalizer_version": NORMALIZER_VERSION}
