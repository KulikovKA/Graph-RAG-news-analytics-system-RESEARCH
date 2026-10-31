"""Experimental noun-head normalization; historical N1 remains in normalization.py."""
import re

from .normalization import LinguisticNormalizer, safe_unicode, detect_abbreviations


class ScientificNormalizer(LinguisticNormalizer):
    """Only inflect the final lexical noun; never lemmatize scientific modifiers.

    This intentionally leaves internal plurals unresolved. It is a conservative
    search form, not a canonical entity identity or an abbreviation expansion.
    """

    def normalize(self, text, start, end, *, analyzed=None, abbreviations=None):
        doc = self.analyze(text) if analyzed is None else analyzed
        base = super().normalize(text, start, end, analyzed=doc, abbreviations=[])
        tokens = base["tokens"]
        lexical = [t for t in tokens if any(c.isalnum() for c in t["text"])]
        head = lexical[-1] if lexical else None
        pieces, cursor, changes = [], start, []
        for t in tokens:
            a, b, raw = t["start"], t["end"], t["text"]
            replacement = raw
            # Attached symbols/digits protect formula fragments and identifiers.
            attached = ((a > start and not text[a - 1].isspace() and text[a - 1] not in "'‘\"") or
                        (b < end and not text[b].isspace() and text[b] not in "'’\""))
            if t["possessive"]:
                replacement = ""
            elif (t is head and not t["partial_token"] and not attached and
                  raw.isascii() and raw.isalpha() and raw.islower() and
                  t["pos"] == "NOUN" and t["tag"] == "NNS" and
                  t["morph"].get("Number") == "Plur" and
                  t["lemma"].isascii() and t["lemma"].isalpha() and
                  t["lemma"].islower()):
                replacement = t["lemma"]
            pieces.extend((text[cursor:a], replacement))
            if replacement != raw:
                changes.append({"start": a, "end": b, "original": raw,
                                "replacement": replacement,
                                "rule": "possessive_POS" if t["possessive"] else "plural_noun_head"})
            cursor = b
        pieces.append(text[cursor:end])
        form = safe_unicode("".join(pieces)) or safe_unicode(text[start:end])
        definitions = detect_abbreviations(text) if abbreviations is None else abbreviations
        candidates = [d for d in definitions if d["short_form"] == text[start:end] or
                      (d["long_start"] == start and d["long_end"] == end)]
        return {**base, "normalized_form": form, "lemma": form,
                "abbreviation_candidates": candidates, "changes": changes,
                "normalizer_version": "scientific-noun-head-nfc-v1"}
