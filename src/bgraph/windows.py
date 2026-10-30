"""Lossless Unicode sentence windows and unambiguous occurrence mapping."""
import re


def sentences(text):
    # Decimal dots, chemical symbols and abbreviations do not end a sentence.
    boundaries = [0]
    for m in re.finditer(r'[.!?](?:["\u201d\u2019)]*)\s+(?=[A-Z(])', text):
        prefix = text[:m.start() + 1]
        if re.search(r'\b(?:e\.g|i\.e|et al|Fig|Figs|Dr|vs|Ref)\.$', prefix):
            continue
        boundaries.append(m.end())
    boundaries.append(len(text))
    result = []
    for a, b in zip(boundaries, boundaries[1:]):
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if a < b:
            result.append(dict(id=len(result), start=a, end=b, text=text[a:b]))
    return result


def windows(text, cfg):
    rows = sentences(text)
    i = 0
    while i < len(rows):
        j = i + 1
        while (j < len(rows) and j - i < cfg['sentences']
               and rows[j]['end'] - rows[i]['start'] <= cfg['max_chars']):
            j += 1
        yield rows[i:j]
        if j == len(rows):
            break
        i = max(i + 1, j - cfg['overlap'])


def anchor(record, rows):
    row = {r['id']: r for r in rows}.get(record['n'])
    if row is None or not record['s'] or type(record['o']) is not int or record['o'] < 1:
        raise ValueError('INVALID_OCCURRENCE_ANCHOR')
    # Overlapping exact substrings are counted; never normalize or default to find().
    hits = [m.start() for m in re.finditer('(?=' + re.escape(record['s']) + ')', row['text'])]
    # A unique exact surface resolves even when a model emits an invalid ordinal.
    # A repeated surface never falls back to a guessed first occurrence.
    if len(hits) == 1:
        return row['start'] + hits[0], row['start'] + hits[0] + len(record['s'])
    if len(hits) < record['o']:
        raise ValueError('SURFACE_NOT_AT_OCCURRENCE')
    a = row['start'] + hits[record['o'] - 1]
    return a, a + len(record['s'])
