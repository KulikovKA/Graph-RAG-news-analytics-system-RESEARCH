"""Bounded Stage 2F.4 split-context experiment; existing adapter, offline only.

prepare freezes source-only inputs/design/scope; infer resumes fsynced pair rows.
No database, NER, embedding, training, or external model calls are made.
"""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import re
from pathlib import Path
import subprocess
import time

import experiment_cross_document_ontology as onto

prior, core, ROOT = onto.prior, onto.core, onto.ROOT
OUT = onto.previous.OUT / 'split_context_experiment'
SOURCE = onto.SOURCE / 'candidate_evidence.jsonl'
require, banner, progress = onto.require, onto.banner, onto.progress
sha = prior.base.sha
THRESHOLDS = [x / 100 for x in range(35, 91, 5)]
EXPECTED = {
    'P01': ('STAGE_OF', 'E2_TO_E3'),
    'P02': ('POTENTIALLY_APPLICABLE_TO', 'E2_TO_E3'),
    'P03': ('HAS_PARTICIPANT', 'E2_TO_E3'),
    'P04': ('SUBTYPE_OF', 'E2_TO_E3'),
    'P05': ('PART_OF', 'E3_TO_E2'),
    'P06': ('POTENTIALLY_APPLICABLE_TO', 'E2_TO_E3'),
}
SCOPE = {
    'P01': ('BACKGROUND-KNOWLEDGE DEPENDENT', 'Neither abstract defines ESRD as a stage of CKD; the taxonomy is external.'),
    'P02': ('HYPOTHETICAL TRANSFER', 'CKD/low-Hb therapy and a dialysis/TKA cohort do not establish anemia, drug exposure or eligibility in that cohort; also requires the external ESRD/CKD bridge.'),
    'P03': ('TEXT-SUPPORTED SPLIT', 'Only class-level lexical composition: T cell activity in A has T cells as participants; B names T-cells recognizing epitopes. No population identity or Breg-to-rHuEPO mechanism follows. The participant meaning is already recoverable from A; this is not an independent multi-premise scientific reasoning positive.'),
    'P04': ('BACKGROUND-KNOWLEDGE DEPENDENT', 'The sources use overexpression/coexpression/expression for genes but do not explicitly define overexpression as a subtype of gene expression. Conservatively the scientific definition is external; lexical specialization is suggestive, not a stated taxonomy. No yeast copy-number mechanism follows.'),
    'P05': ('BACKGROUND-KNOWLEDGE DEPENDENT', 'The image-transmitting fiber optic bundle is not explicitly described as consisting of individual optical fibers. Component-type inference needs the meaning of bundle; B fibers are not identified as NSTX components.'),
    'P06': ('HYPOTHETICAL TRANSFER', 'MRI OB-NLM and mammographic density processing do not establish cross-modality applicability or preservation of density/risk features; a testable preprocessing proposal only.'),
}


def save(name, obj):
    core.atomic_json(OUT / name, obj)


def load(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def csvout(name, rows):
    import csv
    require(bool(rows), 'Refusing empty CSV ' + name)
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def labels():
    return {r['code']: r['label'] for r in json.loads(
        (onto.OUT / 'cross_document_ontology.json').read_text(encoding='utf-8'))['relations']}


def sentences(text):
    # Deterministic punctuation heuristic, including OpenAlex missing whitespace.
    cuts = [0] + [m.end() for m in re.finditer(r'(?<=[.!?])(?:\s+|(?=[A-Z]))', text)] + [len(text)]
    return [(a, b) for a, b in zip(cuts, cuts[1:]) if a < b]


def enclosing(text, start, end):
    spans = [(a, b) for a, b in sentences(text) if a < end and b > start]
    return min(a for a, _ in spans), max(b for _, b in spans)


def merge_ranges(ranges):
    merged = []
    for a, b in sorted(set(ranges)):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def minimal_range(doc, p, limit=96):
    a, b = enclosing(doc['text'], p['mention_start'], p['mention_end'])
    words = list(core.WORDS.finditer(doc['text']))
    inside = [m for m in words if a <= m.start() < b]
    if len(inside) <= limit:
        return a, b
    first = next(i for i, m in enumerate(words) if m.start() == p['mention_start'])
    last = next(i for i, m in enumerate(words) if m.end() == p['mention_end'])
    room = max(0, limit - (last - first + 1))
    return max(a, words[max(0, first - room // 2)].start()), min(b, words[min(len(words)-1, last + room-room//2)].end())


def build(record, docs, ranges, variant, retrieval=None):
    text, spans, fragments = '', [], []
    for slot in ('E2', 'E3'):
        p = record[slot]['profile']
        d = docs[p['document_id']]
        text += f"[DOCUMENT {d['document_id']}]\n"
        hits = 0
        for index, (a, b) in enumerate(merge_ranges(ranges[slot])):
            if index:
                text += '\n[PASSAGE BREAK]\n'
            offset = len(text)
            text += d['text'][a:b]
            fragments.append(dict(target_slot=slot, document_id=d['document_id'], revision_id=d['revision_id'],
                source_url=d['source_url'], source_reference=d['source_reference'], start=a, end=b,
                text=d['text'][a:b], model_text_start=offset, model_text_end=len(text),
                selected_source_entity_id=p['selected_source_entity_id'],
                source_entity_ids=p['source_entity_ids'], source_mention_ids=p['source_mention_ids'],
                original_mention_start=p['mention_start'], original_mention_end=p['mention_end']))
            if a <= p['mention_start'] < p['mention_end'] <= b:
                hits += 1
                spans.append(dict(entity_id=p['canonical_entity_id'], target_slot=slot,
                    start=offset+p['mention_start']-a, end=offset+p['mention_end']-a,
                    surface_text=p['mention_text'], entity_types=p['entity_types']))
        require(hits == 1, 'Must supply exactly one mention of each target')
        text += f"\n[END DOCUMENT {d['document_id']}]\n\n"
    return dict(pair_id=record['audit_id'], configuration=variant, pool='main', text=text,
        fixed_spans=spans, source_fragments=fragments,
        endpoints={s: record[s]['profile'] for s in ('E2', 'E3')},
        source_audit_id=record['audit_id'], source_audit_file=SOURCE.relative_to(ROOT).as_posix(),
        evidence_ids=record['evidence_ids'], retrieval_refs=retrieval or [],
        synthetic=True, is_source_evidence=False, local_relations_inserted=False,
        input_sha256=core.digest([text, spans]))


def token_count(model, cfg, item, rels):
    words, _, _ = model.prepare_inputs([item['text']])
    prompted, _ = model.data_processor.prepare_inputs(words, cfg['entity_labels'], relations=list(rels.values()))
    return len(model.data_processor.transformer_tokenizer(prompted, is_split_into_words=True,
        truncation=False, padding=False)['input_ids'][0])


def extended(record, docs, initial, model, cfg, rels):
    full = {s: [(0, len(docs[record[s]['profile']['document_id']]['text']))] for s in initial}
    if token_count(model, cfg, build(record, docs, full, 'S2'), rels) <= 512:
        return full
    ranges = {s: list(v) for s, v in initial.items()}
    # Add nearest full sentences while balancing source subword contributions.
    # A final word-boundary fill uses the remaining budget without truncation.
    for mode in ('sentence', 'word'):
        while True:
            candidates = []
            for slot in ('E2', 'E3'):
                d = docs[record[slot]['profile']['document_id']]
                a, b = ranges[slot][0]
                size = len(model.data_processor.transformer_tokenizer(d['text'][a:b], add_special_tokens=False)['input_ids'])
                units = sentences(d['text']) if mode == 'sentence' else [(m.start(), m.end()) for m in core.WORDS.finditer(d['text'])]
                left = [x for x in units if x[0] < a]
                right = [x for x in units if x[1] > b]
                expansions = ([(left[-1][0], b)] if left else []) + ([(a, right[0][1])] if right else [])
                for na, nb in expansions:
                    candidate = {s: list(v) for s, v in ranges.items()}
                    candidate[slot] = [(na, nb)]
                    n = token_count(model, cfg, build(record, docs, candidate, 'S2'), rels)
                    if n <= 512:
                        candidates.append((size, (a-na)+(nb-b), slot, na, nb))
            if not candidates:
                break
            _, _, slot, a, b = min(candidates)
            ranges[slot] = [(a, b)]
    return ranges


def enriched(record, docs, initial, model, cfg, rels):
    ranges = {s: list(v) for s, v in initial.items()}
    candidates = []
    for slot in ('E2', 'E3'):
        p = record[slot]['profile']
        d = docs[p['document_id']]
        for fact in record['local_factual_neighborhood'][slot]:
            require(fact['document_id'] == d['document_id'], 'Nonlocal retrieval fact')
            for ep in fact['source_endpoints']:
                a, b = enclosing(d['text'], ep['mention_start'], ep['mention_end'])
                distance = max(0, a-p['mention_end'], p['mention_start']-b)
                candidates.append((distance, slot, a, b, fact['canonical_triple_id']))
    refs = []
    for distance, slot, a, b, fid in sorted(set(candidates)):
        before = merge_ranges(ranges[slot])
        candidate = {s: list(v) for s, v in ranges.items()}
        candidate[slot] = merge_ranges(before + [(a, b)])
        if candidate[slot] == before:
            refs.append(dict(target_slot=slot, canonical_triple_id=fid, start=a, end=b, status='already_present'))
        elif token_count(model, cfg, build(record, docs, candidate, 'S3'), rels) <= 512:
            ranges = candidate
            refs.append(dict(target_slot=slot, canonical_triple_id=fid, start=a, end=b, status='added'))
        else:
            refs.append(dict(target_slot=slot, canonical_triple_id=fid, start=a, end=b, status='excluded_budget'))
    return ranges, refs


def protected(paths=None):
    if paths is None:
        paths = [p for p in subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0') if p]
    return {p: sha(ROOT/p) for p in progress(paths, desc='Protected source hashes', unit='file')}


def prepare():
    banner('НАЧИНАЮ ФИКСАЦИЮ SPLIT-CONTEXT ДИЗАЙНА, 28 ПАР И ИСХОДНЫХ КОНТЕКСТОВ')
    require(not (OUT/'manifest.json').exists(), 'Already frozen; use infer/resume')
    OUT.mkdir(parents=True, exist_ok=True)
    require((OUT/'experiment_design.md').exists(), 'Write design before preparation')
    records = sorted(core.read_jsonl(SOURCE), key=lambda r: r['audit_id'])
    require(len(records) == 28 and len({r['audit_id'] for r in records}) == 28, 'Incorrect sample')
    docs, _, _ = onto.data()
    rels = labels()
    require(len(rels) == 9 and len(set(rels.values())) == 9, 'Wrong D ontology')
    model, cfg, hashes = prior.initialize()
    rows = {s: [] for s in ('S1', 'S2', 'S3')}
    scope = []
    for r in progress(records, desc='Freezing source-only contexts', unit='pair'):
        require(r['E2']['profile']['document_id'] != r['E3']['profile']['document_id'], 'Intra-document target')
        for slot in ('E2', 'E3'):
            p, d = r[slot]['profile'], docs[r[slot]['profile']['document_id']]
            require(d['text'] == r[slot]['source_document']['text'] and d['revision_id'] == r[slot]['source_document']['revision_id'], 'Changed source')
            require(d['text'][p['mention_start']:p['mention_end']] == p['mention_text'], 'Wrong native offsets')
        initial = {s: [minimal_range(docs[r[s]['profile']['document_id']], r[s]['profile'])] for s in ('E2', 'E3')}
        for limit in (96, 64, 40, 20, 0):
            initial = {s: [minimal_range(docs[r[s]['profile']['document_id']], r[s]['profile'], limit)] for s in initial}
            if token_count(model, cfg, build(r, docs, initial, 'S1'), rels) <= 512:
                break
        else:
            raise ValueError('Targets do not fit')
        s2 = extended(r, docs, initial, model, cfg, rels)
        s3, refs = enriched(r, docs, initial, model, cfg, rels)
        for variant, ranges in [('S1', initial), ('S2', s2), ('S3', s3)]:
            item = build(r, docs, ranges, variant, refs if variant == 'S3' else None)
            n, starts, ends = prior.token_audit(model, cfg, item['text'], rels)
            require(n <= 512 and len(item['fixed_spans']) == 2, 'Budget/targets error')
            for sp in item['fixed_spans']:
                require(item['text'][sp['start']:sp['end']] == sp['surface_text'], 'Wrong joint offsets')
                require(sp['start'] in starts and sp['end'] in ends, 'Unaligned fixed span')
            item['token_counts'] = {variant: n}
            item['full_abstracts'] = {s: ranges[s] == [(0, len(docs[r[s]['profile']['document_id']]['text']))] for s in initial}
            item['equivalent_to'] = next((v for v in rows if v != variant and rows[v] and rows[v][-1]['pair_id'] == item['pair_id'] and rows[v][-1]['input_sha256'] == item['input_sha256']), None)
            rows[variant].append(item)
        category, reason = SCOPE.get(r['audit_id'], ('NEGATIVE / AMBIGUOUS', r['audit_reason'] + ' ' + r['limitation']))
        expected, direction = EXPECTED.get(r['audit_id'], ('', ''))
        sr = dict(pair_id=r['audit_id'], inference_scope=category, expected_D_code=expected,
            expected_direction=direction, reason=reason,
            lexical_only=r['audit_id']=='P03', genuine_two_source_reasoning_positive=False,
            source_audit_file=SOURCE.relative_to(ROOT).as_posix(), evidence_ids=json.dumps(r['evidence_ids'], ensure_ascii=False))
        for s in ('E2', 'E3'):
            p, d = r[s]['profile'], r[s]['source_document']
            sr.update({s+'_'+k: v for k, v in dict(canonical_entity_id=p['canonical_entity_id'],
                native_entity_ids=json.dumps(p['source_entity_ids']), native_mention_ids=json.dumps(p['source_mention_ids']),
                selected_native_entity_id=p['selected_source_entity_id'], source_canonical_ids=json.dumps(p['source_canonical_ids']),
                document_id=p['document_id'], revision_id=d['revision_id'], mention_start=p['mention_start'],
                mention_end=p['mention_end'], mention_text=p['mention_text'], full_source_text=d['text'],
                provenance=json.dumps(d['source_reference'], ensure_ascii=False)).items()})
        scope.append(sr)
    csvout('pair_scope_review.csv', scope)
    # Entirely equivalent S3 has no duplicate files. Partial equivalences are explicit pair-level aliases.
    s3_all_equal = all(a['input_sha256'] == b['input_sha256'] for a, b in zip(rows['S2'], rows['S3']))
    variants = ['S1', 'S2'] + ([] if s3_all_equal else ['S3'])
    for v in variants:
        core.write_jsonl(OUT/f'contexts_{v}.jsonl', rows[v])
    frozen_names = ['experiment_design.md', 'pair_scope_review.csv'] + [f'contexts_{v}.jsonl' for v in variants]
    common = dict(model_id=prior.old.MODEL_ID, checkpoint_sha256=hashes, ontology_file=(onto.OUT/'cross_document_ontology.json').relative_to(ROOT).as_posix(),
        ontology_sha256=sha(onto.OUT/'cross_document_ontology.json'), relation_labels=rels,
        adapter_sha256=sha(Path(core.__file__)), runner_sha256=sha(Path(__file__)),
        entity_labels=cfg['entity_labels'], device='cpu', threads=cfg['threads'], seed=0, deterministic=True,
        token_limit=512, raw_storage_cutoff=-1.0, thresholds=THRESHOLDS, threshold_operator='>=',
        runtime={k: version(k) for k in ['torch', 'gliner', 'transformers', 'tqdm']})
    save('manifest.json', dict(stage='2F.4 split-context experiment', source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        created_utc=datetime.now(timezone.utc).isoformat(), client_date='2026-11-13', variants=variants,
        S3_globally_equivalent_to_S2=s3_all_equal, common_inference=common,
        signatures={v:core.digest(dict(common, configuration=v)) for v in variants},
        frozen_sha256={n:sha(OUT/n) for n in frozen_names}, protected_files_sha256=protected(),
        source_audit_sha256=sha(SOURCE), unique_pairs=28, independent_gold=False,
        model_forwards=0, neo4j_calls=0, ner_calls=0, embedding_calls=0, training_calls=0,
        checkpoints='fsync JSONL after every pair; signature and input hash checked before reuse'))
    save('validation.json', dict(preparation_passed=True, pairs=28, exact_native_and_joint_spans=True,
        tokenizer_and_actual_collator_checked=True, all_nine_label_indexes_verified=True,
        max_tokens={v:max(i['token_counts'][v] for i in rows[v]) for v in variants},
        source_only_fragments=True, additional_target_spans=False,
        scope_counts={c:sum(s['inference_scope']==c for s in scope) for c in sorted({s['inference_scope'] for s in scope})},
        genuine_two_source_reasoning_positives=0, lexical_composition_positives=1,
        prior_sanity_validation_sha256=sha(onto.OUT/'validation.json'),
        prior_adapter_compatibility=onto.load('validation.json')['adapter_compatibility']))


def checked_manifest():
    m = load('manifest.json')
    require(protected(m['protected_files_sha256']) == m['protected_files_sha256'], 'Protected files changed')
    require(sha(Path(__file__)) == m['common_inference']['runner_sha256'], 'Frozen runner changed')
    require(labels() == m['common_inference']['relation_labels'], 'D changed')
    for n, h in m['frozen_sha256'].items():
        require(sha(OUT/n) == h, 'Frozen artifact changed: ' + n)
    return m


def infer():
    m = checked_manifest()
    banner('НАЧИНАЮ OFFLINE FIXED-SPAN RELEX S1/S2/S3; ВСЕ SCORES, ОБА НАПРАВЛЕНИЯ, CHECKPOINT ПОСЛЕ КАЖДОЙ ПАРЫ')
    started = time.perf_counter()
    model, cfg, hashes = prior.initialize()
    require(hashes == m['common_inference']['checkpoint_sha256'], 'Checkpoint changed')
    rels = labels()
    cached_inputs = {}
    forwards = 0
    aliases = 0
    for v in m['variants']:
        path = OUT/f'predictions_{v}.jsonl'
        cached = onto.checkpoint(path, m['signatures'][v])
        contexts = core.read_jsonl(OUT/f'contexts_{v}.jsonl')
        require({pid for _, pid in cached} <= {c['pair_id'] for c in contexts}, 'Foreign pair in checkpoint')
        banner(f'НАЧИНАЮ ВАРИАНТ {v}: 28 ПАР, ПОВТОРНЫЕ ИДЕНТИЧНЫЕ ВХОДЫ ПЕРЕИСПОЛЬЗУЮТСЯ')
        for c in progress(contexts, desc=f'{v} fixed-span inference', unit='pair'):
            key = (c['pair_id'], c['input_sha256'])
            if (v, c['pair_id']) in cached:
                result = cached[v, c['pair_id']]
                require(result['input_sha256'] == c['input_sha256'], 'Checkpoint input changed')
            else:
                if key in cached_inputs:
                    old = cached_inputs[key]
                    result = dict(old, configuration=v, inference_signature=m['signatures'][v],
                        reused_from=dict(configuration=old['configuration'], pair_id=c['pair_id'], input_sha256=c['input_sha256']))
                    aliases += 1
                else:
                    # -1 preserves even exact sigmoid zero; validate all 18 entries below.
                    raw, audit = core.predict_window(model, dict(text=c['text'], start=0, end=len(c['text'])),
                        c['fixed_spans'], cfg['entity_labels'], -1.0, 512, relation_labels=rels)
                    require(audit['tokens'] == c['token_counts'][v] and audit['spans'] == audit['pairs'] == 2, 'Truncation/targets')
                    require(audit['relation_codes'] == list(rels) and audit['relation_prompts'] == list(rels.values()), 'Wrong head prompts')
                    lookup = {(s['start'],s['end']):s for s in c['fixed_spans']}
                    reverse = {label: code for code,label in rels.items()}
                    scores = []
                    for r in raw:
                        h,t = [lookup[r[s]['start'],r[s]['end']] for s in ('head','tail')]
                        require(h != t and all(r[s]['text'] == p['surface_text'] for s,p in [('head',h),('tail',t)]), 'Wrong endpoints')
                        scores.append(dict(head_id=h['entity_id'],tail_id=t['entity_id'],head_slot=h['target_slot'],tail_slot=t['target_slot'],
                            relation_code=reverse[r['relation']],relation_prompt=r['relation'],confidence=r['score']))
                    expected = {(h,t,code) for h,t in [('E2','E3'),('E3','E2')] for code in rels}
                    require(len(scores)==18 and {(s['head_slot'],s['tail_slot'],s['relation_code']) for s in scores}==expected, 'Incomplete raw vector')
                    require(all(0 <= s['confidence'] <= 1 for s in scores), 'Nonfinite raw scores')
                    result = dict(configuration=v,pair_id=c['pair_id'],pool='main', input_sha256=c['input_sha256'],
                        inference_signature=m['signatures'][v],audit=audit,raw_scores=scores,
                        relations=[s for s in scores if s['confidence']>=.35])
                    forwards += 1
                onto.append(path, result)
            cached_inputs[key] = result
    # Reconstruct exact logical forward count from durable checkpoints, including resume.
    all_results = [r for v in m['variants'] for r in core.read_jsonl(OUT/f'predictions_{v}.jsonl')]
    m['model_forwards'] = sum('reused_from' not in r for r in all_results)
    m['reused_identical_inputs'] = sum('reused_from' in r for r in all_results)
    m.setdefault('runs', []).append(dict(new_forwards=forwards,new_aliases=aliases,elapsed_seconds=time.perf_counter()-started,completed_utc=datetime.now(timezone.utc).isoformat()))
    m['prediction_sha256'] = {f'predictions_{v}.jsonl':sha(OUT/f'predictions_{v}.jsonl') for v in m['variants']}
    save('manifest.json',m)
    val = load('validation.json')
    val.update(inference_passed=True,raw_vectors=len(all_results),raw_values=len(all_results)*18,
        all_two_directions_nine_labels=True,no_truncation=True,identical_checkpoint=True,
        threshold_forwards=0,model_forwards=m['model_forwards'],reused_identical_inputs=m['reused_identical_inputs'])
    save('validation.json',val)
    checked_manifest()
    print(json.dumps(dict(forwards=m['model_forwards'],aliases=m['reused_identical_inputs'],positive_rows=sum(len(r['relations']) for r in all_results))),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare','infer'], required=True)
    args = parser.parse_args()
    prepare() if args.stage == 'prepare' else infer()
