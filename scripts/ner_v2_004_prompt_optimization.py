"""NER-V2-004: frozen schema comparison; no training or model changes."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cgraph.ner import windows
from cgraph.ner_v2.research import SHORT, Resources
from cgraph.ner_v2.detection import postprocess

OUT = ROOT / 'reports/ner_v2/prompt_optimization'
HIST = ROOT / 'reports/ner_v2/research_run2'
MODEL = ROOT / 'artifacts/ner_v2/models/gliner25'
LOCK = ROOT / 'contracts/ner_v2/gliner25_research_lock.json'
ONTOLOGY = ROOT / 'contracts/ner_v2/ontology.v2.json'
TYPES = list(SHORT)
V1 = {
 'OBJECT': 'A physical, biological, technical or digital object, system, component or artifact; a concrete entity rather than an abstract concept or method.',
 'ACTOR': 'A person, organization, institution, state or other participant acting as a subject in context; a physical object causing change is not automatically an actor.',
 'CONCEPT': 'An abstract concept, theory, mathematical construction, scientific category or hypothesis, rather than a concrete object or method.',
 'METHOD': 'A defined way to perform, study, analyze or compute something: an algorithm, procedure, technology or experimental technique; the way to solve a task rather than the task or its execution.',
 'PROCESS': 'An action, operation, event, change or sequence of occurring actions; the occurrence itself, rather than a method, object or necessarily an explanatory mechanism.',
 'DATA_RESOURCE': 'A dataset, corpus, registry, archive, sample or source of organized information, distinct from the object described by the information.',
 'ATTRIBUTE': 'A property, parameter, measurable characteristic or metric; the characteristic itself rather than a particular obtained result or an explicit condition.',
 'RESULT': 'An observed, measured, computed or established outcome of a study, experiment, method or process; a particular obtained outcome rather than the metric itself.',
 'CONDITION': 'An explicit experimental, observational or applicability condition under which a statement holds; not simply an abstract concept or a characteristic.',
 'TASK': 'A task, goal, problem or application area to be addressed, rather than the method used to solve it.',
 'MECHANISM': 'A mechanism described in the source that explains how or why a phenomenon occurs; not every process and not merely an object involved.'}
EXAMPLES = {
 'OBJECT': 'Examples: a robot; a concrete neural network model.',
 'ACTOR': 'Examples: a university; a government.',
 'CONCEPT': 'Examples: game theory; a topological space.',
 'METHOD': 'Examples: spectroscopy; the Monte Carlo method.',
 'PROCESS': 'Examples: cell division; chemical reaction.',
 'DATA_RESOURCE': 'Examples: a text corpus; a medical registry.',
 'ATTRIBUTE': 'Examples: mass; classification accuracy as a metric.',
 'RESULT': 'Examples: 94% accuracy; a measured reduction in mortality.',
 'CONDITION': 'Examples: room temperature; fasting state.',
 'TASK': 'Examples: image classification; route optimization.',
 'MECHANISM': 'Examples: immune response mechanism; disease causal mechanism.'}
SCHEMAS = {'V0': dict(SHORT), 'V1': V1, 'V2': {k: V1[k] + ' ' + EXAMPLES[k] for k in TYPES}}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def identity(row): return row['document_id'], int(row['start']), int(row['end'])


def split_docs(docs):
    groups = {}
    for doc in docs: groups.setdefault(doc['discipline'], []).append(doc['document_id'])
    result = {}
    for discipline, ids in sorted(groups.items()):
        ids = sorted(ids); random.Random(4004).shuffle(ids)
        for i, doc_id in enumerate(ids): result[doc_id] = 'test' if i == 0 else 'dev'
    return result


def load_gold(path, docs, candidates):
    rows = jsonl(path); by_doc = {d['document_id']: d['text'] for d in docs}
    expected = {identity(r) for r in candidates}
    if len(expected) != 1211: raise ValueError(f'Expected 1211 unique E3 spans, got {len(expected)}')
    gold = {}
    for row in rows:
        k = identity(row); label = row.get('entity_type', row.get('gold_type'))
        if k in gold or (label not in TYPES and label != "NON_ENTITY"): raise ValueError(f'Duplicate span or invalid gold type: {k}, {label}')       
        doc, start, end = k
        if doc not in by_doc or not 0 <= start < end <= len(by_doc[doc]): raise ValueError(f'Invalid gold offsets: {k}')
        surface = row.get('surface_text', row.get('surface'))
        if surface is not None and surface != by_doc[doc][start:end]: raise ValueError(f'Gold surface mismatch: {k}')
        gold[k] = label
    if set(gold) != expected:
        raise ValueError(f'Gold must cover exactly the original 1211 E3 spans: missing={len(expected-set(gold))}, extra={len(set(gold)-expected)}')
    return gold


def score(gold, predictions):
    # Top score across all labels and windows, stable ontology-order tie break.
    best = {}
    for p in predictions:
        k = identity(p); rank = (float(p['score']), -TYPES.index(p['ner_label_hint']))
        if k not in best or rank > best[k][0]: best[k] = (rank, p['ner_label_hint'])
    matched = set(gold) & set(best); correct = sum(best[k][1] == gold[k] for k in matched)
    entity_keys = {k for k, label in gold.items() if label in TYPES}
    nonentity_keys = set(gold) - entity_keys
    unknown = {gold[k] for k in nonentity_keys} - {'NON_ENTITY'}
    if unknown: raise ValueError(f'Unknown Gold labels: {sorted(unknown)}')
    matched_entities = matched & entity_keys
    matrix = {t: {p: 0 for p in TYPES + ['MISSED']} for t in TYPES + ['NON_ENTITY']}
    for k, label in gold.items(): matrix[label][best[k][1] if k in best else 'MISSED'] += 1
    per = []
    for t in TYPES:
        tp = matrix[t][t]; fn = sum(matrix[t].values()) - tp
        fp = sum(matrix[g][t] for g in TYPES + ['NON_ENTITY'] if g != t)
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        per.append(dict(type=t, support=sum(matrix[t].values()), tp=tp, fp=fp, fn=fn, f1=f1))
    values = [r['f1'] for r in per if r['f1'] is not None]
    return dict(gold=len(gold), matched=len(matched), coverage=len(matched)/len(gold) if gold else None,
                typing_accuracy=correct/len(matched_entities) if matched_entities else None,
                typing_accuracy_all_matched=correct/len(matched) if matched else None,
                gold_entities=len(entity_keys), matched_entities=len(matched_entities),
                gold_nonentity=len(nonentity_keys), matched_nonentity=len(matched & nonentity_keys), correct=correct,
                correct_entity_fraction=correct/len(entity_keys) if entity_keys else None,
                correct_fraction=correct/len(gold) if gold else None,
                macro_f1=sum(values)/len(values) if values else None, new_spans=len(set(best)-set(gold))), per, matrix, best


def split_window(text, a, b):
    word = list(re.finditer(r'\S+', text[a:b]))
    if len(word) < 8: raise ValueError('Schema and minimum text do not fit encoder budget')
    bounds = [m.end() for m in re.finditer(r'[.!?]\s+', text[a:b])]
    middle = min(bounds, key=lambda x: abs(x-(b-a)/2)) if bounds else (b-a)//2
    pivot = min(range(1, len(word)), key=lambda x: abs(word[x].start()-middle))
    overlap = min(25, max(1, len(word)//6))
    le = min(len(word)-1, pivot+overlap//2); rs = max(1, pivot-overlap//2)
    parts = [(a, a+word[le-1].end()), (a+word[rs].start(), b)]
    if any(y-x >= b-a for x,y in parts): raise ValueError('Window splitting made no progress')
    return parts


def shared_windows(text, processor, schemas):
    queue = list(windows(text, 110, 25)); accepted = []; i = 0
    while i < len(queue):
        a,b = queue[i]; i += 1
        records = [processor.transform_record(text[a:b], schema, max_len=None) for schema in schemas]
        sizes = [len(r.input_ids) for r in records]
        if max(sizes) > 512:
            queue[i:i] = split_window(text, a, b)
        else:
            # Verify the processor retained every word token and every schema prompt.
            for record in records:
                source = text[a:b]
                # GLiNER2 appends a period when the input lacks terminal
                # punctuation. Compare against the actual normalized input,
                # while rejecting any changes to the source itself.
                if record.text not in (source, source + '.'):
                    raise ValueError('Processor changed the source text')
                expected = list(processor.word_splitter(record.text, lower=True))
                if (record.start_token_idx != [token[1] for token in expected] or
                        record.end_token_idx != [token[2] for token in expected]):
                    raise ValueError('Text was truncated or token offsets changed')
            accepted.append((a,b,sizes))
    return accepted


def source_prediction(window, item, description):
    """Map a GLiNER result to exact source offsets, excluding its added period."""
    a, b, surface = item['start'], item['end'], item['text']
    if type(a) is not int or type(b) is not int or not 0 <= a < b:
        raise ValueError(f'Invalid model offsets [{a}, {b}) for window length {len(window)}')
    normalized = window if window.endswith(('.', '!', '?')) else window + '.'
    if b > len(normalized) or normalized[a:b].strip() != surface:
        raise ValueError(f'Model span is inconsistent with its window: [{a}, {b}), {surface!r}, length={len(window)}')
    if b > len(window):
        # This prediction includes GLiNER's synthetic punctuation. Do not
        # silently clip it into a different, unpredicted source span.
        return None
    fragment = window[a:b]
    start = a + len(fragment) - len(fragment.lstrip())
    end = b - (len(fragment) - len(fragment.rstrip()))
    if not start < end or window[start:end] != surface:
        raise ValueError(f'Model span cannot be mapped exactly: [{a}, {b}), {surface!r}')
    return dict(start=start, end=end, text=surface, label=description, score=item['confidence'])


def freeze_manifest(out, manifest, resume):
    frozen = out/'manifest.json'
    previous = read_json(frozen) if frozen.exists() else None
    manifest['compatible_checkpoint_hashes'] = (previous.get('compatible_checkpoint_hashes', [])
        if previous and previous.get('script_sha256') == manifest['script_sha256'] else [])
    manifest['config_hash'] = fingerprint({k:v for k,v in manifest.items() if k != 'config_hash'})
    if previous and previous != manifest:
        substantive = lambda m: {k:v for k,v in m.items() if k not in ('script_sha256', 'config_hash', 'compatible_checkpoint_hashes')}
        existing = [p for part in ('dev', 'test') for p in (out/part).glob('*/*.json')]
        # Only the known predecessor can reuse completed, source-validated
        # checkpoints: synthetic spans made it abort before saving a document.
        predecessors = {'a0fda94f0835fbc16178d0deb19636681f42bbcb40f2d2a4a2ecb9b35317f88f',
                        'a374d5e9cabed5675e3783b18baece123235405381482b25e476bddb72e1e87f'}
        reusable = previous.get('script_sha256') in predecessors
        if not resume or (existing and not reusable) or substantive(previous) != substantive(manifest):
            raise ValueError('Frozen manifest mismatch; do not mix experiment configurations')
        backup = out/('manifest_preflight_' + previous['config_hash'] + '.json')
        if backup.exists() and read_json(backup) != previous: raise ValueError('Manifest backup mismatch')
        if not backup.exists(): write_json(backup, previous)
        if existing:
            manifest['compatible_checkpoint_hashes'] = list(dict.fromkeys([previous['config_hash'], *previous.get('compatible_checkpoint_hashes', [])]))
            manifest['config_hash'] = fingerprint({k:v for k,v in manifest.items() if k != 'config_hash'})
    write_json(frozen, manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gold', type=Path, help='JSONL with document_id,start,end,entity_type and optional surface_text')
    p.add_argument('--dry-run', action='store_true', help='Validate inputs and split without loading tokenizer or model')
    p.add_argument('--split', choices=['dev','test'], default='dev')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=OUT/'run.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', encoding='utf-8')
    docs = jsonl(HIST/'input_abstracts.jsonl'); candidates = jsonl(HIST/'E3/mentions_035.jsonl')
    corpus = read_json(HIST/'corpus_manifest.json')
    if digest(HIST/'input_abstracts.jsonl') != corpus['input_sha256']: raise ValueError('Frozen corpus hash mismatch')
    if len(docs) != 30 or len({d['document_id'] for d in docs}) != 30: raise ValueError('Expected 30 unique documents')
    if len({identity(x) for x in candidates}) != 1211: raise ValueError('Expected 1211 unique E3 candidate spans')
    if set(TYPES) != set(read_json(ONTOLOGY)['entity_types']): raise ValueError('Ontology types do not match schemas')
    if SCHEMAS['V0'] != read_json(HIST/'E3/config.json')['labels']: raise ValueError('V0 differs from E3')
    gold = load_gold(args.gold, docs, candidates) if args.gold else None
    splits = split_docs(docs)
    manifest = dict(splits=splits, split_rule='per discipline, sorted IDs shuffled with seed 4004; first test, remaining development',
                    schemas=SCHEMAS, corpus_sha256=digest(HIST/'input_abstracts.jsonl'),
                    candidates_sha256=digest(HIST/'E3/mentions_035.jsonl'), ontology_sha256=digest(ONTOLOGY),
                    model_lock=read_json(LOCK), threshold=0.35, threads=4, window_words=110, overlap_words=25,
                    shared_adaptive_windows=True, gold_sha256=digest(args.gold) if args.gold else None,
                    script_sha256=digest(__file__))
    manifest['config_hash'] = fingerprint(manifest)
    if args.dry_run:
        write_json(OUT/'dry_run.json', dict(status='passed', model_inference=False, tokenizer_checked=False,
                   documents=len(docs), candidate_spans=len({identity(x) for x in candidates}),
                   gold_validated=gold is not None, split_counts={s:list(splits.values()).count(s) for s in ['dev','test']}, manifest=manifest))
        print('Dry-run passed; no model or tokenizer loaded. Gold validated:', gold is not None)
        return
    if gold is None: p.error('--gold is required for inference and evaluation')
    versions = {name: importlib.metadata.version(name) for name in
                ('gliner2', 'torch', 'transformers', 'spacy', 'tqdm', 'psutil')}
    if versions['gliner2'] != '2.1.0': raise ValueError('E3 requires gliner2==2.1.0')
    manifest['package_versions'] = versions
    manifest = freeze_manifest(OUT, manifest, args.resume)
    if args.split == 'test':
        if not all((OUT/'dev'/v/'complete.json').exists() for v in SCHEMAS):
            raise ValueError('Complete all development variants before held-out test')
    for name, meta in manifest['model_lock']['files'].items():
        file = MODEL/name
        if not file.is_file() or digest(file) != meta['sha256']: raise ValueError(f'Checkpoint hash mismatch: {file}')
    os.environ['OMP_NUM_THREADS'] = '4'; os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from tqdm import tqdm
    from gliner2 import AutoExtractor
    from cgraph.ner_v2.scientific_normalization import ScientificNormalizer
    from cgraph.ner_v2.normalization import detect_abbreviations
    torch.set_num_threads(4); torch.manual_seed(42); random.seed(42); torch.use_deterministic_algorithms(True)
    selected = [d for d in docs if splits[d['document_id']] == args.split]
    selected_gold = {k:v for k,v in gold.items() if splits[k[0]] == args.split}
    with Resources() as resources:
        model = AutoExtractor.from_pretrained(str(MODEL), map_location='cpu', local_files_only=True, attention_backend='standard')
        model.to('cpu').float().eval(); normalizer = ScientificNormalizer()
        schemas = [model.create_schema().entities(SCHEMAS[v]) for v in SCHEMAS]
        plans = {d['document_id']: shared_windows(d['text'], model.processor, schemas) for d in selected}
        encoder = [m for m in model.modules() if type(m).__name__ == 'DebertaV2Model']
        if len(encoder) != 1: raise ValueError('Unexpected encoder layout')
        def check_encoder(module, positional, kwargs):
            ids = kwargs.get('input_ids', positional[0] if positional else None)
            if ids is None or ids.shape[-1] > 512: raise ValueError('Unexpected encoder length')
        encoder[0].register_forward_pre_hook(check_encoder, with_kwargs=True)
        all_metrics = []; per_rows = []; matrices = []
        for variant, descriptions in SCHEMAS.items():
            folder = OUT/args.split/variant; folder.mkdir(parents=True, exist_ok=True)
            results = []; seconds = 0.0; dropped = 0
            for doc in tqdm(selected, desc=f'{args.split} {variant}', unit='doc'):
                checkpoint = folder/(doc['document_id']+'.json')
                if checkpoint.exists():
                    if not args.resume: raise ValueError(f'Checkpoint exists, use --resume: {checkpoint}')
                    saved = read_json(checkpoint)
                    if saved['config_hash'] not in [manifest['config_hash'], *manifest['compatible_checkpoint_hashes']] or saved['text_sha256'] != doc['text_sha256']: raise ValueError('Resume mismatch')
                    for item in saved['mentions']:
                        if item['document_id'] != doc['document_id'] or not 0 <= item['start'] < item['end'] <= len(doc['text']) or doc['text'][item['start']:item['end']] != item['surface_text']:
                            raise ValueError('Resume checkpoint has invalid source span')
                else:
                    raw = []; duration = 0.0; synthetic_dropped = 0
                    for a,b,sizes in plans[doc['document_id']]:
                        window = doc['text'][a:b]; start = time.perf_counter()
                        with torch.inference_mode():
                            response = model.extract_entities(window, descriptions, threshold=.35, include_confidence=True, include_spans=True, overlap_policy='allow', max_len=None)
                        duration += time.perf_counter()-start
                        predictions = []
                        for label, items in response['entities'].items():
                            for x in items:
                                prediction = source_prediction(window, x, descriptions[label])
                                if prediction is None:
                                    synthetic_dropped += 1
                                    logging.warning('Dropped synthetic punctuation span: variant=%s doc=%s window=[%s,%s) label=%s span=[%s,%s)', variant, doc['document_id'], a, b, label, x['start'], x['end'])
                                else: predictions.append(prediction)
                        raw.append((a,predictions))
                    analyzed = normalizer.analyze(doc['text']); abbreviations = detect_abbreviations(doc['text'])
                    mentions = []
                    for item in postprocess(doc['text'],raw,{v:k for k,v in descriptions.items()}):
                        normalized = normalizer.normalize(doc['text'],item['start'],item['end'],analyzed=analyzed,abbreviations=abbreviations)
                        mentions.append({'document_id': doc['document_id'], **item, **normalized})
                    saved = dict(config_hash=manifest['config_hash'], text_sha256=doc['text_sha256'], mentions=mentions, inference_seconds=duration, windows=plans[doc['document_id']], synthetic_punctuation_spans_dropped=synthetic_dropped)
                    write_json(checkpoint,saved)
                seconds += saved['inference_seconds']; dropped += saved.get('synthetic_punctuation_spans_dropped', 0); results.extend(saved['mentions'])
            metrics, per, matrix, best = score(selected_gold, results)
            metrics.update(variant=variant, inference_seconds=seconds, peak_rss_bytes=resources.peak,
                           synthetic_punctuation_spans_dropped=dropped)
            all_metrics.append(metrics)
            for row in per: per_rows.append(dict(variant=variant, **row))
            for label, columns in matrix.items():
                for predicted,count in columns.items(): matrices.append(dict(variant=variant,gold=label,predicted=predicted,count=count))
            new = [dict(document_id=k[0],start=k[1],end=k[2],predicted_type=v[1]) for k,v in sorted(best.items()) if k not in selected_gold]
            write_json(folder/'new_spans.json',new)
            write_json(folder/'complete.json',dict(config_hash=manifest['config_hash'],metrics=metrics))
        fields = list(all_metrics[0]); write_csv(OUT/args.split/'metrics.csv',all_metrics,fields)
        base = {r['type']:r['f1'] for r in per_rows if r['variant']=='V0'}
        for r in per_rows: r['delta_f1_vs_v0'] = r['f1']-base[r['type']] if r['f1'] is not None and base[r['type']] is not None else None
        write_csv(OUT/args.split/'per_type.csv',per_rows,list(per_rows[0]))
        write_csv(OUT/args.split/'confusion.csv',matrices,list(matrices[0]))
        lines = ['# NER-V2-004 — '+args.split, '', 'Candidate-set evaluation; new spans are unlabelled and excluded from precision/F1. Coverage and correct / candidates use all original candidates, including NON_ENTITY. Typing accuracy uses matched gold entities only; metrics.csv also gives accuracy across all matches, where matched NON_ENTITY is an error. A missed NON_ENTITY is not a correctly detected and typed entity. NON_ENTITY predictions count as false positives for the predicted type, and appear as a separate confusion-matrix row; they are not an ontology type. Missed gold entities count as FN. Macro-F1 averages types with a nonzero F1 denominator. Top score chooses one type per exact span; ties use ontology order.', '', '| Variant | Coverage | Typing accuracy on matched entities | Correct / candidates | Macro-F1 | New spans | Inference s | Peak RSS bytes |', '|---|---:|---:|---:|---:|---:|---:|---:|']
        for m in all_metrics:
            fmt = lambda x: 'n/a' if x is None else f'{x:.4f}'
            lines.append(f"| {m['variant']} | {fmt(m['coverage'])} | {fmt(m['typing_accuracy'])} | {fmt(m['correct_fraction'])} | {fmt(m['macro_f1'])} | {m['new_spans']} | {m['inference_seconds']:.2f} | {m['peak_rss_bytes']} |")
        lines += ['', 'Predictions touching GLiNER-added punctuation are excluded; counts are in metrics.csv and details in run.log.', 'Peak RSS is process-wide sampled working set (10 ms), cumulative within this invocation and includes model loading; resumed times sum saved per-document inference time. New spans cannot establish full NER recall. See per_type.csv and confusion.csv.']
        (OUT/args.split/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    logging.info('Completed %s config=%s', args.split, manifest['config_hash'])
    print('Completed', args.split, 'reports:', OUT/args.split)

if __name__ == '__main__': main()
