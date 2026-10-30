"""Offline Gold evaluation; the prediction run must already be complete and frozen."""
from collections import Counter
from pathlib import Path
from cgraph.common import file_sha256, fingerprint, read_json, write_json
from cgraph.evaluation import entity_key, evaluate, prf
from cgraph.inputs import load_inputs, load_ontology
from cgraph.labels import load_labels
from cgraph.validation import validate_chunk


def evaluate_run(cfg, directory, output):
    directory = Path(directory)
    run = read_json(directory / 'run.json')
    if (run['architecture_id'] != 'B-001' or run['training_performed'] is not False
            or not run['access_audit_complete'] or run['config_sha256'] != fingerprint(cfg)):
        raise ValueError('Run is not a completed frozen B-001 extraction')
    for p, digest in run['raw_files_sha256'].items():
        if file_sha256(directory / p) != digest:
            raise ValueError('Raw prediction trace has changed')
    for p, digest in run['source_code_sha256'].items():
        if file_sha256(p) != digest:
            raise ValueError('Extraction code has changed since prediction')
    for k, p in [('prompt_sha256', cfg['prompt']), ('schema_sha256', cfg['schema']), ('ontology_sha256', cfg['ontology'])]:
        if file_sha256(p) != run[k]:
            raise ValueError(f'Frozen contract changed: {k}')
    docs, ontology = load_inputs(cfg, 'all'), load_ontology(cfg)
    labels = load_labels(cfg, docs)
    predictions = []
    per_relation = {name: {'gold': [], 'raw': [], 'final': []} for name in ontology['relations']}
    examples = {'matched': [], 'unmatched': []}
    rejection = Counter()
    unmappable_by_relation = Counter()
    model_abstain = relation_records = 0
    for doc in docs:
        name = doc['document_code']
        path = directory / f'{name}.json'
        if file_sha256(path) != run['prediction_files_sha256'][name]:
            raise ValueError('Prediction bytes changed')
        pred = read_json(path)
        if pred['architecture_id'] != 'B-001' or pred['config_sha256'] != fingerprint(cfg):
            raise ValueError('Mixed prediction run')
        lookup = {c['chunk_id']: c for c in doc['chunks']}
        for chunk in pred['chunks']:
            validate_chunk(doc, lookup[chunk['chunk_id']], chunk, ontology)
            rejection.update(x['reason'] for x in chunk['rejected'])
            for item in chunk['rejected']:
                if item['stage'] == 'relation_mapping':
                    unmappable_by_relation[item['prediction'].get('r', 'UNKNOWN')] += 1
            relation_records += len(chunk['structurally_validated'])
            model_abstain += sum('MODEL_ABSTAIN' in a['reasons'] for a in chunk['structurally_validated'])
        predictions.append(pred)
        gold = labels[name]
        gm = {m['mention_id']: m for m in gold['mentions']}
        pm = {m['mention_id']: dict(m, chunk_id=c['chunk_id']) for c in pred['chunks'] for m in c['entities']}

        def edge(a, mentions):
            h, t = mentions[a['head_mention_id']], mentions[a['tail_mention_id']]
            return (entity_key(h['chunk_id'], h), entity_key(t['chunk_id'], t), a['relation_type'])

        ge = [a for a in gold['assertions'] if a['support_status'] == 'SUPPORTED']
        gold_counts = Counter(edge(a, gm) for a in ge)
        for a in ge:
            per_relation[a['relation_type']]['gold'].append(edge(a, gm))
        for chunk in pred['chunks']:
            for stage, field in [('raw', 'raw_proposals'), ('final', 'assertions')]:
                for a in chunk[field]:
                    key = edge(a, pm)
                    per_relation[a['relation_type']][stage].append(key)
                    if stage == 'final':
                        matched = gold_counts[key] > 0
                        if matched:
                            gold_counts[key] -= 1
                        category = 'matched' if matched else 'unmatched'
                        if len(examples[category]) < 20:
                            examples[category].append(dict(document=name, relation=a['relation_type'],
                                head=pm[a['head_mention_id']]['surface_text'], head_kind=pm[a['head_mention_id']]['kind'],
                                tail=pm[a['tail_mention_id']]['surface_text'], tail_kind=pm[a['tail_mention_id']]['kind'],
                                polarity=a['polarity'], modality=a['modality'], evidence=a['evidence'][0]['text'],
                                note='Exact Gold match' if matched else 'Unmatched to annotation; not independently adjudicated'))
    result = evaluate(predictions, labels)
    # Count unaddressable LLM relation proposals as false positives as well;
    # the unchanged C metric remains available separately for direct comparison.
    mappable = result['micro']['relation_raw_end_to_end']
    additional = sum(unmappable_by_relation.values())
    tp, ng, np = mappable['tp'], mappable['gold'], mappable['predicted'] + additional
    all_raw = dict(tp=tp, fp=np-tp, fn=ng-tp, gold=ng, predicted=np,
                   precision=tp/np if np else None, recall=tp/ng if ng else None,
                   f1=2*tp/(ng+np) if ng+np else None, unmappable_counted_as_fp=additional)
    result.update(architecture_id='B-001', subset='all', training_performed=False,
                  inferential_status='Exploratory reused DEV; not an independent final test',
                  relation_metrics_scope='C-compatible raw metric uses mappable typed endpoints and evidence. raw_all_proposals additionally counts unmappable relation outputs as FP.',
                  raw_all_proposals=all_raw, unmappable_raw_by_relation=dict(unmappable_by_relation),
                  candidate_policy='C single-sentence directed pair definition applied to B entities; B itself serves wider windows.',
                  model_abstention=dict(abstained=model_abstain, structurally_validated_relation_records=relation_records,
                      rate=model_abstain/relation_records if relation_records else None),
                  rejection_reasons=dict(rejection),
                  per_relation={k: {s: prf(v['gold'], v[s]) for s in ('raw', 'final')} for k, v in per_relation.items()},
                  examples=examples, performance={k: run[k] for k in ('cost', 'llm_calls', 'input_tokens', 'output_tokens', 'retries', 'json_errors', 'request_errors')},
                  d024_limitation='D024 relation annotations are incomplete; unmatched predictions are not necessarily semantically false.',
                  prediction_run_sha256=file_sha256(directory / 'run.json'))
    write_json(output, result)
    return result['micro']
