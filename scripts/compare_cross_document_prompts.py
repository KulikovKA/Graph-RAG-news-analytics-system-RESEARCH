"""Paired A/B/C ReLEx on frozen E2/E3 bridge spans. No graph writes.

Stages: select; pilot (prepares shared input); infer (requires pilot content gate).
Uses the existing local checkpoint and fixed-span adapter, never NER/embeddings.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import analyze_cross_document_re_2f4 as previous

core, base, old = previous.core, previous.base, previous.old
ROOT = base.ROOT
SOURCE = previous.OUT
OUT = SOURCE / 'prompt_comparison'
SEMANTIC = {
    'RELATED_TO': 'related to', 'CONCEPTUALLY_RELATED_TO': 'conceptually related to',
    'SIMILAR_IN_MEANING_TO': 'similar to', 'STUDIES_SAME_PHENOMENON_AS': 'studies the same phenomenon as',
    'USES_SAME_METHOD_AS': 'uses the same method as', 'COMPLEMENTS': 'complements',
    'BUILDS_ON': 'builds on', 'INFLUENCES': 'influences',
    'ADDRESSES_RELATED_PROBLEM': 'addresses a related problem to',
}
HYBRID = ['IS_A', 'HAS_PART', 'HAS_RESULT', 'USES', 'USED_FOR', 'APPLIED_TO',
          'PRODUCES', 'DERIVED_FROM', 'DEPENDS_ON', 'CAUSES', 'ASSOCIATED_WITH']
CONFIGS = {'A': dict(core.PREDICATES), 'B': SEMANTIC,
           'C': {**{k: core.PREDICATES[k] for k in HYBRID}, **SEMANTIC}}
require, progress, banner, dumps = previous.require, previous.progress, previous.banner, previous.dumps


def save(name, obj):
    core.atomic_json(OUT / name, obj)


def load(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def jsonl(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    core.write_jsonl(OUT / name, rows)


def protected():
    paths = [p for p in base.ER.rglob('*') if p.is_file() and not p.is_relative_to(OUT)]
    paths += [base.DOCUMENTS, old.CONFIG, old.ONTOLOGY, ROOT / 'scripts/add_cross_document_relex.py',
              ROOT / 'scripts/analyze_cross_document_re_2f4.py', ROOT / 'scripts/reconstruct_global_kg_2f3.py']
    return {p.relative_to(ROOT).as_posix(): base.sha(p) for p in sorted(set(paths))}


def select():
    banner('НАЧИНАЮ ОТБОР E2–E3 BRIDGE-ПАР; ВНЕШНИЕ E1/E4 НЕ ЯВЛЯЮТСЯ TARGETS')
    OUT.mkdir(parents=True, exist_ok=True)
    require(not (OUT / 'predictions_A.jsonl').exists(), 'Frozen selection already used; refusing overwrite')
    original, mapping, entities, docs, facts, semantic, concepts, profiles = previous.inputs()
    before = previous.live_snapshot()
    pool = {}
    incoming, outgoing = defaultdict(list), defaultdict(list)
    for f in facts:
        h, t = mapping[f['head_id']], mapping[f['tail_id']]
        if h != t:
            incoming[t].append(f['canonical_triple_id'])
            outgoing[h].append(f['canonical_triple_id'])
    def add(a, b, source, source_id):
        if a == b or profiles[a]['document_id'] == profiles[b]['document_id']:
            return
        ids = sorted((a, b))
        pid = previous.identity('bridgepair:', ids)
        if pid not in pool:
            # Prefer the diagram E1 -> E2 ... E3 -> E4 when local edges exist.
            orientations = [ids, ids[::-1]]
            a, b = min(orientations, key=lambda p: (-(bool(incoming[p[0]]) + bool(outgoing[p[1]])), p))
            pool[pid] = dict(pair_id=pid, E2_id=a, E3_id=b, E2_name=profiles[a]['name'], E3_name=profiles[b]['name'],
                E2_document_id=profiles[a]['document_id'], E3_document_id=profiles[b]['document_id'],
                sources={}, source_path_ids=[], local_incoming_E2=sorted(incoming[a]), local_outgoing_E3=sorted(outgoing[b]),
                patient_control=any('patient' in profiles[x]['name'].casefold() for x in (a, b)),
                generic_control=any(profiles[x]['name'].casefold() in {'results', 'study', 'paper', 'author', 'test', 'testing'} for x in (a, b)),
                scientific_identity_assumed=False, targets='E2_E3_only', directions=['E2_to_E3', 'E3_to_E2'])
        row = pool[pid]
        row['sources'].setdefault(source, set()).add(source_id)
        if source == 'evidence_path_intermediates':
            row['source_path_ids'].append(source_id)
    for s in semantic:
        if s['original_cross_document']:
            add(s['head_id'], s['tail_id'], 'cross_SIMILAR_TO', s['semantic_link_id'])
    groups = defaultdict(list)
    for c in concepts:
        groups[c['tail_id']].append(c)
    for cid, links in groups.items():
        for a, b in combinations(links, 2):
            add(a['head_id'], b['head_id'], 'shared_concept', cid)
    for path in progress(core.read_jsonl(SOURCE / 'all_evidence_paths.jsonl'), desc='Selecting actual path intermediates', unit='path'):
        add(*path['intermediate_entity_ids'], 'evidence_path_intermediates', path['path_id'])
    for r in core.read_jsonl(SOURCE / 'candidate_pairs.jsonl'):
        if 'D' in r['sources']:
            add(r['left_id'], r['right_id'], 'cross_component_exploratory', r['pair_id'])
    # Keep every non-patient bridge (including rare scientific terms and explicit
    # homonym controls); diversify twenty patient cases across document pairs.
    selected = [r for r in pool.values() if not r['patient_control']]
    patients_by_docs = defaultdict(list)
    for r in pool.values():
        if r['patient_control']:
            patients_by_docs[tuple(sorted((r['E2_document_id'], r['E3_document_id'])))].append(r)
    for group in patients_by_docs.values():
        group.sort(key=lambda r: (-(bool(r['local_incoming_E2']) + bool(r['local_outgoing_E3'])), r['pair_id']))
    patient_order = []
    while any(patients_by_docs.values()):
        for ds in sorted(patients_by_docs):
            if patients_by_docs[ds]:
                patient_order.append(patients_by_docs[ds].pop(0))
    selected += patient_order[:20]
    selected_ids = {r['pair_id'] for r in selected}
    for r in pool.values():
        r['sources'] = {k: sorted(v) for k, v in r['sources'].items()}
        r['source_path_ids'] = sorted(set(r['source_path_ids']))
        r['selection_reason'] = ('Balanced independent-patient/cohort diagnostic control' if r['patient_control'] else
                                  'All available non-patient bridge pairs, including scientific concepts and ambiguity controls')
        r['selected'] = r['pair_id'] in selected_ids
    rows = sorted(selected, key=lambda r: r['pair_id'])
    require(60 <= len(rows) <= 100, 'Unexpected bridge sample size')
    baseline, memberships = previous.components(profiles, previous.graph_pairs(mapping, facts, semantic, concepts), {c['tail_id'] for c in concepts})
    for r in pool.values():
        r['cross_baseline_components'] = memberships[r['E2_id']] != memberships[r['E3_id']]
    pilot = []
    conditions = [lambda r: r['E2_name'].casefold() == r['E3_name'].casefold() == 'results',
                  lambda r: r['patient_control'], lambda r: 'cross_component_exploratory' in r['sources'],
                  lambda r: set((r['E2_name'].casefold(), r['E3_name'].casefold())) == {'supply line', 'electric supply line'},
                  lambda r: set((r['E2_name'].casefold(), r['E3_name'].casefold())) == {'grains', 'grain'},
                  lambda r: 'energy consumption' in r['E2_name'].casefold(),
                  lambda r: 'breast cancer' in r['E2_name'].casefold(),
                  lambda r: 'medical complications' in r['E2_name'].casefold()]
    for condition in conditions:
        item = next((r for r in rows if condition(r) and r not in pilot), None)
        if item:
            pilot.append(item)
    for discipline in sorted({p['discipline'] for p in profiles.values()}):
        item = next((r for r in rows if discipline in [profiles[r[k]]['discipline'] for k in ('E2_id', 'E3_id')] and r not in pilot), None)
        if item:
            pilot.append(item)
    pilot += [r for r in rows if r not in pilot][:max(0, 14 - len(pilot))]
    for r in rows:
        r['pilot'] = r in pilot
    jsonl('bridge_pool.jsonl', sorted(pool.values(), key=lambda r: r['pair_id']))
    jsonl('selected_bridge_pairs.jsonl', progress(rows, desc='Freezing common sample', unit='pair'))
    save('configurations.json', {k: dict(name={'A': 'Original', 'B': 'Semantic', 'C': 'Hybrid'}[k],
        relation_labels=v, ordered_relation_labels=list(v.items()), relation_prompt_count=len(v)) for k, v in CONFIGS.items()})
    require([len(v) for v in CONFIGS.values()] == [20, 9, 20], 'Incorrect label counts')
    require(all(len(set(v.values())) == len(v) for v in CONFIGS.values()), 'Duplicate prompt text')
    input_paths = [SOURCE / name for name in ('candidate_pairs.jsonl', 'all_evidence_paths.jsonl', 'entity_profiles.jsonl', 'source_factual_evidence.jsonl')]
    summary = dict(pool=len(pool), selected=len(rows), patient_controls=sum(r['patient_control'] for r in rows),
        generic_controls=sum(r['generic_control'] for r in rows), pilot_pairs=len(pilot),
        selected_documents=len({r[k] for r in rows for k in ('E2_document_id', 'E3_document_id')}),
        selected_disciplines=sorted({profiles[r[k]]['discipline'] for r in rows for k in ('E2_id', 'E3_id')}),
        by_source={s: dict(pool=sum(s in r['sources'] for r in pool.values()), selected=sum(s in r['sources'] for r in rows)) for s in sorted({s for r in pool.values() for s in r['sources']})},
        baseline=baseline)
    save('manifest.json', dict(stage='2F.4 paired prompt comparison', source_commit='2e7c740',
        created_utc=datetime.now(timezone.utc).isoformat(), source_inputs_sha256={str(p.relative_to(ROOT)): base.sha(p) for p in input_paths},
        protected_files_sha256=protected(), before_neo4j=before, selection=summary,
        old_adapter_sha256=__import__('hashlib').sha256(subprocess.check_output(['git', 'show', '2e7c740:scripts/build_relex_graphs.py'])).hexdigest(),
        adapter_sha256=base.sha(Path(core.__file__)), adapter_change='Optional relation_labels parameter; original default unchanged',
        neo4j_import=False, thresholds=previous.THRESHOLDS))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def initialize():
    cfg = json.loads(old.CONFIG.read_text(encoding='utf-8'))
    hashes = {name: base.sha(old.MODEL_PATH / name) for name in cfg['models']['base']}
    require(hashes == cfg['models']['base'], 'Local checkpoint changed')
    _, _, _, docs, _, _, _, _ = previous.inputs()
    old.initialize(cfg, old._entities, docs)
    return old._model, cfg, hashes


def common_context(pair, profiles, fact_index, radius, include_local):
    text, members, fragments, local = '', [], [], []
    for slot, field in [('E2', 'local_incoming_E2'), ('E3', 'local_outgoing_E3')]:
        p = profiles[pair[slot + '_id']]
        doc = old._documents[p['document_id']]
        context = p['source_context']
        low, high = max(context['start'], p['mention_start'] - radius), min(context['end'], p['mention_end'] + radius)
        text += f"[DOCUMENT {p['document_id']}]\n"
        start = len(text) + p['mention_start'] - low
        end = len(text) + p['mention_end'] - low
        text += doc['text'][low:high] + '\n'
        members.append(dict(entity_id=p['canonical_entity_id'], target_slot=slot, start=start, end=end,
            surface_text=p['mention_text'], entity_types=p['entity_types']))
        fragments.append(dict(document_id=p['document_id'], start=low, end=high, text=doc['text'][low:high],
            revision_id=doc['revision_id'], source_url=doc['source_url'],
            source_entity_id=p['selected_source_entity_id'], source_mention_ids=p['source_mention_ids'],
            original_mention_start=p['mention_start'], original_mention_end=p['mention_end']))
        if include_local and pair[field]:
            # One actual directed local edge per target. Never bridge to outer endpoints.
            fid = pair[field][0]
            f = fact_index[fid]
            a, b = f['source_endpoints']
            text += f"[LOCAL RELATION] {a['name']} --{core.PREDICATES[f['relation_type']]}--> {b['name']}\n"
            local.append(dict(target_slot=slot, **f))
        text += f"[END DOCUMENT {p['document_id']}]\n"
    # Provenance identifies the existing bridge but does not insert a target label
    # (e.g. 'similar to') into the text, which would confound the label comparison.
    return dict(pair_id=pair['pair_id'], text=text, fixed_spans=members, source_fragments=fragments,
        local_factual_context=local, source_path_ids=pair['source_path_ids'], synthetic=True,
        is_source_evidence=False, context_radius=radius, local_context_included=include_local,
        context_recipe='two exact target mentions + one directed local factual flank per target if available; no outer targets')


def token_audit(model, cfg, text, labels):
    tokens, starts, ends = model.prepare_inputs([text])
    prompted, _ = model.data_processor.prepare_inputs(tokens, cfg['entity_labels'], relations=list(labels.values()))
    count = len(model.data_processor.transformer_tokenizer(prompted, is_split_into_words=True,
        truncation=False, padding=False)['input_ids'][0])
    if count > 512:
        return count, starts[0], ends[0]  # Reject this common window before any truncating collator.
    # Actual collator check as well, before any model forward or sample inference.
    collator = model.data_collator_class(model.config, data_processor=model.data_processor, return_tokens=True,
        return_entities=True, return_id_to_classes=True, return_rel_id_to_classes=True, prepare_labels=False)
    batch = collator(model.prepare_base_input(tokens), entity_types=cfg['entity_labels'], relation_types=list(labels.values()))
    require(batch['input_ids'].shape[1] == count, 'Tokenizer and collator token counts differ')
    mapping = batch['rel_id_to_classes']
    if isinstance(mapping, list):
        mapping = mapping[0]
    require([mapping[i + 1] for i in range(len(labels))] == list(labels.values()), 'Prepared relation mapping differs')
    return count, starts[0], ends[0]


def prepare_contexts(model, cfg, pairs):
    _, _, _, _, _, _, _, profiles = previous.inputs()
    fact_index = {f['canonical_triple_id']: f for f in core.read_jsonl(SOURCE / 'source_factual_evidence.jsonl')}
    rows = []
    for pair in progress(pairs, desc='Freezing identical A/B/C contexts', unit='pair'):
        for include_local, radius in [(True, 160), (True, 120), (True, 80), (True, 40), (True, 0), (False, 80), (False, 0)]:
            item = common_context(pair, profiles, fact_index, radius, include_local)
            counts = {}
            for name, labels in CONFIGS.items():
                n, starts, ends = token_audit(model, cfg, item['text'], labels)
                counts[name] = n
                for m in item['fixed_spans']:
                    require(item['text'][m['start']:m['end']] == m['surface_text'], 'Incorrect common offsets')
                    require(m['start'] in starts and m['end'] in ends, 'Unaligned target span')
            if max(counts.values()) <= 512:
                item['token_counts'] = counts
                item['input_sha256'] = core.digest([item['text'], item['fixed_spans']])
                rows.append(item)
                break
        else:
            raise ValueError('Complete target spans cannot fit every configuration')
    jsonl('prepared_contexts.jsonl', rows)
    return rows


def checkpoints(name, signature):
    path = OUT / f'predictions_{name}.jsonl'
    if not path.exists():
        return {}
    data = path.read_bytes()
    if data and not data.endswith(b'\n'):
        prefix = data[:data.rfind(b'\n') + 1]
        save(f'checkpoint_recovery_{name}.json', dict(discarded_incomplete_tail_bytes=len(data) - len(prefix)))
        path.write_bytes(prefix)
    rows = core.read_jsonl(path)
    require(len({r['pair_id'] for r in rows}) == len(rows), 'Duplicate cached pair')
    require(all(r['inference_signature'] == signature and r['configuration'] == name for r in rows), 'Mixed configuration checkpoint')
    return {r['pair_id']: r for r in rows}


def predict(model, cfg, context, name, signature, use_default=False):
    window = dict(text=context['text'], start=0, end=len(context['text']),
        window_id='bridgeinput:' + context['input_sha256'], document_id='synthetic:' + context['pair_id'])
    spans = [dict(m, document_id=window['document_id']) for m in context['fixed_spans']]
    kwargs = {} if use_default else dict(relation_labels=CONFIGS[name])
    raw, audit = core.predict_window(model, window, spans, cfg['entity_labels'], .35, 512, **kwargs)
    require(audit['tokens'] == context['token_counts'][name] and audit['spans'] == 2 and audit['pairs'] == 2, 'Inference truncation/span mismatch')
    if not use_default:
        require(audit['relation_prompt_count'] == len(CONFIGS[name]) and audit['relation_prompts'] == list(CONFIGS[name].values()), 'Wrong relation prompts in head')
    lookup = {(m['start'], m['end']): m for m in spans}
    reverse = {label: code for code, label in CONFIGS[name].items()}
    relations = []
    for row in raw:
        h, t = [lookup[(row[s]['start'], row[s]['end'])] for s in ('head', 'tail')]
        require(all(row[s]['text'] == m['surface_text'] for s, m in zip(('head', 'tail'), (h, t))), 'Wrong target span mapping')
        code = reverse[row['relation']]
        relations.append(dict(prediction_id=previous.identity('prompt:' + name + ':', [h['entity_id'], code, t['entity_id']]),
            head_id=h['entity_id'], tail_id=t['entity_id'], head_slot=h['target_slot'], tail_slot=t['target_slot'],
            relation_code=code, relation_prompt=row['relation'], confidence=row['score'],
            experimental=True, direct_text_support=False))
    result = dict(configuration=name, pair_id=context['pair_id'], inference_signature=signature,
        input_sha256=context['input_sha256'], input_file='prepared_contexts.jsonl', audit=audit,
        relations=sorted(relations, key=lambda r: r['prediction_id']), status='positive' if relations else 'negative')
    return result, raw


def inference(stage):
    manifest = load('manifest.json')
    require(protected() == manifest['protected_files_sha256'], 'Source files changed')
    require(base.sha(Path(core.__file__)) == manifest['adapter_sha256'], 'Adapter changed since sample freeze')
    pairs = core.read_jsonl(OUT / 'selected_bridge_pairs.jsonl')
    banner('НАЧИНАЮ ЛОКАЛЬНУЮ ЗАГРУЗКУ GLINER RELEX; OFFLINE, CPU, БЕЗ ОБУЧЕНИЯ')
    started = time.perf_counter()
    model, cfg, model_hashes = initialize()
    if (OUT / 'prepared_contexts.jsonl').exists():
        contexts = core.read_jsonl(OUT / 'prepared_contexts.jsonl')
    else:
        require(stage == 'pilot', 'Prepare shared contexts during pilot first')
        contexts = prepare_contexts(model, cfg, pairs)
    by_id = {r['pair_id']: r for r in contexts}
    common_signature = dict(model=old.MODEL_ID, checkpoint_sha256=model_hashes,
        adapter_sha256=base.sha(Path(core.__file__)), runner_sha256=base.sha(Path(__file__)),
        selected_pairs_sha256=base.sha(OUT / 'selected_bridge_pairs.jsonl'), contexts_sha256=base.sha(OUT / 'prepared_contexts.jsonl'),
        configurations_sha256=base.sha(OUT / 'configurations.json'), min_threshold=.35, token_limit=512,
        entity_labels=cfg['entity_labels'], threads=cfg['threads'], device='cpu', seed=0,
        runtime={k: version(k) for k in ('torch', 'gliner', 'transformers', 'tqdm')})
    signatures = {name: core.digest(dict(common_signature, configuration=name, ordered_relation_labels=list(labels.items())))
                  for name, labels in CONFIGS.items()}
    if stage == 'infer':
        gate = load('pilot_validation.json')
        require(gate['technical_passed'] and gate.get('content_reviewed') and gate['signatures'] == signatures, 'Pilot content/technical gate not passed')
        require(base.sha(OUT / 'pilot_review.csv') == gate['content_review_sha256'], 'Pilot review changed after gate')
    else:
        # Prove the old default path is unchanged by the adapter extension: same
        # checkpoint/input, historical function vs current default vs explicit A.
        import types
        historical = types.ModuleType('historical_core_2e7c740')
        historical.__file__ = str(core.__file__)
        exec(compile(subprocess.check_output(['git', 'show', '2e7c740:scripts/build_relex_graphs.py']).decode('utf-8'),
                     historical.__file__, 'exec'), historical.__dict__)
        item = next(by_id[p['pair_id']] for p in pairs if p['E2_name'].casefold() == p['E3_name'].casefold() == 'results')
        current, current_raw = predict(model, cfg, item, 'A', signatures['A'], use_default=True)
        explicit, explicit_raw = predict(model, cfg, item, 'A', signatures['A'])
        w = dict(text=item['text'], start=0, end=len(item['text']))
        old_raw, old_audit = historical.predict_window(model, w, item['fixed_spans'], cfg['entity_labels'], .35, 512)
        require(current_raw == explicit_raw == old_raw and current['audit'] == old_audit, 'Default intra-document adapter behavior changed')
        save('adapter_compatibility.json', dict(passed=True, exact_raw_scores=True, default_audit_unchanged=True,
            original_default_equals_current_default_equals_explicit_A=True, test_pair_id=item['pair_id'],
            historical_adapter_sha256=manifest['old_adapter_sha256'], new_adapter_sha256=manifest['adapter_sha256']))
    statistics = {}
    for name, labels in CONFIGS.items():
        banner(f'НАЧИНАЮ {"PILOT" if stage == "pilot" else "ОСНОВНОЙ ПРОГОН"} {name}: {len(labels)} RELATION LABELS')
        cached = checkpoints(name, signatures[name])
        require(set(cached) <= set(by_id), 'Unknown checkpoint pair')
        tasks = [p for p in pairs if stage == 'infer' or p['pilot']]
        with (OUT / f'predictions_{name}.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
            pending = [p for p in tasks if p['pair_id'] not in cached]
            for pair in progress(pending, desc=f'{name} fixed-span {stage}', unit='pair'):
                result, _ = predict(model, cfg, by_id[pair['pair_id']], name, signatures[name])
                stream.write(dumps(result) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                cached[pair['pair_id']] = result
        statistics[name] = dict(completed_pairs=len(cached), positive_pairs=sum(bool(r['relations']) for r in cached.values()),
            predictions=sum(len(r['relations']) for r in cached.values()), actual_new_forward_calls=len(pending))
        if stage == 'infer':
            require(set(cached) == set(by_id), 'Incomplete configuration')
        print(json.dumps(statistics[name]), flush=True)
    if stage == 'pilot':
        save('pilot_validation.json', dict(technical_passed=True, content_reviewed=False, signatures=signatures,
            shared_pilot_ids=[p['pair_id'] for p in pairs if p['pilot']], statistics=statistics,
            offsets_exact=True, target_spans_only=True, all_label_maps_verified=True,
            prompt_counts={k: len(v) for k, v in CONFIGS.items()}, common_contexts_frozen=True))
    manifest['inference'] = dict(common_signature, signatures=signatures, statistics=statistics,
        stage=stage, elapsed_current_run_seconds=time.perf_counter() - started,
        single_inference_per_configuration_pair=True, negative_results_saved=True, checkpoint_resume=True)
    save('manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['select', 'pilot', 'infer'], required=True)
    args = parser.parse_args()
    if args.stage == 'select':
        select()
    else:
        inference(args.stage)


if __name__ == '__main__':
    main()
