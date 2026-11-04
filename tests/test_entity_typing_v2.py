"""All Qwen tests use a fake client. No test contacts a model or loads weights."""
import copy
import json
from pathlib import Path

import pytest

from cgraph.entity_typing_v2 import analysis, runner
from cgraph.entity_typing_v2.cli import export_results, parser
from cgraph.entity_typing_v2.dataset import (gliner_rows, load_candidates, load_frozen,
                                           prepare, stratified_sample)
from cgraph.entity_typing_v2.ollama import (OllamaClient, OllamaError, allocate_context,
                                          context_choices, validate_config)
from cgraph.entity_typing_v2.parsing import parse_response
from cgraph.entity_typing_v2.prompting import batch_record, messages_for, plan_batches
from cgraph.entity_typing_v2.storage import (digest, exclusive_lock, read_json,
                                           read_jsonl, write_json, write_jsonl)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def forbid_real_http(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Unit tests must never contact Ollama or the network')
    monkeypatch.setattr('urllib.request.urlopen', forbidden)


@pytest.fixture
def config():
    cfg = read_json(ROOT / 'config/entity_typing_v2.json')
    cfg['retry_delay_seconds'] = 0
    return cfg


@pytest.fixture(scope='module')
def saved():
    return load_candidates(read_json(ROOT / 'config/entity_typing_v2.json'), ROOT)


def decision(cid, label='METHOD'):
    return dict(candidate_id=cid, entity_type=label, typing_status='TYPED' if label not in ('UNKNOWN', 'NON_ENTITY') else label,
                confidence_estimate=.83, short_justification='An analytical procedure.')


def fake_model():
    return dict(tag='qwen3.5:4b-q4_K_M', digest='fixed-checkpoint', size_bytes=3_000_000_000,
                details={'quantization_level': 'Q4_K_M'}, ollama_version='0.35.0', native_context=262144,
                running_models=[], model_info={'general.architecture': 'qwen35',
                'general.parameter_count': 4_600_000_000, 'qwen35.context_length': 262144,
                'qwen35.block_count': 32, 'qwen35.attention.head_count_kv': [0, 0, 0, 4] * 8,
                'qwen35.attention.key_length': 256, 'qwen35.attention.value_length': 256})


class FakeClient:
    def __init__(self, action=None):
        self.calls = []
        self.action = action
        self.model = fake_model()
        self.ctx = None

    def discover(self, config):
        return self.model

    def request(self, path, payload=None, metadata=False):
        self.calls.append((path, copy.deepcopy(payload)))
        if path == '/api/generate':
            assert 'prompt' not in payload and payload['options']['num_predict'] == 0
            self.ctx = payload['options']['num_ctx']
            return {'done': True, 'done_reason': 'load', 'load_duration': 1000}
        if path == '/api/ps':
            return {'models': [{'digest': self.model['digest'], 'context_length': self.ctx}]}
        assert path == '/api/chat'
        items = json.loads(payload['messages'][1]['content'])['mentions']
        response = dict(done=True, done_reason='stop', model=self.model['tag'],
                        message={'content': json.dumps({'decisions': [decision(c['candidate_id']) for c in items]}),
                                 'thinking': 'separate thinking' if payload['think'] else ''},
                        prompt_eval_count=100, eval_count=50, prompt_eval_duration=10_000_000,
                        eval_duration=20_000_000, total_duration=40_000_000, load_duration=1000)
        return self.action(payload, response, self) if self.action else response

    @property
    def chats(self):
        return [payload for path, payload in self.calls if path == '/api/chat']


@pytest.fixture
def sandbox(tmp_path, saved, config, monkeypatch):
    candidates, ontology, _ = saved
    # Six source-exact candidates from one document; two initial batches.
    rows = candidates[:6]
    frozen = dict(candidate_digest=digest(rows), ontology_digest=digest(ontology),
                  candidate_ids=[c['candidate_id'] for c in rows], dataset='pilot')
    write_json(tmp_path / 'manifest.json', {'frozen': frozen, 'runtime': None, 'candidate_count': len(rows)})
    write_jsonl(tmp_path / 'candidate_dataset.jsonl', rows)
    write_json(tmp_path / 'ontology.json', ontology)
    write_jsonl(tmp_path / 'failure_log.jsonl', [])
    config['batch_size'] = 4
    monkeypatch.setattr(runner, 'metadata_preflight', lambda client, config: (
        client.model, {'ram_available_bytes': 40_000_000_000}, [262144, 131072], {}))
    monkeypatch.setattr(runner, 'verify_server_serial', lambda: {'mock': True})
    return tmp_path, rows, ontology


def test_real_saved_spans_and_n3(saved):
    rows, ontology, hashes = saved
    assert len(rows) == 1211 and len(ontology['entity_types']) == 11
    assert len({c['document_id'] for c in rows}) == 30
    assert len({c['candidate_id'] for c in rows}) == 1211
    assert sum(len(c['alternative_labels']) for c in rows) == 1327
    assert all(c['context'][c['start']:c['end']] == c['surface_text'] for c in rows)
    assert all(c['top1_score'] >= .35 for c in rows)
    assert all(c['score_margin'] is None for c in rows if c['top2_type'] is None)
    assert all(hashes.values())


def test_sampling_seed_strata_and_input_order(saved):
    rows = saved[0]
    pilot = stratified_sample(rows)
    assert len(pilot) == 240 and pilot == stratified_sample(list(reversed(rows)))
    assert pilot != stratified_sample(rows, seed=4402)
    assert {(c['discipline'], c['top1_type']) for c in pilot} == {(c['discipline'], c['top1_type']) for c in rows}


def test_prepare_idempotent_and_freeze_integrity(tmp_path, config):
    first, rows = prepare(config, ROOT, tmp_path, 'pilot')
    second, again = prepare(config, ROOT, tmp_path, 'pilot')
    assert first == second and rows == again
    broken = read_jsonl(tmp_path / 'candidate_dataset.jsonl')
    broken[0]['surface_text'] = 'tampered'
    write_jsonl(tmp_path / 'candidate_dataset.jsonl', broken)
    with pytest.raises(ValueError, match='integrity'):
        load_frozen(tmp_path)


def test_prompt_no_hints_gold_normalized_form_and_exact_definitions(saved):
    rows, ontology, _ = saved
    c = dict(rows[0], top1_type='LEAK_SENTINEL', normalized_form='NORMALIZATION_SENTINEL', assisted_label='GOLD_SENTINEL')
    prompt = messages_for([c], ontology)
    encoded = json.dumps(prompt, ensure_ascii=False)
    assert all(token not in encoded for token in ('LEAK_SENTINEL', 'GOLD_SENTINEL', 'NORMALIZATION_SENTINEL', 'top1_score', 'alternative_labels'))
    system = prompt[0]['content']
    for label, fields in ontology['entity_types'].items():
        assert label in system and fields['definition_ru'] in system


@pytest.mark.parametrize('label', ['METHOD', 'UNKNOWN', 'NON_ENTITY'])
def test_parser_valid_statuses(label):
    row = decision('c1', label)
    valid, errors = parse_response(json.dumps({'decisions': [row]}), ['c1'], ['METHOD'])
    assert valid == {'c1': row} and not errors


def test_parser_salvages_only_complete_json_objects():
    text = '{"decisions":[' + json.dumps(decision('c1')) + ',{"candidate_id":"c2","entity_type":'
    valid, errors = parse_response(text, ['c1', 'c2'], ['METHOD'])
    assert list(valid) == ['c1']
    assert {e['kind'] for e in errors} >= {'invalid_json', 'missing_ids'}


@pytest.mark.parametrize('wrapper', ['fence', 'single_object'])
def test_parser_recovers_structural_wrappers_without_changing_fields(wrapper):
    row = decision('c1')
    text = json.dumps(row)
    if wrapper == 'fence':
        text = '```json\n{"decisions":[' + text + ']}\n```'
    valid, errors = parse_response(text, ['c1'], ['METHOD'])
    assert valid == {'c1': row} and errors


@pytest.mark.parametrize('mutation', ['type', 'missing_field', 'mismatch', 'confidence', 'duplicate', 'foreign', 'nan'])
def test_parser_rejects_invalid_semantics(mutation):
    row = decision('c1')
    rows = [row]
    if mutation == 'type': row['entity_type'] = 'INVENTED'
    if mutation == 'missing_field': del row['short_justification']
    if mutation == 'mismatch': row['typing_status'] = 'UNKNOWN'
    if mutation == 'confidence': row['confidence_estimate'] = 2
    if mutation == 'duplicate': rows.append(dict(row))
    if mutation == 'foreign': row['candidate_id'] = 'other'
    if mutation == 'nan': row['confidence_estimate'] = float('nan')
    valid, errors = parse_response(json.dumps({'decisions': rows}), ['c1'], ['METHOD'])
    assert not valid and errors


def test_parser_does_not_read_objects_embedded_in_json_strings():
    text = '{"decisions": ["' + json.dumps(decision('c1')).replace('"', '\\"') + '"'
    assert not parse_response(text, ['c1'], ['METHOD'])[0]


def test_context_uses_per_layer_hybrid_kv_metadata(config):
    model = fake_model()
    choices, estimate = context_choices(model, {'ram_available_bytes': 13_000_000_000}, config)
    assert estimate['kv_bytes_per_token'] == 32768
    assert choices[0] == 131072
    config['strict_max_context'] = True
    with pytest.raises(OllamaError, match='Strict'):
        context_choices(model, {'ram_available_bytes': 13_000_000_000}, config)


def test_context_rejects_impossible_ram_and_honors_runtime_limit(config):
    with pytest.raises(OllamaError, match='No context'):
        context_choices(fake_model(), {'ram_available_bytes': 1_000_000}, config)
    config['runtime_context_limit'] = 32768
    assert context_choices(fake_model(), {'ram_available_bytes': 100_000_000_000}, config)[0][0] == 32768


def test_load_only_allocation_checks_actual_context(config):
    client = FakeClient()
    ctx, evidence = allocate_context(client, client.model, [262144], config)
    assert ctx == 262144 and evidence['context_verified'] and not client.chats


def test_load_only_does_not_silently_accept_smaller_context(config):
    class WrongContext(FakeClient):
        def request(self, path, payload=None, metadata=False):
            if path == '/api/ps': return {'models': [{'digest': self.model['digest'], 'context_length': 4096}]}
            return super().request(path, payload, metadata)
    config['strict_max_context'] = True
    client = WrongContext()
    with pytest.raises(OllamaError, match='context mismatch'):
        allocate_context(client, client.model, [262144], config)


def test_batch_context_overflow_is_preflight(saved, config):
    with pytest.raises(ValueError, match='Context overflow'):
        plan_batches(saved[0][:1], saved[1], config, 100)
    plans = plan_batches(saved[0], saved[1], config, 65536)
    assert sum(map(len, plans)) == 1211
    assert all(len({c['document_id'] for c in b}) == 1 for b in plans)
    assert all(len(b) <= 8 for b in plans)


def test_dry_run_never_loads_or_generates(sandbox, config):
    output, rows, ontology = sandbox
    client = FakeClient()
    result = runner.dry_run(output, config, client)
    assert result['model_available'] and result['inference_calls'] == result['model_load_calls'] == 0
    assert client.calls == []


def test_resume_complete_skips_chat_and_load(sandbox, config):
    output, rows, _ = sandbox
    client = FakeClient()
    runner.run_qwen(output, config, client, 'no-thinking')
    assert len(client.chats) == 2
    resumed = FakeClient()
    completed = runner.run_qwen(output, config, resumed, 'no-thinking', resume=True)
    assert len(completed) == 6 and resumed.calls == []


def test_resume_after_interrupt_only_sends_remaining_ids(sandbox, config):
    output, rows, _ = sandbox
    def interrupt(payload, response, client):
        if len(client.chats) == 2: raise KeyboardInterrupt
        return response
    with pytest.raises(KeyboardInterrupt):
        runner.run_qwen(output, config, FakeClient(interrupt), 'no-thinking')
    assert sum(r['processing_status'] == 'COMPLETE' for r in read_jsonl(output / runner.MODE_FILES['no-thinking'])) == 4
    client = FakeClient()
    runner.run_qwen(output, config, client, 'no-thinking', resume=True)
    ids = [m['candidate_id'] for p in client.chats for m in json.loads(p['messages'][1]['content'])['mentions']]
    assert set(ids) == {c['candidate_id'] for c in rows[4:]}


def test_resume_replays_raw_journal_after_snapshot_loss(sandbox, config):
    output, rows, _ = sandbox
    runner.run_qwen(output, config, FakeClient(), 'no-thinking')
    (output / runner.MODE_FILES['no-thinking']).unlink()
    client = FakeClient()
    assert len(runner.run_qwen(output, config, client, 'no-thinking', resume=True)) == len(rows)
    assert not client.calls


def test_resume_recovers_raw_response_before_parser_commit(sandbox, config):
    output, rows, _ = sandbox
    runner.run_qwen(output, config, FakeClient(), 'no-thinking')
    (output / runner.MODE_FILES['no-thinking']).unlink()
    for path in (output / 'attempts/no-thinking').glob('*.json'):
        record = read_json(path)
        record.pop('parsed_finalized')
        record['parsed'] = {}
        write_json(path, record)
    client = FakeClient()
    assert len(runner.run_qwen(output, config, client, 'no-thinking', resume=True)) == len(rows)
    assert not client.calls
    assert all(read_json(p)['parsed_finalized'] for p in (output / 'attempts/no-thinking').glob('*.json'))


def test_partial_json_retries_only_missing_candidates(sandbox, config):
    output, rows, _ = sandbox
    def partial(payload, response, client):
        if len(client.chats) == 1:
            first = json.loads(response['message']['content'])['decisions'][0]
            response['message']['content'] = '{"decisions":[' + json.dumps(first) + ',{"candidate_id":'
        return response
    client = FakeClient(partial)
    completed = runner.run_qwen(output, config, client, 'no-thinking')
    assert len(completed) == 6 and len(client.chats) == 4
    first_id = rows[0]['candidate_id']
    assert all(first_id not in [m['candidate_id'] for m in json.loads(p['messages'][1]['content'])['mentions']] for p in client.chats[1:])
    assert 'invalid_json' in {r['kind'] for r in read_jsonl(output / 'failure_log.jsonl')}


def test_retry_budget_is_bounded_and_failure_rows_are_explicit(sandbox, config):
    output, rows, _ = sandbox
    def fail(*args): raise OllamaError('mock timeout')
    client = FakeClient(fail)
    with pytest.raises(RuntimeError, match='NOT_PROCESSED'):
        runner.run_qwen(output, config, client, 'thinking')
    attempts = [read_json(p) for p in (output / 'attempts/thinking').glob('*.json')]
    for c in rows:
        assert sum(c['candidate_id'] in a['candidate_ids'] for a in attempts) == 3
    assert all(r['processing_status'] == 'NOT_PROCESSED' for r in read_jsonl(output / runner.MODE_FILES['thinking']))


def test_thinking_only_changes_api_think_and_checkpoint_is_locked(sandbox, config):
    output, rows, _ = sandbox
    a, b = FakeClient(), FakeClient()
    runner.run_qwen(output, config, a, 'no-thinking')
    runner.run_qwen(output, config, b, 'thinking')
    for off, on in zip(a.chats, b.chats):
        assert off['think'] is False and on['think'] is True
        assert {k: v for k, v in off.items() if k != 'think'} == {k: v for k, v in on.items() if k != 'think'}
    b.model['digest'] = 'different-checkpoint'
    # A complete resume validates stored provenance offline and does not use a live checkpoint.
    assert len(runner.run_qwen(output, config, b, 'thinking', resume=True)) == len(rows)
    (output / runner.MODE_FILES['thinking']).unlink()
    for path in (output / 'attempts/thinking').glob('*.json'):
        path.unlink()
    with pytest.raises(ValueError, match='changed'):
        runner.run_qwen(output, config, b, 'thinking', resume=True)


@pytest.mark.parametrize('threshold', [.50, .55, .60, .65])
def test_hybrid_threshold_boundary_risk_and_absent_margin(threshold):
    g = dict(top1_score=threshold, score_margin=None)
    assert analysis.route(g, threshold, True)[0] == 'gliner'
    g['score_margin'] = .149
    assert analysis.route(g, threshold, True)[0] == 'qwen'
    g['score_margin'] = .15
    assert analysis.route(g, threshold, True)[0] == 'gliner'
    g['top1_score'] -= .00001
    assert analysis.route(g, threshold)[0] == 'qwen'


def test_offline_hybrid_metrics_export_and_later_gold(sandbox, config, tmp_path):
    output, rows, _ = sandbox
    write_jsonl(output / 'T1_gliner.jsonl', gliner_rows(rows))
    for mode in runner.MODE_FILES:
        runner.run_qwen(output, config, FakeClient(), mode)
    hybrid = analysis.hybrid(output, config)  # this function accepts no client
    assert len(hybrid) == 16 * len(rows)
    metrics = analysis.summarize(output)
    assert len(metrics['hybrid']) == 16
    assert metrics['runtime']['thinking']['input_tokens'] == 200
    assert metrics['comparisons']['T21_vs_T22']['agreement'] == 1
    assert all(v['measured_hybrid_inference_latency_seconds'] is None for v in metrics['hybrid'].values())
    assert 'micro_f1' not in json.dumps(metrics)
    gold_path = output / 'human_gold.jsonl'
    write_jsonl(gold_path, [{'candidate_id': c['candidate_id'], 'entity_type': 'METHOD', 'review_status': 'HUMAN_REVIEWED'} for c in rows])
    evaluation = analysis.evaluate(output, gold_path)
    assert evaluation['methods']['T21']['micro_f1'] == 1
    (output / 'weights.safetensors').write_bytes(b'not a real model')
    destination = output.parent / 'test-results.zip'
    export_results(output, destination, ROOT)
    import zipfile
    with zipfile.ZipFile(destination) as z:
        assert 'manifest.json' in z.namelist() and 'metrics.json' in z.namelist()
        assert 'weights.safetensors' not in z.namelist()
        assert 'archive_sha256_manifest.json' in z.namelist()


def test_discovery_never_substitutes_other_qwen_models(config):
    class MetadataClient(OllamaClient):
        def request(self, path, payload=None, metadata=False):
            if path == '/api/version': return {'version': '0.35.0'}
            if path == '/api/tags': return {'models': [{'name': 'qwen3:4b'}, {'name': 'tev1:4b'}, {'name': 'qwen3.5:9b'}]}
            raise AssertionError('Unexpected request')
    with pytest.raises(OllamaError, match='absent'):
        MetadataClient('unused').discover(config)


def test_config_and_cli_defaults(config):
    validate_config(config)
    args = parser().parse_args(['qwen', '--mode', 'thinking', '--resume', '--num-thread', '3'])
    assert args.dataset is None and config['default_dataset'] == 'full' and args.num_thread == 3 and args.resume
    config['num_parallel'] = 2
    with pytest.raises(ValueError, match='num_parallel'):
        validate_config(config)


def test_process_lock_blocks_second_writer(tmp_path):
    with exclusive_lock(tmp_path / 'lock'):
        with pytest.raises(RuntimeError, match='Another process'):
            with exclusive_lock(tmp_path / 'lock'): pass
