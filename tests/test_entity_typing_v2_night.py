"""Unattended orchestration tests: synthetic model outputs, no network or weights."""
import json
import zipfile

import pytest

from test_entity_typing_v2 import (ROOT, FakeClient, config, forbid_real_http,
                                  sandbox, saved)
from cgraph.entity_typing_v2 import analysis, night, runner
from cgraph.entity_typing_v2.cli import export_results, main
from cgraph.entity_typing_v2.completeness import check_completeness
from cgraph.entity_typing_v2.ollama import OllamaError
from cgraph.entity_typing_v2.storage import read_json, read_jsonl, write_jsonl


@pytest.fixture
def orchestration(sandbox, config, monkeypatch):
    output, rows, ontology = sandbox
    monkeypatch.setattr(night, 'prepare', lambda *args: None)
    destination = output.parent / (output.name + '_night.zip')
    return output, rows, destination


def launch(orchestration, config, client):
    output, _, destination = orchestration
    return night.run_night(output, ROOT, config, 'pilot', client, destination, export_results)


def test_night_all_stages_completeness_zip_and_offline_skip(orchestration, config, monkeypatch):
    output, rows, destination = orchestration
    client = FakeClient()
    assert launch(orchestration, config, client) == 0
    report = read_json(output / 'completeness.json')
    assert report['success'] and report['all_ids_accounted_for']
    assert len(report['methods']) == 19
    assert all(m['complete'] == len(rows) for m in report['methods'].values())
    manifest = read_json(output / 'manifest.json')
    run = list(manifest['execution']['night_launches'].values())[-1]
    assert [s['stage'] for s in run['stages']] == ['T1', 'T21', 'T22', 'T3', 'summarize', 'export-results']
    assert manifest['execution']['thinking_flags'] == {'T21': False, 'T22': True}
    assert manifest['execution']['runtime_timings']['thinking']['attempt_count'] == 2
    with zipfile.ZipFile(destination) as archive:
        assert 'completeness.csv' in archive.namelist()
        assert json.loads(archive.read('completeness.json'))['success']
        archived_runs = json.loads(archive.read('manifest.json'))['execution']['night_launches']
        assert list(archived_runs.values())[-1]['stages'][-1]['status'] == 'COMPLETE'
        checksums = json.loads(archive.read('archive_sha256_manifest.json'))
        import hashlib
        assert all(hashlib.sha256(archive.read(name)).hexdigest() == expected for name, expected in checksums.items())
        assert any(name.startswith('night_runs/') for name in archive.namelist())
    # Even an unavailable server cannot stop offline skipping of completed modes.
    def forbidden(*args): raise AssertionError('Completed modes must not query metadata')
    monkeypatch.setattr(runner, 'metadata_preflight', forbidden)
    resumed = FakeClient()
    assert launch(orchestration, config, resumed) == 0
    assert resumed.calls == []
    launches = read_json(output / 'manifest.json')['execution']['night_launches']
    latest = max(launches.values(), key=lambda r: r['updated_at'])
    assert all(s['status'] == 'SKIPPED_COMPLETE' for s in latest['stages'] if s['stage'] in ('T21', 'T22'))


def test_night_failed_t21_continues_t22_and_exports_partial(orchestration, config):
    output, rows, destination = orchestration
    def fail_off(payload, response, client):
        if payload['think'] is False:
            raise OllamaError('mock permanent failure for OFF')
        return response
    client = FakeClient(fail_off)
    assert launch(orchestration, config, client) == 3
    assert any(p['think'] is True for p in client.chats)
    off = read_jsonl(output / runner.MODE_FILES['no-thinking'])
    on = read_jsonl(output / runner.MODE_FILES['thinking'])
    assert len(off) == len(on) == len(rows)
    assert all(r['processing_status'] == 'NOT_PROCESSED' and r['error_reason'] for r in off)
    assert all(r['processing_status'] == 'COMPLETE' for r in on)
    report = check_completeness(output)
    assert not report['success'] and report['all_ids_accounted_for']
    assert 'PARTIAL' in (output / 'experiment_summary.md').read_text(encoding='utf-8')
    assert any(r['kind'] == 'stage_error' for r in read_jsonl(output / 'failure_log.jsonl'))
    assert destination.exists()


def test_night_completed_off_proceeds_directly_to_on(orchestration, config):
    output, rows, _ = orchestration
    runner.run_qwen(output, config, FakeClient(), 'no-thinking')
    client = FakeClient()
    assert launch(orchestration, config, client) == 0
    assert client.chats and all(p['think'] is True for p in client.chats)


def test_one_bad_candidate_does_not_stop_other_batches(sandbox, config):
    output, rows, _ = sandbox
    bad_id = rows[0]['candidate_id']
    def bad_candidate(payload, response, client):
        mentions = json.loads(payload['messages'][1]['content'])['mentions']
        if any(c['candidate_id'] == bad_id for c in mentions):
            raise OllamaError('mock one irrecoverable candidate')
        return response
    with pytest.raises(RuntimeError, match='NOT_PROCESSED'):
        runner.run_qwen(output, config, FakeClient(bad_candidate), 'no-thinking')
    results = read_jsonl(output / runner.MODE_FILES['no-thinking'])
    assert len(results) == len(rows)
    assert sum(r['processing_status'] == 'COMPLETE' for r in results) == len(rows) - 1
    assert next(r for r in results if r['candidate_id'] == bad_id)['processing_status'] == 'NOT_PROCESSED'


def test_night_server_unavailable_still_accounts_for_all_ids(orchestration, config, monkeypatch):
    output, rows, destination = orchestration
    def unavailable(*args): raise OllamaError('mock server unavailable')
    monkeypatch.setattr(runner, 'metadata_preflight', unavailable)
    client = FakeClient()
    assert launch(orchestration, config, client) == 3
    assert not client.calls and destination.exists()
    report = check_completeness(output)
    assert report['all_ids_accounted_for'] and not report['success']
    assert all(report['methods'][m]['not_processed'] == len(rows) for m in ('T21', 'T22'))
    assert len(read_jsonl(output / 'T3_hybrid.jsonl')) == 16 * len(rows)


def test_night_interrupt_exports_and_does_not_start_on(orchestration, config):
    output, rows, destination = orchestration
    def interrupt(payload, response, client): raise KeyboardInterrupt
    client = FakeClient(interrupt)
    assert launch(orchestration, config, client) == 130
    assert destination.exists() and all(p['think'] is False for p in client.chats)
    report = check_completeness(output)
    assert report['all_ids_accounted_for'] and not report['success']


def test_completeness_detects_missing_duplicate_and_stale_hybrid(orchestration, config):
    output, rows, _ = orchestration
    assert launch(orchestration, config, FakeClient()) == 0
    hybrid = read_jsonl(output / 'T3_hybrid.jsonl')
    write_jsonl(output / 'T3_hybrid.jsonl', hybrid[1:])
    assert not check_completeness(output)['success']
    write_jsonl(output / 'T3_hybrid.jsonl', hybrid + [hybrid[0]])
    assert not check_completeness(output)['all_ids_accounted_for']
    write_jsonl(output / 'T3_hybrid.jsonl', hybrid)
    pending = read_jsonl(output / runner.MODE_FILES['no-thinking'])
    pending[0]['processing_status'] = 'NOT_PROCESSED'
    pending[0]['entity_type'] = None
    pending[0]['typing_status'] = 'NOT_PROCESSED'
    write_jsonl(output / runner.MODE_FILES['no-thinking'], pending)
    report = check_completeness(output)
    assert not report['success'] and any('Stale hybrid' in e for e in report['errors'])


def test_full_default_cli_dry_run_never_calls_qwen(tmp_path, monkeypatch):
    from cgraph.entity_typing_v2 import cli
    called = {}
    def prepare(config, root, output, dataset):
        called.update(dataset=dataset, output=output)
        return {'frozen': {'dataset': dataset}}, []
    def dry(*args):
        called['dry'] = True
        return {'model_available': True}
    monkeypatch.setattr(cli, 'prepare', prepare)
    monkeypatch.setattr(cli, 'dry_run', dry)
    monkeypatch.setattr(night, 'run_night', lambda *args: pytest.fail('Dry-run must not run orchestration'))
    assert main(['night', '--dry-run', '--output', str(tmp_path)]) == 0
    assert called['dataset'] == 'full' and called['output'] == tmp_path / 'full' and called['dry']


def test_malformed_chat_envelope_is_bounded_retry(sandbox, config):
    output, rows, _ = sandbox
    def malformed(payload, response, client):
        if len(client.chats) == 1: response['message'] = None
        return response
    completed = runner.run_qwen(output, config, FakeClient(malformed), 'no-thinking')
    assert len(completed) == len(rows)
    # The rejected envelope remains recoverable as an error, never as a successful decision.
    runner.run_qwen(output, config, FakeClient(), 'no-thinking', resume=True)
