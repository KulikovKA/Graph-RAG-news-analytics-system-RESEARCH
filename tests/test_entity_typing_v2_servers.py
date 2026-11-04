"""Independent server scheduling and process lifecycle; all inference is mocked."""
import copy
import json
import threading
from types import SimpleNamespace

import pytest

from test_entity_typing_v2 import (ROOT, FakeClient, config, fake_model,
                                  forbid_real_http, sandbox, saved)
from cgraph.entity_typing_v2 import runner, servers
from cgraph.entity_typing_v2.ollama import OllamaClient, OllamaError, context_choices, allocate_context, validate_config
from cgraph.entity_typing_v2.storage import read_json, read_jsonl


class FakePool:
    def __init__(self, clients):
        self.clients = clients
        self.model = fake_model()
        for index, client in enumerate(clients):
            client.base_url = f'http://127.0.0.1:{11434 + index}'
        self.model['instances'] = [dict(fake_model(), base_url=c.base_url) for c in clients]

    def request(self, path, payload=None, metadata=False):
        if path == '/api/ps':
            return {'models': [{**c.request(path, metadata=metadata)['models'][0], 'base_url': c.base_url} for c in self.clients]}
        assert path == '/api/generate'
        responses = [c.request(path, payload) for c in self.clients]
        return {'done': True, 'load_duration': sum(r['load_duration'] for r in responses)}


def configure_pair(config):
    config['server_urls'] = ['http://127.0.0.1:11434', 'http://127.0.0.1:11435']


def test_two_servers_really_overlap_and_resume_without_calls(sandbox, config, monkeypatch):
    output, rows, _ = sandbox
    configure_pair(config)
    barrier = threading.Barrier(2)
    active = {'count': 0, 'peak': 0}
    lock = threading.Lock()
    def concurrent(payload, response, client):
        with lock:
            active['count'] += 1
            active['peak'] = max(active['peak'], active['count'])
        barrier.wait(timeout=5)  # fails if the scheduler sends requests serially
        with lock: active['count'] -= 1
        return response
    pool = FakePool([FakeClient(concurrent), FakeClient(concurrent)])
    assert len(runner.run_qwen(output, config, pool, 'no-thinking')) == len(rows)
    assert active['peak'] == 2
    attempts = [read_json(p) for p in (output / 'attempts/no-thinking').glob('*.json')]
    assert {a['server_url'] for a in attempts} == set(config['server_urls'])
    assert all(a['request']['think'] is False for a in attempts)
    assert {cid for a in attempts for cid in a['candidate_ids']} == {c['candidate_id'] for c in rows}
    def forbidden(*args): raise AssertionError('complete resume must be offline')
    monkeypatch.setattr(runner, 'metadata_preflight', forbidden)
    resumed = FakePool([FakeClient(), FakeClient()])
    runner.run_qwen(output, config, resumed, 'no-thinking', resume=True)
    assert not any(c.calls for c in resumed.clients)


def test_two_replicas_memory_budget_and_reclaim_not_double_counted(config):
    a = dict(fake_model(), base_url='http://localhost:11434', observed_url='http://localhost:11434',
             running_models=[{'digest': 'fixed-checkpoint', 'size': 1_000_000_000, 'size_vram': 0}])
    b = dict(a, base_url='http://127.0.0.1:11435', observed_url='http://127.0.0.1:11434')
    model = dict(a, instances=[a, b])
    choices, proof = context_choices(model, {'ram_available_bytes': 30_000_000_000}, config)
    assert proof['model_instances'] == 2 and proof['context_is_per_instance']
    assert proof['reclaimable_loaded_model_bytes'] == 1_000_000_000
    assert proof['estimates'][0]['estimated_ram_bytes'] > 25_000_000_000
    with pytest.raises(OllamaError, match='No context'):
        context_choices(model, {'ram_available_bytes': 5_000_000_000}, config)


def test_allocation_requires_context_on_both_servers(config):
    configure_pair(config)
    pool = FakePool([FakeClient(), FakeClient()])
    ctx, proof = allocate_context(pool, pool.model, [32768], config)
    assert ctx == 32768 and len(proof['observed_running_models']) == 2
    assert all(c.ctx == 32768 for c in pool.clients)


def test_discovery_rejects_different_worker_checkpoint(config, monkeypatch):
    configure_pair(config)
    def discover(self, config):
        return dict(fake_model(), digest='different' if self.base_url.endswith('11435') else 'fixed-checkpoint')
    monkeypatch.setattr(OllamaClient, 'discover', discover)
    with pytest.raises(OllamaError, match='identical'):
        servers.OllamaPool(config).discover(config)


def test_dry_run_metadata_fallback_never_starts_server(config, monkeypatch):
    configure_pair(config)
    def discover(self, config):
        if self.base_url.endswith('11435'): raise OllamaError('not started')
        return fake_model()
    monkeypatch.setattr(OllamaClient, 'discover', discover)
    monkeypatch.setattr(servers.subprocess, 'Popen', lambda *a, **k: pytest.fail('Dry-run must not start a server'))
    metadata = servers.OllamaPool(config, dry_run=True).discover(config)
    assert [i['ready'] for i in metadata['instances']] == [True, False]


def test_managed_server_reuses_existing_and_cleans_only_owned(config, tmp_path, monkeypatch):
    configure_pair(config)
    spawned = []
    state = {'second_started': False}
    class Process:
        pid = 1234
        stopped = False
        def poll(self): return 0 if self.stopped else None
        def terminate(self): self.stopped = True
        def wait(self, timeout): return 0
    process = Process()
    def popen(argv, **kwargs):
        spawned.append((argv, kwargs))
        state['second_started'] = True
        return process
    def request(self, path, payload=None, metadata=False):
        assert path == '/api/version'
        if self.base_url.endswith('11435') and not state['second_started']: raise OllamaError('missing')
        return {'version': '0.35.0'}
    monkeypatch.setattr(OllamaClient, 'request', request)
    monkeypatch.setattr(servers.shutil, 'which', lambda name: 'ollama.exe')
    monkeypatch.setattr(servers.subprocess, 'Popen', popen)
    with servers.ManagedServers(config, tmp_path) as manager:
        assert len(manager.owned) == 1 and not process.stopped
        assert len(spawned) == 1
        assert spawned[0][1]['env']['OLLAMA_NUM_PARALLEL'] == '1'
        assert spawned[0][1]['env']['OLLAMA_HOST'] == '127.0.0.1:11435'
    assert process.stopped


def test_one_failed_server_preserves_every_candidate(sandbox, config):
    output, rows, _ = sandbox
    configure_pair(config)
    def fail(*args): raise OllamaError('mock worker unavailable')
    pool = FakePool([FakeClient(), FakeClient(fail)])
    try:
        runner.run_qwen(output, config, pool, 'thinking')
    except RuntimeError as exc:
        assert 'NOT_PROCESSED' in str(exc)
    results = read_jsonl(output / runner.MODE_FILES['thinking'])
    assert len(results) == len(rows)
    assert {r['candidate_id'] for r in results} == {c['candidate_id'] for c in rows}
    assert all(r['processing_status'] in ('COMPLETE', 'NOT_PROCESSED') for r in results)


def test_alias_endpoints_cannot_share_same_server(config):
    config['server_urls'] = ['http://localhost:11434', 'http://127.0.0.1:11434']
    with pytest.raises(ValueError, match='distinct'):
        validate_config(config)


def test_unusable_allocated_context_is_not_frozen(sandbox, config, monkeypatch):
    output, _, _ = sandbox
    monkeypatch.setattr(runner, 'allocate_context', lambda *a, **k: (8192, {}))
    client = FakeClient()
    with pytest.raises(ValueError, match='Context overflow'):
        runner.run_qwen(output, config, client, 'no-thinking')
    assert read_json(output / 'manifest.json')['runtime'] is None
    assert not client.chats


def test_client_cancel_shuts_down_active_sockets():
    client = OllamaClient('http://localhost:11434')
    closed = []
    connection = SimpleNamespace(sock=SimpleNamespace(shutdown=lambda how: closed.append(how)))
    client._connections.add(type('Connection', (), {'sock': connection.sock})())
    client.cancel_pending()
    assert closed and client._cancelled.is_set()


def test_interrupt_cancels_other_worker_and_recovers_received_response(sandbox, config):
    output, rows, _ = sandbox
    configure_pair(config)
    entered = threading.Barrier(2)
    cancelled = threading.Event()
    def interrupt(payload, response, client):
        entered.wait(timeout=5)
        raise KeyboardInterrupt
    def survivor(payload, response, client):
        entered.wait(timeout=5)
        assert cancelled.wait(timeout=5)
        return response  # response arrived at the time of cancellation; must not be lost
    a, b = FakeClient(interrupt), FakeClient(survivor)
    a.cancel_pending = b.cancel_pending = cancelled.set
    with pytest.raises(KeyboardInterrupt):
        runner.run_qwen(output, config, FakePool([a, b]), 'no-thinking')
    assert cancelled.is_set()
    resumed = FakePool([FakeClient(), FakeClient()])
    assert len(runner.run_qwen(output, config, resumed, 'no-thinking', resume=True)) == len(rows)
    sent = [m['candidate_id'] for c in resumed.clients for p in c.chats
            for m in json.loads(p['messages'][1]['content'])['mentions']]
    assert set(sent) == {c['candidate_id'] for c in rows[:4]}


def test_parallel_journal_reads_and_replaces_are_safe_on_windows(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from cgraph.entity_typing_v2.storage import write_json
    path = tmp_path / 'attempt.json'
    write_json(path, {'n': 0})
    def writer():
        for n in range(30): write_json(path, {'n': n})
    def reader():
        for _ in range(100): assert isinstance(read_json(path)['n'], int)
    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.submit(writer), pool.submit(reader)
        a.result(); b.result()


def test_chat_transport_cancellation_unblocks_response_read(monkeypatch):
    waiting, released = threading.Event(), threading.Event()
    class Connection:
        def __init__(self, *args, **kwargs):
            self.sock = SimpleNamespace(shutdown=lambda how: released.set())
        def request(self, *args): pass
        def getresponse(self):
            # Simulate Connection: close, with the body retaining the underlying socket.
            self.sock = None
            return self
        status = 200
        def read(self):
            waiting.set()
            assert released.wait(timeout=3)
            raise OSError('cancelled socket')
        def close(self): pass
    monkeypatch.setattr('http.client.HTTPConnection', Connection)
    client = OllamaClient('http://127.0.0.1:11434')
    errors = []
    def request():
        try: client.request('/api/chat', {'think': False})
        except OllamaError as exc: errors.append(str(exc))
    thread = threading.Thread(target=request)
    thread.start()
    assert waiting.wait(timeout=3)
    client.cancel_pending()
    thread.join(timeout=3)
    assert not thread.is_alive() and errors == ['cancelled socket']
