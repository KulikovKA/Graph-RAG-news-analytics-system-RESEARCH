"""Runtime boundary tests; all inputs are synthetic, no real model or labels."""
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import pytest
from bgraph.backend import InferenceFailure, Qwen, verify
from cgraph.common import read_json


def test_inference_guard_blocks_labels_and_external_network(tmp_path):
    path = tmp_path / 'labels' / 'synthetic.json'
    path.parent.mkdir()
    path.write_text('{}')
    code = '''
from bgraph.cli import install_guard
import socket,sys
install_guard()
for action in [lambda: open(sys.argv[1]), lambda: socket.socket().connect(('192.0.2.1',443))]:
    try: action()
    except PermissionError: pass
    else: raise AssertionError('access guard did not block forbidden action')
'''
    env = dict(os.environ, PYTHONPATH=str(Path('src').resolve()))
    result = subprocess.run([sys.executable, '-c', code, str(path)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_truncated_generation_is_not_a_success_or_identically_retried():
    cfg = read_json('config/b001.json')
    model = Qwen(cfg, {}, 'synthetic prompt')
    response = dict(response='{"entities":[', done=True, done_reason='length', prompt_eval_count=10, eval_count=20)
    with patch('bgraph.backend.request', return_value=response):
        with pytest.raises(InferenceFailure) as failure:
            model.extract([dict(id=0,text='Invented sentence.')])
    assert model.calls == 1 and model.retries == 0 and model.json_errors == 1
    assert model.input_tokens == 10 and model.output_tokens == 20
    assert failure.value.attempts[0]['response'] == response['response']


def test_changed_model_digest_is_rejected_before_inference():
    cfg = read_json('config/b001.json')
    with patch('bgraph.backend.request', side_effect=[{'version':cfg['ollama_version']},
            {'models':[dict(name=cfg['model'],digest='incorrect')]}]):
        with pytest.raises(ValueError, match='digest mismatch'):
            verify(cfg)
