"""Pinned local Qwen, strict structured output and explicit failure accounting."""
import json
import urllib.request
from pathlib import Path

from cgraph.common import file_sha256


def request(base, route, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(base + route, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def verify(cfg):
    base = cfg['endpoint']
    if base != 'http://127.0.0.1:11434':
        raise ValueError('Only the fixed local inference endpoint is permitted')
    version = request(base, '/api/version')['version']
    if version != cfg['ollama_version']:
        raise ValueError(f'Ollama version mismatch: {version}')
    matches = [m for m in request(base, '/api/tags')['models'] if m['name'] == cfg['model']]
    if len(matches) != 1 or matches[0]['digest'] != cfg['model_digest']:
        raise ValueError('Qwen model digest mismatch')
    contract = json.loads(Path(cfg['model_contract']).read_text(encoding='utf-8'))
    for item in contract['blobs']:
        path = Path.home() / '.ollama' / 'models' / 'blobs' / ('sha256-' + item['sha256'])
        if not path.is_file() or path.stat().st_size != item['size'] or file_sha256(path) != item['sha256']:
            raise ValueError(f"Model blob mismatch: {item['sha256']}")
    return dict(ollama_version=version, model=matches[0])


class InferenceFailure(RuntimeError):
    def __init__(self, attempts):
        super().__init__('Inference failed after fixed failure policy')
        self.attempts = attempts


class Qwen:
    def __init__(self, cfg, schema, prompt):
        self.cfg, self.schema, self.prompt = cfg, schema, prompt
        self.calls = self.retries = self.json_errors = self.request_errors = 0
        self.input_tokens = self.output_tokens = 0

    def extract(self, rows):
        from jsonschema import Draft202012Validator
        source = '\n'.join(f"[{r['id']}] {r['text']}" for r in rows)
        payload = dict(model=self.cfg['model'], system=self.prompt, prompt=source, format=self.schema,
                       options=self.cfg['decoding'], think=self.cfg['think'], stream=False, keep_alive='30m')
        attempts = []
        for i in range(self.cfg['max_retries'] + 1):
            self.calls += 1
            if i:
                self.retries += 1
            try:
                result = request(self.cfg['endpoint'], '/api/generate', payload, self.cfg['timeout_seconds'])
                self.input_tokens += result.get('prompt_eval_count', 0)
                self.output_tokens += result.get('eval_count', 0)
                attempt = {k: v for k, v in result.items() if k not in {'context', 'thinking'}}
                attempts.append(attempt)
                if result.get('done_reason') != 'stop':
                    raise ValueError('GENERATION_TRUNCATED')
                value = json.loads(result['response'])
                Draft202012Validator(self.schema).validate(value)
                return value, attempts
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self.json_errors += 1
                attempts.append({'error': type(exc).__name__, 'detail': str(exc)[:500]})
                # Greedy decoding of the same truncated response cannot repair it.
                if str(exc) == 'GENERATION_TRUNCATED':
                    break
            except Exception as exc:
                # Includes schema validation and request failures; never silently turn failure into empty predictions.
                if exc.__class__.__name__ == 'ValidationError':
                    self.json_errors += 1
                else:
                    self.request_errors += 1
                attempts.append({'error': type(exc).__name__, 'detail': str(exc)[:500]})
        raise InferenceFailure(attempts)
