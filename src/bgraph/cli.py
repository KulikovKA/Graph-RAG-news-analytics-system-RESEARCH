"""B-001 commands; extraction is source-only; evaluation is a separate process."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

from cgraph.common import file_sha256, fingerprint, read_json, write_json


def install_guard():
    def guard(event, args):
        if event == 'socket.connect':
            address = args[1]
            if not isinstance(address, tuple) or address[0] != '127.0.0.1' or address[1] != 11434:
                raise PermissionError('Only loopback Ollama inference is permitted')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.fsdecode(args[0]).replace('\\', '/').lower()
            if (any(x in path for x in ('/labels/', '/gold/', '/calibration/', '/holdout/', '/re_model/'))
                    or path.endswith(('/ontology.v1.0.json', '/cgraph/labels.py', '/cgraph/relations.py'))):
                raise PermissionError('Gold, closed splits, original examples and fitted RE forbidden in inference')
        if event == 'import' and args[0] in {'cgraph.labels', 'cgraph.relations'}:
            raise PermissionError('Label/training module import forbidden in inference')
    sys.addaudithook(guard)


def extract(cfg, directory):
    install_guard()
    from cgraph.inputs import load_inputs, load_ontology
    from .backend import Qwen, verify
    from .contract import schema, system_prompt
    from .mapping import process, finish_document
    from .resources import Meter
    from .windows import windows
    from jsonschema import Draft202012Validator

    path = Path(directory)
    if path.exists() and any(path.iterdir()):
        raise ValueError('Output must be new or empty')
    path.mkdir(parents=True, exist_ok=True)
    with Meter() as meter:
        docs, ontology = load_inputs(cfg, 'all'), load_ontology(cfg)
        frozen_schema = read_json(cfg['schema'])
        if frozen_schema != schema(ontology):
            raise ValueError('Frozen schema disagrees with ontology')
        Draft202012Validator.check_schema(frozen_schema)
        identity = verify(cfg)
        qwen = Qwen(cfg, frozen_schema, system_prompt(cfg, ontology))
        output_hashes, trace_hashes = {}, {}
        for doc in docs:
            begin, calls = time.perf_counter(), qwen.calls
            chunks = []
            for chunk in doc['chunks']:
                results = []
                for rows in windows(chunk['text'], cfg['windows']):
                    value, attempts = qwen.extract(rows)
                    results.append((rows, value))
                    trace_path = path / 'raw' / doc['document_code'] / f"{chunk['chunk_id']}-{rows[0]['id']}.json"
                    write_json(trace_path, dict(chunk_id=chunk['chunk_id'], sentence_ids=[r['id'] for r in rows],
                                                attempts=attempts, parsed=value))
                    trace_hashes[trace_path.relative_to(path).as_posix()] = file_sha256(trace_path)
                    print(f"{doc['document_code']} chunk {chunk['ordinal']} window {rows[0]['id']}: "
                          f"calls={qwen.calls} in={qwen.input_tokens} out={qwen.output_tokens}", file=sys.stderr, flush=True)
                chunks.append(process(doc, chunk, results, ontology))
            prediction = finish_document(doc, chunks, fingerprint(cfg),
                dict(wall_ms=1000*(time.perf_counter()-begin), llm_calls=qwen.calls-calls))
            filename = path / (doc['document_code'] + '.json')
            write_json(filename, prediction)
            output_hashes[doc['document_code']] = file_sha256(filename)
            print(f"{doc['document_code']} complete: {sum(len(c['entities']) for c in chunks)} entities, "
                  f"{sum(len(c['assertions']) for c in chunks)} assertions", file=sys.stderr, flush=True)
    result = dict(architecture_id='B-001', subset='all', documents=[d['document_code'] for d in docs],
                  training_performed=False, model=identity, config=cfg, config_sha256=fingerprint(cfg),
                  config_file_sha256=file_sha256('config/b001.json'), prompt_sha256=file_sha256(cfg['prompt']),
                  schema_sha256=file_sha256(cfg['schema']), ontology_sha256=file_sha256(cfg['ontology']),
                  source_code_sha256={p.as_posix(): file_sha256(p) for p in sorted(Path('src/bgraph').glob('*.py'))},
                  prediction_files_sha256=output_hashes, raw_files_sha256=trace_hashes,
                  cost=meter.result(), llm_calls=qwen.calls, input_tokens=qwen.input_tokens,
                  output_tokens=qwen.output_tokens, retries=qwen.retries, json_errors=qwen.json_errors,
                  request_errors=qwen.request_errors, access_guard='Python audit hook: Gold/closed paths forbidden; loopback Ollama only',
                  access_audit_complete=True)
    write_json(path / 'run.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description='B-001 source-grounded graph extraction')
    parser.add_argument('--config', default='config/b001.json')
    sub = parser.add_subparsers(dest='command', required=True)
    x = sub.add_parser('extract')
    x.add_argument('--output', required=True)
    e = sub.add_parser('evaluate')
    e.add_argument('--predictions', required=True)
    e.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = read_json(args.config)
    if args.command == 'extract':
        result = extract(cfg, args.output)
    else:
        from .evaluate import evaluate_run
        result = evaluate_run(cfg, args.predictions, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
