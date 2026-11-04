"""Sequential Qwen execution with durable attempt journals and candidate-level resume."""
import time
import uuid
from collections import deque

from tqdm import tqdm

from . import PROTOCOL_VERSION
from .dataset import load_frozen
from .ollama import (MemoryMonitor, OllamaError, allocate_context, context_choices,
                     hardware_info, sampling_options, verify_server_serial)
from .parsing import parse_response
from .prompting import (PROMPT_VERSION, batch_record, output_budget, plan_batches, response_schema)
from .storage import digest, file_hash, now, read_json, read_jsonl, write_json, write_jsonl

MODE_FILES = {"no-thinking": "T21_qwen_no_thinking.jsonl", "thinking": "T22_qwen_thinking.jsonl"}


def implementation_digest():
    from pathlib import Path
    return digest({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def common_settings(config, model, num_ctx):
    return {"protocol_version": PROTOCOL_VERSION, "prompt_version": PROMPT_VERSION,
            "implementation_digest": implementation_digest(),
            "model": model["tag"], "digest": model["digest"], "details": model["details"],
            "ollama_version": model["ollama_version"], "native_context": model["native_context"],
            "num_ctx": num_ctx, "options": sampling_options(config, num_ctx, config["num_predict"]),
            "num_predict_thinking": config["num_predict_thinking"],
            "batch_size": config["batch_size"], "output_tokens_per_candidate": config["output_tokens_per_candidate"],
            "thinking_reserve_tokens": config["thinking_reserve_tokens"],
            "prompt_overhead_tokens": config["prompt_overhead_tokens"],
            "structured_output": config["structured_output"], "keep_alive": config["keep_alive"],
            "stream": False, "num_parallel": 1}


def metadata_preflight(client, config):
    model = client.discover(config)
    hardware = hardware_info()
    choices, estimate = context_choices(model, hardware, config)
    return model, hardware, choices, estimate


def dry_run(output, config, client):
    manifest, candidates, ontology = load_frozen(output)
    report = {"created_at": now(), "inference_calls": 0, "model_load_calls": 0,
              "output_directory": str(output.resolve()), "candidate_count": len(candidates),
              "token_count_method": "conservative UTF-8 byte upper bound + schema + renderer reserve"}
    try:
        model, hardware, choices, estimate = metadata_preflight(client, config)
        ctx = manifest["runtime"]["num_ctx"] if manifest.get("runtime") else choices[0]
        if ctx not in choices:
            raise OllamaError("Frozen num_ctx no longer fits available RAM/runtime")
        settings = common_settings(config, model, ctx)
        if manifest.get("runtime") and manifest["runtime"] != settings:
            raise ValueError("Runtime settings/checkpoint/code differ from frozen experiment")
        report.update(model_available=True, model=model, selected_num_ctx=ctx, context_selection=estimate,
                      settings=settings, hardware=hardware, server_concurrency=verify_server_serial())
        write_json(output / "hardware.json", hardware)
        # Proposed settings are explicitly marked; never replace frozen actual settings.
        if not manifest.get("runtime"):
            write_json(output / "model_settings.json", {"status": "DRY_RUN_PROPOSED", **settings})
    except OllamaError as exc:
        ctx = config["max_model_context"]
        report.update(model_available=False, error=str(exc), selected_num_ctx=None)
    batches = plan_batches(candidates, ontology, config, ctx)
    plans = [batch_record(batch, ontology, config) for batch in batches]
    report.update(batch_count=len(batches), candidate_ids=manifest["frozen"]["candidate_ids"],
                  max_prompt_token_upper_bound=max(p["input_token_upper_bound"] for p in plans),
                  total_prompt_token_upper_bound=sum(p["input_token_upper_bound"] for p in plans),
                  planned_output_budget={m: output_budget(config, m) for m in MODE_FILES},
                  runtime_verified=False)
    write_json(output / "dry_run.json", report)
    manifest["preflight"] = report
    write_json(output / "manifest.json", manifest)
    write_jsonl(output / "dry_run_batches.jsonl", plans)
    tqdm.write(f"Dry-run: candidates={len(candidates)}, batches={len(batches)}, model_available={report['model_available']}, "
               f"num_ctx={report['selected_num_ctx']}; no loading or inference")
    return report


def _recover(output, mode, run_digest, candidates, labels):
    """Raw response is committed first, so interrupted materialization can be replayed."""
    expected = {c["candidate_id"] for c in candidates}
    completed = {}
    attempts = []
    for path in sorted((output / "attempts" / mode).glob("*.json")):
        record = read_json(path)
        if record["run_digest"] != run_digest or record["mode"] != mode:
            raise ValueError("Attempt journal provenance mismatch")
        if not set(record["candidate_ids"]) <= expected:
            raise ValueError("Attempt journal includes unfrozen IDs")
        attempts.append(record)
        raw = record.get("response")
        if raw:
            message = raw.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                valid, recovered_errors = parse_response(message["content"], record["candidate_ids"], labels)
            else:
                valid, recovered_errors = {}, [{"kind": "invalid_response_envelope"}]
            if record.get("parsed_finalized"):
                if any(cid not in valid or valid[cid] != row for cid, row in record["parsed"].items()):
                    raise ValueError("Parsed journal differs from raw response")
                valid = record["parsed"]
            else:
                if raw.get("done_reason") == "length" or raw.get("eval_count", 0) >= record["request"]["options"]["num_predict"]:
                    recovered_errors.append({"kind": "output_budget_reached", "budget": record["request"]["options"]["num_predict"]})
                if (raw.get("done") is not True or
                        raw.get("prompt_eval_count", 0) > record["input_token_upper_bound"] or
                        raw.get("prompt_eval_count", 0) + record["request"]["options"]["num_predict"] > record["request"]["options"]["num_ctx"]):
                    valid = {}
                    recovered_errors.append({"kind": "invalid_runtime_accounting_on_recovery"})
                record["parsed"] = valid
                record["errors"] = recovered_errors
                record["parsed_finalized"] = True
                record["thinking_token_count"] = raw.get("thinking_eval_count")
                record["thinking_token_note"] = "recovered from raw; separate count unavailable unless returned by runtime"
                write_json(path, record)
            for cid, decision in valid.items():
                result = {**decision, "processing_status": "COMPLETE", "mode": mode,
                          "attempt_id": record["attempt_id"], "run_digest": run_digest,
                          "prompt_sha256": record["prompt_sha256"]}
                if cid in completed and completed[cid]["entity_type"] != result["entity_type"]:
                    raise ValueError("Conflicting successful decisions in attempt journal")
                completed[cid] = result
    # A canonical result must have its durable raw response; detect deletion/tampering.
    for row in read_jsonl(output / MODE_FILES[mode]):
        if row.get("processing_status") == "COMPLETE":
            cid = row["candidate_id"]
            if completed.get(cid) != row:
                raise ValueError("Canonical Qwen result differs from durable raw journal")
    return completed, attempts


def _save_results(output, mode, candidates, completed, run_digest):
    rows = [completed.get(c["candidate_id"], {"candidate_id": c["candidate_id"],
            "processing_status": "NOT_PROCESSED", "entity_type": None,
            "typing_status": "NOT_PROCESSED", "mode": mode, "run_digest": run_digest,
            "error_reason": "No validated decision; inspect attempt journal or stage error"}) for c in candidates]
    write_jsonl(output / MODE_FILES[mode], rows)


def materialize_pending(output, mode, reason):
    """Give every frozen ID a row even if model discovery/allocation failed."""
    manifest, candidates, ontology = load_frozen(output)
    run_digest = digest([manifest["frozen"], manifest["runtime"]]) if manifest.get("runtime") else None
    completed, _ = _recover(output, mode, run_digest, candidates, ontology["entity_types"])
    _save_results(output, mode, candidates, completed, run_digest)
    rows = read_jsonl(output / MODE_FILES[mode])
    for row in rows:
        if row["processing_status"] != "COMPLETE":
            row["error_reason"] = reason
    write_jsonl(output / MODE_FILES[mode], rows)


def completed_mode(output, config, mode):
    """Validate a completed mode offline; the other mode still verifies live metadata."""
    manifest, candidates, ontology = load_frozen(output)
    runtime = manifest.get("runtime")
    if not runtime:
        return None
    saved_model = {"tag": runtime["model"], **{k: runtime[k] for k in
                   ("digest", "details", "ollama_version", "native_context")}}
    if runtime != common_settings(config, saved_model, runtime["num_ctx"]):
        raise ValueError("Frozen settings/code changed; use a new --output")
    run_digest = digest([manifest["frozen"], runtime])
    completed, _ = _recover(output, mode, run_digest, candidates, ontology["entity_types"])
    if len(completed) != len(candidates):
        return None
    _save_results(output, mode, candidates, completed, run_digest)
    rebuild_failure_log(output)
    tqdm.write(f"{mode}: all {len(candidates)} candidates complete; skipping without Ollama access")
    return completed


def rebuild_failure_log(output):
    failures = read_jsonl(output / "stage_errors.jsonl")
    for path in sorted((output / "attempts").glob("*/*.json")):
        attempt = read_json(path)
        for error in attempt.get("errors", []):
            failures.append({"attempt_id": attempt["attempt_id"], "batch_id": attempt["batch_id"],
                             "mode": attempt["mode"], "candidate_ids": attempt["candidate_ids"],
                             "messages": attempt["messages"], "created_at": attempt["created_at"], **error})
    write_jsonl(output / "failure_log.jsonl", failures)


def run_qwen(output, config, client, mode, resume=False):
    manifest, candidates, ontology = load_frozen(output)
    existing = (output / MODE_FILES[mode]).exists() or any((output / "attempts" / mode).glob("*.json"))
    if existing and not resume:
        raise ValueError("Results already exist. Use --resume or a new --output directory")
    if resume:
        complete = completed_mode(output, config, mode)
        if complete is not None:
            return complete
    model, hardware, choices, estimate = metadata_preflight(client, config)
    pinned = manifest["runtime"]["num_ctx"] if manifest.get("runtime") else None
    if pinned is not None:
        if pinned not in choices:
            raise OllamaError("Frozen num_ctx cannot safely fit RAM now; free memory, do not change the experiment")
        if manifest["runtime"] != common_settings(config, model, pinned):
            raise ValueError("Checkpoint/settings/code changed; use a new --output for a new experiment")
    server = verify_server_serial()
    # Check already-complete resumes before any model allocation.
    if pinned is not None:
        run_digest = digest([manifest["frozen"], manifest["runtime"]])
        completed, previous = _recover(output, mode, run_digest, candidates, ontology["entity_types"])
        if len(completed) == len(candidates):
            _save_results(output, mode, candidates, completed, run_digest)
            rebuild_failure_log(output)
            tqdm.write(f"{mode}: all {len(candidates)} candidates complete; no model load/inference")
            return completed
    selected, allocation = allocate_context(client, model, choices, config, pinned=pinned)
    settings = common_settings(config, model, selected)
    manifest["runtime"] = settings
    write_json(output / "manifest.json", manifest)  # pinned before the first typing call
    write_json(output / "hardware.json", hardware)
    write_json(output / "model_settings.json", {"status": "RUNTIME_CONTEXT_VERIFIED", **settings,
               "model_metadata": model, "context_selection": estimate, "allocation": allocation,
               "server_concurrency": server,
               "effective_sampling_verification": "Not observable through public Ollama API; exact requested options are saved per call"})
    run_digest = digest([manifest["frozen"], settings])
    completed, previous = _recover(output, mode, run_digest, candidates, ontology["entity_types"])
    batches = plan_batches(candidates, ontology, config, selected)
    plans = [batch_record(batch, ontology, config) for batch in batches]
    plan_path = output / "batch_plan.jsonl"
    if plan_path.exists() and read_jsonl(plan_path) != plans:
        raise ValueError("Frozen batch plan changed")
    write_jsonl(plan_path, plans)
    _save_results(output, mode, candidates, completed, run_digest)
    queue = deque((batch, 0) for batch in batches if any(c["candidate_id"] not in completed for c in batch))
    session_id = uuid.uuid4().hex
    started = time.monotonic()
    input_tokens = sum((a.get("response") or {}).get("prompt_eval_count", 0) for a in previous)
    output_tokens = sum((a.get("response") or {}).get("eval_count", 0) for a in previous)
    errors = sum(bool(a.get("errors")) for a in previous)
    initial_completed = len(completed)
    tqdm.write(f"ENTITY-TYPING-V2-001 | Model: {model['tag']} {model['details']['quantization_level']} | "
               f"num_ctx={selected} | Thinking {'ON' if mode == 'thinking' else 'OFF'} | Candidates: {len(candidates)}")
    if selected != model["native_context"]:
        tqdm.write(f"Explicit context reduction: native={model['native_context']} -> {selected}; memory/runtime evidence in model_settings.json")
    stage = "T2.2 Thinking ON" if mode == "thinking" else "T2.1 Thinking OFF"
    candidate_bar = tqdm(total=len(candidates), initial=len(completed), desc=stage + " candidates", unit="candidate")
    batch_bar = tqdm(total=len(queue), desc=stage + " batches", unit="batch")
    monitor = MemoryMonitor()
    try:
        with monitor:
            while queue:
                batch, depth = queue.popleft()
                batch = [c for c in batch if c["candidate_id"] not in completed]
                if not batch:
                    batch_bar.update(1)
                    continue
                plan = batch_record(batch, ontology, config)
                budget = output_budget(config, mode)
                if plan["input_token_upper_bound"] + budget > selected:
                    raise ValueError("Preflight input + output context overflow")
                attempt_id = uuid.uuid4().hex
                record = {**plan, "attempt_id": attempt_id, "session_id": session_id, "mode": mode,
                          "run_digest": run_digest, "created_at": now(), "retry_depth": depth,
                          "errors": [], "response": None, "parsed": {}}
                payload = {"model": model["tag"], "messages": plan["messages"], "think": mode == "thinking",
                           "stream": False, "keep_alive": config["keep_alive"],
                           "options": sampling_options(config, selected, budget),
                           "format": response_schema(ontology["entity_types"]) if config["structured_output"] else "json"}
                record["request"] = payload
                attempt_path = output / "attempts" / mode / (attempt_id + ".json")
                before = time.monotonic()
                valid = {}
                candidate_bar.set_postfix(batch=plan["batch_id"], errors=errors, input=input_tokens, output=output_tokens)
                try:
                    response = client.request("/api/chat", payload)
                    record["response"] = response
                    record["wall_seconds"] = time.monotonic() - before
                    # Persist raw content/thinking before parsing or materializing canonical outputs.
                    write_json(attempt_path, record)
                    if (not isinstance(response.get("message"), dict) or
                            not isinstance(response["message"].get("content"), str)):
                        raise OllamaError("Invalid /api/chat message envelope; raw response saved")
                    valid, validation_errors = parse_response(response.get("message", {}).get("content", ""),
                                                              plan["candidate_ids"], ontology["entity_types"])
                    record["errors"].extend(validation_errors)
                    if response.get("done_reason") == "length" or response.get("eval_count", 0) >= budget:
                        record["errors"].append({"kind": "output_budget_reached", "budget": budget})
                    if response.get("prompt_eval_count", 0) + budget > selected:
                        record["errors"].append({"kind": "actual_context_overflow"})
                        valid = {}  # cannot trust a context-truncated result
                    if response.get("prompt_eval_count", 0) > plan["input_token_upper_bound"]:
                        record["errors"].append({"kind": "token_bound_exceeded"})
                        valid = {}
                    if response.get("done") is not True:
                        record["errors"].append({"kind": "incomplete_transport_response"})
                        valid = {}
                    record["thinking_token_count"] = response.get("thinking_eval_count")
                    record["thinking_token_note"] = "separate runtime counter" if "thinking_eval_count" in response else "unavailable; eval_count may include thinking; no fabricated split"
                    input_tokens += response.get("prompt_eval_count", 0)
                    output_tokens += response.get("eval_count", 0)
                except OllamaError as exc:
                    record["wall_seconds"] = time.monotonic() - before
                    record["errors"].append({"kind": "transport", "detail": str(exc)})
                record["parsed"] = valid
                record["parsed_finalized"] = True
                write_json(attempt_path, record)
                for cid, decision in valid.items():
                    completed[cid] = {**decision, "processing_status": "COMPLETE", "mode": mode,
                                      "attempt_id": attempt_id, "run_digest": run_digest,
                                      "prompt_sha256": plan["prompt_sha256"]}
                _save_results(output, mode, candidates, completed, run_digest)
                rebuild_failure_log(output)
                errors += bool(record["errors"])
                candidate_bar.update(len(valid))
                remaining = [c for c in batch if c["candidate_id"] not in valid]
                if remaining and depth < config["max_retries"]:
                    middle = max(1, len(remaining) // 2)
                    children = [remaining[:middle]] + ([remaining[middle:]] if remaining[middle:] else [])
                    queue.extend((child, depth + 1) for child in children)
                    batch_bar.total += len(children)
                    time.sleep(config["retry_delay_seconds"])
                elapsed = max(time.monotonic() - started, .001)
                candidate_bar.set_postfix(batch=plan["batch_id"], errors=errors, input=input_tokens, output=output_tokens,
                                         candidates_min=round((len(completed) - initial_completed) / elapsed * 60, 2))
                batch_bar.update(1)
                tqdm.write(f"{plan['batch_id']} | entities={len(valid)}/{len(batch)} | {record['wall_seconds']:.2f}s | "
                           f"in/out={record.get('response', {}).get('prompt_eval_count', 0) if record['response'] else 0}/"
                           f"{record.get('response', {}).get('eval_count', 0) if record['response'] else 0} | "
                           f"thinking={mode == 'thinking'} | JSON={'valid' if not record['errors'] else 'errors saved'}")
    finally:
        candidate_bar.close()
        batch_bar.close()
        _save_results(output, mode, candidates, completed, run_digest)
        write_json(output / "sessions" / (session_id + ".json"), {
            "session_id": session_id, "mode": mode, "wall_seconds": time.monotonic() - started,
            "completed_this_session": len(completed) - initial_completed,
            "remaining": len(candidates) - len(completed), "memory": monitor.result(),
            "allocation": allocation, "finished_at": now()})
    if len(completed) != len(candidates):
        raise RuntimeError(f"{len(candidates) - len(completed)} candidates NOT_PROCESSED; inspect failure_log.jsonl and --resume")
    return completed
