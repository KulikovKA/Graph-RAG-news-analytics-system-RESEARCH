"""Unattended sequential orchestration. Never called for a metadata-only dry-run."""
import importlib.metadata
import platform
import sys
import time
import traceback
import uuid
from contextlib import contextmanager, redirect_stderr, redirect_stdout

from tqdm import tqdm

from .analysis import hybrid, runtime_metrics, summarize
from .completeness import check_completeness
from .dataset import gliner_rows, load_frozen, prepare
from .ollama import hardware_info
from .runner import MODE_FILES, completed_mode, materialize_pending, rebuild_failure_log, run_qwen
from .storage import now, read_json, read_jsonl, write_json, write_jsonl


class _Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.log.write(text)
        self.log.flush()
        return self.terminal.write(text)

    def flush(self):
        self.log.flush()
        self.terminal.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


@contextmanager
def technical_log(output, launch_id):
    path = output / "night_runs" / (launch_id + ".log")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            tqdm.write(f"[{now()}] Night log: {path.resolve()}")
            yield


def run_night(output, root, config, dataset, client, destination, exporter):
    """Caller holds both OS locks for the entire night, including stage transitions."""
    launch_id = uuid.uuid4().hex
    stages = []
    started = time.monotonic()

    def snapshot(status):
        manifest = read_json(output / "manifest.json")
        history = manifest.setdefault("execution", {})
        history.update({"requested_config": config, "python_version": platform.python_version(),
                        "package_versions": {p: importlib.metadata.version(p) for p in ("psutil", "tqdm", "jsonschema")},
                        "thinking_flags": {"T21": False, "T22": True},
                        "runtime_timings": runtime_metrics(output)})
        launches = history.setdefault("night_launches", {})
        launches[launch_id] = {"status": status, "stages": stages, "updated_at": now(),
                               "wall_seconds": time.monotonic() - started,
                               "technical_log": f"night_runs/{launch_id}.log"}
        settings_path = output / "model_settings.json"
        if settings_path.exists():
            history["model_settings"] = read_json(settings_path)
        write_json(output / "manifest.json", manifest)

    def perform(name, action, think=None, launch_status="RUNNING"):
        stage = {"stage": name, "started_at": now(), "status": "RUNNING", "think": think}
        stages.append(stage)
        before = time.monotonic()
        snapshot(launch_status)
        tqdm.write(f"[{now()}] START {name}")
        try:
            result = action()
            stage["status"] = "SKIPPED_COMPLETE" if isinstance(result, str) and result == "SKIPPED_COMPLETE" else "COMPLETE"
        except KeyboardInterrupt:
            stage["status"] = "INTERRUPTED"
            raise
        except Exception as exc:
            stage.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
            previous = read_jsonl(output / "stage_errors.jsonl")
            write_jsonl(output / "stage_errors.jsonl", previous + [{
                "kind": "stage_error", "stage": name, "attempt_id": launch_id,
                "created_at": now(), "detail": stage["error"]}])
            rebuild_failure_log(output)
            traceback.print_exc()
        finally:
            stage.update(finished_at=now(), wall_seconds=time.monotonic() - before)
            snapshot(launch_status)
            tqdm.write(f"[{now()}] END {name}: {stage['status']} ({stage['wall_seconds']:.2f}s)")
        return stage["status"] in ("COMPLETE", "SKIPPED_COMPLETE")

    def qwen(mode):
        if completed_mode(output, config, mode) is not None:
            return "SKIPPED_COMPLETE"
        return run_qwen(output, config, client, mode, resume=True)

    with technical_log(output, launch_id):
        prepare(config, root, output, dataset)
        write_json(output / "hardware.json", hardware_info())
        _, candidates, _ = load_frozen(output)
        if not (output / "model_settings.json").exists():
            write_json(output / "model_settings.json", {"status": "NOT_YET_VERIFIED", "requested_config": config})
        snapshot("RUNNING")
        for mode in MODE_FILES:
            materialize_pending(output, mode, "Stage not completed; no validated saved decision")
        interrupted = False
        try:
            perform("T1", lambda: write_jsonl(output / "T1_gliner.jsonl",
                    tqdm(gliner_rows(candidates), desc="T1 saved GLiNER", unit="candidate")))
            for mode, stage in (("no-thinking", "T21"), ("thinking", "T22")):
                # Rows exist even if discovery/allocation fails before the first chat call.
                if not perform(stage, lambda mode=mode: qwen(mode), think=mode == "thinking"):
                    materialize_pending(output, mode, stages[-1].get("error", "Stage interrupted"))
            perform("T3", lambda: hybrid(output, config))
        except KeyboardInterrupt:
            interrupted = True
            tqdm.write("Interrupted; finalizing saved results without further Qwen calls")
        finally:
            # Do not replace existing complete results; create explicit missing-mode rows.
            for mode in MODE_FILES:
                try:
                    if not (output / MODE_FILES[mode]).exists():
                        materialize_pending(output, mode, "Interrupted or stage failed; inspect night log")
                except Exception:
                    traceback.print_exc()
            if interrupted:
                perform("T3_after_interrupt", lambda: hybrid(output, config))
            perform("summarize", lambda: summarize(output))
            completeness = check_completeness(output)
            success = completeness["success"] and not interrupted and all(
                s["status"] in ("COMPLETE", "SKIPPED_COMPLETE") for s in stages)
            status = "COMPLETE" if success else "INTERRUPTED" if interrupted else "PARTIAL"
            snapshot(status)
            export_started = time.monotonic()
            def final_archive_manifest():
                stages[-1].update(status="COMPLETE", finished_at=now(), wall_seconds=time.monotonic() - export_started,
                                  timing_note="ZIP payload copied; final manifest/checksum metadata and archive close excluded")
                snapshot(status)
            exported = perform("export-results", lambda: exporter(output, destination, root,
                               before_manifest=final_archive_manifest), launch_status=status)
            snapshot(status if exported else "EXPORT_FAILED")
            tqdm.write(f"[{now()}] NIGHT {status if exported else 'EXPORT_FAILED'}; "
                       f"IDs accounted for={completeness['all_ids_accounted_for']}; ZIP={destination}")
        return 130 if interrupted else 0 if success and exported else 3
