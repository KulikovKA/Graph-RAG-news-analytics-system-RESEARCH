"""Two independent Ollama servers, one outstanding inference request per server."""
import os
import shutil
import subprocess
import time
import uuid
from urllib.parse import urlsplit

from tqdm import tqdm

from .ollama import OllamaClient, OllamaError
from .storage import now, write_json


def server_urls(config):
    return config.get("server_urls") or [config["base_url"]]


def lock_identity(url):
    parts = urlsplit(url)
    host = "127.0.0.1" if parts.hostname in ("localhost", "127.0.0.1", "::1") else parts.hostname
    return f"{parts.scheme}://{host}:{parts.port or (443 if parts.scheme == 'https' else 80)}"


class OllamaPool:
    """Metadata facade. Actual /chat calls must select a specific endpoint."""
    def __init__(self, config, dry_run=False):
        self.clients = [OllamaClient(url, config["timeout_seconds"], config["metadata_timeout_seconds"])
                        for url in server_urls(config)]
        self.fallback = OllamaClient(config["base_url"], config["timeout_seconds"], config["metadata_timeout_seconds"])
        self.dry_run = dry_run
        self.last_discovery = None

    def discover(self, config):
        instances = []
        for client in self.clients:
            try:
                metadata = client.discover(config)
                ready = True
                observed_url = client.base_url
            except OllamaError:
                if not self.dry_run or not config.get("manage_servers"):
                    raise
                metadata = self.fallback.discover(config)
                ready = False
                observed_url = self.fallback.base_url
            instances.append({**metadata, "base_url": client.base_url, "ready": ready,
                              "observed_url": observed_url})
        first = instances[0]
        for instance in instances[1:]:
            if any(instance[k] != first[k] for k in ("tag", "digest", "details", "native_context", "ollama_version")):
                raise OllamaError("Worker servers must expose the identical Qwen checkpoint, quantization and Ollama version")
        self.last_discovery = {**first, "instances": instances}
        return self.last_discovery

    def request(self, path, payload=None, metadata=False):
        if path == "/api/chat":
            raise RuntimeError("Pool inference must be dispatched to an explicit worker")
        results = [(client, client.request(path, payload, metadata)) for client in self.clients]
        if path == "/api/ps":
            return {"models": [{**m, "base_url": client.base_url} for client, result in results
                               for m in result.get("models", [])]}
        if path == "/api/generate":
            return {"done": all(result.get("done") for _, result in results),
                    "load_duration": sum(result.get("load_duration", 0) for _, result in results),
                    "eval_count": sum(result.get("eval_count", 0) for _, result in results),
                    "prompt_eval_count": sum(result.get("prompt_eval_count", 0) for _, result in results),
                    "server_responses": [{"base_url": c.base_url, "response": r} for c, r in results]}
        return results[0][1]

    def cancel_pending(self):
        for client in self.clients:
            client.cancel_pending()


def make_client(config, dry_run=False):
    if config.get("server_urls"):
        return OllamaPool(config, dry_run=dry_run)
    return OllamaClient(config["base_url"], config["timeout_seconds"], config["metadata_timeout_seconds"])


class ManagedServers:
    """Start missing local servers only on a real user launch. Reused servers stay alive."""
    def __init__(self, config, output):
        self.config, self.output = config, output
        self.owned = []
        self.records = []

    def __enter__(self):
        if not self.config.get("server_urls"):
            return self
        try:
            for url in server_urls(self.config):
                parts = urlsplit(url)
                client = OllamaClient(url, metadata_timeout=1)
                try:
                    version = client.request("/api/version", metadata=True)["version"]
                    self.records.append({"base_url": url, "owned": False, "version": version, "status": "REUSED"})
                    continue
                except (OllamaError, KeyError):
                    if not self.config.get("manage_servers"):
                        raise OllamaError(f"Worker unavailable: {url}")
                if parts.hostname not in ("localhost", "127.0.0.1") or parts.scheme != "http":
                    raise ValueError("Automatic server startup requires a local HTTP endpoint")
                executable = shutil.which("ollama")
                if executable is None:
                    raise OllamaError("ollama executable is absent from PATH; no installation/download is attempted")
                logs = self.output / "server_logs"
                logs.mkdir(parents=True, exist_ok=True)
                log_path = logs / f"ollama_{parts.port}_{uuid.uuid4().hex}.log"
                log = log_path.open("ab", buffering=0)
                env = dict(os.environ, OLLAMA_HOST=f"127.0.0.1:{parts.port}",
                           OLLAMA_NUM_PARALLEL="1", OLLAMA_MAX_LOADED_MODELS="1")
                try:
                    process = subprocess.Popen([executable, "serve"], env=env, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                except BaseException:
                    log.close()
                    raise
                self.owned.append((process, log))
                deadline = time.monotonic() + self.config.get("server_start_timeout_seconds", 60)
                while True:
                    if process.poll() is not None:
                        raise OllamaError(f"Worker {url} exited during startup; inspect {log_path}")
                    try:
                        version = client.request("/api/version", metadata=True)["version"]
                        break
                    except (OllamaError, KeyError):
                        if time.monotonic() >= deadline:
                            raise OllamaError(f"Timed out starting {url}; inspect {log_path}")
                        time.sleep(.2)
                self.records.append({"base_url": url, "owned": True, "pid": process.pid,
                                     "version": version, "status": "STARTED", "log": str(log_path)})
                tqdm.write(f"Worker server ready: {url}, PID={process.pid}; model not loaded yet")
            write_json(self.output / "server_processes.json", {"created_at": now(), "servers": self.records})
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        for process, log in reversed(self.owned):
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            finally:
                log.close()
        self.owned.clear()
