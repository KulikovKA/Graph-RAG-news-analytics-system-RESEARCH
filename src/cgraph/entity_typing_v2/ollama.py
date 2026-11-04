"""Ollama metadata, per-instance context selection and cancellable HTTP calls."""
import json
import http.client
import math
import platform
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

import psutil
from tqdm import tqdm


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url, timeout=900, metadata_timeout=10):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.metadata_timeout = metadata_timeout
        self._connections = set()
        self._connection_lock = threading.Lock()
        self._cancelled = threading.Event()

    def request(self, path, payload=None, metadata=False):
        if path == "/api/chat":
            return self._chat_request(path, payload)
        body = json.dumps(payload, allow_nan=False).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(self.base_url + path, body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.metadata_timeout if metadata else self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise OllamaError(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:2000]}") from exc
        except (OSError, ValueError) as exc:
            raise OllamaError(str(exc)) from exc
        if not isinstance(result, dict):
            raise OllamaError("Invalid Ollama response: expected a JSON object")
        if result.get("error"):
            raise OllamaError(str(result["error"]))
        return result

    def _chat_request(self, path, payload):
        """Keep sockets cancellable so Ctrl+C does not wait for the HTTP timeout."""
        parts = urlsplit(self.base_url)
        connection_type = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        connection = connection_type(parts.hostname, parts.port, timeout=self.timeout)
        with self._connection_lock:
            if self._cancelled.is_set():
                raise OllamaError("Request cancelled")
            self._connections.add(connection)
        try:
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            connection.request("POST", parts.path.rstrip("/") + path, body, {"Content-Type": "application/json"})
            connection._typing_socket = connection.sock
            if self._cancelled.is_set():
                raise OllamaError("Request cancelled")
            response = connection.getresponse()
            data = response.read()
            if response.status >= 400:
                raise OllamaError(f"HTTP {response.status}: {data.decode('utf-8', errors='replace')[:2000]}")
            result = json.loads(data)
            if not isinstance(result, dict) or result.get("error"):
                raise OllamaError("Invalid/error Ollama response: " + str(result)[:2000])
            return result
        except (OSError, ValueError, http.client.HTTPException) as exc:
            raise OllamaError(str(exc)) from exc
        finally:
            with self._connection_lock:
                self._connections.discard(connection)
            connection.close()

    def cancel_pending(self):
        self._cancelled.set()
        with self._connection_lock:
            for connection in self._connections:
                active_socket = connection.sock or getattr(connection, "_typing_socket", None)
                if active_socket:
                    try:
                        active_socket.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass

    def discover(self, config):
        version = self.request("/api/version", metadata=True)
        tags = self.request("/api/tags", metadata=True)["models"]
        candidates = [m for m in tags if re.fullmatch(r"qwen3\.5:4b(?:-(?:q[0-9][a-zA-Z0-9_]*|fp16|bf16))?", m["name"], re.I)]
        requested = config["model"]
        if requested != "qwen3.5:4b":
            candidates = [m for m in candidates if m["name"] == requested]
        candidates.sort(key=lambda m: (m.get("details", {}).get("quantization_level") != config["preferred_quantization"],
                                       m["name"] != requested, m["name"]))
        if not candidates:
            raise OllamaError("Qwen3.5-4B is absent. Install explicitly: ollama pull qwen3.5:4b-q4_K_M")
        tag = candidates[0]
        show = self.request("/api/show", {"model": tag["name"]}, metadata=True)
        info = show.get("model_info", {})
        architecture = info.get("general.architecture")
        count = info.get("general.parameter_count", 0)
        if architecture != "qwen35" or not 3_500_000_000 <= count <= 5_500_000_000:
            raise OllamaError("Installed model is not the expected Qwen3.5-4B architecture/size")
        if "thinking" not in show.get("capabilities", []):
            raise OllamaError("This checkpoint/runtime does not advertise thinking support")
        thinking = show.get("thinking", {}).get("values")
        if thinking is not None and not (False in thinking and True in thinking):
            raise OllamaError("Runtime does not advertise boolean think=false/true")
        native = info.get(architecture + ".context_length")
        if not isinstance(native, int) or native <= 0:
            raise OllamaError("Checkpoint native context length is missing")
        ps = self.request("/api/ps", metadata=True)
        # Exclude licenses/absolute model paths and retain all inference-relevant metadata.
        return {"tag": tag["name"], "digest": tag["digest"], "size_bytes": tag["size"],
                "details": show.get("details", tag.get("details")), "model_info": info,
                "native_context": native, "capabilities": show.get("capabilities"),
                "thinking": show.get("thinking"), "model_parameters": show.get("parameters"),
                "template": show.get("template"), "ollama_version": version["version"],
                "running_models": ps.get("models", [])}


def hardware_info():
    vm = psutil.virtual_memory()
    return {"recorded_at_unix": time.time(), "platform": platform.platform(),
            "python": platform.python_version(), "cpu": platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False), "logical_cores": psutil.cpu_count(),
            "ram_total_bytes": vm.total, "ram_available_bytes": vm.available}


def context_choices(model, hardware, config):
    info = model["model_info"]
    prefix = info["general.architecture"] + "."
    layers = info[prefix + "block_count"]
    heads = info[prefix + "attention.head_count_kv"]
    if isinstance(heads, list):
        if len(heads) != layers:
            raise ValueError("KV layer metadata mismatch")
        effective_heads = sum(heads)
    else:
        # Conservative for hybrid architectures if per-layer information is absent.
        effective_heads = layers * heads
    head_dim = info.get(prefix + "attention.key_length")
    if not head_dim:
        head_dim = info[prefix + "embedding_length"] // info[prefix + "attention.head_count"]
    value_dim = info.get(prefix + "attention.value_length", head_dim)
    kv_bytes_per_token = effective_heads * (head_dim + value_dim) * 2  # f16 K/V, never assume quantized KV
    if kv_bytes_per_token <= 0:
        raise ValueError("Cannot estimate memory without attention metadata")
    instances = model.get("instances", [model])
    reclaimable, observed = 0, set()
    from .servers import lock_identity
    for instance in instances:
        identity = lock_identity(instance.get("observed_url", instance.get("base_url", "http://localhost:11434")))
        if identity not in observed:
            reclaimable += sum(m.get("size", 0) - m.get("size_vram", 0) for m in instance.get("running_models", [])
                               if m.get("digest") == instance["digest"])
            observed.add(identity)
    available = hardware["ram_available_bytes"] + reclaimable
    ceiling = min(model["native_context"], config["max_model_context"],
                  config.get("runtime_context_limit") or config["max_model_context"])
    estimates = []
    for ctx in config["context_ladder"]:
        estimate = math.ceil(sum(i["size_bytes"] for i in instances) + len(instances) * config["runtime_overhead_bytes"] +
                             ctx * kv_bytes_per_token * len(instances) * config["kv_safety_factor"])
        estimates.append({"num_ctx": ctx, "estimated_ram_bytes": estimate,
                          "fits": ctx <= ceiling and estimate <= available * config["memory_fraction"]})
    choices = [e["num_ctx"] for e in estimates if e["fits"]]
    if not choices:
        raise OllamaError("No context >=8192 fits the model/runtime/RAM budget; free RAM and retry")
    if config["strict_max_context"] and choices[0] != model["native_context"]:
        raise OllamaError(f"Strict native context {model['native_context']} unavailable; estimated maximum {choices[0]}")
    return choices, {"method": "sum of independent model weights + per-instance f16 KV and runtime/SSM/scratch reserve",
                     "model_instances": len(instances), "context_is_per_instance": True,
                     "kv_bytes_per_token": kv_bytes_per_token, "reclaimable_loaded_model_bytes": reclaimable,
                     "available_plus_reclaimable_bytes": available, "estimates": estimates,
                     "runtime_verification": "pending load-only allocation; dry-run does not allocate"}


def sampling_options(config, num_ctx, budget):
    keys = ("temperature", "seed", "top_p", "top_k", "min_p", "repeat_penalty",
            "presence_penalty", "frequency_penalty", "num_thread", "num_gpu", "num_batch")
    return {**{k: config[k] for k in keys}, "num_ctx": num_ctx, "num_predict": budget}


def validate_config(config):
    if config["backend"] != "ollama" or config["num_parallel"] != 1 or config["stream"] is not False:
        raise ValueError("Protocol requires Ollama, num_parallel=1, stream=false")
    from .servers import server_urls, lock_identity
    urls = server_urls(config)
    if len(urls) not in (1, 2) or len({lock_identity(url) for url in urls}) != len(urls):
        raise ValueError("Use one or two distinct Ollama server endpoints")
    if any(urlsplit(url).scheme not in ("http", "https") or not urlsplit(url).hostname for url in urls):
        raise ValueError("Invalid Ollama server URL")
    if (config["temperature"] != 0 or config["top_p"] != 1 or config["top_k"] != 0 or config["num_gpu"] != 0):
        raise ValueError("Primary CPU comparison requires temperature=0, top_p=1, top_k=0, num_gpu=0")
    for key in ("num_thread", "num_batch", "num_predict", "num_predict_thinking", "batch_size",
                "output_tokens_per_candidate", "prompt_overhead_tokens", "timeout_seconds", "metadata_timeout_seconds"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["max_retries"]) is not int or not 0 <= config["max_retries"] <= 5:
        raise ValueError("max_retries must be between 0 and 5")
    if not 0 < config["memory_fraction"] <= 1 or config["kv_safety_factor"] < 1:
        raise ValueError("Invalid memory safety configuration")
    if (config["max_model_context"] > 262144 or config["max_model_context"] < 8192 or
            config["context_ladder"] != [262144, 131072, 65536, 32768, 16384, 8192]):
        raise ValueError("Use the native-context ladder without RoPE/YaRN scaling")


def verify_server_serial():
    """Inspect local server environment where accessible; never set a client-only fake option."""
    evidence = []
    for process in psutil.process_iter(["name", "cmdline"]):
        try:
            if "ollama" not in (process.info["name"] or "").lower():
                continue
            env = process.environ()
            value = env.get("OLLAMA_NUM_PARALLEL")
            if value is not None and value != "1":
                raise OllamaError("Restart Ollama with OLLAMA_NUM_PARALLEL=1; server currently has " + value)
            evidence.append({"pid": process.pid, "OLLAMA_NUM_PARALLEL": value,
                             "command": process.info["cmdline"]})
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return {"observed_local_processes": evidence,
            "note": "One request per independent server; two endpoints may run concurrently. Server slot count is an environment setting; null is not verified."}


def allocate_context(client, model, choices, config, pinned=None):
    """Only called by a real qwen command. Empty /generate loads without token generation."""
    failures = []
    for ctx in choices:
        if pinned is not None and ctx != pinned:
            continue
        tqdm.write(f"Allocating num_ctx={ctx} (load only, no prompt)")
        try:
            result = client.request("/api/generate", {"model": model["tag"], "stream": False,
                "keep_alive": config["keep_alive"], "options": sampling_options(config, ctx, 0)})
            if result.get("eval_count", 0) or result.get("prompt_eval_count", 0):
                raise OllamaError("Unexpected token evaluation in load-only response")
            running = client.request("/api/ps", metadata=True).get("models", [])
            matched_models = [m for m in running if m["digest"] == model["digest"]]
            count = len(model.get("instances", [model]))
            if len(matched_models) != count or any(m.get("context_length") != ctx for m in matched_models):
                raise OllamaError(f"Runtime context mismatch: requested {ctx} on {count} servers, observed {[m.get('context_length') for m in matched_models]}")
            matched = matched_models[0]
            return ctx, {"load_only_response": result, "observed_running_model": matched,
                         "observed_running_models": matched_models,
                         "fallbacks": failures, "context_verified": True,
                         "options_validation": "load endpoint accepted options; sampling settings are not echoed by Ollama"}
        except OllamaError as exc:
            failures.append({"num_ctx": ctx, "error": str(exc)})
            tqdm.write(f"Context {ctx} rejected: {exc}")
            if config["strict_max_context"] or pinned is not None:
                raise
            if not any(word in str(exc).lower() for word in ("memory", "allocate", "context", "out of", "requires")):
                raise
    raise OllamaError(f"Runtime cannot allocate a safe context: {failures}")


class MemoryMonitor:
    """Sample client and Ollama RSS separately; these are process RSS, not total system RAM."""
    def __init__(self):
        self.stop = threading.Event()
        self.client_peak = 0
        self.ollama_peak = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def sample(self):
        self.client_peak = max(self.client_peak, psutil.Process().memory_info().rss)
        rss = 0
        for p in psutil.process_iter(["name", "memory_info"]):
            try:
                if "ollama" in (p.info["name"] or "").lower():
                    rss += p.info["memory_info"].rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        self.ollama_peak = max(self.ollama_peak, rss)

    def _run(self):
        while not self.stop.is_set():
            self.sample()
            self.stop.wait(.25)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=2)
        self.sample()

    def result(self):
        return {"client_peak_rss_bytes": self.client_peak, "ollama_processes_peak_rss_bytes": self.ollama_peak,
                "sampling_interval_seconds": .25}
