"""Клиент документированного Groq Chat Completions API."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import re
from time import perf_counter, sleep
from urllib.parse import urlsplit

import httpx


DEFAULT_URL = "https://api.groq.com/openai/v1"
SECRET_PATTERN = re.compile(r"gsk_[a-zA-Z0-9]{20,}")


class ProviderError(RuntimeError):
    def __init__(self, code, trace=None):
        super().__init__(code)
        self.code = code
        self.trace = trace if trace is not None else {}


@dataclass(frozen=True)
class GenerationSettings:
    model: str
    temperature: float
    max_output_tokens: int
    timeout: float
    max_retries: int
    max_input_tokens: int
    context_window: int
    reasoning_effort: str = "low"

    @classmethod
    def from_environment(cls):
        try:
            settings = cls(os.environ["LLM_MODEL"], float(os.getenv("LLM_TEMPERATURE", "0")),
                           int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1000")),
                           float(os.getenv("LLM_TIMEOUT", "30")), int(os.getenv("LLM_MAX_RETRIES", "1")),
                           int(os.getenv("LLM_MAX_INPUT_TOKENS", "6000")),
                           int(os.getenv("LLM_CONTEXT_WINDOW", "131072")),
                           os.getenv("LLM_REASONING_EFFORT", "low"))
        except (KeyError, ValueError):
            raise ProviderError("invalid_llm_configuration") from None
        if (not re.fullmatch(r"[a-zA-Z0-9._/-]+", settings.model) or SECRET_PATTERN.search(settings.model)
                or not 0 <= settings.temperature <= 2 or not 0 <= settings.max_retries <= 3
                or not 0 < settings.timeout <= 120 or settings.max_input_tokens < 1
                or settings.max_output_tokens < 1
                or settings.reasoning_effort not in ("low", "medium", "high")
                or settings.max_input_tokens + settings.max_output_tokens > settings.context_window):
            raise ProviderError("invalid_llm_configuration")
        return settings


class HarmonyTokenCounter:
    # Резерв покрывает дополнительный framing провайдера при JSON mode.
    reserve = 256

    def __init__(self, path: Path, expected_sha: str, reasoning_effort: str):
        from tokenizers import Tokenizer

        data = path.read_bytes()
        self.sha256 = sha256(data).hexdigest()
        if self.sha256 != expected_sha:
            raise ProviderError("tokenizer_checksum_mismatch")
        self.tokenizer = Tokenizer.from_buffer(data)
        self.reasoning_effort = reasoning_effort
        for token in ("<|start|>", "<|message|>", "<|end|>", "<|channel|>"):
            if self.tokenizer.token_to_id(token) is None:
                raise ProviderError("incompatible_harmony_tokenizer")

    def __call__(self, messages):
        rendered = (f"<|start|>system<|message|>Reasoning: {self.reasoning_effort}\n"
                    "# Valid channels: analysis, commentary, final.<|end|>")
        rendered += "".join(
            f"<|start|>{'developer' if m['role'] == 'system' else m['role']}<|message|>"
            f"{m['content']}<|end|>" for m in messages)
        rendered += "<|start|>assistant"
        return len(self.tokenizer.encode(rendered, add_special_tokens=False).ids) + self.reserve


def _identifier(value):
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,150}", value) and not SECRET_PATTERN.search(value):
        return value
    return None


class GroqClient:
    def __init__(self, settings: GenerationSettings, *, stream=httpx.stream, pause=sleep, jitter=random.uniform):
        self._key = os.getenv("GROQ_API_KEY", "").strip()
        if not self._key:
            raise ProviderError("missing_GROQ_API_KEY")
        if any(char.isspace() for char in self._key):
            raise ProviderError("invalid_GROQ_API_KEY")
        self.settings = settings
        self._base = os.getenv("LLM_BASE_URL", DEFAULT_URL).rstrip("/")
        try:
            url = urlsplit(self._base)
        except ValueError:
            raise ProviderError("invalid_LLM_BASE_URL") from None
        if (url.scheme != "https" or not url.hostname or url.username is not None
                or url.password is not None or url.query or url.fragment):
            raise ProviderError("invalid_LLM_BASE_URL")
        self._stream = stream
        self._pause, self._jitter = pause, jitter

    def complete(self, messages: list[dict[str, str]], counted_input: int) -> tuple[str, dict]:
        settings = self.settings
        trace = {"provider": "groq", "model": settings.model, "timestamp": datetime.now(timezone.utc).isoformat(),
                 "settings": asdict(settings), "attempts": [], "retries": 0, "usage": None,
                 "response_format": "json_object", "include_reasoning": False,
                 "request_id": None, "status": "INVALID_RESPONSE", "latency_seconds": 0.0}
        started = perf_counter()
        try:
            if counted_input > settings.max_input_tokens:
                raise ProviderError("context_budget_exceeded")
            payload = {"model": settings.model, "messages": messages, "temperature": settings.temperature,
                       "max_completion_tokens": settings.max_output_tokens,
                       "reasoning_effort": settings.reasoning_effort, "include_reasoning": False,
                       "response_format": {"type": "json_object"}}
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            if self._key.encode("utf-8") in body or SECRET_PATTERN.search(body.decode("utf-8")):
                raise ProviderError("secret_in_prompt")
            result = self._send(body, trace)
            content = self._parse(result, trace)
            if trace["usage"]["input_tokens"] > settings.max_input_tokens:
                raise ProviderError("provider_context_budget_exceeded")
            trace["status"] = "OK"
            return content, trace
        except ProviderError as exc:
            trace["status"] = exc.code
            raise ProviderError(exc.code, trace) from None
        finally:
            trace["latency_seconds"] = perf_counter() - started

    def _send(self, body, trace):
        for attempt in range(self.settings.max_retries + 1):
            record = {"attempt": attempt + 1, "request_id": None}
            trace["attempts"].append(record)
            trace["retries"] = attempt
            retry_after = None
            started = perf_counter()
            try:
                with self._stream("POST", self._base + "/chat/completions", content=body,
                                  headers={"Authorization": "Bearer " + self._key,
                                           "Content-Type": "application/json"},
                                  timeout=self.settings.timeout, follow_redirects=False) as response:
                    code = response.status_code
                    record["http_status"] = code
                    record["request_id"] = _identifier(response.headers.get("x-request-id"))
                    if code == 200:
                        data = bytearray()
                        for chunk in response.iter_bytes(chunk_size=8192):
                            data.extend(chunk)
                            if len(data) > 2_000_000:
                                raise ProviderError("invalid_provider_response")
                        try:
                            return json.loads(data)
                        except (ValueError, RecursionError):
                            raise ProviderError("invalid_provider_response") from None
                    if code in (401, 403):
                        raise ProviderError("authentication_error")
                    if code == 404:
                        raise ProviderError("model_or_endpoint_not_found")
                    if code in (400, 413):
                        data = next(response.iter_bytes(chunk_size=8192), b"").lower() if code == 400 else b""
                        is_context = code == 413 or any(marker in data for marker in (
                            b"context_length_exceeded", b"context length", b"context window"))
                        raise ProviderError("context_budget_exceeded" if is_context else "invalid_api_request")
                    if code != 429 and not 500 <= code < 600:
                        raise ProviderError("http_error")
                    error = "rate_limited" if code == 429 else "provider_unavailable"
                    retry_after = response.headers.get("Retry-After")
            except (httpx.TimeoutException, TimeoutError):
                error = "timeout"
            except (httpx.RequestError, OSError):
                error = "network_error"
            finally:
                record["latency_seconds"] = perf_counter() - started
                trace["request_id"] = record["request_id"]
            record["error"] = error
            if attempt == self.settings.max_retries:
                raise ProviderError(error)
            delay = min(8.0, 2 ** attempt + self._jitter(0.0, 0.5))
            if retry_after:
                try:
                    after = float(retry_after)
                except ValueError:
                    # Значение в формате HTTP-date оставляем вызывающей стороне.
                    raise ProviderError(error) from None
                if not math.isfinite(after) or after > 10:
                    raise ProviderError(error)
                delay = max(delay, after)
            self._pause(delay)
        raise AssertionError("Retry loop exhausted")

    def _parse(self, result, trace):
        try:
            trace["request_id"] = trace["request_id"] or _identifier(result.get("id"))
            trace["system_fingerprint"] = _identifier(result.get("system_fingerprint"))
            usage = result["usage"]
            counts = [usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")]
            if any(type(n) is not int or n < 0 for n in counts) or counts[0] + counts[1] != counts[2]:
                raise ValueError
            trace["usage"] = dict(zip(("input_tokens", "output_tokens", "total_tokens"), counts))
            if result["model"] != self.settings.model:
                raise ProviderError("unexpected_provider_model")
            choice, = result["choices"]
            if choice["finish_reason"] != "stop":
                raise ProviderError("incomplete_generation")
            content = choice["message"]["content"]
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            raise ProviderError("invalid_provider_response") from None
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("empty_model_response")
        if self._key in content or SECRET_PATTERN.search(content):
            raise ProviderError("secret_in_model_response")
        return content
