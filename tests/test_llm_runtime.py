from io import StringIO
import json
import sys
from types import SimpleNamespace

import pytest

from cgraph import llm_qa, llm_retrieval


def test_runtime_reuses_lazy_resources_after_query_error(monkeypatch):
    profile = {"graph_fingerprint": "graph", "snapshot_sha256": "snapshot", "seed": 42}
    events = []

    class Lazy:
        def __init__(self, name):
            self.name = name
            self.model = None
            self.cache = object()

        def call(self, text, *_):
            if self.model is None:
                self.model = object()
                events.append(("load", self.name))
            events.append((self.name, text))
            return text

    analyzer, linker = Lazy("analyzer"), Lazy("linker")
    cache = linker.cache

    def retrieve(question, _, driver, database, timings, *, analyze, link):
        assert analyze.__self__ is analyzer and link.__self__ is linker
        result = link(analyze(question), 10)
        if question == "broken":
            raise RuntimeError
        return {"result": result}

    def snapshot(driver, database):
        events.append(("snapshot", database))
        return {k: profile[k] for k in ("graph_fingerprint", "snapshot_sha256")}

    driver = SimpleNamespace(close=lambda: events.append("close"))
    analyzer.analyze, linker.link_seeds = analyzer.call, linker.call
    monkeypatch.setattr(llm_retrieval, "retrieve_c", retrieve)
    runtime = llm_retrieval.LiveRetrievalRuntime(profile, driver, "neo4j", analyzer, linker, snapshot,
                                               lambda seed: events.append(("seed", seed)))
    with runtime:
        first_timings = {}
        assert runtime.retrieve("first", first_timings)["result"] == "first"
        with pytest.raises(RuntimeError):
            runtime.retrieve("broken", {})
        assert runtime.retrieve("last", {})["result"] == "last"
        assert linker.cache is cache
    assert [e for e in events if e[0] == "load"] == [("load", "analyzer"), ("load", "linker")]
    assert events.count(("snapshot", "neo4j")) == 5
    assert events.count(("seed", 42)) == 3
    assert {"retrieval_preflight", "retrieval_postflight"} <= first_timings.keys()
    assert runtime._analyzer is None and runtime._linker is None
    runtime.close()
    assert events.count("close") == 1
    with pytest.raises(llm_retrieval.RetrievalError, match="closed"):
        runtime.retrieve("closed", {})


def test_runtime_closes_on_graph_drift(monkeypatch):
    verified = {"graph_fingerprint": "graph", "snapshot_sha256": "snapshot"}
    observations = iter([verified, dict(verified, snapshot_sha256="changed")])
    component = SimpleNamespace(analyze=lambda _: {}, link_seeds=lambda *_: [])
    monkeypatch.setattr(llm_retrieval, "retrieve_c", lambda *_, **__: {})
    closed = []
    driver = SimpleNamespace(close=lambda: closed.append(True))
    with pytest.raises(llm_retrieval.RetrievalError, match="changed"):
        with llm_retrieval.LiveRetrievalRuntime(dict(verified, seed=42), driver, None, component,
                                               component, lambda *_: next(observations), lambda _: None) as runtime:
            runtime.retrieve("question", {})
    assert closed == [True]


@pytest.fixture
def runtime_cli(monkeypatch):
    events = []
    profile = llm_retrieval.load_profile()

    class Runtime:
        def __enter__(self):
            events.append("open")
            return self

        def __exit__(self, *_):
            events.append("close")

        def retrieve(self, question, timings):
            events.append(question)
            if question == "broken":
                raise RuntimeError("untrusted driver detail")
            return {"paths": []}

    def open_runtime(frozen, **options):
        runtime = Runtime()
        runtime.profile = frozen
        return runtime

    def tokenizer(*_):
        events.append("tokenizer_load")
        return SimpleCounter()

    monkeypatch.setattr(llm_qa, "load_profile", lambda: profile)
    monkeypatch.setattr(llm_qa, "open_live_runtime", open_runtime)
    monkeypatch.setattr(llm_qa, "HarmonyTokenCounter", tokenizer)
    monkeypatch.setattr(llm_qa, "peak_rss_bytes", lambda: 123)
    monkeypatch.setenv("LLM_MODEL", "fixture")
    monkeypatch.setenv("LLM_TOKENIZER_PATH", "unused")
    monkeypatch.setenv("LLM_TOKENIZER_SHA256", "unused")
    return events


class SimpleCounter:
    sha256, reserve = "fixture", 0

    def __call__(self, messages):
        return 100


def test_interactive_reuses_runtime_and_tokenizer_continues_after_error(monkeypatch, runtime_cli, tmp_path, capsys):
    monkeypatch.setattr(sys, "stdin", StringIO("first\n\nbroken\nlast\n exit \nafter exit\n"))
    assert llm_qa.main(["--interactive", "--output", str(tmp_path)]) == 2
    assert runtime_cli == ["open", "first", "tokenizer_load", "broken", "last", "close"]
    records = [json.loads(path.read_bytes()) for path in sorted(tmp_path.glob("*.json"))]
    assert [r["session_query"] for r in records] == [1, 2, 3]
    assert [r["status"] for r in records] == ["INSUFFICIENT_EVIDENCE", "ERROR", "INSUFFICIENT_EVIDENCE"]
    assert all(record["peak_rss_bytes"] == 123 for record in records)
    assert "runtime_setup" in records[0]["timings_seconds"] and "runtime_setup" not in records[2]["timings_seconds"]
    report = capsys.readouterr()
    assert "untrusted driver detail" not in report.out + report.err


@pytest.mark.parametrize("interrupt", [False, True])
def test_interactive_closes_on_eof_and_interrupt(monkeypatch, runtime_cli, interrupt):
    class Input(StringIO):
        def readline(self, *args):
            if interrupt and self.tell():
                raise KeyboardInterrupt
            return super().readline(*args)

    monkeypatch.setattr(sys, "stdin", Input("first\n"))
    assert llm_qa.main(["--interactive"]) == (130 if interrupt else 0)
    assert runtime_cli == ["open", "first", "tokenizer_load", "close"]


def test_interactive_exits_without_opening_runtime(monkeypatch, runtime_cli):
    monkeypatch.setattr(sys, "stdin", StringIO("exit\n"))
    assert llm_qa.main(["--interactive"]) == 0
    assert runtime_cli == []


def test_interactive_sanitizes_startup_failure(monkeypatch, runtime_cli, capsys):
    def fail(*args, **kwargs):
        raise RuntimeError("sensitive connection details")

    monkeypatch.setattr(llm_qa, "open_live_runtime", fail)
    monkeypatch.setattr(sys, "stdin", StringIO("question\n"))
    assert llm_qa.main(["--interactive"]) == 2
    stderr = capsys.readouterr().err
    assert "startup failed (RuntimeError)" in stderr
    assert "sensitive connection details" not in stderr


@pytest.mark.parametrize("flags", [["--interactive", "--question", "Q"], ["--interactive", "--replay"]])
def test_cli_rejects_conflicting_modes(flags):
    with pytest.raises(SystemExit, match="2"):
        llm_qa.main(flags)


def test_one_shot_still_closes_and_returns_preflight_time(monkeypatch):
    events = []

    class Runtime:
        def __enter__(self):
            events.append("enter")
            return self

        def retrieve(self, question, timings):
            events.append(question)
            timings["retrieval_preflight"] = 1.0
            return {"paths": []}

        def __exit__(self, *_):
            events.append("close")

    monkeypatch.setattr(llm_retrieval, "open_live_runtime", lambda *_, **__: Runtime())
    timings = {}
    result = llm_retrieval.live_retrieval("Q", {}, timings, neo4j_env=None, entity_cache=None, gliner_path=None)
    assert events == ["enter", "Q", "close"]
    assert result == {"paths": []} and timings["retrieval_preflight"] >= 1.0
    assert "runtime_setup" not in timings
