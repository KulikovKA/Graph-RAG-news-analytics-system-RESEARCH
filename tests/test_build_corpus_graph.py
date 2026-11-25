from pathlib import Path
import json
import subprocess
import sys
import time

import psutil
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_corpus_graph import checkpoint, load_checkpoint, source_identity
from corpus_graph_inference import e3_source_span
from corpus_graph_payload import build_payload, graph_statistics, resolve_entities


def test_process_interruption_reuses_completed_document(tmp_path):
    driver = tmp_path / "fixture_runner.py"
    driver.write_text(f"""
from pathlib import Path
import sys
import time

sys.path.insert(0, {str(SCRIPTS)!r})
import build_corpus_graph as runner

def factory(config, paths):
    def infer(doc, upstream):
        did = doc['document_id']
        with open(paths['calls'], 'a', encoding='utf-8') as log:
            log.write(did + '\\n')
        if did != 'first' and not Path(paths['ready']).exists():
            time.sleep(60)
        return {{'test': did}}
    return infer

if __name__ == '__main__':
    out = Path(sys.argv[1])
    runner.preflight = lambda *args: None
    docs = [dict(document_id=did, revision_id='r1', text_sha256=did)
            for did in ('first', 'second', 'third')]
    with runner.MemorySampler() as memory:
        run = runner.Run(out, len(sys.argv) > 2, dict(threads=1, model_files={{}}), memory)
        run.documents('fixture', docs, factory, dict(calls=out / 'calls', ready=out / 'ready'), ())
""", encoding="utf-8")
    cmd = [sys.executable, str(driver), str(tmp_path)]
    first = tmp_path / "checkpoints/fixture/first.json"
    with (tmp_path / "aborted.log").open("w") as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 20
            while not first.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.05)
            assert first.exists()
            assert not (tmp_path / "checkpoints/fixture/second.json").exists()
        finally:
            if process.poll() is None:
                children = psutil.Process(process.pid).children(recursive=True)
                process.kill()
                for child in children:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                psutil.wait_procs(children, timeout=8)
            process.wait(timeout=10)
    original = first.read_bytes()
    (tmp_path / "ready").touch()
    # Оставшийся после сбоя временный файл и повреждённый итог требуют повтора.
    (first.parent / "second.json.tmp").write_text("{")
    (first.parent / "third.json").write_text("{")
    subprocess.run([*cmd, "--resume"], check=True, capture_output=True, timeout=30)
    assert first.read_bytes() == original
    assert (tmp_path / "calls").read_text().splitlines().count("first") == 1
    for did in ("second", "third"):
        assert json.loads((first.parent / f"{did}.json").read_text())["results"] == {"test": did}
    state = json.loads((tmp_path / "progress.json").read_text())["stages"]["fixture"]
    assert (state["processed"], state["remaining"], state["recovered"]) == (3, 0, 1)


def test_checkpoint_rejects_corruption_and_incompatible_identity(tmp_path):
    path = tmp_path / "checkpoint.json"
    identity = dict(source_identity(dict(document_id="d", revision_id="r", text_sha256="sha")),
                    config_fingerprint="config", model_fingerprint="model")
    checkpoint(path, identity, [], 1)
    assert load_checkpoint(path, identity, lambda _: None) == []
    assert load_checkpoint(path, identity | {"revision_id": "new"}, lambda _: None) is None
    assert load_checkpoint(path, identity | {"config_fingerprint": "new"}, lambda _: None) is None
    value = json.loads(path.read_text())
    value["results"] = ["tampered"]
    path.write_text(json.dumps(value))
    assert load_checkpoint(path, identity, lambda _: None) is None


def test_e3_removes_only_model_appended_window_punctuation():
    assert e3_source_span("value 161.2", dict(start=6, end=12, text="161.2.")) == {
        "start": 6, "end": 11, "text": "161.2"}
    assert e3_source_span("value 161.2.", dict(start=6, end=12, text="161.2.")) == {
        "start": 6, "end": 12, "text": "161.2."}
    with pytest.raises(ValueError, match="Offsets"):
        e3_source_span("value 161.2", dict(start=6, end=12, text="261.2."))


def test_global_er_preserves_offsets_parallel_edges_and_self_loops():
    text = "Long Term (LT). Long Term works."
    docs = {did: dict(document_id=did, revision_id="rev", title=did, text=text) for did in ("a", "b")}

    def entity(did, start, end):
        eid = f"{did}:{start}"
        span = dict(document_id=did, start=start, end=end, surface_text=text[start:end])
        mention = dict(span, revision_id="rev", mention_id="m:" + eid, ner_label_hint="METHOD", tokens=[])
        return dict(span, entity_id=eid, source_mentions=[mention], normalized_forms=[text[start:end]],
                    confidence=.9, entity_types=["METHOD"])

    def relation(head, tail, n):
        return dict(triple_id=str(n), document_id="a", head_id=head, tail_id=tail,
                    relation_type="USES", confidence=.9, evidence=[dict(start=0, end=len(text), text=text,
                    entity_ids=["a:0", "a:11", "a:16"])])

    ner = {did: dict(mentions=[], abbreviations=[dict(short_form="LT", long_form="Long Term",
           long_start=0, long_end=9, short_start=11, short_end=13)]) for did in docs}
    re = {did: dict(entities=[entity(did, 0, 9), entity(did, 11, 13), entity(did, 16, 25)],
          relations=[], raw_relation_count=0) for did in docs}
    re["a"].update(relations=[relation("a:0", "a:11", 1), relation("a:0", "a:16", 2),
                             relation("a:11", "a:16", 3)], raw_relation_count=3)
    resolved = resolve_entities(docs, ner, re, .46)
    assert resolved["mapping"] == {"a:0": "a:0", "a:11": "a:0", "a:16": "a:16",
                                   "b:0": "b:0", "b:11": "b:0", "b:16": "b:16"}
    empty_concepts = dict(concepts=[], memberships=[])
    nodes, edges = build_payload("fixture", "fp", docs, resolved, [], empty_concepts)
    facts = [r for r in edges.values() if r["type"] == "RELEX_REL"]
    assert len(facts) == 3 and sum(r["head"] == r["tail"] for r in facts) == 1
    assert all(r["head"].startswith("fixture:") and r["tail"].startswith("fixture:") for r in edges.values())
    stats = graph_statistics(nodes, edges, resolved, ner, re)
    assert (stats["entities_before_resolution"], stats["entities_after_resolution"]) == (6, 4)
    assert (stats["factual_edges"], stats["unique_graph_triples"], stats["mean_degree"]) == (3, 2, 1.5)
    resolved["relations"][0]["evidence"][0]["text"] = "invalid"
    with pytest.raises(ValueError, match="Offsets"):
        build_payload("fixture", "fp", docs, resolved, [], empty_concepts)
