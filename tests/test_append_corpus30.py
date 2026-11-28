from pathlib import Path
import sys
from threading import Barrier

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import append_corpus30 as append


@pytest.mark.parametrize("value", ["0", "-1", "31", "1.5"])
def test_invalid_worker_counts_stop_before_preparation(value, monkeypatch, tmp_path):
    monkeypatch.setattr(append, "WORK", tmp_path / "work")
    with pytest.raises(SystemExit) as error:
        append.main(["--workers", value])
    assert error.value.code == 2
    assert not append.WORK.exists()


def test_document_checkpoints_resume_with_different_worker_count(tmp_path, monkeypatch):
    documents = [dict(document_id=f"doc{i}", revision_id="r1", text_sha256="text") for i in range(3)]
    saved = {doc["document_id"]: {"scored": True} for doc in documents}
    rendezvous = Barrier(3, timeout=5)

    def recover(doc, evidence, config):
        rendezvous.wait()
        return evidence

    monkeypatch.setattr(append, "recover_ner", recover)
    monkeypatch.setattr(append.upstream, "validate_document_result", lambda *args: None)
    original = append.stage_documents("ner", documents, saved, {}, {}, {}, tmp_path, 3, "recipe")

    def unexpected(*args):
        pytest.fail("A completed checkpoint was recomputed")

    monkeypatch.setattr(append, "recover_ner", unexpected)
    assert append.stage_documents("ner", documents, saved, {}, {}, {}, tmp_path, 1, "recipe") == original


def test_re_checkpoint_is_invalidated_by_changed_ner(tmp_path, monkeypatch):
    doc = dict(document_id="doc", revision_id="r1", text_sha256="text")

    def validate(stage, document, result, ner, config):
        assert stage == "re"
        assert result == ner

    monkeypatch.setattr(append, "recover_re", lambda doc, saved, ner, config: ner)
    monkeypatch.setattr(append.upstream, "validate_document_result", validate)
    for revision in ("first", "second"):
        dependency = dict(origin="historical", result={"evidence": revision})
        result = append.stage_documents("re", [doc], {"doc": {}}, {"doc": dependency},
                                        {}, {}, tmp_path, 2, "recipe")
        assert result == {"doc": dependency}


def test_sealed_delta_rejects_changed_parent_and_content(tmp_path):
    delta = tmp_path / "delta"
    delta.mkdir()
    files = {name: delta / name for name in ("documents.jsonl", "entity_resolution.json")}
    for path in files.values():
        path.write_text("{}", encoding="utf-8")
    append.write_checked(delta / "manifest.json", dict(base_generation="base", files_sha256={
        name: append.upstream.sha(path) for name, path in files.items()}))
    assert append.prepare_delta(tmp_path, "base", [], 2) == delta
    with pytest.raises(ValueError, match="another parent"):
        append.prepare_delta(tmp_path, "other", [], 2)
    files["documents.jsonl"].write_text('{"drift":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        append.prepare_delta(tmp_path, "base", [], 2)
