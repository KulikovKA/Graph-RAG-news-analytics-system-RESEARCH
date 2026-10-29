from pathlib import Path

import numpy as np
import pytest

from cgraph.common import config, read_json
from cgraph.inputs import load_inputs, load_ontology
from cgraph.labels import load_labels
from cgraph.relations import RelationClassifier, train


@pytest.mark.historical_training
def test_real_classifier_training_serialization_and_split(tmp_path):
    cfg = config()
    cfg["relations"]["model_dir"] = str(tmp_path / "model")
    ontology = load_ontology(cfg)
    docs = load_inputs(cfg, "train")
    metadata = train(cfg, ontology, docs, load_labels(cfg, docs))
    assert len(metadata["labels"]) == 24 and metadata["positive_support"]["NO_RELATION"] > 0
    assert set(metadata["train_documents"]).isdisjoint(read_json(cfg["partition"])["validation"])
    # Incomplete D024 is positives-only, never a source of assumed negatives.
    assert next(row for row in metadata["training_inventory"] if row["document_code"] == "D024")["hard_negatives"] == 0
    model = RelationClassifier(cfg, ontology)
    assert np.any(model.coef != 0) and model.labels[-1] == "NO_RELATION"
    from cgraph.candidates import generate_pairs
    from cgraph.labels import gold_entities
    chunk = docs[0]["chunks"][0]
    labels = load_labels(cfg, [docs[0]])
    pairs, _ = generate_pairs(chunk, gold_entities(labels[docs[0]["document_code"]], chunk), ontology, cfg)
    first = model.predict(chunk, pairs)
    assert first == RelationClassifier(cfg, ontology).predict(chunk, pairs)
    assert first and all(d["status"] in {"RELATION", "NO_RELATION", "ABSTAIN"} for d in first)


def test_inference_modules_have_no_annotation_imports():
    import ast
    for name in ("pipeline", "inputs", "ner", "rules", "candidates", "validation"):
        tree = ast.parse(Path(f"src/cgraph/{name}.py").read_text(encoding="utf-8"))
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert "labels" not in imports


def test_provenance_does_not_open_the_annotation_module():
    import subprocess
    import sys
    code = """
import sys
def guard(event, args):
    if event == 'open' and isinstance(args[0], str) and args[0].replace('\\\\', '/').endswith('cgraph/labels.py'):
        raise PermissionError('Forbidden annotation module access')
sys.addaudithook(guard)
from cgraph.cli import inference_code_hashes
hashes = inference_code_hashes()
assert hashes and all(not p.endswith('/labels.py') for p in hashes)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
