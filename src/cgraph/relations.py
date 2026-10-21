"""Small, trained, multi-label marked-pair classifiers. No generative backend."""
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .candidates import marked_text
from .common import fingerprint, read_json, sha256, write_json

NO_RELATION = "NO_RELATION"
ABSTAIN = "ABSTAIN"


def new_vectorizer(cfg, vocabulary=None):
    return TfidfVectorizer(lowercase=False, ngram_range=(1, 2), sublinear_tf=True,
                           token_pattern=r"(?u)\b[\w]+\b", max_features=cfg["relations"]["max_features"],
                           vocabulary=vocabulary, dtype=np.float64)


class RelationClassifier:
    def __init__(self, cfg, ontology):
        self.cfg = cfg
        directory = Path(cfg["relations"]["model_dir"])
        self.metadata = read_json(directory / "metadata.json")
        expected = list(ontology["relations"]) + [NO_RELATION]
        if self.metadata["labels"] != expected:
            raise ValueError("Relation model/ontology label mismatch")
        payload = (directory / "parameters.npz").read_bytes()
        if sha256(payload) != self.metadata["parameters_sha256"]:
            raise ValueError("Relation weights SHA-256 mismatch")
        features = read_json(directory / "features.json")
        if fingerprint(features) != self.metadata["features_sha256"]:
            raise ValueError("Relation tokenizer/features SHA-256 mismatch")
        self.vectorizer = new_vectorizer(cfg, features["vocabulary"])
        self.vectorizer.idf_ = np.array(features["idf"])
        with np.load(directory / "parameters.npz", allow_pickle=False) as data:
            self.coef = data["coef"]
            self.intercept = data["intercept"]
            self.trained = data["trained"].astype(bool)
        self.labels = expected
        self.calls = 0

    def predict(self, chunk, pairs):
        if not pairs:
            return []
        self.calls += 1
        x = self.vectorizer.transform([marked_text(chunk, p) for p in pairs])
        logits = np.asarray(x @ self.coef.T) + self.intercept
        scores = 1 / (1 + np.exp(-np.clip(logits, -60, 60)))
        scores[:, ~self.trained] = 0
        decisions = []
        for pair, row in zip(pairs, scores, strict=True):
            candidates = [(label, float(row[i])) for i, label in enumerate(self.labels[:-1]) if label in pair["allowed_labels"]]
            positives = [(label, score) for label, score in candidates if score >= self.cfg["relations"]["threshold"]]
            no_score = float(row[-1])
            unavailable = [label for label in pair["allowed_labels"] if not self.trained[self.labels.index(label)]]
            if positives and no_score < self.cfg["relations"]["no_relation_threshold"]:
                decisions.append({"status": "RELATION", "predictions": positives, "no_relation_score": no_score})
            elif not positives and not unavailable and no_score >= self.cfg["relations"]["no_relation_threshold"]:
                decisions.append({"status": NO_RELATION, "predictions": [], "no_relation_score": no_score})
            else:
                decisions.append({"status": ABSTAIN, "predictions": [], "reason": "LOW_CONFIDENCE_OR_CONFLICT", "no_relation_score": no_score})
            decisions[-1]["unsupported_labels"] = unavailable
        return decisions


def train(cfg, ontology, documents, gold):
    from sklearn.linear_model import LogisticRegression
    from .candidates import generate_pairs
    from .labels import gold_entities

    labels = list(ontology["relations"]) + [NO_RELATION]
    texts, targets, inventory = [], [], []
    for doc in documents:
        g = gold[doc["document_code"]]
        positives = {}
        for a in g["assertions"]:
            if a["support_status"] == "SUPPORTED":
                positives.setdefault((a["head_mention_id"], a["tail_mention_id"]), set()).add(a["relation_type"])
        rows, negatives = [], []
        excluded = 0
        seen = set()
        for chunk in doc["chunks"]:
            entities = gold_entities(g, chunk)
            pairs, _ = generate_pairs(chunk, entities, ontology, cfg)
            for pair in pairs:
                key = pair["head"]["mention_id"], pair["tail"]["mention_id"]
                target = positives.get(key, set())
                # Unknown sentences are not hard negatives. This is intentionally conservative.
                unknown = any(e["chunk_id"] == chunk["chunk_id"] and e["start"] < pair["end"] and pair["start"] < e["end"]
                              for u in g.get("unresolved", []) for e in u.get("evidence", []))
                if not target and (unknown or not g["exhaustive_relations"]):
                    excluded += 1
                    continue
                entry = (marked_text(chunk, pair), [int(l in target if target else l == NO_RELATION) for l in labels])
                if target:
                    rows.append(entry)
                    seen.add(key)
                else:
                    # Reversed positives first; then close same-sentence compatible entities.
                    negatives.append(((0 if (key[1], key[0]) in positives else 1, pair["gap"], key), entry))
        negatives.sort(key=lambda x: x[0])
        rows.extend(x[1] for x in negatives[:cfg["relations"]["max_negatives_per_document"]])
        inventory.append(dict(document_code=doc["document_code"], examples=len(rows), positive_pairs=len(seen),
                              gold_positive_pairs=len(positives), excluded_unresolved_negatives=excluded,
                              exhaustive_relations=g["exhaustive_relations"],
                              hard_negatives=min(len(negatives), cfg["relations"]["max_negatives_per_document"])))
        for text, target in rows:
            texts.append(text)
            targets.append(target)
    if not texts:
        raise ValueError("No training examples")
    vectorizer = new_vectorizer(cfg)
    x = vectorizer.fit_transform(texts)
    y = np.asarray(targets)
    coef = np.zeros((len(labels), x.shape[1]))
    intercept = np.zeros(len(labels))
    trained = np.zeros(len(labels), dtype=bool)
    supports = {}
    for i, label in enumerate(labels):
        supports[label] = int(y[:, i].sum())
        if len(np.unique(y[:, i])) != 2:
            continue
        estimator = LogisticRegression(C=cfg["relations"]["C"], class_weight="balanced", max_iter=cfg["relations"]["max_iter"],
                                       solver="liblinear", random_state=cfg["seed"])
        estimator.fit(x, y[:, i])
        coef[i], intercept[i], trained[i] = estimator.coef_[0], estimator.intercept_[0], True
    directory = Path(cfg["relations"]["model_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "parameters.npz", coef=coef, intercept=intercept, trained=trained)
    features = dict(vocabulary={k: int(v) for k, v in vectorizer.vocabulary_.items()}, idf=vectorizer.idf_.tolist())
    write_json(directory / "features.json", features)
    metadata = dict(format="marked-pair-tfidf-ovr-v1", labels=labels, seed=cfg["seed"], tokenizer="sklearn TfidfVectorizer: Unicode word regex, case-sensitive, unigram+bigram",
                    pretrained_checkpoint=None, license="Repository code license; newly fitted research parameters, no external RE checkpoint",
                    train_documents=[d["document_code"] for d in documents], training_inventory=inventory,
                    positive_support=supports, untrained_labels=[l for l, flag in zip(labels, trained) if not flag],
                    parameters_sha256=sha256((directory / "parameters.npz").read_bytes()), features_sha256=fingerprint(features),
                    config_sha256=fingerprint(cfg), training_labels_sha256=fingerprint({d["document_code"]: read_json(cfg["manifest"])["documents"][d["document_code"]]["gold_sha256"] for d in documents}))
    write_json(directory / "metadata.json", metadata)
    return metadata
