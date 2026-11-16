"""Frozen name-vector retrieval; candidates do not assert entity identity."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cgraph.query_analysis import QueryAnalysis
from cgraph.retrieval_types import Seed, TARGET_NAMESPACE, validate_seed
from cgraph.semantic_concepts import cosine_matrix, pool_e5

CACHE_FINGERPRINT = "4c72a8d17a451298dcede72a3ccc6d4c5260865f63f1f8af1b5a61696bac1710"
PINNED_HASHES = {
    "embedding_manifest.json": "8482ad1b6d4835e85fc7225f27ff93b2250f928faf00eb09c9ffae30a9da9021",
    "representations.json": "5f045f6bdc413544d473999d932c45f988d9de82a4c8c1f6a97ef8a245176390",
    "baseline.json": "e57c323852162262f2039b8b59a75668aa985e3345240fd6f786520655a6f511",
    "text_name_vectors.npy": "0b986f44c8a83a4e4ee3fc78f99405f150257c8026c3e5e9504f924e2e9de327",
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class EntityCache:
    entities: list[dict]
    vectors: np.ndarray
    fingerprint: str
    model: dict


def load_entity_cache(cache_dir: str | Path = ".cache/semantic_concept_sweep") -> EntityCache:
    root = Path(cache_dir)
    for name, expected in PINNED_HASHES.items():
        if _sha(root / name) != expected:
            raise ValueError(f"Frozen cache hash mismatch: {name}")
    manifest = json.loads((root / "embedding_manifest.json").read_text(encoding="utf-8"))
    if (manifest["fingerprint"] != CACHE_FINGERPRINT or manifest["prefix"] != "query: " or
            manifest["pooling"] != "masked_mean" or manifest["max_length"] != 512 or
            not manifest["l2_normalization"] or
            manifest["cache_files_sha256"]["text_name_vectors.npy"] != PINNED_HASHES["text_name_vectors.npy"]):
        raise ValueError("Unexpected frozen embedding recipe")
    reps = json.loads((root / "representations.json").read_text(encoding="utf-8"))
    rows = reps["entities"]
    nodes = json.loads((root / "baseline.json").read_text(encoding="utf-8"))["nodes"]
    entities = [node["properties"] for node in nodes if "RELEXEntity" in node["labels"]]
    by_id = {node["entity_id"]: node for node in entities}
    ids = [row["entity_id"] for row in rows]
    if len(rows) != 811 or len(entities) != 811 or len(by_id) != 811 or len(set(ids)) != 811 or set(ids) != set(by_id):
        raise ValueError("Entity IDs do not match the frozen snapshot")
    for row in rows:
        node = by_id[row["entity_id"]]
        if row["document_id"] != node["document_id"] or row["text_name"] != node["name"]:
            raise ValueError("Entity document/name differs from frozen snapshot")
    vectors = np.load(root / "text_name_vectors.npy", allow_pickle=False)
    if (vectors.shape != (888, 384) or vectors.dtype != np.float32 or not np.isfinite(vectors).all() or
            not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-6)):
        raise ValueError("Invalid frozen vectors")
    return EntityCache(rows, vectors[:811], manifest["fingerprint"], manifest["model"])


class E5Encoder:
    def __init__(self, model: dict):
        self.description = model
        self._tokenizer = None
        self._model = None

    def _load(self):
        desc = self.description
        for item in desc["files"]:
            if _sha(Path(item["path"])) != item["sha256"]:
                raise ValueError("Frozen E5 model hash mismatch")
        import torch
        from tokenizers import Tokenizer
        from transformers import AutoModel
        tokenizer = Tokenizer.from_file(desc["tokenizer"])
        tokenizer.no_padding()
        tokenizer.enable_truncation(max_length=512)
        if tokenizer.token_to_id("[PAD]") != 0:
            raise ValueError("E5 padding mismatch")
        model = AutoModel.from_pretrained(desc["root"], local_files_only=True, trust_remote_code=False,
                                         use_safetensors=True, dtype=torch.float32).cpu().eval()
        self._tokenizer, self._model = tokenizer, model

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        if self._model is None:
            self._load()
        import torch
        outputs = []
        for start in range(0, len(texts), 8):
            batch = self._tokenizer.encode_batch(["query: " + text for text in texts[start:start + 8]])
            width = max(len(item.ids) for item in batch)
            inputs = {key: np.zeros((len(batch), width), dtype=np.int64)
                      for key in ("input_ids", "attention_mask", "token_type_ids")}
            for i, item in enumerate(batch):
                n = len(item.ids)
                inputs["input_ids"][i, :n] = item.ids
                inputs["attention_mask"][i, :n] = item.attention_mask
                inputs["token_type_ids"][i, :n] = item.type_ids
            with torch.inference_mode():
                hidden = self._model(**{key: torch.from_numpy(value) for key, value in inputs.items()}).last_hidden_state.numpy()
            outputs.append(pool_e5(hidden, inputs["attention_mask"]))
        return np.concatenate(outputs)


class EntityLinker:
    """Без min_score выдаёт только кандидатов; порог задаёт вызывающий код.

    Совпадение имени и cosine не подтверждают идентичность сущности. Seeds —
    кандидаты для обхода, допущенные явно заданной политикой min_score.
    """

    def __init__(self, cache: EntityCache, encoder=None, *, namespace: str = TARGET_NAMESPACE,
                 min_score: float | None = None):
        if not namespace:
            raise ValueError("Namespace is required")
        if min_score is not None and (isinstance(min_score, bool) or not isinstance(min_score, (float, int)) or
                                      not math.isfinite(min_score) or not -1 <= min_score <= 1):
            raise ValueError("Query cutoff must be between -1 and 1")
        self.cache, self.encoder, self.namespace, self.min_score = cache, encoder or E5Encoder(cache.model), namespace, min_score
        vectors = cache.vectors
        if (vectors.ndim != 2 or len(vectors) != len(cache.entities) or not np.isfinite(vectors).all() or
                np.any(np.linalg.norm(vectors, axis=1) == 0)):
            raise ValueError("Invalid entity cache")
        ids = [row["entity_id"] for row in cache.entities]
        if len(ids) != len(set(ids)) or any(not row["entity_id"] or not row["document_id"] or not row["text_name"] for row in cache.entities):
            raise ValueError("Invalid entity identities")

    def link(self, query: QueryAnalysis, top_k: int = 5) -> dict:
        """Возвращает режим, кандидатов и причину отказа от выдачи seeds.

        candidates_only сохраняет кандидатов даже с высоким сходством, но не
        выдаёт seeds без политики. abstained относится именно к выдаче seeds.
        """
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        mentions = query["mentions"]
        searches = [(m["mention_id"], m["surface_text"]) for m in mentions]
        if not searches and query["text"].strip():
            searches = [(None, query["text"])]
        candidates, seeds, unresolved = [], [], []
        if searches:
            vectors = np.asarray(self.encoder.encode([text for _, text in searches]))
            if (vectors.shape != (len(searches), self.cache.vectors.shape[1]) or
                    not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0)):
                raise ValueError("Invalid query embeddings")
            scores = cosine_matrix(vectors, self.cache.vectors)
            for i, (mention_id, text) in enumerate(searches):
                order = sorted(range(len(self.cache.entities)), key=lambda j: (
                    0 if mention_id is not None and self.cache.entities[j]["text_name"].casefold() == text.casefold() else 1,
                    -float(scores[i, j]), self.cache.entities[j]["entity_id"]))[:top_k]
                accepted = False
                for rank, j in enumerate(order, 1):
                    row = self.cache.entities[j]
                    exact = mention_id is not None and row["text_name"].casefold() == text.casefold()
                    method = "exact" if exact else ("fallback" if mention_id is None else "vector")
                    seed: Seed = {"node_key": self.namespace + ":" + row["entity_id"], "entity_id": row["entity_id"],
                                  "document_id": row["document_id"], "mention_id": mention_id,
                                  "score": float(scores[i, j]), "match_method": method}
                    validate_seed(seed, self.namespace)
                    candidates.append({**seed, "rank": rank})
                    if self.min_score is not None and seed["score"] >= self.min_score:
                        seeds.append(seed)
                        accepted = True
                if mention_id is not None and not accepted:
                    unresolved.append(mention_id)
        abstention_reason = None
        if not seeds:
            if not searches:
                abstention_reason = "empty_query"
            elif self.min_score is None:
                abstention_reason = "no_linking_policy"
            elif not candidates:
                abstention_reason = "no_candidates"
            else:
                abstention_reason = "below_min_score"
        return {"namespace": self.namespace, "cache_fingerprint": self.cache.fingerprint,
                "candidates": candidates, "seeds": seeds, "unresolved_mentions": unresolved,
                "cutoff": self.min_score, "abstained": not bool(seeds),
                "mode": "candidates_only" if self.min_score is None else "threshold",
                "abstention_reason": abstention_reason}

    def link_seeds(self, query: QueryAnalysis, top_k: int = 5) -> list[Seed]:
        """Вход для pipeline: требует политику, при слабом сходстве возвращает []."""
        if self.min_score is None:
            raise ValueError("link_seeds requires an explicit min_score policy; use link() for candidates only")
        return self.link(query, top_k)["seeds"]
