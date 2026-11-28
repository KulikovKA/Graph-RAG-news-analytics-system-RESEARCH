"""Двухканальный Entity Linking для замороженного графа Corpus100."""
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from cgraph.entity_linking import E5Encoder
from cgraph.query_analysis import QueryAnalysis
from cgraph.retrieval_types import Seed, validate_seed
from cgraph.semantic_concepts import cosine_matrix
from cgraph.shared_concepts_v2 import DIMENSION, Entity, unit

NAMESPACE = "relex_corpus100_shared_concepts_v2"


@dataclass(frozen=True)
class LinkingPolicy:
    threshold_name: float
    alpha: float
    top_k: int
    threshold_score: float | None = None

    def __post_init__(self):
        for name, value, lower in (("threshold_name", self.threshold_name, -1), ("alpha", self.alpha, 0),
                                   ("threshold_score", self.threshold_score, -1)):
            if name == "threshold_score" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not lower <= value <= 1:
                raise ValueError(f"Invalid {name}")
        if type(self.top_k) is not int or self.top_k < 1:
            raise ValueError("top_k must be a positive integer")

    @property
    def variant(self):
        return "C1" if self.threshold_score is None else "C2"


@dataclass(frozen=True)
class Similarities:
    name: np.ndarray
    context: np.ndarray

    def __post_init__(self):
        if (self.name.ndim != 2 or self.name.shape != self.context.shape
                or not np.isfinite(self.name).all() or not np.isfinite(self.context).all()
                or np.any(np.abs(self.name) > 1) or np.any(np.abs(self.context) > 1)):
            raise ValueError("Invalid name/context similarity matrices")


def rank_mentions(similarities: Similarities, policy: LinkingPolicy) -> list[list[tuple[int, float]]]:
    """Столбцы упорядочены по ID сущности, стабильная сортировка сохраняет этот порядок при равенстве."""
    rankings = []
    for names, contexts in zip(similarities.name, similarities.context, strict=True):
        rows = np.flatnonzero(names >= policy.threshold_name)
        scores = policy.alpha * names[rows] + (1 - policy.alpha) * contexts[rows]
        if policy.threshold_score is not None:
            keep = scores >= policy.threshold_score
            rows, scores = rows[keep], scores[keep]
        top = np.argsort(-scores, kind="stable")[:policy.top_k]
        rankings.append(list(zip(rows[top].tolist(), scores[top].tolist(), strict=True)))
    return rankings


def merge_mentions(rankings: list[list[tuple[int, float]]]) -> list[tuple[int, int, float]]:
    """Один seed на сущность с максимальным score и provenance первого лучшего упоминания."""
    best: dict[int, tuple[int, float]] = {}
    for mention, ranking in enumerate(rankings):
        for entity, score in ranking:
            if entity not in best or score > best[entity][1]:
                best[entity] = mention, score
    return [(entity, *best[entity]) for entity in sorted(best, key=lambda i: (-best[i][1], i))]


class EntityLinkerV2:
    def __init__(self, entities: Iterable[Entity], model: dict,
                 context_inputs: Callable[[QueryAnalysis], list[str]], *, encoder=None):
        self.entities = tuple(sorted(entities, key=lambda e: e.entity_id))
        if not self.entities or len({e.entity_id for e in self.entities}) != len(self.entities):
            raise ValueError("Expected unique Corpus100 entity IDs")
        self.vectors = np.stack([e.vectors for e in self.entities], axis=1)
        if self.vectors.shape != (2, len(self.entities), DIMENSION):
            raise ValueError("Invalid Corpus100 vector layout")
        unit(self.vectors)
        self.encoder = encoder if encoder is not None else E5Encoder(model)
        self.context_inputs = context_inputs

    def similarities(self, query: QueryAnalysis) -> Similarities:
        names = [mention["surface_text"] for mention in query["mentions"]]
        if not names:
            empty = np.empty((0, len(self.entities)), dtype=np.float64)
            return Similarities(empty, empty)
        contexts = self.context_inputs(query)
        if len(contexts) != len(names):
            raise ValueError("Expected one context input per mention")
        texts = list(dict.fromkeys(names + contexts))
        vectors = np.asarray(self.encoder.encode(texts), dtype=self.vectors.dtype)
        if vectors.shape != (len(texts), DIMENSION):
            raise ValueError("Invalid query vector layout")
        unit(vectors)
        by_text = dict(zip(texts, vectors, strict=True))
        return Similarities(*(cosine_matrix(np.stack([by_text[t] for t in inputs]), stored)
                              for inputs, stored in zip((names, contexts), self.vectors, strict=True)))

    def _seed(self, query: QueryAnalysis, mention: int, row: int, score: float) -> Seed:
        entity = self.entities[row]
        seed: Seed = dict(node_key=f"{NAMESPACE}:{entity.entity_id}", entity_id=entity.entity_id,
                          document_id=entity.document_id, mention_id=query["mentions"][mention]["mention_id"],
                          score=score, match_method="vector")
        validate_seed(seed, NAMESPACE)
        return seed

    def link(self, query: QueryAnalysis, policy: LinkingPolicy) -> dict:
        similarities = self.similarities(query)
        rankings = rank_mentions(similarities, policy)
        candidates = []
        for mention, ranking in enumerate(rankings):
            candidates.extend(dict(self._seed(query, mention, row, score), rank=rank,
                                   s_name=float(similarities.name[mention, row]),
                                   s_context=float(similarities.context[mention, row]))
                              for rank, (row, score) in enumerate(ranking, 1))
        seeds = [self._seed(query, mention, row, score) for row, mention, score in merge_mentions(rankings)]
        return dict(namespace=NAMESPACE, variant=policy.variant, candidates=candidates, seeds=seeds,
                    abstained=not seeds,
                    unresolved_mentions=[m["mention_id"] for m, ranking in zip(query["mentions"], rankings, strict=True)
                                         if not ranking])

    def link_seeds(self, query: QueryAnalysis, policy: LinkingPolicy) -> list[Seed]:
        return [self._seed(query, mention, row, score) for row, mention, score in
                merge_mentions(rank_mentions(self.similarities(query), policy))]
