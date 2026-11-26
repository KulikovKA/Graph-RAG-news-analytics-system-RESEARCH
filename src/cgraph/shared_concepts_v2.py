"""Математика и жизненный цикл каталога Shared Concepts V2."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from fractions import Fraction
from hashlib import sha256
from heapq import heapify, heappop, heappush
from itertools import combinations
import json
from time import perf_counter

import numpy as np

MODEL = "intfloat/e5-small-v2"
DIMENSION = 384
WINDOW = "S1"
EE = (.94, .86)
COSINE_BINS = np.linspace(-1., 1., 201)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value):
    return sha256(canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Thresholds:
    name: float
    context: float

    def __post_init__(self):
        if any(type(v) not in (int, float) or not -1 <= v <= 1 for v in (self.name, self.context)):
            raise ValueError("Cosine thresholds must lie in [-1, 1]")

    def passes(self, scores):
        return scores[0] >= self.name and scores[1] >= self.context


@dataclass(frozen=True)
class Config:
    ec: Thresholds
    cc: Thresholds
    delta: Fraction

    @classmethod
    def from_dict(cls, values):
        if set(values) != {"ec", "cc", "delta"} or type(values["delta"]) not in (str, int, float, Decimal):
            raise ValueError("Specify only EC, CC and delta explicitly")

        def thresholds(kind):
            if any(isinstance(v, Decimal) and (not v.is_finite() or not -1 <= v <= 1)
                   for v in values[kind].values()):
                raise ValueError("Cosine thresholds must lie in [-1, 1]")
            return Thresholds(**{k: float(v) if isinstance(v, Decimal) else v for k, v in values[kind].items()})

        return cls(thresholds("ec"), thresholds("cc"), Fraction(str(values["delta"])))

    def __post_init__(self):
        if not isinstance(self.delta, Fraction) or not 0 <= self.delta < 1:
            raise ValueError("delta must be an exact fraction in [0, 1)")

    def to_dict(self):
        return dict(ec=asdict(self.ec), cc=asdict(self.cc), delta=str(self.delta))


def unit(values):
    """Временная нормализация, применяемая только при вычислении cosine."""
    values = np.asarray(values, dtype=np.float64)
    lengths = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or not np.all(np.isfinite(lengths) & (lengths > 0)):
        raise ValueError("Nonfinite or zero vector")
    return values / lengths


def aligned_cosine(first, second):
    a, b = unit(first), unit(second)
    scores = np.sum(a * b, axis=-1)
    return np.where(np.all(a == b, axis=-1), 1., np.clip(scores, -1, 1))


@dataclass(frozen=True, eq=False)
class Entity:
    entity_id: str
    document_id: str
    name: str
    vectors: np.ndarray
    profile_json: str
    windows_json: str
    recipe_hash: str
    fingerprint: str

    @classmethod
    def create(cls, profile, windows, vectors, recipe_hash):
        entity_id, document, name = (profile[key] for key in ("entity_id", "document_id", "name"))
        if (not all(isinstance(v, str) and v.strip() for v in (entity_id, document, name, recipe_hash))
                or ("document_ids" in profile and set(profile["document_ids"]) != {document})
                or not profile["entity_types"] or not profile["evidence"]
                or any(row["document_id"] != document for row in profile["evidence"])
                or not windows or any(not row["text"].strip() or not row["provenance"] for row in windows)):
            raise ValueError(f"Invalid profile or unsupported multiple documents: {entity_id}")
        data = np.asarray(vectors, dtype="<f8")
        if data.shape != (2, DIMENSION):
            raise ValueError(f"Expected two {DIMENSION}-dimensional vectors")
        unit(data)
        if not np.isclose(np.linalg.norm(data[0]), 1., atol=2e-6, rtol=0):
            raise ValueError("Expected individually L2-normalized name vectors")
        raw = data.tobytes()
        fingerprint = digest([profile, windows, recipe_hash, sha256(raw).hexdigest()])
        return cls(entity_id, document, name, np.frombuffer(raw, dtype="<f8").reshape(data.shape),
                   canonical(profile), canonical(windows), recipe_hash, fingerprint)

    @property
    def profile(self):
        return json.loads(self.profile_json)


@dataclass(frozen=True)
class Concept:
    concept_id: str
    version: int
    members: tuple[str, ...]
    sums: tuple[tuple[float, ...], ...]
    documents: tuple[str, ...]
    representative: str
    name: str
    scores: tuple[tuple[float, float], ...]
    seeds: tuple[str, ...]
    events: tuple[int, ...]
    set_members: frozenset[str]

    @property
    def key(self):
        return self.concept_id, self.version

    @property
    def centroids(self):
        return np.asarray(self.sums) / len(self.members)

    def record(self):
        row = {key: value for key, value in vars(self).items() if key != "set_members"}
        return row | dict(member_count=len(self.members), member_fingerprint=digest(self.members))

    @classmethod
    def from_record(cls, row):
        counts = row.pop("member_count"), row.pop("member_fingerprint")
        if counts != (len(row["members"]), digest(row["members"])):
            raise ValueError("Invalid membership fingerprint")
        row = {k: tuple(v) if isinstance(v, list) else v for k, v in row.items()}
        row.update({k: tuple(map(tuple, row[k])) for k in ("sums", "scores")})
        return cls(**row,
                   set_members=frozenset(row["members"]))


def nesting_pairs(concepts):
    """Полная проверка вложенности для независимой сверки каталога."""
    pairs = set()
    for a, b in combinations(concepts, 2):
        if a.set_members < b.set_members:
            pairs.add((a.concept_id, b.concept_id))
        elif b.set_members < a.set_members:
            pairs.add((b.concept_id, a.concept_id))
    return pairs


@dataclass
class ECResult:
    reason: str
    members: tuple[str, ...]
    sums: np.ndarray
    scores: np.ndarray | None
    rounds: list
    excluded: tuple[str, ...]

    def diagnostic(self):
        return dict(reason=self.reason, members=self.members, rounds=self.rounds, excluded=self.excluded)


class Catalog:
    def __init__(self, entities, config, namespace, recipe_hash=None, *, track_cc=False):
        self.entities = {e.entity_id: e for e in entities}
        if len(self.entities) != len(entities):
            raise ValueError("Duplicate canonical entity ID")
        recipes = {e.recipe_hash for e in entities}
        if recipe_hash is not None:
            recipes.add(recipe_hash)
        if len(recipes) > 1 or not namespace:
            raise ValueError("Mixed vector recipes or missing namespace")
        self.config = config
        self.namespace = namespace
        self.config_hash = digest([config.to_dict(), MODEL, WINDOW, EE, DIMENSION,
                                   next(iter(recipes), None)])
        self.concepts = {}
        self.versions = {}
        self.by_members = {}
        self.by_entity = defaultdict(set)
        self.unit_sums = {}
        self.adjacency = defaultdict(set)
        self.events = []
        self.event_seconds = {}
        self.cc_counts = Counter()
        self.cc_histogram = np.zeros((2, len(COSINE_BINS) - 1), dtype=np.int64) if track_cc else None
        self.queue = []
        self.nesting_rejected = []

    def add_edges(self, pairs):
        pairs = tuple(pairs)
        for a, b in pairs:
            if a == b or self.entities[a].document_id == self.entities[b].document_id:
                raise ValueError("EE edge must connect distinct documents")
        for a, b in pairs:
            self.adjacency[a].add(b)
            self.adjacency[b].add(a)

    def vectors(self, members):
        return np.asarray([self.entities[key].vectors for key in members])

    def sums(self, members):
        return self.vectors(sorted(members)).sum(axis=0) if members else np.zeros((2, DIMENSION))

    def documents(self, members):
        return tuple(sorted({self.entities[key].document_id for key in members}))

    def event(self, operation, *, started=None, **fields):
        index = len(self.events)
        self.events.append(dict(event_id=index, operation=operation, **fields))
        if started is not None:
            self.event_seconds[str(index)] = perf_counter() - started
        return index

    def stabilize(self, members, sums=None):
        initial = current = tuple(sorted(set(members)))
        rounds = []
        budget = int(self.config.delta * len(initial))
        if sums is None:
            sums = self.sums(current)

        def result(reason, scores=None):
            return ECResult(reason, current, sums, scores, rounds, tuple(sorted(set(initial) - set(current))))

        if len(self.documents(current)) < 2:
            return result("MIN_MEMBERS_OR_DOCS")
        while True:
            try:
                scores = aligned_cosine(self.vectors(current), sums)
            except ValueError:
                return result("INVALID_SUM")
            keep = (scores[:, 0] >= self.config.ec.name) & (scores[:, 1] >= self.config.ec.context)
            bad = tuple(key for key, ok in zip(current, keep, strict=True) if not ok)
            removed = len(initial) - len(current) + len(bad)
            rounds.append(dict(before=current, bad=bad, scores=scores.tolist(), cumulative_removed=removed,
                               cumulative_fraction=removed / len(initial)))
            if not bad:
                return result("PASS", scores)
            remaining = tuple(key for key, ok in zip(current, keep, strict=True) if ok)
            if removed > budget:
                return result("EXCLUSION_BUDGET")
            if len(self.documents(remaining)) < 2:
                return result("MIN_MEMBERS_OR_DOCS")
            sums = sums - self.sums(bad)
            current = remaining

    def concept(self, concept_id, version, trial, seeds, events):
        members = trial.members
        frequency = Counter(self.entities[key].name for key in members)
        best = min(range(len(members)), key=lambda i: (-trial.scores[i, 0],
                   -frequency[self.entities[members[i]].name], members[i]))
        representative = members[best]
        return Concept(concept_id, version, members, tuple(map(tuple, trial.sums.tolist())),
                       self.documents(members), representative, self.entities[representative].name,
                       tuple(map(tuple, trial.scores.tolist())), tuple(sorted(set(seeds))),
                       tuple(sorted(set(events))), frozenset(members))

    def fresh_id(self, lineage):
        return "scv2:" + digest([self.namespace, self.config_hash, lineage])

    def put(self, concept):
        duplicate = self.by_members.get(concept.members)
        if duplicate is not None and duplicate != concept.concept_id:
            raise ValueError("Duplicate active membership set")
        existing = self.concepts.get(concept.concept_id)
        if existing is not None and existing.members != concept.members:
            raise ValueError("Membership changes require an atomic replacement")
        self.concepts[concept.concept_id] = concept
        self.versions[concept.concept_id] = concept.version
        self.unit_sums[concept.concept_id] = unit(concept.sums)
        self.by_members[concept.members] = concept.concept_id
        for key in concept.members:
            self.by_entity[key].add(concept.concept_id)

    def remove(self, concept_id):
        concept = self.concepts.pop(concept_id)
        del self.unit_sums[concept_id]
        del self.by_members[concept.members]
        for key in concept.members:
            self.by_entity[key].remove(concept_id)

    def neighborhoods(self, seeds):
        groups = defaultdict(list)
        for seed in sorted(seeds):
            if self.adjacency.get(seed):
                groups[tuple(sorted(self.adjacency[seed] | {seed}))].append(seed)
        return groups

    def initialize(self, seeds, progress=iter, check=lambda: None):
        groups = self.neighborhoods(seeds)
        self.event("neighborhoods", seeds=len(seeds), nonisolated=sum(map(len, groups.values())),
                   raw_size_histogram=dict(Counter(str(len(group)) for group, roots in groups.items() for _ in roots)),
                   exact_groups=len(groups))
        for members, roots in progress(sorted(groups.items())):
            check()
            started = perf_counter()
            trial = self.stabilize(members)
            event = self.event("initialize", started=started, source=members, seeds=roots, **trial.diagnostic())
            if trial.reason != "PASS":
                continue
            duplicate = self.by_members.get(trial.members)
            if duplicate is not None:
                existing = self.concepts[duplicate]
                self.put(replace(existing, seeds=tuple(sorted(set(existing.seeds) | set(roots))),
                                 events=existing.events + (event,)))
                self.event("exact_dedup", source_event=event, survivor=duplicate)
                continue
            concept_id = self.fresh_id(["initialize", trial.members])
            while concept_id in self.concepts:
                # Ранее расширенный концепт может сосуществовать с исходным окружением.
                concept_id = self.fresh_id(["initialize", trial.members, self.concepts[concept_id].key])
            version = self.versions.get(concept_id, 0) + 1
            self.put(self.concept(concept_id, version, trial, roots, [event]))

    def replacement(self, parents, proposed, keep_id=False, blockers=None):
        """Проверяет затронутые старые включения, включая точный дубликат."""
        sources = set(parents)
        duplicate = self.by_members.get(proposed.members)
        if duplicate in parents:
            duplicate = None
        if duplicate is not None:
            old = self.concepts[duplicate]
            target = proposed if keep_id else old
            proposed = replace(target, version=target.version + (not keep_id),
                               seeds=tuple(sorted(set(old.seeds) | set(proposed.seeds))),
                               events=tuple(sorted(set(old.events) | set(proposed.events))))
            parents = (*parents, duplicate)
        changed = set(parents)
        for parent in parents:
            before = self.concepts[parent]
            neighbors = set.union(*(self.by_entity[key] for key in before.members))
            for other_id in sorted(neighbors - {parent}, key=lambda key: self.concepts[key].members):
                other = self.concepts[other_id].set_members
                after = proposed.set_members if other_id in changed else other
                if ((before.set_members < other and not proposed.set_members < after)
                        or (other < before.set_members and not after < proposed.set_members)):
                    if blockers is not None:
                        blockers.update(self.concepts[key].key for key in {parent, other_id} - sources)
                    return None
        return parents, proposed

    def commit(self, replacement):
        parents, proposed = replacement
        for parent in parents:
            self.remove(parent)
        self.put(proposed)

    def append(self, concept_id, entity_id):
        started = perf_counter()
        before = self.concepts[concept_id]
        if entity_id in before.set_members:
            return "NOOP"
        value = self.entities[entity_id].vectors
        pre = aligned_cosine(value, before.sums)
        details = dict(started=started, parent=before.key, entity_id=entity_id, pre_scores=pre.tolist())
        if not self.config.ec.passes(pre):
            self.event("append", reason="PRE_EC", **details)
            return "PRE_EC"
        members = tuple(sorted(before.set_members | {entity_id}))
        sums = np.asarray(before.sums) + value
        try:
            scores = aligned_cosine(self.vectors(members), sums)
        except ValueError:
            self.event("append", reason="INVALID_SUM", **details)
            return "INVALID_SUM"
        details.update(members=members, scores=scores.tolist())
        if not all(self.config.ec.passes(row) for row in scores):
            self.event("append", reason="POST_EC", **details)
            return "POST_EC"
        trial = ECResult("PASS", members, sums, scores, [], ())
        proposed = self.concept(concept_id, before.version + 1, trial, before.seeds,
                                (*before.events, len(self.events)))
        replacement = self.replacement((concept_id,), proposed, keep_id=True)
        if replacement is None:
            self.event("append", reason="NESTING", **details)
            return "NESTING"
        parents, result = replacement
        self.event("append", reason="PASS", retired=sorted(set(parents) - {result.concept_id}),
                   survivor=result.key, **details)
        self.commit(replacement)
        return "PASS"

    def enqueue(self, concept, others):
        if not others:
            return
        a_unit = self.unit_sums[concept.concept_id]
        b_unit = np.asarray([self.unit_sums[other.concept_id] for other in others])
        scores = np.where(np.all(a_unit == b_unit, axis=-1), 1.,
                          np.clip(np.sum(a_unit * b_unit, axis=-1), -1, 1))
        scored = []
        for index, (other, score) in enumerate(zip(others, scores, strict=True)):
            a, b = concept.set_members, other.set_members
            if a < b or b < a:
                self.cc_counts["nested"] += 1
                continue
            if self.cc_histogram is not None:
                scored.append(index)
            intersection = len(a & b)
            kind = "partial" if intersection else "disjoint"
            self.cc_counts[kind] += 1
            name_ok = score[0] >= self.config.cc.name
            context_ok = score[1] >= self.config.cc.context
            self.cc_counts["name_pass"] += int(name_ok)
            self.cc_counts["context_pass"] += int(context_ok)
            if not (name_ok and context_ok):
                self.cc_counts["threshold_reject"] += 1
                continue
            self.cc_counts["eligible"] += 1
            first, second = sorted((concept, other), key=lambda row: row.members)
            priority = -Fraction(intersection, len(a) + len(b) - intersection)
            heappush(self.queue, (priority, (first.members, second.members),
                                  first.key, second.key, tuple(score.tolist())))
        if scored:
            for channel, values in enumerate(scores[scored].T):
                self.cc_histogram[channel] += np.histogram(values, bins=COSINE_BINS)[0]

    def prepare_cc(self, progress=iter, check=lambda: None):
        ordered = sorted(self.concepts)
        for i in progress(range(len(ordered))):
            check()
            self.enqueue(self.concepts[ordered[i]], [self.concepts[key] for key in ordered[:i]])

    def active(self, key):
        concept = self.concepts.get(key[0])
        return concept is not None and concept.version == key[1]

    def retry_nesting(self):
        still_blocked = []
        for entry, dependencies in self.nesting_rejected:
            if self.active(entry[2]) and self.active(entry[3]):
                if any(not self.active(key) for key in dependencies):
                    heappush(self.queue, entry)
                else:
                    still_blocked.append((entry, dependencies))
        self.nesting_rejected = still_blocked

    def merge_next(self):
        while self.queue:
            entry = heappop(self.queue)
            priority, _, first, second, scores = entry
            if not (self.active(first) and self.active(second)):
                continue
            started = perf_counter()
            a, b = self.concepts[first[0]], self.concepts[second[0]]
            members = a.set_members | b.set_members
            trial = self.stabilize(members, np.asarray(a.sums) + b.sums - self.sums(a.set_members & b.set_members))
            info = dict(started=started, parents=(first, second), jaccard=str(-priority), cc_scores=scores,
                        source=tuple(sorted(members)), **trial.diagnostic())
            if trial.reason != "PASS":
                self.event("merge", **info)
                return True
            event = len(self.events)
            # ID по-прежнему зависит от отсортированных версий родителей.
            concept_id = self.fresh_id(["merge", *sorted((first, second)), trial.members])
            version = self.versions.get(concept_id, 0) + 1
            proposed = self.concept(concept_id, version, trial, a.seeds + b.seeds, (*a.events, *b.events, event))
            blockers = set()
            replacement = self.replacement((first[0], second[0]), proposed, blockers=blockers)
            if replacement is None:
                dependencies = tuple(sorted(blockers))
                if dependencies:
                    self.nesting_rejected.append((entry, dependencies))
                self.event("merge", **(info | dict(reason="NESTING", dependencies=dependencies)))
                return True
            parents, survivor = replacement
            self.event("merge", **(info | dict(retired=sorted(set(parents) - {survivor.concept_id}),
                                               survivor=survivor.key, exact_collision=len(parents) == 3)))
            self.commit(replacement)
            self.enqueue(survivor, [self.concepts[key] for key in sorted(self.concepts) if key != survivor.concept_id])
            self.retry_nesting()
            return True
        return False

    def snapshot(self):
        def pack(entry):
            priority, members, a, b, scores = entry
            return str(priority), members, a, b, scores

        return dict(concepts=[self.concepts[key].record() for key in sorted(self.concepts)],
                    versions=dict(self.versions),
                    edges=[(a, b) for a in sorted(self.adjacency) for b in sorted(self.adjacency[a]) if a < b],
                    events=self.events, cc_counts=dict(self.cc_counts), queue=[pack(e) for e in sorted(self.queue)],
                    nesting_rejected=[(pack(e), keys) for e, keys in sorted(self.nesting_rejected)],
                    cc_histogram=self.cc_histogram.tolist() if self.cc_histogram is not None else None)

    def restore(self, state):
        def unpack(entry):
            priority, members, a, b, scores = entry
            return Fraction(priority), tuple(map(tuple, members)), tuple(a), tuple(b), tuple(scores)

        for row in state["concepts"]:
            self.put(Concept.from_record(dict(row)))
        self.versions.update(state["versions"])
        self.add_edges(state["edges"])
        self.events = list(state["events"])
        self.cc_counts = Counter(state["cc_counts"])
        if self.cc_histogram is not None:
            self.cc_histogram[:] = state["cc_histogram"]
        self.queue = [unpack(entry) for entry in state["queue"]]
        heapify(self.queue)
        self.nesting_rejected = [(unpack(entry), tuple(map(tuple, keys))) for entry, keys in state["nesting_rejected"]]
