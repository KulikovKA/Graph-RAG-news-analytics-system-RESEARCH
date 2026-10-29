# C-RESTART-001: audit and fixed experiment protocol

Recorded before opening DEV labels for this experiment, 2026-10-29 (Europe/Moscow).

Base: research origin/main `5bd3ff21d84ce6dcb465264dc4a11f2fa27516ef`.
Existing staged C-002 work is preserved in the original worktree. This branch uses
an isolated worktree. README, C001_REPORT, all source modules, configs, contracts,
and both historical test files were reviewed. No CALIBRATION/HOLDOUT content is
needed or opened. Frozen ontology/source contracts remain byte-identical.

C-001 audit: TF-IDF/LogisticRegression is supervised training on eight DEV
documents; it cannot implement the requested zero-shot protocol. It remains a
historical baseline, with no imported fitted parameters. Candidate generation
limited same-sentence distance to 300 characters and 400 pairs/chunk. Wrong NER
boundaries/types account for most missing endpoints. The gate required a narrow
lexical cue between endpoints and mistook uppercase chemical NO for negation.
It removed all five correct raw relations on the four legacy validation docs.

Reuse: hash-verified DEV loader, label-free normative ontology projection,
GLiNER-small implementation and pinned files, exact evidence utilities,
conservative canonicalization, prediction schema, structural validator and exact
multiset evaluator. Keep the historical CLI/tests/reports; do not invoke training
commands or the historical training test (even that test fits on DEV).

Fixed selection, without a model sweep: retain GLiNER-small 0.35 and its 12
descriptions to isolate the RE replacement. GLiREL-large-v0 is an available
separate encoder RE checkpoint accepting externally supplied mention spans.
CPU float32, four threads, seed 42. GLiREL threshold 0.5 is its API default;
multi-label output top_k=-1. Do not search thresholds, synonyms or checkpoints
using Gold. Enumerate both directions in one/two adjacent sentences without
type, character-gap or pair-budget pruning. Deduplicate repeated predictions by
maximum score; no Gold-assisted boundary repairs. No coreference claim.

GLiREL 1.2.1 lacks a declared loguru dependency; pin loguru explicitly. Its
backbone loader normally fetches base weights that the full RE checkpoint
immediately replaces. Construct the local backbone skeleton instead, then load
the complete unchanged RE checkpoint with strict=True, weights_only=True. No
missing/unexpected/random task heads are allowed. Frozen parameters, eval mode,
inference_mode, deterministic PyTorch operations. No optimizer/backward/fit.

All 23 capabilities are declared in config. 22 prompts are EXPERIMENTAL_ZERO_SHOT,
not claims of validated scientific semantics. MEETS_REQUIREMENT is UNSUPPORTED:
joint_requirement=all and collective scope require n-ary interpretation.
CHANGES_PARAMETER additionally needs an explicit deterministic physical direction.
No fabricated NO_RELATION head: no score above threshold means ABSTAIN.

After raw RE: frozen domain/range, clause-level negation/modality, required
qualifiers, exact evidence and canonicalization. Remove the narrow predicate cue
gate. Chemical NO/NOx/NO2 is distinct from linguistic negation. Preserve raw
numeric/unit evidence without arbitrary attachment. Attach only simple explicit
trailing temperatures and required direction cues. Completeness remains UNKNOWN;
strict qualified TP is unavailable rather than falsely declaring full support.
Unresolved optional qualifiers do not erase the provisional binary relation.
Exact evidence establishes provenance, not independently verified semantics.

One real guarded fresh-process run of all 12 DEV documents. Gold only in a later
separate evaluation process; no Gold-entity diagnostic inference. Report exact
typed entities, directed raw/final relations, strict qualified relations,
candidate recall, abstention, wall/CPU, process peak RAM, API and encoder calls.
Report all-DEV results and the identical four-document legacy validation view.
Record a loss funnel: untyped endpoints -> typed endpoints -> candidates ->
model-served pairs -> raw -> domain/range -> scope -> evidence -> final. Retain
per-relation aggregate counts and per-gate losses. No tuning after scoring.

Historical C-001 is not an equal no-training comparison; eight DEV docs were
training data. DEV is exploratory and was already examined in C-001. D024 has
incomplete relation annotation, limiting the interpretation of all-DEV precision.
No confirmatory HOLDOUT claim, no downstream reranker, no generative LLM.

Sources inspected:
- https://github.com/jackboyla/GLiREL (separate RE architecture, API, inclusive input spans)
- https://huggingface.co/jackboyla/glirel-large-v0/tree/40a523e12a8432d6da364cf2a195a28755ff04d3
- https://arxiv.org/abs/2501.03172 (FewRel/WikiZSL evidence, not scientific ontology validation)
- https://huggingface.co/urchade/gliner_small-v2.1
- https://huggingface.co/microsoft/deberta-v3-large

Licensing: GLiNER Apache-2.0; DeBERTa cards MIT; GLiREL model card/repository text
CC BY-NC-SA 4.0, despite Apache-2.0 in package metadata. Treat GLiREL as
noncommercial research use; do not label the checkpoint permissively licensed.
Joint GLiNER-Relex/GLiNER2 approaches belong to F. Generative RE is excluded.
