# Repository Guidelines

## Project Structure & Module Organization

- `src/cgraph/`: extraction, validation, concept linking; `ner_v2/` and `entity_typing_v2/`: research workflows.
- `scripts/`: runners, audits, Neo4j importers; `tests/`: pytest suites.
- `config/`: settings; `contracts/`: schemas, ontologies, partitions, provenance locks.
- `docs/architecture/`: KG design; `reports/`: metrics and evidence.
- `data/`, `artifacts/`: Git-excluded inputs, labels, weights, predictions.

## Build, Test, and Development Commands

Python 3.11; repository root:

- `uv sync --frozen --extra test`: install locked dependencies.
- `uv run --no-sync cgraph fetch-dev`: fetch permitted DEV data.
- `uv run --no-sync cgraph download-models`: download pinned assets.
- `uv run --no-sync cgraph train`: train baseline classifier.
- `uv run --no-sync cgraph extract --subset validation --output artifacts/new_run`: extract predictions.
- `uv run --no-sync python -m pytest tests/test_semantic_concepts.py -q`: focused tests.

Workflow extras (see `docs/`): `ner-v2`, `research`, `relex`.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions/modules, `PascalCase` classes, `UPPER_SNAKE_CASE` constants, and descriptive stage identifiers. Preserve UTF-8 and exact evidence offsets. Follow nearby style; no formatter/linter is configured.

## Testing Guidelines

Use `test_*.py` files and `test_*` functions. Cover contracts, split isolation, evidence integrity, and algorithm boundaries. Separate synthetic fixtures from corpus metrics. Some tests require data/models; no coverage threshold.

## Commit & Pull Request Guidelines

Follow recent `feat:`/`research:` subjects, including stages. PRs should describe changes, affected stages, validation/results, prerequisites, limitations, and linked issues/reports.

## Research Integrity & Configuration

Preserve frozen contracts/completed runs; use new output directories. Keep Gold labels outside inference and train/validation separate. Record seeds, thresholds, revisions, hashes, provenance. Keep credentials in untracked `.env`; exclude weights.

## AI Agent Instructions

- **Strict scope:** perform only explicitly requested work; avoid unnecessary refactoring, optimization, experiments, or architecture changes.
- Stage 2G.3 Knowledge Graph is complete. Changing existing Neo4j namespaces, data, thresholds, embeddings, or factual relations requires explicit permission.
- Distinguish factual `RELEX_REL`, conceptual `HAS_SHARED_CONCEPT`, and semantic `SIMILAR_TO`. Never infer factual dependencies from similarity. Preserve provenance.
- Avoid unnecessary heavy models, inference, training, or research. Default: one heavy model worker; maximum: two across all agents.
- Before lengthy operations, display a clear UPPERCASE warning. Use real `tqdm` for lengthy batch processing.
- Check Git branch/status before editing. Never run `reset --hard` or delete existing artifacts without explicit permission.
- After completing requested work and necessary checks, commit and push task-related changes to the current branch without further confirmation, unless instructed otherwise. Exclude unrelated changes; never force-push.
- Before committing or pushing, verify the current branch and staged diff. Never stage unrelated, untracked, or secret files. In parallel workflows, push only the agent's designated feature branch.
- Reuse existing components and minimal dependencies; justify additional frameworks/abstractions.
- Spawn subagents only when explicitly requested; assign independent files or Git worktrees.
- Run only necessary tests for affected components; avoid the full heavy pipeline for small changes.
