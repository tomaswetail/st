# Agent Guide — Stryktipset / Football Analytics (`st`)

This repository is a **Python research and data pipeline** for Stryktipset coupon analysis: historical football ingestion, feature engineering, and 1X2 probability modeling. It is **not** a web application and has **no authentication layer**.

Application Python packages live under **`src/`**. First-party imports are **`src.`-prefixed absolute imports** (`from src.calc...`, `from src.utils...`), with the **repository root as the sole path root**. Run everything from the repo root using `python -m` — **no `PYTHONPATH` needed** (see [DEC-013](docs/DECISIONS.md)):

```bash
python -m pytest tests/
python -m src.scripts.calculate_probabilities --draw-number <N>
```

Before implementing features, read the knowledge layer under `docs/`.

## Before implementing a task

1. Read [`docs/BUSINESS.md`](docs/BUSINESS.md) — product purpose, workflows, business rules.
2. Read [`docs/DOMAIN.md`](docs/DOMAIN.md) — terminology (ST Match vs Fixture, draw_number vs round id, etc.).
3. Read [`docs/wiki.md`](docs/wiki.md) — modeling & evaluation jargon (slice, HGB residual, logit, shrink, Brier).
4. Read relevant [`docs/product/`](docs/product/) docs for the area you are changing.
5. Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) when touching ingestion, modeling, or cross-cutting behavior.
6. Read [`docs/DECISIONS.md`](docs/DECISIONS.md) for established patterns and uncertain areas.
7. Skim [`docs/project_status.md`](docs/project_status.md) for current ship state and experiment context (gate status, what is off).
8. Inspect existing implementation **and tests** before assuming behavior.

Also useful: [`docs/football_data_ingestion.md`](docs/product/football_data_ingestion.md) (operational ingestion guide — verify against code), [`docs/production_profile.md`](docs/production_profile.md) (live scoring profile), [`docs/wiki.md`](docs/wiki.md) (modeling jargon). Experiment / phase write-ups live under [`docs/reports/`](docs/reports/) (`injury/`, `ml_draw/`, `ml_residual/`) — not product correctness docs.

## Product correctness

- Preserve documented **business invariants** (especially **no future-data leakage** in feature pipelines).
- **Do not invent** domain behavior. If unclear, mark `UNKNOWN / NEEDS PRODUCT CONFIRMATION`.
- Treat as **high-risk** when changing:
  - Entity resolution / team matching (`src/data_sources/entity_resolver.py`)
  - Feature cutoff dates in `src/calc/strength_calculator.py` and related calculators
  - Market baseline + residual ML wiring in `src/calc/probability_manager.py`
  - Fixture upsert keys and EntityResolver / team&league `external_id`

## Implementation conventions

- **Layers:** `src/scripts/` / `src/main.py` orchestrate → `src/calc/` business logic → `src/objects/repositories/` data access → `src/objects/models/` ORM.
- **Config:** use `DataSourceConfig` (`src/objects/schema/data_classes/data_sources.py`); avoid new magic numbers in scripts. Repo-root `config/` stays outside `src/`.
- **DTOs:** Pydantic schemas in `src/objects/schema/` for I/O; dataclass feature objects in `src/objects/schema/data_classes/`.
- **Providers:** xG via `ExtendedMatchDataService` + provider protocol; do not bypass entity resolution.
- **Imports:** `src.`-prefixed absolute imports everywhere; residual ML is `from src.calc.residual_ml import ...`.
- **Minimal diffs:** match surrounding style; avoid unrelated refactors.
- **Do not commit secrets** — API keys belong in environment variables.

## Testing

**Run tests from repo root, with no `PYTHONPATH` set:**

```bash
python -m pytest tests/
python -m pytest tests/test_calc/ -q
python -m pytest tests/football_data/ -q
```

Use `python -m pytest` (not bare `pytest`) so `src.scripts.*` imports resolve.

**Requires:** Python deps (SQLAlchemy, pydantic, numpy, scipy, pytest, httpx, requests; scikit-learn for ML tests).

**No CI config in repo** — run relevant test directories locally before finishing.

Add/update tests when changing business rules, entity resolution, probability logic, or fixing regressions.

## Database

PostgreSQL via SQLAlchemy. Connection: `DATABASE_URL` or `POSTGRES_*` env vars.

Schema bootstrapped with `init_db()` (`create_all`). **No Alembic migrations.**

## Completion checklist

An implementation is not complete until:

- [ ] Acceptance criteria met
- [ ] Documented business invariants still hold (especially leakage-safe features)
- [ ] Relevant tests added or updated
- [ ] `python -m pytest tests/<relevant>/` passes
- [ ] No secrets committed

## Cursor rules

Project-specific agent rules: [`.cursor/rules/`](.cursor/rules/)

Subagent roles: [`.cursor/agents/`](.cursor/agents/) (project-leader, developer, verifier)
