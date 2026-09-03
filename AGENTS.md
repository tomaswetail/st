# Agent Guide — Stryktipset / Football Analytics (`st`)

This repository is a **Python research and data pipeline** for Stryktipset coupon analysis: historical football ingestion, feature engineering, and 1X2 probability modeling. It is **not** a web application and has **no authentication layer**.

Before implementing features, read the knowledge layer under `docs/`.

## Before implementing a task

1. Read [`docs/BUSINESS.md`](docs/BUSINESS.md) — product purpose, workflows, business rules.
2. Read [`docs/DOMAIN.md`](docs/DOMAIN.md) — terminology (ST Match vs Fixture, draw_number vs round id, etc.).
3. Read relevant [`docs/product/`](docs/product/) docs for the area you are changing.
4. Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) when touching ingestion, modeling, or cross-cutting behavior.
5. Read [`docs/DECISIONS.md`](docs/DECISIONS.md) for established patterns and uncertain areas.
6. Inspect existing implementation **and tests** before assuming behavior.

Also useful: [`README.md`](README.md), [`docs/football_data_ingestion.md`](docs/football_data_ingestion.md) (operational ingestion guide — verify against code).

## Product correctness

- Preserve documented **business invariants** (especially **no future-data leakage** in feature pipelines).
- **Do not invent** domain behavior. If unclear, mark `UNKNOWN / NEEDS PRODUCT CONFIRMATION`.
- Treat as **high-risk** when changing:
  - Entity resolution / team matching (`data_sources/entity_resolver.py`)
  - Feature cutoff dates in `calc/strength_calculator.py` and related calculators
  - Probability blend weights and ML baseline logic (`calc/probability_manager.py`)
  - Fixture upsert keys and EntityResolver / team&league `external_id`
  - DC league params JSON used in production backtests

## Implementation conventions

- **Layers:** `scripts/` / `services/` orchestrate → `calc/` business logic → `objects/repositories/` data access → `objects/models/` ORM.
- **Config:** use `DataSourceConfig` (`objects/schema/data_classes/data_sources.py`); avoid new magic numbers in scripts.
- **DTOs:** Pydantic schemas in `objects/schema/` for I/O; dataclass feature objects in `objects/schema/data_classes/`.
- **Providers:** xG via `ExtendedMatchDataService` + provider protocol; do not bypass entity resolution.
- **Imports:** residual ML lives in `calc/residual_ml/` package.
- **Minimal diffs:** match surrounding style; avoid unrelated refactors.
- **Do not commit secrets** — API keys belong in environment variables.

## Testing

**Run tests from repo root:**

```bash
PYTHONPATH=. python -m pytest tests/
PYTHONPATH=. python -m pytest tests/test_calc/ -q
PYTHONPATH=. python -m pytest tests/football_data/ -q
```

Use `python -m pytest` (not bare `pytest`) so `scripts.*` imports resolve.

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
- [ ] `PYTHONPATH=. python -m pytest tests/<relevant>/` passes
- [ ] No secrets committed

## Cursor rules

Project-specific agent rules: [`.cursor/rules/`](.cursor/rules/)

Subagent roles: [`.cursor/agents/`](.cursor/agents/) (project-leader, developer, verifier)
