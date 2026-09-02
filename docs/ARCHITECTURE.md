# System Overview

`st` is a **Python backend/research codebase** with no web frontend. It:

1. Persists football and Stryktipset data in **PostgreSQL**
2. Ingests from **Svenska Spel**, **API-Football**, and **SofaScore/FotMob**
3. Computes match features and **1X2 probabilities** via statistical and ML models
4. Supports **offline backtesting and hyperparameter optimization**

There is **no HTTP API layer**, **no authentication**, and **no job queue** in-repo.

---

# Technology Stack

| Layer | Technology | Evidence |
|-------|------------|----------|
| Language | Python 3.10+ | Type syntax, `.idea` config |
| Database | PostgreSQL | `database/__init__.py` |
| ORM | SQLAlchemy 2.x | `database/models.py`, repositories |
| Validation/DTOs | Pydantic | `objects/schema/` |
| Numerics | NumPy, SciPy | `calc/dixon_coles/` |
| ML | scikit-learn (lazy import) | `calc/residual_ml/trainer.py` |
| HTTP clients | httpx, requests | `data_sources/football_data/http_client.py`, `api_football_client.py` |
| Testing | pytest | `tests/` |
| Migrations | None (SQLAlchemy `create_all`) | `database/init_db()` |

**Not present:** Docker, CI workflows, requirements.txt/pyproject.toml (dependencies implicit).

---

# Major Components

## `database/`

- SQLAlchemy engine, `SessionLocal`, `init_db()` (`create_all` + optional `pg_trgm`/`unaccent`)
- Env: `DATABASE_URL` or `POSTGRES_*`

## `objects/`

Domain persistence layer.

| Subfolder | Responsibility |
|-----------|----------------|
| `models/` | SQLAlchemy ORM tables |
| `repositories/` | Query/upsert logic (`BaseRepository`) |
| `schema/db/` | Pydantic create/read DTOs |
| `schema/data_classes/` | Feature vectors, config, provider DTOs |

## `data_sources/`

External API clients and ingestion orchestration.

| Module | Role |
|--------|------|
| `svenskaspel_api_client.py` | Stryktipset draws |
| `api_football_client.py` | Fixtures, teams, leagues |
| `data_collector.py` | Bulk fixture import |
| `entity_resolver.py` | Cross-provider entity linking |
| `football_data/service.py` | xG/shots enrichment |
| `football_data/providers/` | SofaScore, FotMob adapters |
| `classic_dc_config.py` | DC grid/params JSON loading |

## `calc/`

Business logic for features and probabilities (no DB writes in most modules).

| Module | Role |
|--------|------|
| `probability_manager.py` | End-to-end coupon probability pipeline |
| `strength_calculator.py` | xG-based strength + DC matrix |
| `dixon_coles/` | Classic DC fit/optimize/walk-forward |
| `residual_ml/` | ML dataset, features, train, predict |
| `home_advantage_calculator.py` | HA features |
| `league_behavior_calculator.py` | League draw/low-score rates |
| `rest_congestion_calculator.py` | Fixture congestion |
| `balance_and_environment.py` | Match context features |
| `market_probabilities.py` | Odds → probs wrapper |
| `probability_metrics.py` | Log-loss, RPS |

## `services/`

Thin orchestration over repositories + clients.

- `draw_manager.py` — Stryktipset import
- `bet_distribution.py` — coupon double-coverage heuristics

## `scripts/`

CLI entry points for batch jobs (DC optimize, ML train/backtest, missing stats).

## `config/`

JSON league maps, team aliases, DC params/grids; `stryktipset.py` draw window constants.

## `utils/`

Shared helpers: seasons, team matching, repo paths, time splits, common constants.

---

# Request/Data Flow

This system has no HTTP request flow. Typical **batch flow**:

```
CLI / main.py
  → Service or Manager (e.g. STDrawManager, ProbabilityManager, ExtendedMatchDataService)
    → Repository (SQLAlchemy queries/upserts)
      → PostgreSQL
    → External API client (optional)
  → Pydantic result / stdout / CSV / model files
```

**Probability flow (per coupon match):**

```
ProbabilityManager.process_match
  → ResidualMLFeatureAssembler.assemble(STMatchModel)
      → StrengthCalculator, HomeAdvantageCalculator, DixonColesService, etc.
  → market_baseline + engine_baseline → blend_baselines
  → ResidualMLModel.predict_proba (if enabled)
  → STMatchProbabilityResult
```

Evidence: `calc/probability_manager.py`, `calc/residual_ml/feature_assembler.py`

---

# Domain Boundaries

| Concern | Location | Do not put here |
|---------|----------|-----------------|
| SQL / persistence | `objects/repositories/` | Probability math |
| Provider HTTP | `data_sources/` | Feature formulas |
| Feature engineering | `calc/` | Raw SQL in calculators |
| Config defaults | `objects/schema/data_classes/data_sources.py` | Hardcoded magic in scripts |
| Orchestration | `services/`, `scripts/`, `main.py` | Complex feature math |

---

# Database Architecture

## Main entities

See `docs/DOMAIN.md` entity diagram.

## Tenancy

**Single-tenant.** No `user_id`, `org_id`, or row-level security in schema.

## Constraints

- `fixtures.fixture_id` unique
- `stryktipset_matches.external_id` unique
- `stryktipset_rounds (product_id, draw_number)` unique
- `external_entity_mapping` dual uniqueness on external and internal ids per provider+type

## Schema management

`init_db()` → `Base.metadata.create_all()`. No Alembic.

## Soft deletion

Not used. Rows are upserted or truncated during reimports (see ingestion doc SQL examples).

## Transactions

Repositories use SQLAlchemy session; callers commit/close sessions. `session_scope()` context manager available.

---

# Authentication

**None.** Database and API keys come from environment variables on the machine running scripts.

---

# Authorization

**None in application code.**

---

# External Integrations

## Svenska Spel

- **Purpose:** Stryktipset draw JSON (matches, odds, bet %)
- **Code:** `data_sources/svenskaspel_api_client.py`
- **Config:** `SvenskaSpelConfig`, `SVENSKASPEL_ACCESS_KEY`
- **Failure handling:** Disk cache with TTL by draw state; `DrawNotFoundError`
- **Evidence:** `tests/data_sources/test_svenskaspel_api_client.py`

## API-Football

- **Purpose:** Leagues, teams, fixtures (results source of truth per docs)
- **Code:** `data_sources/api_football_client.py`, `data_collector.py`
- **Cache:** Disk cache under `data/cache/api-football/`
- **Evidence:** `docs/football_data_ingestion.md`

## SofaScore (preferred xG per docs)

- **Purpose:** Advanced stats + shots
- **Code:** `data_sources/football_data/providers/sofascore.py`
- **Default when** `FOOTBALL_DATA_PROVIDER=sofascore`

## FotMob (alternate xG)

- **Purpose:** Same as SofaScore; still referenced in `main.py`
- **Code:** `data_sources/football_data/providers/fotmob.py`

## Throttling / retries

`ThrottledHttpClient` in `data_sources/football_data/http_client.py` — delay, retry 429/5xx, disk cache.

---

# Background Processing

**No queue/worker system.** Long jobs run as CLI processes (DC optimization, ML training). Shell pipeline: `scripts/run_classic_dc_ml_pipeline.sh`.

---

# Error Handling

| Pattern | Where |
|---------|-------|
| `ValueError` for missing entities / invalid inputs | `ProbabilityManager`, `EntityResolver`, odds conversion |
| Unresolved matches logged to CSV | `DataSourceConfig.unresolved_matches_csv_path` |
| Provider HTTP errors | `FootballDataHttpError`, `NotFoundError` |
| ML prediction failures | Caught in `ProbabilityManager`; falls back to blend with `ml_notes` |

---

# Testing Architecture

## Layout

| Directory | Scope |
|-----------|-------|
| `tests/test_calc/` | Calculators, DC, ML, probability (~20 files) |
| `tests/football_data/` | Ingestion, providers (JSON fixtures) |
| `tests/data_sources/` | API clients, collector |
| `tests/repositories/` | DB query logic (mocked sessions) |
| `tests/test_utils/` | Utilities |

## Strategy

- **Unit tests** with `unittest.mock` / `MagicMock` for DB and HTTP
- **JSON fixtures** for provider responses (`tests/football_data/fixtures/`)
- **No E2E** against live DB/API in default suite (external tests not present)

## Running tests

```bash
PYTHONPATH=. python -m pytest tests/
PYTHONPATH=. python -m pytest tests/test_calc/ -q
PYTHONPATH=. python -m pytest tests/football_data/test_ingestion.py -q
```

Use `python -m pytest` (not bare `pytest`) for reliable imports of `scripts.*`.

## Factories

No formal factory library; tests use `SimpleNamespace`, `MagicMock`, local helpers (`make_match`, `make_calculator`).

---

# Architectural Constraints

## ARCH-001 — Business logic lives in `calc/`, not repositories

Repositories perform queries/upserts; probability and feature math belongs in `calc/`.

Evidence: separation across `objects/repositories/` vs `calc/`

Confidence: **HIGH**

## ARCH-002 — Configuration centralized in `DataSourceConfig`

Feature tuning, paths, provider URLs, ML weights flow through Pydantic config.

Evidence: `objects/schema/data_classes/data_sources.py`

Confidence: **HIGH**

## ARCH-003 — Provider abstraction for xG ingestion

`FootballDataProvider` protocol; `ExtendedMatchDataService` selects SofaScore or FotMob.

Evidence: `data_sources/football_data/protocol.py`, `service.py`

Confidence: **HIGH**

## ARCH-004 — Entity resolution is mandatory for cross-provider linking

Do not assume provider team IDs match internal `teams.id`.

Evidence: `data_sources/entity_resolver.py`, ingestion tests

Confidence: **HIGH**

## ARCH-005 — No future-data leakage in feature pipelines

Any new feature calculator must filter history with `before` / `before_date` cutoff.

Evidence: `calc/strength_calculator.py` docstring + tests

Confidence: **HIGH**

## ARCH-006 — Residual ML package is the canonical ML location

Flat `calc/residual_ml_*.py` files were consolidated into `calc/residual_ml/`.

Evidence: package structure, imports in `calc/probability_manager.py`

Confidence: **HIGH**

---

# Dangerous Areas

| Area | Risk |
|------|------|
| `EntityResolver.resolve_team` / match linking | Wrong team → wrong features for entire coupon |
| `calc/strength_calculator.py` cutoff logic | Leakage breaks backtest validity |
| `objects/repositories/fixture_repository.py` upsert | Data loss on bad reimport |
| `ProbabilityManager` blend/ML fallback | Silent probability changes |
| `config/classic_dc_league_params.json` | Wrong DC params → systematic bias |
| `utils/team_mappings.py` | Large static maps; stale mappings cause unresolved teams |
| `init_db()` / TRUNCATE workflows | Destructive full reimports |
