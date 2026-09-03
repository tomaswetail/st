# Product Overview

This repository (`st`) is a **research and analysis pipeline for Stryktipset**, the Swedish pool betting product operated by Svenska Spel. It combines:

1. **Historical football data** (match results, teams, leagues, xG/shots)
2. **Stryktipset coupon data** (draws, matches, odds, public bet distribution)
3. **Statistical models** that estimate **1 / X / 2** (home win / draw / away win) probabilities per coupon match

Evidence: `README.md`, `calc/probability_manager.py`, `services/draw_manager.py`.

## Problem it solves

Analysts and model developers need a repeatable way to:

- Import and normalize football data from multiple external providers
- Link Stryktipset coupon matches to historical fixtures and advanced stats
- Compute baseline and ML-adjusted match probabilities
- Backtest models against historical coupon outcomes (log-loss, RPS, etc.)

## Who uses it

**UNKNOWN / NEEDS PRODUCT CONFIRMATION** — no user-facing application, authentication, or role model exists in this repository. Usage appears to be **internal CLI/script workflows** for data scientists or engineers.

Evidence: entry points are `main.py` and `scripts/`; no web server or API routes.

## Major user types (inferred)

| Persona | Likely use | Confidence |
|---------|------------|------------|
| Data engineer | Ingest fixtures, xG, team mappings | HIGH |
| Quant / ML researcher | Train/backtest DC and residual ML models | HIGH |
| Analyst | Run probability pipeline on a draw | MEDIUM |

---

# Core Product Areas

| Area | Purpose | Primary code |
|------|---------|--------------|
| Stryktipset ingestion | Fetch draws, matches, odds, bet % from Svenska Spel | `services/draw_manager.py`, `data_sources/svenskaspel_api_client.py` |
| Historical fixtures | Import API-Football results into `fixtures` | `data_sources/data_collector.py`, `data_sources/api_football_client.py` |
| Advanced stats | Attach xG/shots to existing fixtures | `data_sources/football_data/service.py` |
| Entity resolution | Map provider IDs/names → internal teams/leagues/fixtures | `data_sources/entity_resolver.py` |
| Match probability engine | Market + Dixon–Coles + optional ML blend | `calc/probability_manager.py`, `calc/residual_ml/` |
| Model research | DC optimization, ML dataset build, backtests | `scripts/`, `calc/dixon_coles/` |
| Coupon strategy helpers | Double-coverage heuristics (experimental) | `services/bet_distribution.py` |

See also `docs/product/` for deeper area docs.

---

# Primary User Workflows

## WF-001 — Import a Stryktipset draw

```
Operator runs STDrawManager.import_draw(draw_number)
  → SvenskaSpelClient.fetch_draw(draw_number)
  → EntityResolver resolves home/away teams (provider=svenska-spel)
  → Upsert stryktipset_rounds, stryktipset_matches, odds, bet distribution
```

Evidence: `services/draw_manager.py`, `main.py` (`calc()` imports draw 4750).

## WF-002 — Refresh historical fixtures

```
DataCollector.refresh_all_data(seasons)
  → APIFootballClient fetches leagues/teams/fixtures
  → FixtureRepository upsert (keyed on fixture_id)
```

Evidence: `data_sources/data_collector.py`, `tests/repositories/test_historical_match_same_date.py`.

## WF-003 — Enrich fixtures with xG/shots

```
ExtendedMatchDataService.fetch_and_store_* (league history or missing stats)
  → Resolve fixture via EntityResolver
  → Fetch provider match details (SofaScore or FotMob)
  → Persist match_advanced_stats + match_shots
```

Evidence: `data_sources/football_data/service.py`, `docs/football_data_ingestion.md`.

## WF-004 — Compute probabilities for a draw

```
ProbabilityManager.process(draw_number)
  → Load round + matches from DB
  → For each match: ResidualMLFeatureAssembler.assemble()
  → market_baseline (Svenska Spel odds) + engine_baseline (DC) → blend
  → Optional ResidualMLModel.predict_proba() → STMatchProbabilityResult
```

Evidence: `calc/probability_manager.py`.

**Note:** Results are returned as Pydantic objects (`STMatchProbabilityResult`); no DB table for persisted predictions was found.

## WF-005 — ML research pipeline

```
optimize_classic_dixon_coles.py  → per-league DC params JSON
build_residual_ml_dataset.py     → CSV from finished ST matches (draws 4760–4960)
train_residual_ml.py             → train/sweep model
backtest_residual_ml.py          → validation metrics
```

Evidence: `scripts/run_classic_dc_ml_pipeline.sh`, `config/stryktipset.py`.

---

# Roles and Permissions

**No authentication or authorization model exists in this repository.**

- No user accounts, roles, or permission checks
- Database access is direct via SQLAlchemy session (local/dev assumption)
- External API access uses environment API keys

**UNKNOWN / NEEDS PRODUCT CONFIRMATION:** Whether any downstream system enforces access control when consuming model outputs.

---

# Important Business Rules

## BR-001 — Stryktipset outcomes are 1, X, or 2

Match results are stored as a single character: home win (`1`), draw (`X`), away win (`2`).

Evidence:
- `objects/repositories/utils.py` (`stryktipset_result`)
- `objects/repositories/st_match_repository.py` (`stryktipset_result.in_(("1", "X", "2"))`)

Confidence: **HIGH**

## BR-002 — ML/backtest training uses finished matches with odds and valid results

`find_finished_with_odds` requires:
- `stryktipset_result` in `("1", "X", "2")`
- related odds row exists
- `start_time` set
- optional draw-number window filter

Evidence: `objects/repositories/st_match_repository.py`

Confidence: **HIGH**

## BR-003 — Feature history must not include the target match (no leakage)

Team strength and related features use only matches **strictly before** the feature cutoff date.

Evidence:
- Docstring in `calc/strength_calculator.py`
- Tests: `tests/test_calc/test_strength_calculator.py`, `tests/test_calc/test_strength_calculator_functions.py`

Confidence: **HIGH**

## BR-004 — Market probabilities must be on a consistent 0–1 scale

`ensure_unit_probabilities` rejects mixed unit-scale and percentage-scale inputs in the same row.

Evidence: `utils/common.py`, `calc/residual_ml/baseline.py`

Confidence: **HIGH**

## BR-005 — Decimal odds must be positive to convert to probabilities

`odds_to_probabilities` raises if any of home/draw/away odds ≤ 0.

Evidence: `utils/common.py`

Confidence: **HIGH**

## BR-006 — Fixture identity is keyed on API-Football `fixture_id`

Upsert uses unique constraint `uq_fixtures_fixture_id`; re-import is idempotent on that key.

Evidence: `objects/models/fixture.py`, `tests/repositories/test_historical_match_same_date.py`

Confidence: **HIGH**

## BR-007 — Provider IDs resolve via EntityResolver and team/league external_id

Provider team/league/match IDs must be resolved through `EntityResolver`. Canonical API-Football ids live on `teams.external_id` and `leagues.external_id`. There is no live cross-provider mapping table in application code (an orphaned `external_entity_mapping` table may still exist in older DBs).

Evidence: `data_sources/entity_resolver.py`, `objects/models/team.py`, `objects/models/league.py`

Confidence: **HIGH**

## BR-008 — Residual ML draw window for dataset building

Research scripts default to draws **4760–4960 inclusive** (`STRYKETIPSET_DRAW_MIN` / `STRYKETIPSET_DRAW_MAX`).

Evidence: `config/stryktipset.py`, `calc/residual_ml/dataset.py`

Confidence: **HIGH** (for research pipeline; not necessarily a product rule for live use)

## BR-009 — Default probability blend weights market over engine

When both baselines exist: default **70% market / 30% Dixon–Coles** (`residual_ml_market_weight=0.7`, `residual_ml_dc_weight=0.3`).

Evidence: `objects/schema/data_classes/data_sources.py`, `calc/probability_manager.py`

Confidence: **HIGH** (configurable via env)

## BR-010 — Residual ML is off by default

`RESIDUAL_ML_ENABLED` defaults to false; pipeline uses market+DC blend unless explicitly enabled and model file exists.

Evidence: `objects/schema/data_classes/data_sources.py`, `calc/probability_manager.py`

Confidence: **HIGH**

---

# Business Invariants

| ID | Invariant | Evidence |
|----|-----------|----------|
| INV-001 | Team strength features computed only from data available before match kickoff | `calc/strength_calculator.py`, strength tests |
| INV-002 | Opponent strength for adjustment uses only prior matches | `_get_opponent_strength_before` in `calc/strength_calculator.py` |
| INV-003 | Dixon–Coles 1X2 probabilities renormalize to sum ≈ 1 | `calc/strength_calculator.py` (`dixon_coles_matrix`), tests |
| INV-004 | ST match `external_id` is unique (Svenska Spel matchId) | `objects/models/st_match.py` |
| INV-005 | ST round unique on `(product_id, draw_number)` | `objects/models/st_round.py` |
| INV-006 | Advanced stats attach to existing fixtures; unresolved fixtures are skipped and logged | `data_sources/football_data/service.py`, ingestion tests |

---

# Known Product Limitations

1. **No persisted prediction store** — `STMatchProbabilityResult` is computed in memory; no ORM model for saved predictions.
2. **No production API/UI** — CLI and scripts only.
3. **Schema via `create_all`** — no Alembic migrations; schema changes require manual coordination.
4. **No dependency lockfile** — packages inferred from imports, not pinned in repo.
5. **Documented CLI missing** — `docs/football_data_ingestion.md` references `services/football_data_cli.py` which does not exist.
6. **Dual xG providers** — SofaScore (documented preferred) and FotMob (still used in `main.py`); consolidation incomplete.
7. **Config placeholders** — Understat, FootyStats, The Odds API referenced in config but not implemented as clients.

---

# Product Unknowns

- **UNKNOWN / NEEDS PRODUCT CONFIRMATION** — Is there a downstream consumer (UI, betting tool, automated submission)?
- **UNKNOWN / NEEDS PRODUCT CONFIRMATION** — Intended production xG provider: SofaScore vs FotMob?
- **UNKNOWN / NEEDS PRODUCT CONFIRMATION** — Should computed probabilities be persisted? If so, where?
- **UNKNOWN / NEEDS PRODUCT CONFIRMATION** — Role and status of `BetDistribution` double-coverage helpers in real workflows.
- **UNKNOWN / NEEDS PRODUCT CONFIRMATION** — Target draw numbers for live vs research (`draw_manager` hardcodes 4959–4967 in some methods).
- **UNKNOWN / NEEDS PRODUCT CONFIRMATION** — Whether API keys in `DataSourceConfig` defaults are intentional for dev only.
