# Established Decisions

Capture of patterns and choices evidenced in code, tests, and existing docs. Rationale is inferred unless explicitly documented.

---

## DEC-001 — API-Football is the canonical source for match results

**Decision:** Fixtures (results, teams, leagues) are imported from API-Football only; fixture odds from API-Football are not persisted.

**Reason/rationale:** Documented ingestion architecture.

**Evidence:** `docs/football_data_ingestion.md`, `data_sources/data_collector.py`

**Implications:** ST odds remain on Svenska Spel draw rows only; modeling uses ST odds for market baseline, not API-Football odds.

**Confidence:** HIGH

---

## DEC-002 — Stryktipset odds are the market baseline source

**Decision:** Market probabilities for coupon matches come from Svenska Spel decimal odds stored in `stryktipset_match_odds`.

**Reason/rationale:** `ProbabilityManager` / `MarketProbabilities` use ST odds; `odds_provider` default is `svenskaspel`.

**Evidence:** `calc/market_probabilities.py`, `objects/schema/data_classes/data_sources.py`

**Implications:** Missing ST odds → market baseline unavailable for that match.

**Confidence:** HIGH

---

## DEC-003 — [SUPERSEDED by DEC-015] Dual baseline: market + Dixon–Coles before ML

The blended market + DC baseline was replaced with a market-only baseline. See DEC-015.

---

## DEC-004 — [SUPERSEDED by DEC-015] Residual ML predicts in logit space relative to blend baseline

Residual ML now predicts logit deltas relative to the market baseline (no DC/blend). See DEC-015.

---

## DEC-005 — Repository pattern over raw SQL in features

**Decision:** Data access goes through `objects/repositories/*`; calculators receive sessions and use repos.

**Reason/rationale:** Consistent structure across ST, fixture, stats repos.

**Evidence:** All repositories extend `BaseRepository`

**Implications:** New queries should get repository methods, not inline SQL in `calc/`.

**Confidence:** HIGH

---

## DEC-006 — Provider identity via EntityResolver + external_id (mapping table retired)

**Decision (superseded):** Cross-provider identity was previously designed around an `external_entity_mapping` table with uniqueness constraints.

**Current decision:** Provider identity resolution uses `EntityResolver` plus `external_id` columns on `teams` and `leagues`. The mapping model/repository have been removed from the codebase; production resolution never read mapping rows.

**Reason/rationale:** Code (`EntityResolver`) never consulted the mapping table; keeping it as a documented live mechanism was misleading.

**Evidence:** `data_sources/entity_resolver.py`, `objects/models/team.py`, `objects/models/league.py`

**Implications:** New providers need resolver paths and/or static maps (`utils/team_mappings.py`, aliases), not mapping-table rows. Existing databases may still have an orphaned `external_entity_mapping` physical table (no Alembic; do not assume `create_all` drops it).

**Confidence:** HIGH

---

## DEC-007 — Schema bootstrapped via SQLAlchemy create_all (no migrations)

**Decision:** Database schema created by `init_db()` without Alembic.

**Reason/rationale:** Rationale is not documented.

**Evidence:** `database.py` — no `migrations/` directory

**Implications:** Schema changes need coordinated manual updates; test DBs rely on model definitions.

**Confidence:** HIGH

---

## DEC-008 — [SUPERSEDED by DEC-015] Classic DC params stored per league in JSON

Dixon–Coles is no longer part of the pipeline. See DEC-015.

---

## DEC-009 — Feature shrinkage and recency weighting for team strength

**Decision:** Team strength uses exponential recency decay (default 0.9) and Bayesian shrinkage toward prior (default 8 matches).

**Reason/rationale:** Implemented in `calc/strength_helpers.py`; tested in strength calculator tests.

**Evidence:** `calc/strength_calculator.py`, `DataSourceConfig.team_strength_*`

**Implications:** Tuning these changes all strength-derived features.

**Confidence:** HIGH

---

## DEC-010 — Opponent adjustment is optional (config flag)

**Decision:** Simple opponent-adjusted strength only when `football_data_opponent_adjustment == "simple"` (default `"none"`).

**Reason/rationale:** Rationale is not documented; likely performance/complexity tradeoff.

**Evidence:** `calc/strength_calculator.py` `_accumulate_match_metrics`

**Implications:** Tests asserting opponent-adjusted fields must enable this config.

**Confidence:** HIGH

---

## DEC-011 — Residual ML research window: draws 4760–4960

**Decision:** Dataset builder and scripts use `STRYKETIPSET_DRAW_MIN/MAX` from `config/stryktipset.py`.

**Reason/rationale:** Rationale is not documented; likely historical data availability.

**Evidence:** `config/stryktipset.py`, `calc/residual_ml/dataset.py`

**Implications:** Extending backtest range requires updating constants and verifying data coverage.

**Confidence:** HIGH

---

## DEC-012 — calc/residual_ml/ package consolidation

**Decision:** Residual ML code lives in package `calc/residual_ml/` with submodules (baseline, dataset, trainer, etc.).

**Reason/rationale:** Rationale is not documented; improves maintainability.

**Evidence:** `calc/residual_ml/__init__.py`, imports across codebase

**Implications:** Do not reintroduce flat `calc/residual_ml_*.py` modules.

**Confidence:** HIGH

---

## DEC-013 — `src.`-prefixed absolute imports, repository root as sole path root

**Decision:** All first-party imports in `src/` and `tests/` are `src.`-prefixed absolute imports (`from src.calc.strength_calculator import ...`). The repository root is the single `sys.path` root. Run everything from the repo root with `python -m` and no `PYTHONPATH`:

```bash
python -m pytest tests/
python -m src.scripts.calculate_probabilities --draw-number 4964
```

`config/` sits at the repo root, so `from config....` is already correct and stays bare.

**Reason/rationale:** ~135 files were already in the `src.`-prefixed style after an IDE-wide refactor; only 27 modules and one test still used bare first-party imports. Reverting the majority to bare imports would have been pure churn for no benefit, so the minority was converted instead. A single path root also removes the previous mixed two-root setup, under which a module was importable under two names and produced duplicate module objects.

**Evidence:** `rg "^\s*(from|import)\s+(database|calc|utils|objects|data_sources|scripts)\b" src tests` and `rg "[\"'](database|calc|utils|objects|data_sources|scripts)\." src tests` both return zero matches. No `pytest.ini`, `pyproject.toml`, `src/__init__.py`, or `config/__init__.py` is needed — implicit namespace packages (PEP 420) cover it on Python 3.10+.

**Implications:** `unittest.mock.patch()` target strings must also be `src.`-prefixed, otherwise they patch a different module object than the one under test and fail silently. Do not document or reintroduce a `PYTHONPATH` pointing at `src`; `python -m src.scripts.<name>` is the one script invocation form. Dated write-ups under `docs/reports/` still show the old form and are deliberately left as-is.

**Confidence:** HIGH

---

## DEC-014 — [SUPERSEDED by DEC-015] MLE ρ is the canonical Dixon–Coles

Dixon–Coles is no longer part of the pipeline. See DEC-015.

---

## DEC-015 — Pipeline pivots to market baseline + optional residual ML + optional market shrink

**Decision:** The live and research 1X2 pipeline is:

```
market_baseline  →  (optional) residual ML logit-delta correction  →  (optional) shrink toward market
```

All Dixon–Coles, blend, draw-adjustment, Sarmanov / negative-binomial, and injury λ-shock / logit-shift code, config, tests, docs, and shipped model artifacts are removed. Feature scaffolding (xG, availability, rest/congestion, league behavior, balance/environment) and historical ingestion are kept intact.

**Reason/rationale:** The DC + blend + draw-adj stack repeatedly failed to beat market on held-out slices. The market baseline is a strong, cheap prior; residual ML has room to add value only against a market-anchored target, not against a self-blended one. Collapsing the stack removes bug surface, config surface, and misleading defaults, and makes the shipped HGB retrain trivial (single baseline, one target definition).

**Evidence:** `src/calc/probability_manager.py`, `src/calc/residual_ml/*` (market baseline only), `src/scripts/calculate_probabilities.py`, DoD grep `rg 'dixon_coles|draw_adjustment|blend_weights|sarmanov_nb|lambda_shock|classic_dc_config' src/` returns nothing.

**Implications:**
- Supersedes DEC-003 (dual baseline), DEC-004 (residual vs blend baseline), DEC-008 (classic DC params JSON), DEC-014 (MLE ρ as canonical DC).
- `STMatchProbabilityResult` drops `engine_probabilities`, `draw_boost_score`, `draw_value_gap`; declares `event_number`.
- `ResidualMLFeatures` drops `*_dc*`, `market_vs_dc_*`, and `expected_*_goals` fields.
- `DataSourceConfig` drops `residual_ml_market_weight`, `residual_ml_dc_weight`, `residual_ml_blend_weights_path`, `residual_ml_home_advantage_mode`, `residual_ml_dc_engine`, all `classic_dc_*` fields, and both `dixon_coles_*` fields.
- **Operational impact / OI1:** the currently shipped HGB (`models/residual_ml/sweep_best/`) was trained against the blend baseline. It must be retrained against the market baseline before production use of the ML branch. A pre-pivot snapshot is preserved at `models/residual_ml/sweep_best_pre_market_pivot_bck/`.
- Regression guard: `tests/test_calc/test_no_dc_imports.py` forbids DC-adjacent tokens under `src/`.

**Confidence:** HIGH

---

## DEC-016 — Historical fixture odds from football-data.co.uk

**Decision:** Persist historical bookmaker 1X2 odds from football-data.co.uk onto existing `fixtures` rows in `fixture_odds`, for ML training and evaluation. Live coupon scoring is unchanged.

**Reason/rationale:** Residual ML and backtests need a fixture-level market snapshot. Svenska Spel `stryktipset_match_odds` cover coupon matches only. football-data.co.uk publishes opening and closing 1X2 columns that can be attached to API-Football fixtures via entity resolution.

**Evidence:** `src/objects/models/fixture_odds.py`, `src/data_sources/football_data_odds/`, `src/scripts/ingest_football_data_odds.py`

**Implications:**
- DEC-001 still holds for **API-Football** odds: we do not fetch or persist API-Football `/odds`.
- DEC-002 still holds for **live coupon scoring**: `ProbabilityManager` continues to use Svenska Spel ST odds as the market baseline. `fixture_odds` is a second source for the future ML dataset, not a replacement of that path.
- `snapshot_at` is stored as `kickoff_at` for both opening and closing rows because the archive has no opening timestamp; do not invent one. `price_type` (`opening` | `closing`) distinguishes them.
- Canonical price for later modeling is **closing** primary; opening is stored alongside and can be selected by `FIXTURE_ODDS_PRICE_TYPE`.
- Archive consensus for ML fixtures selects a stored `fixture_odds` bookmaker row (default `Avg` closing; `PS` via `FIXTURE_ODDS_BOOKMAKER`). If the preferred bookmaker is missing and is not `Avg`, fall back to `Avg` at the same `price_type`; never mix opening and closing. Live coupon scoring still uses Svenska Spel ST odds (DEC-002).
- Odds FK is internal `fixtures.id`. Unresolved archive matches are logged to `data/unresolved_football_data_odds.csv`, not invented.

**Confidence:** HIGH

---

# Decisions Requiring Confirmation

## DEC-U01 — SofaScore vs FotMob as production xG provider

Docs say SofaScore only; `main.py` still loops FotMob mappings. **UNKNOWN / NEEDS PRODUCT CONFIRMATION**

Evidence: `docs/football_data_ingestion.md` vs `main.py` `main_extra_data()`

---

## DEC-U02 — Whether to persist probability outputs

`STMatchProbabilityResult` has no ORM table. **UNKNOWN / NEEDS PRODUCT CONFIRMATION**

---

## DEC-U03 — Hardcoded API key default in DataSourceConfig

`api_football_key` has a default string in source. Intentional dev convenience vs security issue — **UNKNOWN / NEEDS PRODUCT CONFIRMATION**

Evidence: `objects/schema/data_classes/data_sources.py` line 35

---

## DEC-U04 — BetDistribution integration

Double-coverage helpers exist but are not wired into draw import or probability pipeline. **UNKNOWN / NEEDS PRODUCT CONFIRMATION**

Evidence: `services/bet_distribution.py` — no callers found in main flows

---

## DEC-U05 — Understat as xg_provider default

Config defaults `xg_provider: "understat"` but no Understat client exists. Likely legacy placeholder.

Evidence: `DataSourceConfig.xg_provider`, no understat client in repo

Confidence: MEDIUM
