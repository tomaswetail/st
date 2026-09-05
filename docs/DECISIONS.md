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

## DEC-003 — Dual baseline: market + Dixon–Coles before ML

**Decision:** Default pipeline blends market (70%) and engine/DC (30%) before optional residual ML adjustment.

**Reason/rationale:** Config defaults and `ProbabilityManager.process_match` flow.

**Evidence:** `calc/probability_manager.py`, `calc/residual_ml/baseline.py`, `DataSourceConfig.residual_ml_market_weight`

**Implications:** Changing weights affects both inference and ML training baseline.

**Confidence:** HIGH

---

## DEC-004 — Residual ML predicts in logit space relative to blend baseline

**Decision:** ML model adjusts baseline probabilities, not raw odds or scores.

**Reason/rationale:** `calc/residual_ml/baseline.py` (logit/inv_logit), trainer predicts residuals.

**Evidence:** `calc/residual_ml/model.py`, `calc/residual_ml/trainer.py`

**Implications:** Baseline must be computable at predict time for ML path.

**Confidence:** HIGH

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

## DEC-008 — Classic DC params stored per league in JSON

**Decision:** Optimized Dixon–Coles hyperparameters persisted in `config/classic_dc_league_params.json`, loaded by `DixonColesService`.

**Reason/rationale:** Enables per-league tuning from optimization scripts.

**Evidence:** `calc/dixon_coles/service.py`, `tests/test_calc/test_dixon_coles_service.py`

**Implications:** Production/research DC behavior depends on this file being current.

**Confidence:** HIGH

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
python -m src.scripts.optimize_classic_dixon_coles --help
```

`config/` sits at the repo root, so `from config....` is already correct and stays bare.

**Reason/rationale:** ~135 files were already in the `src.`-prefixed style after an IDE-wide refactor; only 27 modules and one test still used bare first-party imports. Reverting the majority to bare imports would have been pure churn for no benefit, so the minority was converted instead. A single path root also removes the previous mixed two-root setup, under which a module was importable under two names and produced duplicate module objects.

**Evidence:** `rg "^\s*(from|import)\s+(database|calc|utils|objects|data_sources|scripts)\b" src tests` and `rg "[\"'](database|calc|utils|objects|data_sources|scripts)\." src tests` both return zero matches. No `pytest.ini`, `pyproject.toml`, `src/__init__.py`, or `config/__init__.py` is needed — implicit namespace packages (PEP 420) cover it on Python 3.10+.

**Implications:** `unittest.mock.patch()` target strings must also be `src.`-prefixed, otherwise they patch a different module object than the one under test and fail silently. Do not document or reintroduce a `PYTHONPATH` pointing at `src`; `python -m src.scripts.<name>` is the one script invocation form. Dated write-ups under `docs/reports/` still show the old form and are deliberately left as-is.

**Confidence:** HIGH

---

## DEC-014 — MLE ρ is the canonical Dixon–Coles

**Decision:** Live scoring and official eval use MLE-fitted ρ. `config/classic_dc_league_params.json` is the live MLE params file. `config/classic_dc_league_params_grid_bck.json` is grid-ρ backup only and must not be used as production. Optimizer and live fit default to `fit_rho=True` (`CLASSIC_DC_FIT_RHO` / `DataSourceConfig.classic_dc_fit_rho`, and production `config/classic_dc_optimization_grid.json` `"fit_rho": true`).

**Reason/rationale:** Grid ρ was selection noise on a small 1X2 log-loss slice. Product freeze: one canonical DC for live scoring and official eval.

**Evidence:** `DataSourceConfig.classic_dc_fit_rho` default `"1"`, `config/classic_dc_league_params.json`, `src/scripts/optimize_classic_dixon_coles.py` (`resolve_optimize_fit_rho`).

**Implications:** A bare `python -m src.scripts.optimize_classic_dixon_coles` must not overwrite live params with grid-searched ρ. Use `--no-fit-rho` only when deliberately grid-searching.

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
