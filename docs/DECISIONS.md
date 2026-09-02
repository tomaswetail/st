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

## DEC-006 — External entity mapping table for provider IDs

**Decision:** Cross-provider identity stored in `external_entity_mapping` with uniqueness constraints.

**Reason/rationale:** Documented in ingestion guide; model constraints.

**Evidence:** `objects/models/external_entity_mapping.py`, `docs/football_data_ingestion.md`

**Implications:** New providers require mapping rows + resolver logic.

**Confidence:** HIGH

---

## DEC-007 — Schema bootstrapped via SQLAlchemy create_all (no migrations)

**Decision:** Database schema created by `init_db()` without Alembic.

**Reason/rationale:** Rationale is not documented.

**Evidence:** `database/__init__.py` — no `migrations/` directory

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
