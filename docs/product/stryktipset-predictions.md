# Purpose

Compute **1 / X / 2 probabilities** for each match on a Stryktipset coupon from the Svenska Spel market baseline plus an optional residual ML delta plus an optional shrink back toward the market. See [DEC-015](../DECISIONS.md).

# Core Concepts

- **Market baseline** — normalized implied probabilities from Svenska Spel 1X2 decimal odds
- **Residual ML delta** — optional HGB three-logit correction on top of the market baseline
- **Shrink toward market** — optional convex mix of ML output and market baseline (`α ∈ [0, 1]`)
- **Final probabilities** — ML output when enabled, else the market baseline

Formulas and worked examples: [`docs/probability_calculations.md`](probability_calculations.md).

# Entities

| Entity | Role |
|--------|------|
| `STRoundModel` | Coupon/draw container |
| `STMatchModel` | Single coupon line |
| `STMatchOddsModel` | Decimal odds 1/X/2 |
| `STMatchBetModel` | Public bet % (not used as market baseline) |
| `STMatchProbabilityResult` | Output DTO (not persisted) |
| `ResidualMLFeatures` | Flat feature vector per match |

# Primary Workflows

## Live / ad-hoc scoring

```python
ProbabilityManager(session).process(draw_number)
```

Steps per match (`calc/probability_manager.py`):
1. Validate teams and `start_time` present
2. `ResidualMLFeatureAssembler.assemble(match)` (includes cutoff-safe injuries when snapshots exist)
3. `market_baseline` ← ST odds stored on match (often start odds — see `docs/production_cutoff_alignment.md`)
4. If `residual_ml_enabled` and model loaded: `ResidualMLModel.predict_proba(features, baseline=market)`
5. If `RESIDUAL_ML_FINAL_SHRINK_TO_MARKET > 0`: `shrink_toward_market` on the ML output
6. Return `STMatchProbabilityResult`

## Research / backtest

`scripts/build_residual_ml_dataset.py` → `train_residual_ml.py` → `backtest_residual_ml.py`

Uses finished matches with odds in draw window 4760–4960.

# State/Lifecycle

- Draw imported via `STDrawManager` before scoring
- Match must have resolved `home_team`, `away_team`, `start_time`
- For training: finished result + odds required

# Business Rules

- **BR-009, BR-010** in `docs/BUSINESS.md`
- ML failure falls back to the market baseline; notes appended to `ml_notes`

# Permissions

None (batch CLI).

# Side Effects

- Reads DB extensively (fixtures, stats, ST tables)
- Does **not** write probabilities to DB (current implementation)
- ML model loaded from disk when enabled

# Integrations

- Input: Svenska Spel odds (already in DB)
- Input: historical fixtures + advanced stats for features
- Optional: trained model at `models/residual_ml/...`

# Failure Scenarios

| Failure | Behavior |
|---------|----------|
| Round not found | `ValueError` |
| Missing team/start_time | `ValueError` |
| No market probabilities | `ValueError` |
| ML predict error | Fallback to market baseline, `ml_notes` populated |

# Edge Cases

- `RESIDUAL_ML_ENABLED=false` → no ML path; market baseline is the final output
- Missing odds on a match → `ValueError` (market baseline is required)

# Important Tests

- `tests/test_calc/test_probability_calculations.py` (worked examples vs docs)
- `tests/test_calc/test_probability_manager.py` (import smoke)
- `tests/test_calc/test_residual_ml_feature_assembler.py`
- `tests/test_calc/test_backtest_residual_ml.py`
- `tests/test_calc/test_residual_ml_baseline.py`

# Relevant Code

- `calc/probability_manager.py`
- `calc/residual_ml/feature_assembler.py`
- `calc/residual_ml/baseline.py`
- `calc/residual_ml/model.py`
- `calc/market_probabilities.py`

# Known Limitations

- No persistence of results
- Minimal unit tests on `ProbabilityManager` itself (mostly integration via backtest)
- Feature assembly is expensive (many DB reads per match)
- Market odds are whatever is stored on `STMatchOdds` (often early/start odds), not necessarily closing
- See `docs/production_cutoff_alignment.md` for injury vs market timing
- OI1 — shipped HGB in `models/residual_ml/sweep_best/` was trained against the pre-pivot blend baseline; must be retrained against the market baseline before residuals are enabled (backup at `models/residual_ml/sweep_best_pre_market_pivot_bck/`). See [DEC-015](../DECISIONS.md).

# Unknowns

- **UNKNOWN / NEEDS PRODUCT CONFIRMATION** — Production draw numbers and scheduling
- **UNKNOWN / NEEDS PRODUCT CONFIRMATION** — Whether ML should always be enabled in production
