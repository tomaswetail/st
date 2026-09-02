# Purpose

Compute **1 / X / 2 probabilities** for each match on a Stryktipset coupon by combining market odds, a Dixon–Coles (or strength) engine baseline, and an optional residual ML model.

# Core Concepts

- **Market baseline** — implied probs from Svenska Spel odds
- **Engine baseline** — DC/strength model probs (`p_home_dc`, etc.)
- **Blend baseline** — weighted mix (default 70/30)
- **Final probabilities** — ML output when enabled, else blend/engine/market fallback chain

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
2. `ResidualMLFeatureAssembler.assemble(match)`
3. `market_baseline` ← ST odds in features
4. `engine_baseline` ← DC probs in features
5. `blend_baselines(market, engine)`
6. If `residual_ml_enabled` and model loaded: `predict_proba(features, baseline=blend)`
7. Return `STMatchProbabilityResult`

## Research / backtest

`scripts/build_residual_ml_dataset.py` → `train_residual_ml.py` → `backtest_residual_ml.py`

Uses finished matches with odds in draw window 4760–4960.

# State/Lifecycle

- Draw imported via `STDrawManager` before scoring
- Match must have resolved `home_team`, `away_team`, `start_time`
- For training: finished result + odds required

# Business Rules

- **BR-009, BR-010** in `docs/BUSINESS.md`
- ML failure falls back to blend; notes appended to `ml_notes`
- `draw_boost_score` = ML draw prob − market draw prob (when ML succeeds)

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
| No baseline computable | `ValueError` |
| ML predict error | Fallback to blend, `ml_notes` populated |

# Edge Cases

- `RESIDUAL_ML_ENABLED=false` → no ML path
- Missing DC features → engine baseline may be None → blend degrades to market-only
- `residual_ml_dc_engine`: `"classic"` vs `"strength"` affects engine features

# Important Tests

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

# Unknowns

- **UNKNOWN / NEEDS PRODUCT CONFIRMATION** — Production draw numbers and scheduling
- **UNKNOWN / NEEDS PRODUCT CONFIRMATION** — Whether ML should always be enabled in production
