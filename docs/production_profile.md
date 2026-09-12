# Production profile (market baseline + optional residual ML + optional shrink)

Shipped configuration for live Stryktipset 1X2 scoring after the market baseline pivot (see [DEC-015](DECISIONS.md)).

## Pipeline order

```
ST odds → market_baseline
       →  (optional) HGB residual delta          [RESIDUAL_ML_ENABLED]
       →  (optional) shrink toward market        [RESIDUAL_ML_FINAL_SHRINK_TO_MARKET]
```

`ProbabilityManager` runs the above per coupon match. Missing market probabilities raise `ValueError` for that match; missing / broken ML falls back to the market baseline with `ml_notes`.

## Components

### Market baseline

- Source: Svenska Spel decimal 1X2 odds stored in `stryktipset_match_odds`
- Overround removed via inverse-odds normalization (`utils.common.odds_to_probabilities`)
- Renormalized to sum to 1 by `market_baseline` in `src/calc/residual_ml/baseline.py`

### Residual ML (HGB)

- **Status: off by default** (`RESIDUAL_ML_ENABLED=false`)
- Shipped artifact `models/residual_ml/sweep_best/` was trained against the pre-pivot **blend** baseline. Retrain against the market baseline before enabling in production.
- Pre-pivot snapshot preserved at `models/residual_ml/sweep_best_pre_market_pivot_bck/`.
- Training / prediction produce three independent logit deltas relative to the market baseline; renormalized to a simplex on apply.

### Shrink toward market

- Config: `RESIDUAL_ML_FINAL_SHRINK_TO_MARKET` (default **0.7** when ML is enabled)
- `α = 0` → keep ML; `α = 1` → collapse to market

## Environment variables (production-oriented)

| Variable | Typical / recommended |
|----------|------------------------|
| `RESIDUAL_ML_ENABLED` | `false` (until model is retrained on market baseline) |
| `RESIDUAL_ML_MODEL_PATH` | `models/residual_ml/sweep_best/model.pkl` |
| `RESIDUAL_ML_FINAL_SHRINK_TO_MARKET` | `0.7` |

## Running

```bash
python -m src.scripts.calculate_probabilities --draw-number <N>
```

## Monitoring

```bash
python -m src.scripts.monitor_round_ll --draw-number <N>
```

## Caveats

1. Shipped HGB is not usable as-is (baseline mismatch); requires retrain against the market baseline. See DEC-015 OI1.
2. ST odds are stored match odds, not closing lines.
3. Feature scaffolding (xG, availability, rest/congestion, league behavior, balance/environment) is preserved; nothing feeds into a scoring engine other than the residual ML input when it is enabled.
