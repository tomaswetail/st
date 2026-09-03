# Draw driver analysis

Phase 3.1 discovery (interpretable models only). HGB residual engine is unchanged.

## Protocol

- Train / validation time split: last **20%** by `match_date`
- Holdout draws ≥ 4951 excluded from discovery
- Models always include `p_draw_blend` (incremental vs odds+DC)

## Sample sizes

| Split | Rows |
|-------|------|
| Train | 1756 |
| Validation | 440 |

## Draw metrics (binary X vs not-X)

| Metric | Blend-only | L1 logistic | Δ |
|--------|------------|-------------|---|
| Val draw Brier | 0.1749 | 0.1767 | +0.0019 |
| Val draw log loss | 0.5343 | 0.5394 | +0.0052 |
| Train draw Brier | 0.1894 | 0.1871 | -0.0023 |

## Discovery gate

| Gate | Result |
|------|--------|
| ≥3 stable drivers | PASS (6) |
| OOS draw Brier or LL improves vs blend | FAIL (Brier Δ=+0.0019, LL Δ=+0.0052) |

Home/away LL regression gate applies only after Phase 3.2 ships an explicit draw adjustment.

## Stable L1 drivers (same sign train+val, non-baseline)

| Feature | Train coef | Val coef |
|---------|------------|----------|
| `away_short_rest` | +0.0635 | +0.0281 |
| `rest_day_difference` | -0.0597 | -0.0654 |
| `home_npxg_against` | -0.0466 | -0.0227 |
| `league_draw_rate` | -0.0309 | -0.0476 |
| `combined_low_scoring_rate` | -0.0302 | -0.0192 |
| `league_competitive_balance` | -0.0054 | -0.0650 |

**Stable count:** 6 (discovery gate wants ≥ 3)

## Top L1 coefficients (by |train|)

| Feature | Train | Val | Stable |
|---------|-------|-----|--------|
| `market_balance` | -0.1652 | +0.0000 | no |
| `away_npxg_for` | -0.1401 | +0.0458 | no |
| `home_advantage_log` | -0.0735 | +0.0555 | no |
| `away_short_rest` | +0.0635 | +0.0281 | yes |
| `rest_day_difference` | -0.0597 | -0.0654 | yes |
| `market_vs_dc_draw` | +0.0518 | +0.0000 | no |
| `home_short_rest` | +0.0473 | +0.0000 | no |
| `home_npxg_against` | -0.0466 | -0.0227 | yes |
| `home_npxg_for` | -0.0440 | +0.0000 | no |
| `away_npxg_against` | +0.0399 | +0.0000 | no |
| `expected_goal_total` | -0.0367 | +0.1220 | no |
| `combined_one_goal_match_rate` | +0.0320 | +0.0000 | no |
| `league_draw_rate` | -0.0309 | -0.0476 | yes |
| `combined_low_scoring_rate` | -0.0302 | -0.0192 | yes |
| `congestion_difference` | +0.0168 | -0.0880 | no |

## Top draw-error (surprise) Ridge coefficients

Target: `is_draw - p_draw_blend` (positive = blend undercalls X).

| Feature | Train | Val |
|---------|-------|-----|
| `favourite_strength` | +0.1802 | +0.0646 |
| `market_balance` | -0.1773 | -0.0639 |
| `away_npxg_for` | -0.0251 | +0.0101 |
| `market_vs_dc_draw` | +0.0177 | -0.0020 |
| `home_advantage_log` | -0.0151 | +0.0168 |
| `away_short_rest` | +0.0137 | +0.0145 |
| `rest_day_difference` | -0.0122 | -0.0180 |
| `home_npxg_against` | -0.0106 | -0.0164 |
| `home_short_rest` | +0.0100 | +0.0008 |
| `home_npxg_for` | -0.0099 | -0.0009 |

## Next step (Phase 3.2)

Ship only if OOS gate passes (or after a re-tuned sparse subset beats blend). Preferred: 3–8 stable terms into `config/draw_adjustment.json` via `calc/draw_adjustment.py` (logit adjust on blend, then renormalize).
