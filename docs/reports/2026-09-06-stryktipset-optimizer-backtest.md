# Report: Stryktipset coupon optimizer — chronological OOS backtest

**Date:** 2026-09-06  
**Status:** Diagnostic run complete (eval only)  
**Mode:** **VALUE only** — does **not** validate PREDICTION / MAX_P13 default (see [`2026-09-07-stryktipset-optimizer-prediction.md`](2026-09-07-stryktipset-optimizer-prediction.md)).  
**Production impact:** None — ship gate / ProbabilityManager / residual ML / Dixon–Coles **untouched**.

## Setup

| Item | Value |
|------|-------|
| Source | Local Postgres via `load_rounds_for_backtest` (`rounds/` fixtures absent) |
| Coupons | **195** settled ST rounds |
| Draw range | **4760–4966** |
| Ordering | Chronological by `min(start_time)` else `draw_number` |
| Portfolio size | `row_count=3` |
| Grid | β ∈ {0.5, 1.0, 1.5} × λ ∈ {0, 0.5, 1.0} × C ∈ {100, 500} (18 configs) |
| Seed | 0 |
| Runtime | ~**37 min** (`elapsed_s_total` ≈ 2194) |
| Artifact | [`artifacts/stryktipset_optimizer_backtest/backtest_rich.json`](../../artifacts/stryktipset_optimizer_backtest/backtest_rich.json) |

**Primary rank:** `mean_portfolio_leverage` = mean over rounds of mean selected-row `Σ log(Pm/Pp)` (β-comparable).  
**Hit-rate (`mean_best_correct`, tier counts):** diagnostic only — not the optimizer objective.

## Headline results

| Selector | Params | mean leverage | mean best-correct | ≥13 / ≥12 / ≥11 / ≥10 |
|----------|--------|--------------:|------------------:|----------------------|
| **Primary (leverage)** | β=1.0, λ=0.0, C=100 (=C=500) | **3.655** | 3.88 | 0 / 0 / 0 / 0 |
| **Hit-rate diagnostic** | β=0.5, λ=1.0, C=500 | 0.413 | **6.79** | 0 / 1 / 2 / 11 |

### 1-row baselines (same 195 rounds)

| Baseline | mean correct | ≥11 / ≥10 |
|----------|-------------:|-----------|
| Market favorites | 6.45 | 5 / 11 |
| Public favorites | 6.51 | 2 / 13 |

## Interpretation

- High β (≥1) maximizes public-dilution leverage and **collapses** hit rate (no rounds with ≥9 correct under the primary winner).
- β=0.5 stays nearer favorites: mean best-correct **6.79** slightly above 1-row baselines; still rare high tiers.
- At β≥1 and λ=0, C=100 vs C=500 are **identical** on this grid (top candidates already inside C=100).
- Do **not** promote a “winning” β from hit-rate alone — that fights the stated EV-vs-public objective.

## Limitations (UNKNOWN leakage)

Stored odds have **no timestamp** (typically startOdds on import). Public streckprocent has **no timestamp**. `regCloseTime` exists in the Svenska Spel API but is **not** persisted on `STRound`. Treat snapshots as operable coupon-time data; **do not** claim closing-line safety. No SEK/pool modeling (relative metrics only).

## CLI replay

```bash
python -m src.scripts.backtest_stryktipset_optimizer --json artifacts/stryktipset_optimizer_backtest/backtest_rich.json
```

(Uses DB when `--fixture-dir` is omitted.)
