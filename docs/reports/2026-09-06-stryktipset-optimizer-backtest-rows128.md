# Report: Stryktipset coupon optimizer — OOS backtest `row_count=128`

**Date:** 2026-09-06  
**Status:** Diagnostic run complete (eval only)  
**Mode:** **VALUE only** — does **not** validate PREDICTION / MAX_P13 default (see [`2026-09-07-stryktipset-optimizer-prediction.md`](2026-09-07-stryktipset-optimizer-prediction.md)).  
**Production impact:** None — ship gate / ProbabilityManager / residual ML / Dixon–Coles **untouched**.

## Setup

| Item | Value |
|------|-------|
| Source | Local Postgres via `load_rounds_for_backtest` |
| Coupons | **195** settled ST rounds |
| Draw range | **4760–4966** |
| Ordering | Chronological by `min(start_time)` else `draw_number` |
| Portfolio size | `row_count=128` |
| Grid | β ∈ {0.5, 1.0, 1.5} × λ ∈ {0, 0.5, 1.0} × **C=500 only** (9 configs) |
| Why not default CLI grid | Stock CLI hardcodes `row_count=3` and C ∈ {100, 500}. **C=100 cannot select 128 rows** (`take = min(row_count, len(candidates))`). C=2000 omitted (~4× slower per cell). |
| Seed | 0 |
| Runtime | ~**152 min** (`elapsed_s_total` ≈ 9146) |
| Python | `/tmp/st-diag-venv/bin/python` |
| Artifact | `artifacts/stryktipset_optimizer_backtest/backtest_rich_rows128.json` (source: `/tmp/st_stryktipset_bt/backtest_rich_rows128.json`) |

**Primary rank:** `mean_portfolio_leverage` = mean over rounds of mean selected-row `Σ log(Pm/Pp)` (β-comparable).  
**Hit-rate (`mean_best_correct`, tier counts):** best-of-128 rows — diagnostic only.

## Headline results

| Selector | Params | mean leverage | mean best-correct | ≥13 / ≥12 / ≥11 / ≥10 |
|----------|--------|--------------:|------------------:|----------------------|
| **Primary (leverage)** | β=1.0, λ=0.0, C=500 | **3.444** | 5.76 | 0 / 0 / 0 / 2 |
| **Hit-rate diagnostic** | β=0.5, λ=1.0, C=500 | 0.445 | **8.48** | 0 / 5 / 28 / 63 |

Mean correct **across** the 128 rows (not best-of): primary **3.42**; hit-diag **5.84**.

### vs prior `row_count=3` (same 195 coupons)

| | row=3 (prior) | row=128 (this) |
|--|---------------:|---------------:|
| Primary params | β=1.0, λ=0.0, C=100 (=C=500) | β=1.0, λ=0.0, C=500 |
| Primary mean leverage | **3.655** | **3.444** |
| Primary mean best-correct | 3.88 | 5.76 |
| Primary ≥10 / ≥11 / ≥12 | 0 / 0 / 0 | 2 / 0 / 0 |
| Hit-diag params | β=0.5, λ=1.0, C=500 | β=0.5, λ=1.0, C=500 |
| Hit-diag mean best-correct | 6.79 | **8.48** |
| Hit-diag ≥12 / ≥11 / ≥10 | 1 / 2 / 11 | **5 / 28 / 63** |

1-row market/public baselines unchanged: mean correct **6.45 / 6.51**.

## Interpretation

- Same qualitative story as row=3: high β maximizes dilution leverage and hurts hit rate; β=0.5 maximizes best-of-portfolio hits.
- Expanding 3→128 rows **raises** best-of hit tiers (more draws) while **slightly lowering** mean portfolio leverage under β=1 (averaging deeper into the candidate list).
- λ has almost no effect on leverage at β≥1 with C=500; small hit-rate differences only.
- Do **not** promote from hit-rate alone — that fights the stated EV-vs-public objective.

## Limitations (UNKNOWN leakage)

Stored odds have **no timestamp** (typically startOdds on import). Public streckprocent has **no timestamp**. `regCloseTime` is **not** persisted on `STRound`. Treat snapshots as operable coupon-time data; **do not** claim closing-line safety. No SEK/pool modeling.

## CLI note / replay

Stock `python -m src.scripts.backtest_stryktipset_optimizer` has **no `--rows`** and hardcodes `row_count=3`. This run used the same rich harness as the prior OOS backtest with a custom `ParamConfig` grid (`row_count=128`, `candidate_count=500`).
