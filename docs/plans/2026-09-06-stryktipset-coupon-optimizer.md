# Plan: Stryktipset market + streckprocent coupon optimizer

**Date:** 2026-09-06  
**Status:** Implementation  
**Ship gate:** Unchanged — not wired to ProbabilityManager / residual ML / Dixon–Coles.

## Goal

Originally: optimize Stryktipset coupon rows for **EV vs public dilution** (VALUE).  
**Superseded as default (2026-09-07):** see [`2026-09-07-stryktipset-optimizer-prediction.md`](2026-09-07-stryktipset-optimizer-prediction.md) — PREDICTION / MAX_P13 is now the default; this VALUE path remains as `--mode VALUE` only.

## Architecture

```
src/calc/stryktipset_optimizer/
  fair_probs.py      # strict odds → Pm (do not touch utils.common)
  public_probs.py    # streckprocent normalize + epsilon floor
  value.py           # value_ratio, log_leverage
  candidates.py      # exhaustive top-C via heap (N=13)
  portfolio.py       # greedy Hamming diversity
  analyzer.py        # banker diagnostic + JS match ranking
  data.py            # load ST coupon matches (DB or in-memory)
  optimize.py        # CouponOptimizer orchestration
  params.py          # OptimizerParams
  simulator.py       # Phase 2 MC (relative tiers only)
  backtester.py      # Phase 2 chronological OOS tuner
```

CLI: `python -m src.scripts.optimize_stryktipset_coupon`  
Backtest: `python -m src.scripts.backtest_stryktipset_optimizer`

## Leakage UNKNOWN (document everywhere)

- Odds: no timestamp; typically `startOdds` on import.
- Public %: no timestamp; ints 0–100 on `STMatchBetModel`.
- `regCloseTime` in API but **not** on `STRoundModel`.
- Treat stored snapshot as operable coupon-time data; do **not** claim closing-line safety.

## Non-goals

- SEK payouts
- Wiring into production probability path
- Schema changes for timestamps
