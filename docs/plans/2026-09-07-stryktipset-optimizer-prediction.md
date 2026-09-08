# Plan: Stryktipset optimizer — PREDICTION default (MAX_P13)

**Date:** 2026-09-07  
**Status:** Implementation  
**Ship gate:** Unchanged — not wired to ProbabilityManager / residual ML / Dixon–Coles.

## Goal

Replace the default coupon-optimizer objective with **predictive coverage** using **vig-free market probabilities only**. The previous EV / streckprocent path remains available as **VALUE mode only**.

## Modes (never mix implicitly)

| Mode | Selection uses | Public % |
|------|----------------|----------|
| **PREDICTION** (default) | market Pm only | diagnostic only — must not affect row selection |
| **VALUE** | `Σ log Pm − β Σ log Pp` + Hamming diversity | used in scoring |

## PREDICTION objectives

| Objective | Selection | Notes |
|-----------|-----------|-------|
| **MAX_P13** (default) | Exact top-R by joint `P(row)=∏ Pm` | DFS/heap; no λ_diversity |
| MAX_P12_OR_BETTER | Greedy marginal-gain over top-C | Approximate; documented |
| MAX_P11_OR_BETTER | Greedy marginal-gain over top-C | Approximate; documented |
| MAX_EXPECTED_CORRECT | Greedy marginal-gain over top-C | Approximate; documented |

## Architecture

```
src/calc/stryktipset_optimizer/
  fair_probs.py / public_probs.py / data.py   # unchanged contracts
  prediction.py     # market-only top-R heap (MAX_P13)
  coverage.py       # exact P(best_correct=k) over 3^N
  objectives.py     # greedy approx for non-P13 objectives
  reduced_system.py # doubles/triples marginal-gain builder
  candidates.py / value.py / portfolio.py     # VALUE path
  optimize.py       # branch on mode
  params.py         # mode, objective, VALUE knobs
  backtester.py     # mode-aware primary metric
```

CLI: `python -m src.scripts.optimize_stryktipset_coupon`  
Backtest: `python -m src.scripts.backtest_stryktipset_optimizer --mode PREDICTION|VALUE` (mode required)

## Leakage UNKNOWN (document everywhere)

- Odds: no timestamp; typically `startOdds` on import.
- Public %: no timestamp; ints 0–100 on `STMatchBetModel`.
- `regCloseTime` in API but **not** on `STRoundModel`.
- Treat stored snapshot as operable coupon-time data; do **not** claim closing-line safety.

## Prior VALUE backtests

Reports dated 2026-09-06 (β-leverage OOS, rows=3 / rows=128) apply to **VALUE mode only**. They do **not** validate the PREDICTION default.

## Non-goals

- SEK payouts
- Wiring into production probability path
- Schema changes for timestamps
- Exact combinatorial optimization for MAX_P12+ / MAX_EXPECTED
