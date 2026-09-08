# Report: Stryktipset market + streckprocent coupon optimizer

**Date:** 2026-09-06  
**Status:** **APPROVED** (PL + verifier + model review)  
**Production impact:** None — ship gate / ProbabilityManager / residual ML / Dixon–Coles **untouched**.

## Summary

New standalone package `src/calc/stryktipset_optimizer/` optimizes Stryktipset coupon rows for **EV vs public dilution**, treating fair market odds as true probabilities and streckprocent as public share. Not a prediction-accuracy model.

## Files created / changed

### Package
| File | Role |
|------|------|
| `src/calc/stryktipset_optimizer/__init__.py` | Public exports |
| `fair_probs.py` | Strict odds → Pm (rejects ≤1 / null / NaN / inf) |
| `public_probs.py` | Streckprocent normalize + epsilon floor |
| `value.py` | value_ratio, log_leverage |
| `candidates.py` | Exhaustive top-C via recursive DFS enumeration + min-heap |
| `portfolio.py` | Greedy Hamming diversity |
| `analyzer.py` | Banker diagnostic + Jensen–Shannon ranking |
| `data.py` | DB / in-memory coupon load + leakage note |
| `optimize.py` | `CouponOptimizer` orchestration |
| `params.py` | `OptimizerParams` dataclass |
| `simulator.py` | Phase 2 MC (relative tiers only) |
| `backtester.py` | Phase 2 chronological OOS tuner |

### Schema / CLI / tests / docs
- `src/objects/schema/data_classes/stryktipset_optimizer.py` — `StryktipsetOptimizationResult` DTO
- `src/scripts/optimize_stryktipset_coupon.py` — optimize (+ `--simulate`)
- `src/scripts/backtest_stryktipset_optimizer.py` — historical param grid
- `tests/test_calc/test_stryktipset_optimizer.py` — 31 tests
- `docs/plans/2026-09-06-stryktipset-coupon-optimizer.md`
- `docs/reports/2026-09-06-stryktipset-coupon-optimizer.md` (this file)
- `docs/project_status.md` — Recent work entry

**Not modified:** `src/utils/common.odds_to_probabilities`, `probability_manager.py`, `residual_ml/`, `dixon_coles*`, production profile / ship gate.

## Formulas

1. **Fair market:** `raw = 1/odds`; `Pm = raw / sum(raw)`. Strict helper rejects odds ≤ 1.
2. **Public %:** if any value ``> 1``, treat as **0–100 percentage** (incl. ``1`` = 1%); unit scale only when all values ∈ ``[0, 1]``. Reject negatives / ``> 100``. Renorm to sum=1; epsilon floor then renorm (`public_epsilon` default `1e-6`). Sum=0 → reject.
3. **value_ratio** = Pm/Pp; **log_leverage** = log(Pm/Pp).
4. **row_score** = Σ log Pm[sel] − β Σ log Pp[sel] = Σ a_j(sel_j) with `a_j(i) = log Pm − β log Pp`.
5. **Candidates:** exhaustive 3^N top-C (recursive DFS enumeration + min-heap); diagnostics include outcomes, log sums, row_score, #favorites (market argmax), #homes/#draws/#aways.
6. **Portfolio:** `adjusted = row_score − λ_div · avg_HamSim`; HamSim = identical/N.
7. **Banker (diagnostic):** `log(Pm) + w · log_leverage`.
8. **Match ranking:** Jensen–Shannon (nats) Pm vs Pp; max |log leverage|; highest value outcome + ratio.
9. **MC:** sample truth from Pm; synthetic public rows from Pp; tiers 13/12/11/10; **relative metrics only** (no SEK).
10. **Backtest:** chronological by min(start_time) else draw_number; no time shuffle. Primary ranking: `best_by_mean_portfolio_leverage` = mean over rounds of selected-row `Σ log(Pm/Pp)` (≡ row_score at β=1; comparable across β). Construction `mean_top_row_score` / `mean_correct` / tiers are diagnostics only (row_score inflates with β).

## CLI how-to

```bash
# From fixture (DB-less)
python -m src.scripts.optimize_stryktipset_coupon \
  --fixture path/to/coupon.json --rows 5 --beta 1.0 \
  --lambda-diversity 0.5 --candidate-count 500 --seed 0 --json out.json

# From DB
python -m src.scripts.optimize_stryktipset_coupon --draw-number 4950 --rows 5

# With MC simulation
python -m src.scripts.optimize_stryktipset_coupon --fixture coupon.json --simulate --n-simulations 1000

# Historical tuner
python -m src.scripts.backtest_stryktipset_optimizer --fixture-dir path/to/rounds/ --json bt.json
python -m src.scripts.backtest_stryktipset_optimizer --min-draw 4900 --max-draw 4950
```

## Tests

```bash
/tmp/st-diag-venv/bin/python -m pytest \
  tests/test_calc/test_stryktipset_optimizer.py \
  tests/test_calc/test_probability_calculations.py -q
```

**Result:** 46 passed (31 optimizer + 15 probability_calculations).

Coverage highlights: fair-prob worked example; value_ratio 2.0; beta ranking flip; Hamming 0/1; λ diversity; invalid odds/public/match-count; DTO fields; deterministic MC; chronological backtest ordering; CLI fixture roundtrip; no residual_ml / ProbabilityManager imports on optimize path.

## Benchmark (N=13, microbench in test)

| candidate_count | Wall time |
|-----------------|-----------|
| 500 | **0.417 s** |
| 50_000 | **1.313 s** |

(Measured 2026-09-06 on developer machine via `test_full_13_match_optimize_and_benchmark`.)

## Assumptions / limitations (leakage UNKNOWN)

- Odds: no timestamp in DB; typically `startOdds` on import.
- Public %: no timestamp; ints usually 0–100 on `STMatchBetModel`.
- `regCloseTime` exists in Svenska Spel API but is **not** on `STRoundModel`.
- Stored snapshot treated as operable coupon-time data — **do not claim closing-line safety**.
- MC uses independent synthetic public rows (default book size 1000; not a real pool model).
- No SEK payout modeling.
- Backtest primary ranking is `best_by_mean_portfolio_leverage` (`Σ log(Pm/Pp)` of selected rows; comparable across β). Construction `mean_top_row_score` / correct-pick / tier rates are diagnostics only.

## Rework 2026-09-06 11:05

1. `favorite_count` = market argmax Pm matches (not homes); added `home_count`.
2. Backtest ranking moved off `mean_correct` (superseded 11:10).
3. MC/CLI `public_row_count` default 1000.
4. Status wording: implemented; pending PL/verifier (no premature APPROVED).

## Rework 2026-09-06 11:10

Primary OOS ranking is `best_by_mean_portfolio_leverage` (`Σ log(Pm/Pp)` of selected rows ≡ row_score at β=1). Construction `mean_top_row_score` is diagnostic only — **not comparable across β** (β inflates absolute row_score).

## Rework 2026-09-06 11:13

Public scale: any value `> 1` ⇒ percentage 0–100 (so ST int `1` = 1%). Unit scale only when all ∈ `[0, 1]`. MC tiers relative to coupon size N.

## Production impact

**None.** Optimizer is a separate research/ops tool. Official ship gate (518-row pooled include-holdout, shrink α=0.7, MLE-ρ DC + pinned HGB) unchanged.
