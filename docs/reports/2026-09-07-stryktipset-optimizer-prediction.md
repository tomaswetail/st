# Report: Stryktipset optimizer — PREDICTION default (MAX_P13)

**Date:** 2026-09-07  
**Package:** `src/calc/stryktipset_optimizer/`  
**Production impact:** none (standalone research tool; ship path untouched)

## Summary

Default coupon optimization is now **PREDICTION / MAX_P13**: select the R distinct rows with highest joint vig-free market probability. Public streckprocent is diagnostic only in PREDICTION. The prior EV objective (`Σ log Pm − β Σ log Pp` + Hamming diversity) remains as **VALUE** mode.

## Architecture

| Module | Role |
|--------|------|
| `prediction.py` | Exact top-R by `log Pm` heap DFS (MAX_P13) |
| `coverage.py` | Exact `P(best_correct=k)` via 3^N enumeration |
| `objectives.py` | Greedy approx for MAX_P12+ / MAX_P11+ / MAX_EXPECTED |
| `reduced_system.py` | Doubles/triples marginal-gain reduced systems |
| `candidates.py` / `portfolio.py` / `value.py` | VALUE path (unchanged conceptually) |
| `optimize.py` | Mode branch; attaches coverage to results |
| `params.py` | `mode`, `objective`, VALUE knobs, `reduced_system` |

DTO extensions: `mode`, `objective`, `exact_selection`, `joint_probability`, `coverage`, `system_sign_pattern` / `system_sign_counts`.

## MAX_P13 algorithm (exact)

Assumptions: vig-free Pm are truth; matches independent; rows are mutually exclusive atoms ⇒ `P(union)=Σ P(row)`.

1. Per match, `a_j(i) = log Pm_j(i)`.
2. DFS over outcome indices in OUTCOMES order (`"1","X","2"`), accumulate log-sum.
3. Keep a size-R min-heap of best log-sums (same pattern as VALUE `candidates.py`).
4. Portfolio = heap contents sorted by (−log_sum, enumeration tie-break). No λ_diversity.

**Tie-break:** earlier DFS enumeration order (stable; OUTCOMES-lex at each depth).

Identity: for MAX_P13, `coverage.p_full == Σ joint_probability` of selected rows.

## Approximate objectives

For `MAX_P12_OR_BETTER`, `MAX_P11_OR_BETTER`, `MAX_EXPECTED_CORRECT`:

- Candidate pool = top-C market rows (`prediction_candidate_count`, default 2000).
- Greedy: from empty set, repeatedly add the candidate with largest exact coverage objective gain.
- Results labeled `exact_selection=False` with limitations text.

## Reduced doubles/triples systems

`build_reduced_system(...)` / CLI `--system reduced`:

1. Start from all-favorite singles (size 1).
2. Iteratively upgrade single→double or double→triple choosing largest marginal gain in the active objective, subject to product ≤ `--rows` budget.
3. Expand Cartesian product; attach coverage + sign pattern.

Documented constraint: reduced systems ⊆ arbitrary R-row portfolios; unconstrained MAX_P13 top-R may achieve higher P(full-correct).

## CLI examples

```bash
# Default PREDICTION / MAX_P13
python -m src.scripts.optimize_stryktipset_coupon --fixture coupon.json --rows 8

# Explicit
python -m src.scripts.optimize_stryktipset_coupon --fixture coupon.json \
  --mode PREDICTION --objective MAX_P13 --rows 8

# Reduced doubles/triples
python -m src.scripts.optimize_stryktipset_coupon --fixture coupon.json \
  --mode PREDICTION --system reduced --rows 16

# VALUE (old EV path)
python -m src.scripts.optimize_stryktipset_coupon --fixture coupon.json \
  --mode VALUE --beta 1.0 --lambda-diversity 0.5 --rows 5

# Backtest requires --mode (no silent VALUE leverage on PREDICTION)
python -m src.scripts.backtest_stryktipset_optimizer --mode PREDICTION \
  --objective MAX_P13 --fixture-dir rounds/
python -m src.scripts.backtest_stryktipset_optimizer --mode VALUE \
  --fixture-dir rounds/
```

## Tests

```bash
env -u PYTHONPATH /Users/tomaskircher/wetail/git/st-poisson-distribution/.venv/bin/python \
  -m pytest tests/test_calc/test_stryktipset_optimizer.py -q
```

**Result:** 45 passed.

Coverage includes: MAX_P13 vs brute force (N=3 and N=13 R=3), P(full)=Σ row probs, public-flip invariance, VALUE β/λ regressions, coverage sanity, reduced vs naive closest, no PM/residual_ml/DC imports, CLI roundtrip.

## VALUE path fate

VALUE mode preserves β / λ_diversity / candidate_count behavior. Prior OOS reports:

- [`2026-09-06-stryktipset-optimizer-backtest.md`](2026-09-06-stryktipset-optimizer-backtest.md)
- [`2026-09-06-stryktipset-optimizer-backtest-rows128.md`](2026-09-06-stryktipset-optimizer-backtest-rows128.md)

are **VALUE-objective only** and do **not** validate PREDICTION default.

## Production impact

None. No imports or wiring to `ProbabilityManager`, `residual_ml`, or Dixon–Coles production path.

## Leakage

UNKNOWN: odds/public timestamps absent; `regCloseTime` not on `STRoundModel`. Do not claim closing-line safety.
