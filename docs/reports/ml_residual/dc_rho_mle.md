# Dixon–Coles ρ by MLE instead of grid search

**Status:** implemented behind a default-off flag, measured, **not promoted to production**.

ρ used to be a fixed hyperparameter chosen by grid search on out-of-sample 1X2 log loss. It can now be estimated by maximum likelihood from scorelines inside the fit, alongside attack, defence and home advantage. Terminology: [`docs/wiki.md`](../../wiki.md). Current ship state: [`docs/production_profile.md`](../../production_profile.md).

## Why the grid approach was suspect

In 1X2 space ρ's entire footprint is a single scalar draw shift:

\[
\Delta p_X = -2\lambda\mu\rho e^{-\lambda-\mu},\qquad \Delta p_1 = \Delta p_2 = +\lambda\mu\rho e^{-\lambda-\mu}
\]

So a 3-outcome loss is a very lossy channel for a scoreline-level parameter, and ρ ends up absorbing whatever draw miscalibration a league's validation slice happens to show — the classic DC model has no other draw parameter. Meanwhile the scoreline likelihood sees every 0-0, 1-0, 0-1 and 1-1 in the lookback window.

The shipped `config/classic_dc_league_params.json` showed the symptom: 4 leagues with ρ = −0.20, −0.15, 0.00, +0.05, spanning the whole grid, two of them on a grid boundary.

## What changed

`DixonColesModel` gained `fit_rho` (default **False**). With it enabled, ρ becomes the last element of the L-BFGS-B parameter vector, initialized at the configured ρ and box-bounded to `[rho_min, rho_max]` (default ±0.2) so the τ correction stays positive.

Two implementation details worth recording:

**Home advantage was read as `theta[-1]`.** Appending ρ to the parameter vector would have silently made home advantage `exp(ρ)` ≈ 0.87 and dropped the real parameter — no crash, no convergence failure, just a different model. Home advantage is now indexed explicitly at `2 * n_teams - 1`, with ρ at `2 * n_teams`. The recovery test asserts home advantage too, so this failure cannot hide again.

**ρ enters the likelihood from `theta`, not from `self.rho`.** The value is read inside the `neg_log_likelihood` closure, so there is structurally no path where a fitting run keeps using the stale constructor value — the "MLE that is secretly a no-op" failure mode is impossible by construction rather than by test.

`fit_rho=False` passes no `bounds` to `minimize` and keeps `n_params` unchanged, so the existing path is untouched.

When ρ is fitted, the optimizer drops the ρ grid dimension: **210 combos → 35**.

## Does the estimator work? Synthetic recovery

Scorelines sampled from a known DC process (true ρ = −0.15, true home advantage = 1.35), 8 seeds per row:

| Matches | fitted ρ mean | fitted ρ sd | home advantage mean |
|---:|---:|---:|---:|
| 1,120 | −0.1653 | 0.0354 | 1.3616 |
| 3,360 | −0.1629 | 0.0247 | 1.3318 |
| 16,800 | −0.1522 | 0.0105 | 1.3431 |

Consistent and essentially unbiased — both ρ and home advantage converge to truth as n grows. **Calibration to keep in mind:** sd(ρ) ≈ 0.025 at 3,360 matches, so on realistic lookback windows (a few hundred matches) a single fitted ρ carries sd ≈ 0.03–0.05. ρ is a genuinely noisy quantity even when estimated properly; it is just far less noisy than the grid made it look.

Source: [`artifacts/dc_rho_mle/synthetic_recovery.json`](../../../artifacts/dc_rho_mle/synthetic_recovery.json).

## Head-to-head on real data

Identical data, leagues and windows for both arms: time-split validation, fraction 0.20, `--draw-max=TUNING_DRAW_MAX` so the 4951–4960 holdout stays excluded. 441 validation matches loaded, 5 leagues scored (7 skipped for `< 15` matches), 423 matches scored.

| | Arm A — grid ρ | Arm B — MLE ρ |
|---|---|---|
| Combos per league | 210 | **35** |
| Pooled DC log loss | **1.0311** | 1.0416 |
| Wall clock (`--jobs 4`) | 74m31s | **12m23s** (6.0× faster) |
| ρ on a boundary | **3 of 5 leagues** | **0 of 87 fits** |

Per league:

| League | n | Arm A ρ | Arm A LL | Arm B ρ (median) | Arm B ρ range | Arm B LL |
|---|---:|---:|---:|---:|---|---:|
| 39 | 133 | −0.15 | 0.9663 | −0.0145 | [−0.0466, −0.0012] | 0.9683 |
| 41 | 69 | **−0.20** (bound) | 1.0749 | +0.0063 | [−0.0131, +0.0131] | 1.0808 |
| 42 | 23 | **+0.05** (bound) | 1.0697 | −0.0172 | [−0.0218, +0.0107] | 1.0744 |
| 45 | 19 | 0.00 | 0.9193 | −0.1406 | [−0.1834, −0.0294] | 0.9645 |
| 180 | 179 | **+0.05** (bound) | 1.0694 | −0.1086 | [−0.1251, −0.0756] | 1.0850 |

Full detail: [`artifacts/dc_rho_mle_comparison.json`](../../../artifacts/dc_rho_mle_comparison.json).

## Reading the result

**Arm A wins pooled tuning log loss by 0.0105, and that number should not be trusted.** Arm A is the argmin of 210 candidates scored on the very rows it reports; arm B picks from 35 and lets the likelihood set ρ. The gap is the size of arm A's selection optimism, not evidence that grid ρ models the data better.

The diagnostic that settles it is **agreement, not log loss**. Grid ρ and MLE ρ disagree on the sign of ρ in 4 of 5 leagues — most starkly league 180, where the grid picks +0.05 while the likelihood says −0.109 with a range of [−0.125, −0.076] across 28 fits that never touches a bound. A parameter whose grid-selected value points the opposite way from its own maximum-likelihood estimate is not measuring low-score dependence; it is a draw-adjustment knob being tuned by a noisy 3-outcome objective.

Arm B's estimates behave like estimates: interior in **all 87** fits, tightly clustered within each league, and in the −0.14…+0.01 band. League 41's ρ of +0.006 with range [−0.013, +0.013] is a clean "no detectable low-score dependence in this league" — a legitimate finding, where the grid instead pinned that league at −0.20.

Also note the scale: pooled DC log loss is ≈1.03–1.04 against market ≈1.007, and DC carries 0.3 weight in the blend, so ρ's effect on final published probabilities is second-order regardless of which arm wins.

## Recommendation

**Adopt MLE as the estimator for ρ; do not flip production on this evidence alone.**

The statistical case is settled — grid-searched per-league ρ should stop being read as a league effect, and the 6× speedup makes future ξ/lookback re-tuning much cheaper. But the ship metric is the 518-row include-holdout gate, and nothing here measures it. Promotion requires regenerating the DC feature columns, retraining the HGB residual, and re-running the gate; given DC's 0.3 blend weight and an 0.0105 pooled DC difference, expect a small gate effect in either direction.

Suggested order if promoting: enable `CLASSIC_DC_FIT_RHO=1`, rebuild DC features, retrain, measure the 518-row gate against the current 1.0022, and only then decide. Keep `config/classic_dc_league_params.json` as-is until that run exists.

## Not addressed here

- **Selection contamination is unchanged.** The DC tuning rows are still ~95% of the 518-row gate, so DC feature quality on that gate is optimistically biased in both arms.
- `DixonColesOptimizationResult.pooled_market_log_loss` is still a stub returning `0.0`.
- `min_training_matches` is still checked against the pre-filter match count, before `matches_with_enough_team_history` runs, so effective training size can fall below the threshold.
- ξ and lookback remain grid-searched, and remain boundary-heavy in both arms.

## Reproducing

```bash
# Arm B (rho by MLE, 35 combos)
python -m src.scripts.optimize_classic_dixon_coles --fit-rho --jobs 4 \
  --output artifacts/dc_rho_mle/params_arm_b_fitted_rho.json

# Arm A (grid rho, 210 combos)
python -m src.scripts.optimize_classic_dixon_coles --jobs 4 \
  --output artifacts/dc_rho_mle/params_arm_a_grid.json
```

Always pass `--output`; it defaults to the production `config/classic_dc_league_params.json`.

Config surface: `CLASSIC_DC_FIT_RHO` (default off), `CLASSIC_DC_RHO_MIN` / `CLASSIC_DC_RHO_MAX` (default ±0.2), or `"fit_rho": true` in the grid JSON.
