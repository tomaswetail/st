# DC ρ MLE promotion — Path B execution report

**Date:** 2026-09-04  
**Decision:** PROMOTE (DC params + fit_rho default); **HGB HOLD** (gate regression)  
**Executor:** developer subagent (user override of default HOLD)

---

## Summary

Path B executed end-to-end. MLE-fitted Dixon–Coles ρ and refreshed ξ/lookback were promoted to `config/classic_dc_league_params.json`; `classic_dc_fit_rho` default flipped to **True**. Dataset rebuilt (2589 rows), HGB retrained, and the official 518-row include-holdout gate measured.

**HGB not shipped:** best shrink **1.0046** (α=0.7) vs Phase 6.1 baseline **1.0022** (α=0.5) — regression +0.0024. Production HGB remains Phase 6.1 restore in `models/residual_ml/sweep_best/`.

---

## Phase results

| Phase | Status | Notes |
|-------|--------|-------|
| **B0** Baseline backup | PASS | Grid params backed up; md5s recorded below |
| **B1** MLE ρ re-optimize | PASS | 5 leagues, 0/87 at_bound (0%), 841.6s wall clock |
| **B1b** fit_rho default True | PASS | `DataSourceConfig.classic_dc_fit_rho` default `"1"`; env `=0` still disables |
| **B2** Dataset rebuild | PASS | 2589 rows; DC probs normalize on non-NaN rows |
| **B3** HGB retrain | PASS | val ML raw 1.0149; train LL 0.8372 |
| **B4** Official gate | PASS (measured) | best shrink **1.0046** > 1.0022 → regression |
| **B5** Ship decision | PARTIAL | DC params + fit_rho shipped; HGB held |

---

## Baseline checksums (B0)

| Artifact | md5 |
|----------|-----|
| `config/classic_dc_league_params_grid_bck.json` (grid ρ backup) | `afab9b6444dbaa53cc9b62800b024819` |
| `artifacts/baseline_post_dc_tune_20250902/dataset.csv` | `bd614bdc8a794fc3be5c0467a8dd7e49` |
| `models/residual_ml/sweep_best/model.pkl` | `004bd82d75f2a29c6b579eb672c5bf57` |
| `models/residual_ml/sweep_best/baseline_weights.json` | `090e248e1c70592d6c721431b7cdc58d` |
| `models/residual_ml/sweep_best/feature_schema.json` | `48b18086d8b60cbbd269a7b31734d17b` |
| `models/residual_ml/sweep_best/sweep_results.json` | `b491a0ff78b9b98efbacc44a0aa54612` |

Post-promotion params md5: `2dfab117069e8b4e8671a4c3ceb784ec`  
Promotion dataset md5: `9e789ce5abbadeadb67ffa6f5df176aa`

---

## New ρ table (B1 — MLE walk-forward median)

| League | ξ | Lookback (days) | ρ (MLE) | Grid ρ (old) | Sign change? |
|--------|---|-----------------|---------|--------------|--------------|
| 39 (EPL) | 0.00025 | 1460 | **−0.0145** | −0.20 | same sign, much smaller magnitude |
| 41 (Championship) | 0.0001 | 1460 | **+0.0063** | −0.15 | **yes** |
| 42 (League One) | 0.0001 | 1095 | **−0.0172** | (not in prod) | new league entry |
| 45 (FA Cup) | 0.005 | 730 | **−0.1406** | 0.00 | same sign (was zero) |
| 180 (Scottish Prem) | 0.0001 | 1460 | **−0.1086** | +0.05 | **yes** |

Per-league at_bound: 0% across all fits (0/87 total). Pooled weighted DC log loss on tuning slice: **1.0416**.

Full optimizer output: [`artifacts/dc_rho_mle/params_refit_20250904.json`](../../artifacts/dc_rho_mle/params_refit_20250904.json)  
Log: [`artifacts/dc_rho_mle/refit_20250904.log`](../../artifacts/dc_rho_mle/refit_20250904.log)

---

## B4 gate vs baseline

| Metric | Phase 6.1 baseline | DC ρ MLE candidate | Δ |
|--------|-------------------|-------------------|---|
| Slice | 518-row include-holdout | same | — |
| Market LL | 1.0068 | 1.0068 | 0 |
| Blend LL | 1.0057 → 1.0083* | 1.0083 | +0.0026* |
| ML raw LL | 1.0091 | 1.0149 | +0.0058 |
| **Best shrink LL** | **1.0022** (α=0.5) | **1.0046** (α=0.7) | **+0.0024** |

\*Blend on new dataset differs from archived baseline CSV due to DC rebuild.

Gate threshold: ≤ 1.0022 → **FAILED** (regression).

Artifacts: [`artifacts/dc_rho_mle_promotion/gate_eval.json`](../../artifacts/dc_rho_mle_promotion/gate_eval.json)

---

## B5 ship actions

| Component | Action |
|-----------|--------|
| `config/classic_dc_league_params.json` | **PROMOTED** — MLE ρ + refreshed ξ/lookback |
| `config/classic_dc_league_params_grid_bck.json` | Grid backup retained |
| `classic_dc_fit_rho` default | **True** (`CLASSIC_DC_FIT_RHO=0` disables) |
| `models/residual_ml/sweep_best/` | **UNCHANGED** — Phase 6.1 restore kept |
| `models/residual_ml/dc_rho_mle_candidate/` | Archived candidate (not promoted) |

---

## Dataset verification (B2)

- Rows: **2589** (matches canonical draw window 4760–4960)
- DC prob NaNs: 294 rows (leagues without sufficient classic DC history — same pattern as prior builds)
- Non-NaN DC rows: probabilities sum to **1.000000**
- Canonical baseline CSV **not overwritten**: `artifacts/baseline_post_dc_tune_20250902/dataset.csv`

Build log: [`artifacts/dc_rho_mle_promotion/build_dataset.log`](../../artifacts/dc_rho_mle_promotion/build_dataset.log)

---

## Tests

```
env -u PYTHONPATH python -m pytest tests/ -q
303 passed, 2 xfailed in 10.05s
```

---

## Known limitations / notes

1. **Eval protocol contamination** (~492 tuning rows overlap ~95% of 518 gate rows) — unchanged; noted only.
2. **HGB mismatch:** production HGB still trained on pre-MLE-ρ dataset; live DC scoring now uses MLE ρ at fit time. This is intentional per user override (DC ships regardless of HGB gate).
3. **7 leagues skipped** in optimizer (insufficient validation matches); global fallback still applies.
4. **No git commit** per task contract.

---

## Artifact index

| Path | Purpose |
|------|---------|
| `config/classic_dc_league_params_grid_bck.json` | Grid ρ backup |
| `config/classic_dc_league_params.json` | Promoted MLE params |
| `artifacts/dc_rho_mle/params_refit_20250904.json` | B1 optimizer output |
| `artifacts/dc_rho_mle/refit_20250904.log` | B1 run log |
| `artifacts/dc_rho_mle_promotion/dataset.csv` | B2 rebuilt dataset |
| `artifacts/dc_rho_mle_promotion/build_dataset.log` | B2 build log |
| `artifacts/dc_rho_mle_promotion/train.log` | B3 training log |
| `models/residual_ml/dc_rho_mle_candidate/` | B3 candidate model |
| `artifacts/dc_rho_mle_promotion/gate_eval.json` | B4 official gate |
| `artifacts/dc_rho_mle_promotion/gate_eval.log` | B4 eval stdout |
