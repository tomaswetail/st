# 2026-09-04 — Freeze on MLE ρ (canonical DC + pinned HGB)

**Decision:** **APPROVED**  
**Date:** 2026-09-04 21:45  
**Owner:** Project Leader  
**Implementation:** [developer](6f5215ad-7c81-4a49-bdc8-8b8529407368)  
**Verification:** [verifier](ca4fdade-8fc9-4844-b707-85e04de2723d) — VERIFIED, 60 tests passed

---

## Outcome

Live scoring and official eval are the same **MLE-ρ Dixon–Coles + pinned MLE HGB** system.

| Item | Value |
|------|--------|
| Official 518-row include-holdout ship metric | **1.0046** (α=**0.7**) |
| Stretch ≤0.99 | **MISSED** |
| Phase 6.1 **1.0022** (α=0.5) | **Historical** (grid-DC + old HGB) — not the live number |
| Sweep-on-MLE HGB 1.0063 (α=0.8) | **Not promoted** |

**Backup paths**

- Phase 6.1 HGB: `models/residual_ml/sweep_best_phase61_grid_dc/` (`model.pkl` md5 `004bd82d75f2a29c6b579eb672c5bf57`)
- Grid-ρ DC params: `config/classic_dc_league_params_grid_bck.json`
- Source of live HGB: `models/residual_ml/dc_rho_mle_retrain/` (left in place; `sweep_best/model.pkl` md5 `854efeba53920eeb571c5395c2ea932f`)

---

## Task contract

### Goal

Make live scoring and official eval the same MLE-ρ system. Stop quoting Phase 6.1 1.0022 as the current ship number. Canonical 518-row number is the pinned MLE-aligned HGB: **1.0046** (α=0.7).

### Required behavior

1. Canonical DC = MLE ρ. Keep `CLASSIC_DC_FIT_RHO` default on. Do not revert league params to grid.
2. Canonical residual = pinned MLE HGB (`dc_rho_mle_retrain/` → `sweep_best/`), after backing up Phase 6.1. Live shrink α **0.7**.
3. Canonical eval dataset: `artifacts/dc_rho_mle_promotion/dataset.csv` (2589 rows). Do not rebuild.
4. Harden optimizer so a default re-run cannot overwrite MLE JSON with grid ρ (`fit_rho: true` in production grid; CLI default from config/grid, not silent `store_true` off).
5. Docs: production_profile, project_status, DEC-014, wiki + probability_calculations current-α, this report.
6. Tests only for optimizer/`fit_rho` plumbing. No 20h rebuild.

### Out of scope

Fail-closed classic→strength fallback; ML/shrink-by-year eval reporting; league-gated blend; another HGB sweep or dataset rebuild; git commit/push; promoting the MLE sweep model (1.0063).

### Business invariants

INV-003 (1X2 normalize); no leakage changes; BR-003 / feature cutoffs unchanged; dual baseline remains market + **classic** DC.

### Acceptance criteria

All met (see verifier). Official gate artifact: [`artifacts/dc_rho_mle_promotion/gate_eval_retrain.json`](../../artifacts/dc_rho_mle_promotion/gate_eval_retrain.json).

---

## What changed

| Area | Change |
|------|--------|
| Live HGB | `sweep_best/` is now `dc_rho_mle_retrain` (schema version `dc_rho_mle_retrain`). Phase 6.1 copy at `sweep_best_phase61_grid_dc/`. |
| Shrink | `RESIDUAL_ML_FINAL_SHRINK_TO_MARKET` / `DataSourceConfig` default **0.7**. |
| Optimizer | `config/classic_dc_optimization_grid.json` `"fit_rho": true`. CLI uses `BooleanOptionalAction` + `resolve_optimize_fit_rho` (config default **or** grid key; `--no-fit-rho` wins). |
| DC params | Unchanged in this freeze — already MLE ρ (5 leagues). Grid file remains backup-only. |
| Docs | `production_profile.md`, `project_status.md`, **DEC-014**, wiki + `probability_calculations.md` current α. |
| Tests | `tests/test_calc/test_optimize_classic_dixon_coles.py`; production-grid `fit_rho` assert in `test_classic_dc_config.py`. |

Live DC ρ (unchanged this freeze): 39 −0.0145, 41 +0.0063, 42 −0.0172, 45 −0.1406, 180 −0.1086.

---

## Current official gate (518-row include-holdout)

From `gate_eval_retrain.json`:

| Slice | Market | Blend | ML raw | Best shrink |
|-------|--------|-------|--------|-------------|
| 518 include-holdout | 1.0068 | 1.0083 | 1.0149 | **1.0046** (α=0.7) |

Eval CSV: `artifacts/dc_rho_mle_promotion/dataset.csv`. No holdout-only 4951–4960 number was re-measured on this HGB.

---

## Review

- **Project Leader:** APPROVED after inspecting diff, model checksums, docs, and optimizer plumbing.
- **Verifier:** VERIFIED. Required pytest: **60 passed**. No blocking issues.
- Rework loops used: **0**.

### Non-blocking notes

- Code default `RESIDUAL_ML_MODEL_PATH` is still `models/residual_ml/v1/model.pkl`. Live still requires the documented override to `sweep_best/model.pkl`.
- Historical Recent work entries (15:30 / 16:00 / 17:30) still say “HGB remains held” — those are the earlier-day log, not current ship.
