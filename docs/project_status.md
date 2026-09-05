# Project status

Living snapshot as of the **MLE-ρ freeze**. Ops detail: [`production_profile.md`](production_profile.md). Term definitions (slice, shrink, blend, …): [`wiki.md`](wiki.md).

## Goal

Pooled include-holdout log loss **≤ 0.99** on the official **518-row** gate (`--include-holdout`, 20% time-split validation).

## Current ship

- Model: pinned MLE HGB in `models/residual_ml/sweep_best/` (copy of `dc_rho_mle_retrain/`); Phase 6.1 grid-DC backup at `models/residual_ml/sweep_best_phase61_grid_dc/`
- Shrink α=**0.7**; conditional blend and draw adjustment **off**
- Best shrink on official gate: **1.0046** — gate **MISSED** (still short of ≤0.99)
- DC: **MLE ρ on** — `config/classic_dc_league_params.json`, `CLASSIC_DC_FIT_RHO=1` default
- Full profile: [`production_profile.md`](production_profile.md)

## What is off (and why)

| Component | Status | Why / where |
|-----------|--------|-------------|
| Injuries v1 | **FAIL** (Phase 2 exit) | No pooled + injury-heavy gain; columns kept for monitoring, not in production HGB — [`reports/injury/injury_phase2_results.md`](reports/injury/injury_phase2_results.md) |
| Draw adjustment | **FAIL** (OOS) | Discovery retained; `config/draw_adjustment.json` `enabled: false` — [`reports/ml_draw/draw_driver_analysis.md`](reports/ml_draw/draw_driver_analysis.md) |
| Market-anchored 6.2 | **FAIL** Gate B | Official 518 best shrink worse than Phase 1 / 6.1 — [`reports/ml_residual/phase6_market_anchored.md`](reports/ml_residual/phase6_market_anchored.md) |
| DC ρ by MLE + pinned HGB | **ON** | Live scoring = MLE params + `sweep_best/` retrain, shrink **0.7**, official 518-row gate **1.0046**. Phase 6.1 **1.0022** is historical (grid-DC). — [`reports/2026-09-04-dc-mle-rho-freeze.md`](reports/2026-09-04-dc-mle-rho-freeze.md) |
| Sweep HGB on MLE data | **OFF** | 81-combo sweep gate **1.0063** (α=0.8) worse than pinned retrain; `models/residual_ml/dc_rho_mle_sweep/` not promoted |
| Recency-weighted HGB (half-life 180) | **KILL** | Official 518 shrink@0.7 **1.0088** (also 2026 **1.0310**); candidate kept, not promoted — [`reports/2026-09-05-recency-weighted-hgb.md`](reports/2026-09-05-recency-weighted-hgb.md) |

## Official gate reminder

Do **not** treat tuning-only or holdout-only LL as the ship metric. Ship decisions use the **518-row pooled include-holdout** slice. Holdout 4951–4960 alone can look ≤0.99 (market ≈0.98) without clearing the primary gate.

## Reports index

| Area | Path | Key files |
|------|------|-----------|
| Injuries | [`reports/injury/`](reports/injury/) | `injury_ablation_baseline.md`, `injury_phase2_results.md` |
| Draw ML / adjustment | [`reports/ml_draw/`](reports/ml_draw/) | `draw_driver_analysis.md`, `draw_formula_report.txt` |
| Residual ML / phases | [`reports/ml_residual/`](reports/ml_residual/) | `baseline_after_dc_tune.md`, `phase6_restore_phase1.md`, `phase6_market_anchored.md`, `log_loss_0.99_roadmap.md`, `dc_rho_mle.md`, `dc_rho_mle_promotion_contract.md` |
| MLE-ρ freeze | [`reports/2026-09-04-dc-mle-rho-freeze.md`](reports/2026-09-04-dc-mle-rho-freeze.md) | Canonical DC + pinned HGB; official gate **1.0046** |
| Fail-closed DC + slice ML | [`reports/2026-09-04-dc-failclosed-and-slice-ml.md`](reports/2026-09-04-dc-failclosed-and-slice-ml.md) | Classic miss → market only; ship edge mixed (2025 + league 39) |
| League-gated blend | [`reports/2026-09-04-league-gated-blend.md`](reports/2026-09-04-league-gated-blend.md) | 492 allowlist 39/45/180; official 518 **KILL** (league 39); **not promoted** |
| Recency-weighted HGB | [`reports/2026-09-05-recency-weighted-hgb.md`](reports/2026-09-05-recency-weighted-hgb.md) | Half-life 180 on MLE CSV; official 518 **KILL** (1.0088); **not promoted** |

## Next ideas (non-binding)

Backlog only — **not** production:

- Injuries v2 (counts-only / coverage fixes)
- Draw adjustment v2 (stricter OOS / fewer terms)
- Stretch pooled include-holdout ≤0.99 remains **MISSED** (current ship **1.0046**, α=0.7). The 81-combo sweep on MLE data (**1.0063**, α=0.8) was worse than the pinned retrain and is not the ship.

Prefer [`production_profile.md`](production_profile.md) for anything that affects live scoring.

## Recent work

### 2026-09-05 10:25 — Recency-weighted HGB (KILL, not promoted)

**APPROVED** protocol; gate **KILL**. One pinned HGB on existing MLE CSV with pre-registered train weights (half-life **180** vs max train `match_date`), α=0.7 frozen. Official 518 shrink@0.7 **1.0088** vs ship **1.0046**; 2026 **1.0310** vs **1.0265** (kill >1.0285). League 39 improved (**0.9503** vs **0.9526**) and does not override. Production `sweep_best/` and blend **unchanged**. Candidate kept at `models/residual_ml/recency_180_candidate/`. Tests **236 passed, 2 xfailed**. Report: [`reports/2026-09-05-recency-weighted-hgb.md`](reports/2026-09-05-recency-weighted-hgb.md). Artifact: [`gate_eval_recency_180.json`](../artifacts/dc_rho_mle_promotion/gate_eval_recency_180.json).

### 2026-09-05 00:20 — P0 league-gated blend (KILL, not promoted)

**APPROVED** protocol; gate **KILL**. 492-row allowlist locked first: **39, 45, 180** (CONTINUE because league 41 is a 492 tax). Candidate reblend + one pinned HGB; official 518 shrink@0.7 **1.00450** (HOLD band vs ship **1.0046**) but league 39 **0.9563** vs current **0.9526** (+0.0037, kill line >0.9556). Production `sweep_best/` and `config/blend_weights.json` **unchanged** (still global 70/30). 85/15 / injury / dc-quality stayed off. Tests **230 passed, 2 xfailed**. Report: [`reports/2026-09-04-league-gated-blend.md`](reports/2026-09-04-league-gated-blend.md).

### 2026-09-04 23:10 — Classic DC fail-closed + slice ML/shrink

**APPROVED.** Classic fit/predict miss no longer copies strength DC into `p_*_dc`; blend falls back to market. Official 518-row eval now reports ML raw + shrink (and shrink@0.7) on year and league slices. Pooled ship remains **1.0046** vs market **1.0068**. Current edge is **mixed**: larger per-row in **2025** (−0.0036) and **league 39** (−0.0054); 2026 is most of the rows but a smaller gain (−0.0014); league 41 is flat-to-slightly worse at α=0.7. No α retune, no model promotion, no dataset rebuild. Tests **26 passed**. Report: [`reports/2026-09-04-dc-failclosed-and-slice-ml.md`](reports/2026-09-04-dc-failclosed-and-slice-ml.md). Artifact: [`gate_eval_slices.json`](../artifacts/dc_rho_mle_promotion/gate_eval_slices.json).

### 2026-09-04 21:45 — Freeze on MLE ρ (canonical DC + pinned HGB)

**APPROVED.** Live scoring and official eval are the same MLE-ρ system. `sweep_best/` is the pinned retrain (`dc_rho_mle_retrain`); Phase 6.1 HGB backed up at `models/residual_ml/sweep_best_phase61_grid_dc/`. Shrink default **0.7**. Official 518-row gate **1.0046** (α=0.7); Phase 6.1 **1.0022** is historical. Stretch ≤0.99 still **MISSED**. Optimizer production grid `"fit_rho": true`; bare optimize cannot silently grid-search ρ (**DEC-014**). Sweep HGB 1.0063 not promoted. Tests **60 passed**. Report: [`reports/2026-09-04-dc-mle-rho-freeze.md`](reports/2026-09-04-dc-mle-rho-freeze.md).

### 2026-09-04 17:30 — HGB hyperparameter sweep on MLE dataset

81-combo sweep (`--sweep`) on `artifacts/dc_rho_mle_promotion/dataset.csv`. Best trial by validation LL: `max_depth=4`, `lr=0.03`, `max_iter=100`, **`label_smoothing=0.1`** (Phase 6.1 used 0.05) — raw val LL **1.0137** vs pinned **1.0149**. Official 518-row gate with shrink sweep: **1.0063** (α=0.8) — **worse** than pinned retrain **1.0046** and **+0.0041** vs Phase 6.1 **1.0022**. **HGB remains held.** Model: `models/residual_ml/dc_rho_mle_sweep/model.pkl`. Artifacts: [`sweep_results.json`](../models/residual_ml/dc_rho_mle_sweep/sweep_results.json), [`gate_eval_sweep.json`](../artifacts/dc_rho_mle_promotion/gate_eval_sweep.json).

### 2026-09-04 16:00 — HGB retrain/eval on MLE dataset (no rebuild)

Retrained on existing `artifacts/dc_rho_mle_promotion/dataset.csv` (2589 rows) with Phase 6.1 pinned hyperparams; **no** `build_residual_ml_dataset` rerun. Model: `models/residual_ml/dc_rho_mle_retrain/model.pkl`. Official 518-row gate: best shrink **1.0046** (α=0.7) — **matches** prior promotion candidate; still **+0.0024** vs Phase 6.1 **1.0022**. **HGB remains held.** Artifacts: [`gate_eval_retrain.json`](../artifacts/dc_rho_mle_promotion/gate_eval_retrain.json), [`retrain.log`](../artifacts/dc_rho_mle_promotion/retrain.log).

### 2026-09-04 15:30 — DC ρ MLE promoted (HGB held)

Path B executed per user override. **Promoted:** MLE-fitted ρ + refreshed ξ/lookback in `config/classic_dc_league_params.json` (5 leagues); `classic_dc_fit_rho` default **True**; grid backup at `classic_dc_league_params_grid_bck.json`. **Held:** HGB — official 518-row gate best shrink **1.0046** (α=0.7) vs Phase 6.1 **1.0022** (+0.0024 regression). Production HGB unchanged in `models/residual_ml/sweep_best/`; candidate at `models/residual_ml/dc_rho_mle_candidate/`. Tests **303 passed, 2 xfailed**. Report: [`reports/2026-09-04-dc-rho-mle-promotion.md`](reports/2026-09-04-dc-rho-mle-promotion.md).

### 2026-09-04 12:00 — Dixon–Coles ρ estimated by MLE (not promoted)

ρ can now be estimated by maximum likelihood inside the DC fit instead of grid-searched on 1X2 log loss, behind **`fit_rho` / `CLASSIC_DC_FIT_RHO`, default off**. Production DC params (`config/classic_dc_league_params.json`) are **byte-unchanged**.

Motivation: grid ρ was selection noise. The shipped 4-league config had ρ spanning the whole grid with 2 of 4 on a boundary, chosen as the argmin of **210** combos on as few as **19** matches.

Head-to-head on identical data (5 leagues, 423 scored, tuning slice, holdout excluded):

| | Arm A — grid ρ | Arm B — MLE ρ |
|---|---|---|
| Combos/league | 210 | **35** |
| Pooled DC log loss | **1.0311** | 1.0416 |
| Wall clock (`--jobs 4`) | 74m31s | **12m23s** |
| ρ on a boundary | **3 of 5 leagues** | **0 of 87 fits** |

Arm A's 0.0105 edge is selection optimism — it picked from 6× more candidates on the rows it reports. The decisive diagnostic is that grid ρ and MLE ρ **disagree on the sign in 4 of 5 leagues** (league 180: grid +0.05 vs MLE −0.109 over a [−0.125, −0.076] range, never at a bound), so grid ρ was not measuring low-score dependence. Synthetic recovery confirms the estimator is unbiased (true −0.15 → −0.152 at n=16,800); sd(ρ) ≈ 0.03–0.05 on realistic windows.

Also fixed: `unpack` read home advantage as `theta[-1]`, so appending ρ would have silently made home advantage `exp(ρ)` with no crash. Now indexed explicitly, with a regression test asserting home-advantage recovery.

Tests **303 passed, 2 xfailed**. Report: [`reports/ml_residual/dc_rho_mle.md`](reports/ml_residual/dc_rho_mle.md); data: [`artifacts/dc_rho_mle_comparison.json`](../artifacts/dc_rho_mle_comparison.json).

### 2026-09-04 09:30 — `src/` layout + import root fix (DEC-013)

Application code lives under `src/`, first-party imports are `src.`-prefixed, and the **repo root is the sole path root**. Run everything from the repo root with `python -m` and **no `PYTHONPATH`**:

```bash
python -m pytest tests/
python -m src.scripts.optimize_classic_dixon_coles --help
```

`python src/scripts/X.py` (without `-m`) is **broken** under this layout. Fixed an incomplete conversion where 23 files still did `from database import ...` (needing `src` on the path) while everything else used `src.` (needing the repo root) — the tree only imported with `PYTHONPATH=.:src`, risking a split SQLAlchemy `Base`. Verified: one `src.database` module, one `Base`, 13 tables. Subprocess spawns in the snapshot/ablation scripts now use `python -m` instead of setting `PYTHONPATH`. See **DEC-013** in [`DECISIONS.md`](DECISIONS.md).

### 2026-09-04 00:06 — Modeling & eval wiki

**APPROVED.** Added [`wiki.md`](wiki.md) (slice, HGB residual, logit, shrink/blend, Brier) with pointers from AGENTS / DOMAIN / ARCHITECTURE; fixed stale production α in `product/probability_calculations.md` to **0.5**. Report: [`reports/2026-09-04-modeling-eval-wiki.md`](reports/2026-09-04-modeling-eval-wiki.md).
