# Project status

Living snapshot as of **Phase 6.1 ship**. Ops detail: [`production_profile.md`](production_profile.md). Term definitions (slice, shrink, blend, …): [`wiki.md`](wiki.md).

## Goal

Pooled include-holdout log loss **≤ 0.99** on the official **518-row** gate (`--include-holdout`, 20% time-split validation).

## Current ship

- Model: Phase 6.1 restore in `models/residual_ml/sweep_best/`
- Shrink α=**0.5**; conditional blend and draw adjustment **off**
- Best shrink on official gate: **~1.0022** — gate **MISSED** (still short of ≤0.99)
- Full profile: [`production_profile.md`](production_profile.md); restore write-up: [`reports/ml_residual/phase6_restore_phase1.md`](reports/ml_residual/phase6_restore_phase1.md)

## What is off (and why)

| Component | Status | Why / where |
|-----------|--------|-------------|
| Injuries v1 | **FAIL** (Phase 2 exit) | No pooled + injury-heavy gain; columns kept for monitoring, not in production HGB — [`reports/injury/injury_phase2_results.md`](reports/injury/injury_phase2_results.md) |
| Draw adjustment | **FAIL** (OOS) | Discovery retained; `config/draw_adjustment.json` `enabled: false` — [`reports/ml_draw/draw_driver_analysis.md`](reports/ml_draw/draw_driver_analysis.md) |
| Market-anchored 6.2 | **FAIL** Gate B | Official 518 best shrink worse than Phase 1 / 6.1 — [`reports/ml_residual/phase6_market_anchored.md`](reports/ml_residual/phase6_market_anchored.md) |

## Official gate reminder

Do **not** treat tuning-only or holdout-only LL as the ship metric. Ship decisions use the **518-row pooled include-holdout** slice. Holdout 4951–4960 alone can look ≤0.99 (market ≈0.98) without clearing the primary gate.

## Reports index

| Area | Path | Key files |
|------|------|-----------|
| Injuries | [`reports/injury/`](reports/injury/) | `injury_ablation_baseline.md`, `injury_phase2_results.md` |
| Draw ML / adjustment | [`reports/ml_draw/`](reports/ml_draw/) | `draw_driver_analysis.md`, `draw_formula_report.txt` |
| Residual ML / phases | [`reports/ml_residual/`](reports/ml_residual/) | `baseline_after_dc_tune.md`, `phase6_restore_phase1.md`, `phase6_market_anchored.md`, `log_loss_0.99_roadmap.md` |

## Next ideas (non-binding)

Backlog only — **not** production:

- Injuries v2 (counts-only / coverage fixes)
- Draw adjustment v2 (stricter OOS / fewer terms)

Prefer [`production_profile.md`](production_profile.md) for anything that affects live scoring.

## Recent work

### 2026-09-04 00:06 — Modeling & eval wiki

**APPROVED.** Added [`wiki.md`](wiki.md) (slice, HGB residual, logit, shrink/blend, Brier) with pointers from AGENTS / DOMAIN / ARCHITECTURE; fixed stale production α in `product/probability_calculations.md` to **0.5**. Report: [`reports/2026-09-04-modeling-eval-wiki.md`](reports/2026-09-04-modeling-eval-wiki.md).
