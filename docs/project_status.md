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
| Injuries v2 (T4–T2 wirings) | **KILL** | Official 518 shrink@0.7 T4 **1.0049**, T1 **1.0048**, T3 **1.0047**, T2 **1.0057** vs ship **1.0046**; production still off — [`reports/2026-09-05-injury-wiring-trials.md`](reports/2026-09-05-injury-wiring-trials.md). Follow-up review: **STOP** until XI/starter, fetch-time snapshot, or odds timestamp — [`reports/2026-09-05-injury-calculation-improvements.md`](reports/2026-09-05-injury-calculation-improvements.md) |

## Official gate reminder

Do **not** treat tuning-only or holdout-only LL as the ship metric. Ship decisions use the **518-row pooled include-holdout** slice. Holdout 4951–4960 alone can look ≤0.99 (market ≈0.98) without clearing the primary gate.

## Reports index

| Area | Path | Key files |
|------|------|-----------|
| Injuries | [`reports/injury/`](reports/injury/) | `injury_ablation_baseline.md`, `injury_phase2_results.md` |
| Injury wiring trials | [`plans/2026-09-05-injury-wiring-trials.md`](plans/2026-09-05-injury-wiring-trials.md) | T4–T2 run; all **KILL**; production still off — [`reports/2026-09-05-injury-wiring-trials.md`](reports/2026-09-05-injury-wiring-trials.md) |
| Injury calculation review | [`plans/2026-09-05-injury-calculation-improvements.md`](plans/2026-09-05-injury-calculation-improvements.md) | **STOP** until data quality; I1/I2 documented, not run — [`reports/2026-09-05-injury-calculation-improvements.md`](reports/2026-09-05-injury-calculation-improvements.md) |
| Draw ML / adjustment | [`reports/ml_draw/`](reports/ml_draw/) | `draw_driver_analysis.md`, `draw_formula_report.txt` |
| Residual ML / phases | [`reports/ml_residual/`](reports/ml_residual/) | `baseline_after_dc_tune.md`, `phase6_restore_phase1.md`, `phase6_market_anchored.md`, `log_loss_0.99_roadmap.md`, `dc_rho_mle.md`, `dc_rho_mle_promotion_contract.md` |
| MLE-ρ freeze | [`reports/2026-09-04-dc-mle-rho-freeze.md`](reports/2026-09-04-dc-mle-rho-freeze.md) | Canonical DC + pinned HGB; official gate **1.0046** |
| Fail-closed DC + slice ML | [`reports/2026-09-04-dc-failclosed-and-slice-ml.md`](reports/2026-09-04-dc-failclosed-and-slice-ml.md) | Classic miss → market only; ship edge mixed (2025 + league 39) |
| League-gated blend | [`reports/2026-09-04-league-gated-blend.md`](reports/2026-09-04-league-gated-blend.md) | 492 allowlist 39/45/180; official 518 **KILL** (league 39); **not promoted** |
| Recency-weighted HGB | [`reports/2026-09-05-recency-weighted-hgb.md`](reports/2026-09-05-recency-weighted-hgb.md) | Half-life 180 on MLE CSV; official 518 **KILL** (1.0088); **not promoted** |
| PL vs Allsvenskan flip | [`reports/2026-09-05-pl-vs-allsvenskan-flip.md`](reports/2026-09-05-pl-vs-allsvenskan-flip.md) | Diagnostic; 180 flip is slice arithmetic (518 ⊂ 492; holdout n=0); **no Allsvenskan HGB** |
| DC scoreline leftover | [`reports/2026-09-05-dc-scoreline-leftover.md`](reports/2026-09-05-dc-scoreline-leftover.md) | Diagnostic; leftover exists at locked λ totals but does **not** line up with the env split; **stop named regimes**; production unchanged |
| 100% DC on official 518 | [`reports/2026-09-06-dc-only-518.md`](reports/2026-09-06-dc-only-518.md) | Eval-only; 100% DC **1.0471** (n=465); 53 classic miss; production unchanged |
| Dixon–Coles NBM library | [`reports/2026-09-06-dixon-coles-nbm.md`](reports/2026-09-06-dixon-coles-nbm.md) | New `src/calc/dixon_coles_nbm/`; classic DC untouched; **not** wired to production |
| Sarmanov–NB library | [`reports/2026-09-06-sarmanov-nb.md`](reports/2026-09-06-sarmanov-nb.md) | New `src/calc/sarmanov_nb/`; Michels four-cell Sarmanov×NB2; classic DC + ad-hoc NBM untouched; **not** wired to production |
| 100% NBM on official 518 | [`reports/2026-09-06-dc-nbm-only-518.md`](reports/2026-09-06-dc-nbm-only-518.md) | Eval-only; 100% NBM **1.0143** (n=269); market **0.9908** same rows; production unchanged |
| 100% NBM min_team_matches=3 | [`reports/2026-09-06-dc-nbm-only-518-min3.md`](reports/2026-09-06-dc-nbm-only-518-min3.md) | Eval-only; 100% NBM **1.0322** (n=278) at min=3; vs prior min=5 **1.0143**/269; production default still 5 |
| 100% Sarmanov–NB on official 518 | [`reports/2026-09-06-sarmanov-nb-only-518.md`](reports/2026-09-06-sarmanov-nb-only-518.md) | Eval-only; 100% Sarmanov **1.0173** (n=269); market **0.9908** same rows; vs NBM min5 **1.0143**/269; production unchanged |
| Stryktipset coupon optimizer | [`reports/2026-09-06-stryktipset-coupon-optimizer.md`](reports/2026-09-06-stryktipset-coupon-optimizer.md) | Market+streckprocent EV optimizer; MC + chronological tuner; **not** wired to ship path |
| Stryktipset optimizer OOS backtest | [`reports/2026-09-06-stryktipset-optimizer-backtest.md`](reports/2026-09-06-stryktipset-optimizer-backtest.md) | 195 DB coupons 4760–4966; primary β=1/λ=0/C=100 lev **3.66**; hit-rate diag β=0.5; leakage UNKNOWN |
| Stryktipset optimizer OOS rows=128 | [`reports/2026-09-06-stryktipset-optimizer-backtest-rows128.md`](reports/2026-09-06-stryktipset-optimizer-backtest-rows128.md) | Same 195 coupons; row_count=128, C=500 only; primary lev **3.444**; hit-diag mean correct **8.48** |

## Next ideas (non-binding)

Backlog only — **not** production:

- Injuries calculation review **STOP** until data quality; I1/I2 optional only — [`plans/2026-09-05-injury-calculation-improvements.md`](plans/2026-09-05-injury-calculation-improvements.md)
- Draw adjustment v2 (stricter OOS / fewer terms)
- Stretch pooled include-holdout ≤0.99 remains **MISSED** (current ship **1.0046**, α=0.7). The 81-combo sweep on MLE data (**1.0063**, α=0.8) was worse than the pinned retrain and is not the ship.

Prefer [`production_profile.md`](production_profile.md) for anything that affects live scoring.

## Recent work

### 2026-09-06 23:45 — Stryktipset optimizer OOS backtest row_count=128

Eval-only: same **195** DB coupons (4760–4966), **row_count=128**, reduced 9-cell grid β×λ × **C=500** (dropped C=100 — cannot select 128 rows; C=2000 omitted for runtime). ~**152 min**. Primary (leverage): **β=1.0, λ=0.0, C=500** → lev **3.444**, mean best-correct **5.76**, ≥10=2. Hit-rate diagnostic: **β=0.5, λ=1.0, C=500** → mean best-correct **8.48** (≥12/11/10 = 5/28/63). vs row=3: lev 3.655→3.444; hit best-correct 6.79→8.48. Leakage UNKNOWN; production untouched. Artifact: `artifacts/stryktipset_optimizer_backtest/backtest_rich_rows128.json`. Report: [`reports/2026-09-06-stryktipset-optimizer-backtest-rows128.md`](reports/2026-09-06-stryktipset-optimizer-backtest-rows128.md).

### 2026-09-06 19:50 — Stryktipset optimizer chronological OOS backtest

Eval-only: **195** settled DB coupons (draws **4760–4966**), default 18-point grid, `row_count=3`, ~37 min. Primary (mean portfolio leverage): **β=1.0, λ=0.0, C=100** → lev **3.655**, mean best-correct **3.88**, zero ≥10. Hit-rate diagnostic: **β=0.5, λ=1.0, C=500** → mean correct **6.79** (≥12/11/10 = 1/2/11). Market/public 1-row baselines ~6.45/6.51. Leakage UNKNOWN; ship path untouched. Artifact: `artifacts/stryktipset_optimizer_backtest/backtest_rich.json`. Report: [`reports/2026-09-06-stryktipset-optimizer-backtest.md`](reports/2026-09-06-stryktipset-optimizer-backtest.md).

### 2026-09-06 11:20 — Stryktipset market+streckprocent coupon optimizer (APPROVED)

**APPROVED** (PL + verifier + model review). Standalone `src/calc/stryktipset_optimizer/`: market odds as truth; EV vs streckprocent dilution; top-C search + Hamming portfolio; MC relative tiers; chronological OOS tuner with β-comparable `best_by_mean_portfolio_leverage`; public 1% scale fixed (any value `>1` ⇒ 0–100%). Tests **51 passed**. Ship gate / ProbabilityManager / residual ML / Dixon–Coles **unchanged**. Earlier pending/rework notes below superseded. Plan: [`plans/2026-09-06-stryktipset-coupon-optimizer.md`](plans/2026-09-06-stryktipset-coupon-optimizer.md). Report: [`reports/2026-09-06-stryktipset-coupon-optimizer.md`](reports/2026-09-06-stryktipset-coupon-optimizer.md).

### 2026-09-06 11:13 — Stryktipset optimizer rework iter3 (public 1% scale)

Blocking fix: `normalize_public_shares` treats any value `> 1` as 0–100% (ST ints `1` = 1%). Unit scale only when all ∈ `[0, 1]`. MC tiers relative to N. Favorite-count test keys verified (`"22"`/`"12"`/`"11"`). Pending PL/verifier — not APPROVED.

### 2026-09-06 11:10 — Stryktipset optimizer rework iter2 (cross-β ranking)

Blocking math fix: primary OOS selector is now `best_by_mean_portfolio_leverage` (`Σ log(Pm/Pp)` of selected rows; ≡ row_score at β=1). Construction `mean_top_row_score` kept as diagnostic only — not comparable across β. Test asserts best ≠ trivial max(β) under score inflation. Pending PL/verifier.

### 2026-09-06 11:05 — Stryktipset coupon optimizer rework (pending PL/verifier)

Rework fixes (not PL-approved): (1) `favorite_count` = selections equal to market argmax Pm (ties: OUTCOMES order); added `home_count` for homes; (2) backtest ranking moved off mean_correct (superseded 11:10 by portfolio leverage); realized row_score on settled rounds kept as metric; mean_correct/tiers diagnostic only; (3) MC `public_row_count` default **1000**; (4) docs wording / status honesty. Production scoring path still untouched. Report: [`reports/2026-09-06-stryktipset-coupon-optimizer.md`](reports/2026-09-06-stryktipset-coupon-optimizer.md).

### 2026-09-06 10:55 — Stryktipset market+streckprocent coupon optimizer (implemented; pending PL/verifier)

Implemented (rework applied 11:05). New `src/calc/stryktipset_optimizer/`: fair market Pm + public Pp row scoring, top-C exhaustive search, Hamming-diversified portfolio, banker/JS diagnostics, MC relative tiers, chronological OOS tuner. CLI `optimize_stryktipset_coupon` / `backtest_stryktipset_optimizer`. N=13 microbench candidate_count=500 **0.42s**, 50k **1.31s**. Leakage UNKNOWN documented (no odds/public timestamps; regCloseTime not on STRound). ProbabilityManager / residual ML / Dixon–Coles / ship gate **unchanged**. Plan: [`plans/2026-09-06-stryktipset-coupon-optimizer.md`](plans/2026-09-06-stryktipset-coupon-optimizer.md). Report: [`reports/2026-09-06-stryktipset-coupon-optimizer.md`](reports/2026-09-06-stryktipset-coupon-optimizer.md).

### 2026-09-06 10:34 — 100% Sarmanov–NB on official 518 (eval-only)

Diagnostic only. Official include-holdout 518 from the existing CSV (`select_backtest_rows`, no draw cap). Walk-forward `SarmanovNBModel` (ρ fitted, φ fitted; seed ρ=−0.13 / φ=0.05; no CSV-λ). **100% Sarmanov–NB** log-loss **1.0173** on 269 scored rows; 249 skipped (189 unknown_team, 56 missing_league, 4 fit_fail — same coverage as NBM min5). Same 269: market **0.9908**. vs cited NBM min5 **1.0143** (same n; slightly worse), classic 100% DC **1.0471** (n=465), and ship shrink@0.7 **1.0046** (n=518) is apples-to-oranges on ship/classic n. Intersection of the 269 with classic-valid-DC: Sarmanov **1.0173**, CSV classic **1.0171**, market **0.9908**. Fail-closed Sarmanov-or-market on 518 is **1.0206** — **not** 100% Sarmanov. Production HGB, blend 70/30, shrink α=0.7, classic DC, and the canonical CSV **unchanged**. Sarmanov not wired. Report: [`reports/2026-09-06-sarmanov-nb-only-518.md`](reports/2026-09-06-sarmanov-nb-only-518.md). Artifact: [`artifacts/sarmanov_nb_only_518.json`](../artifacts/sarmanov_nb_only_518.json).

### 2026-09-06 10:15 — Sarmanov–NB library (not production)

**APPROVED.** New `src/calc/sarmanov_nb/`: NB2 marginals × Michels/Karlis four-cell Sarmanov mixer (\(a=\lambda/(1+\varphi\lambda)\)). Classic `dixon_coles/` and ad-hoc `dixon_coles_nbm/` unmodified. Not wired into live scoring, blend, or HGB. Tests **90 passed**. Official ship **1.0046** / 70/30 / α=0.7 **unchanged**. Proper NB joint vs ad-hoc Poisson τ on NB. Report: [`reports/2026-09-06-sarmanov-nb.md`](reports/2026-09-06-sarmanov-nb.md).

### 2026-09-06 09:02 — 100% NBM min_team_matches=3 on official 518 (eval-only)

Diagnostic only. Same official 518; CLI `--min-team-matches 3` (config default still **5**). Walk-forward NBM (ρ=−0.13 fixed, φ fitted; no CSV-λ). **100% NBM** log-loss **1.0322** on 278 scored (+9 vs prior min=5 n=269); 240 skipped (180 unknown_team vs prior 189; 56 missing_league; 4 fit_fail). Same 278: market **0.9986**. vs prior min=5 NBM **1.0143** (worse LL despite more coverage). vs cited classic 100% DC **1.0471** (n=465) and ship **1.0046** (n=518) is apples-to-oranges. Production HGB, blend 70/30, shrink α=0.7, classic DC default min_team_matches=5, and CSV **unchanged**. NBM not wired. Report: [`reports/2026-09-06-dc-nbm-only-518-min3.md`](reports/2026-09-06-dc-nbm-only-518-min3.md). Artifact: [`artifacts/dc_nbm_only_518_min3.json`](../artifacts/dc_nbm_only_518_min3.json).

### 2026-09-06 08:23 — 100% NBM on official 518 (eval-only)

Diagnostic only. Official include-holdout 518 from the existing CSV (`select_backtest_rows`, no draw cap). Walk-forward `DixonColesNBMModel` (ρ=−0.13 fixed, φ fitted; no CSV-λ). **100% NBM** log-loss **1.0143** on 269 scored rows; 249 skipped (189 unknown_team, 56 missing_league, 4 fit_fail). Same 269: market **0.9908**. vs cited classic 100% DC **1.0471** (n=465) and ship shrink@0.7 **1.0046** (n=518) is apples-to-oranges. Intersection of the 269 with classic-valid-DC: NBM **1.0143**, CSV classic **1.0171**, market **0.9908**. Fail-closed NBM-or-market on 518 is **1.0190** — **not** 100% NBM. Production HGB, blend 70/30, shrink α=0.7, classic DC, and the canonical CSV **unchanged**. NBM not wired. Report: [`reports/2026-09-06-dc-nbm-only-518.md`](reports/2026-09-06-dc-nbm-only-518.md).

### 2026-09-06 08:15 — Dixon–Coles NBM library (not production)

**APPROVED.** New `src/calc/dixon_coles_nbm/`: NB2 marginals (one global φ) × classic four-cell τ. Classic `src/calc/dixon_coles/` unmodified. Not wired into live scoring, blend, or HGB. Tests **53 passed**. Official ship **1.0046** / 70/30 / α=0.7 **unchanged**. Not a leftover fix (NB raises P(0-0) vs Poisson). Report: [`reports/2026-09-06-dixon-coles-nbm.md`](reports/2026-09-06-dixon-coles-nbm.md).

### 2026-09-06 07:51 — 100% DC on official 518 (eval-only)

Diagnostic only. Official include-holdout 518 from the existing CSV (`select_backtest_rows`, no draw cap). **100% DC** log-loss **1.0471** on the 465 valid-DC rows; 53 classic misses (no invented DC). Same 465: market **1.0145**, blend **1.0162**. Fail-closed DC-or-market on the full 518 is **1.0361** — **not** 100% DC. 100% DC is worse than cited ship shrink@0.7 **1.0046**, and the comparison is apples-to-oranges (n=465 ≠ 518). Production HGB, blend 70/30, shrink α=0.7, and the canonical CSV **unchanged**. Report: [`reports/2026-09-06-dc-only-518.md`](reports/2026-09-06-dc-only-518.md).

### 2026-09-05 20:15 — DC scoreline leftover diagnostic

Diagnostic only. Joined ST FT scores (2589/2589; fixture fallback unused). Valid-λ n=2295 matches pre-registration. Leftover persists at low/mid λ (DC over-predicts 0-0); mid env persist is one-sided and realized 0-0 is inverted vs low-scoring-rate. **Answer:** Same totals, leftover 0-0/1-1 exists but does not line up with the locked environment split — stop named regimes. Official ship 1.0046 / α=0.7 / 70/30 and production files **unchanged**. Report: [`reports/2026-09-05-dc-scoreline-leftover.md`](reports/2026-09-05-dc-scoreline-leftover.md).

### 2026-09-05 15:20 — Injury calculation review (STOP, plan only)

**APPROVED** review. T4–T2 math was too blunt (squad≠XI, T2 four-factor, T3 on |d|=1, T4 shrank uncovered). Recommendation: **stop** injury 1X2 work until XI/starter, fetch-time snapshot, or odds timestamp. Two optional repaired formulas (I1 gated T3, I2 one-factor λ) documented, not run. Production, CSV, HGB, blend 70/30, shrink α=0.7 **unchanged**. Plan: [`plans/2026-09-05-injury-calculation-improvements.md`](plans/2026-09-05-injury-calculation-improvements.md). Report: [`reports/2026-09-05-injury-calculation-improvements.md`](reports/2026-09-05-injury-calculation-improvements.md).

### 2026-09-05 14:20 — Injury wiring trials (all KILL, not promoted)

**APPROVED** protocol. Preflight locked injury-slice n=191. T4 **1.0049**, T1 **1.0048**, T3 **1.0047** (blend 1.0085), T2 **1.0057** vs ship **1.0046** — all **KILL**. Production HGB, blend 70/30, shrink α=0.7, and canonical CSV **unchanged**. Injuries stay off. Report: [`reports/2026-09-05-injury-wiring-trials.md`](reports/2026-09-05-injury-wiring-trials.md).

### 2026-09-05 — Injury wiring trials plan (not implemented)

A plan was written; injuries were **not** implemented. Four later sittings (T4 shrink, T1 counts-only HGB, T3 post-DC logit, T2 λ shock). Production HGB, blend, and CSV **unchanged**. Plan: [`plans/2026-09-05-injury-wiring-trials.md`](plans/2026-09-05-injury-wiring-trials.md). Report: [`reports/2026-09-05-injury-wiring-trials-plan.md`](reports/2026-09-05-injury-wiring-trials-plan.md).

### 2026-09-05 11:35 — PL vs Allsvenskan flip diagnostic

Diagnostic, not a train. **Cause:** slice arithmetic — official 518 Allsvenskan is a subset of tuning 492 (158 of 201); holdout/518−492 have zero league-180 rows; the 492 win is 43 early-2025 rows that 518 drops. **Decision: no Allsvenskan HGB.** Report: [`reports/2026-09-05-pl-vs-allsvenskan-flip.md`](reports/2026-09-05-pl-vs-allsvenskan-flip.md). Production, CSV, HGB, blend 70/30, shrink α=0.7 **unchanged**.

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
