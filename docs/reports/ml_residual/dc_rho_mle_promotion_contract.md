# Task contract: DC ρ MLE — promote vs hold

**Status:** **PROMOTED (partial)** — 2026-09-04. DC params + MLE ρ default shipped; HGB held due to gate regression (1.0046 > 1.0022). Full report: [`2026-09-04-dc-rho-mle-promotion.md`](../2026-09-04-dc-rho-mle-promotion.md).

**Related:** [`dc_rho_mle.md`](dc_rho_mle.md) (estimator validation), [`artifacts/dc_rho_mle_comparison.json`](../../../artifacts/dc_rho_mle_comparison.json), [`project_status.md`](../../project_status.md), [`production_profile.md`](../../production_profile.md).

---

## Decision question

Should **MLE-fitted Dixon–Coles ρ** replace **grid-selected ρ** in the production scoring path?

| Path | Meaning |
|------|---------|
| **HOLD** | Keep today’s ship: grid ρ from `config/classic_dc_league_params.json`, `CLASSIC_DC_FIT_RHO=0`. No DC/HGB rebuild. |
| **PROMOTE** | Enable `CLASSIC_DC_FIT_RHO=1`, refresh DC inputs, retrain HGB, pass the promotion gate below, then update production profile + params JSON. |

---

## Current ship (unchanged)

| Item | Value |
|------|--------|
| Residual ML | Phase 6.1 restore — `models/residual_ml/sweep_best/model.pkl` |
| Shrink α | **0.5** |
| Blend / draw | **off** (effective 70/30 market/DC) |
| Official gate (518-row include-holdout) | best shrink **~1.0022** — target ≤0.99 **MISSED** |
| DC params | `config/classic_dc_league_params.json` — **grid-selected ρ** (4 leagues in file; live also uses global fallback for others) |
| MLE ρ | Implemented, **`CLASSIC_DC_FIT_RHO=0`** (off) |

---

## Evidence summary (why this decision exists)

**Estimator:** MLE ρ recovers synthetic truth (true −0.15 → −0.152 at n=16,800); **0/87** fits at bounds vs **3/5** grid leagues pinned to ρ grid edge.

**Head-to-head (tuning slice only, holdout excluded):** grid wins pooled DC LL **1.0311** vs MLE **1.0416**, but grid ρ **disagrees in sign with MLE in 4/5 leagues** — grid ρ is treated as selection noise, not a league effect.

**Upside cap:** DC is **30%** of the blend; pooled DC LL ≈1.03 vs market ≈1.007. Expect **small** movement on the 518-row final metric, not a path to ≤0.99 by itself.

**Comparison runs already done** — no need to rerun grid vs MLE unless DB draw window or protocol changes.

---

## Recommendation (project-leader)

**HOLD for production today.** Adopt MLE ρ as the **estimator of record** for future DC work, but do not flip live scoring without a full rebuild + gate measurement.

**PROMOTE only if** the promotion gate (below) shows **no regression** vs 1.0022 and product accepts the operational cost of a DC dataset rebuild + HGB retrain for a likely small LL delta.

---

## Path A — HOLD (default)

### Scope

- Leave `config/classic_dc_league_params.json` as-is.
- Leave `CLASSIC_DC_FIT_RHO=0`.
- No dataset rebuild, no HGB retrain, no production profile change.

### Acceptance (automatic — already true)

- [ ] Official gate remains **~1.0022** on Phase 6.1 artifacts.
- [ ] `docs/production_profile.md` unchanged for DC section.

### When to revisit

- Evaluation protocol fixed (disjoint tuning vs gate slices).
- ξ/lookback grid replaced with less noisy selection.
- DC league coverage extended (7 leagues currently skipped in optimizer).
- A larger modeling change forces a DC rebuild anyway.

---

## Path B — PROMOTE (phased)

Execute in order. **Stop and HOLD** if any phase fails its exit criteria.

### Phase B0 — Sign-off (human)

- [ ] Product confirms PROMOTE path despite expected **small** gate effect.
- [ ] Baseline frozen: record md5 of `config/classic_dc_league_params.json`, `models/residual_ml/sweep_best/`, and `artifacts/baseline_post_dc_tune_20250902/dataset.csv` (or current canonical dataset path).

### Phase B1 — Refresh ξ / lookback under MLE ρ (optional but recommended)

**Goal:** Pick ξ and lookback when ρ is MLE’d inside each walk-forward fit (35 combos/league, not 210).

**Command** (do **not** write to production JSON yet):

```bash
python -m src.scripts.optimize_classic_dixon_coles --fit-rho --jobs 4 \
  --output artifacts/dc_rho_mle/params_promotion_candidate.json
```

**Exit criteria:**

- [ ] Run completes; 5+ leagues scored (same protocol as comparison: `--draw-max=4950`, validation fraction 0.20).
- [ ] Per-league `rho_fitted` diagnostics: **at_bound fraction < 10%** (comparison had **0/87**).
- [ ] Output JSON saved; **production** `config/classic_dc_league_params.json` still untouched.

**Skip allowed:** Use existing `artifacts/dc_rho_mle/params_arm_b_fitted_rho.json` if no DB/protocol change since 2026-09-04 comparison run.

### Phase B2 — Rebuild residual ML dataset with MLE ρ

**Goal:** Regenerate `p_home_dc`, `p_draw_dc`, `p_away_dc` (and dependent blend columns) using `CLASSIC_DC_FIT_RHO=1`.

**Env / config:**

```bash
export CLASSIC_DC_FIT_RHO=1
# Optional: point DC params at promotion candidate
# export CLASSIC_DC_LEAGUE_PARAMS_PATH=artifacts/dc_rho_mle/params_promotion_candidate.json
```

**Command** (canonical pipeline — verify flags against `build_residual_ml_dataset.py --help`):

```bash
python -m src.scripts.build_residual_ml_dataset.py
```

**Exit criteria:**

- [ ] New dataset written (suggest snapshot path, e.g. `artifacts/dc_rho_mle_promotion/dataset.csv`).
- [ ] Row count and date range match prior canonical dataset (no accidental draw-window change).
- [ ] Spot-check: DC probabilities still normalize; no NaNs on scored rows.
- [ ] Leakage invariants unchanged (`production_cutoff_alignment.md`).

### Phase B3 — Retrain HGB (pinned Phase 6.1 hyperparams)

**Goal:** Fair comparison — same HGB recipe as Phase 6.1 restore, new DC inputs only.

Use pinned params from [`phase6_restore_phase1.md`](phase6_restore_phase1.md) / [`baseline_after_dc_tune.md`](baseline_after_dc_tune.md):

- `max_depth=4`, `learning_rate=0.03`, `max_iter=100`, `label_smoothing=0.05`
- No injury columns (same feature set as current production HGB)
- Train on **new** dataset from B2

**Command** (adjust paths):

```bash
python -m src.scripts.train_residual_ml \
  --dataset artifacts/dc_rho_mle_promotion/dataset.csv \
  --output-dir models/residual_ml/dc_rho_mle_candidate/ \
  --version dc_rho_mle_candidate \
  --max-depth 4 \
  --learning-rate 0.03 \
  --max-iter 100 \
  --label-smoothing 0.05 \
  --exclude-injury-features
```

**Exit criteria:**

- [ ] `model.pkl` + schema/weights written under candidate dir.
- [ ] Training/val metrics logged for audit (not used as promotion gate).

### Phase B4 — Official gate evaluation

**Goal:** Measure **518-row pooled include-holdout** — the only ship metric.

```bash
python -m src.scripts.eval_residual_ml \
  --dataset artifacts/dc_rho_mle_promotion/dataset.csv \
  --model models/residual_ml/dc_rho_mle_candidate/model.pkl \
  --include-holdout \
  --json artifacts/dc_rho_mle_promotion/gate_eval.json
```

Run shrink α sweep (or at minimum α=0.5 vs best-on-slice) consistent with [`production_profile.md`](../../production_profile.md).

**Promotion gate (all required):**

| Criterion | Threshold |
|-----------|-------------|
| Best shrink on **518-row include-holdout** | **≤ 1.0022** (no regression vs Phase 6.1) |
| Improvement vs 1.0022 | **Nice-to-have**, not required for “no regression” promote |
| Reach ≤0.99 | **Not expected** — do not block HOLD if missed |
| Tuning-only or holdout-only LL | **Do not** use as promote/ship metric |

**Outcomes:**

| B4 result | Action |
|-----------|--------|
| Best shrink **≤ 1.0022** | Proceed to B5 (ship) |
| Best shrink **> 1.0022** | **HOLD** — keep Phase 6.1; archive candidate artifacts |
| Best shrink **< 1.0022** (improvement) | Proceed to B5; document delta in `production_profile.md` |

### Phase B5 — Ship (only if B4 passed)

- [ ] Copy candidate model → `models/residual_ml/sweep_best/` (backup current to dated folder first).
- [ ] Set `CLASSIC_DC_FIT_RHO=1` in production env / document in `production_profile.md`.
- [ ] Replace or supplement `config/classic_dc_league_params.json` with promotion candidate (ξ/lookback + median fitted ρ metadata).
- [ ] Update `docs/project_status.md` and `docs/production_profile.md`.
- [ ] Re-run `eval_residual_ml` once on shipped paths to confirm reproducibility.

**Rollback:** Restore backed-up `sweep_best/`, revert `CLASSIC_DC_FIT_RHO=0`, restore previous league params JSON.

---

## Explicit non-goals (both paths)

- Fixing evaluation **selection contamination** (~492 tuning rows overlap ~95% of 518 gate rows) — note in results, do not claim as part of this contract unless a separate protocol task is opened.
- Re-tuning shrink α, blend, or draw on the gate slice for this promotion.
- Grid-searching ρ again (210 combos) — deprecated approach.
- Committing `.pyc` or unrelated repo changes.

---

## Risks

| Risk | Mitigation |
|------|------------|
| Small or negative gate effect after rebuild | B4 gate; HOLD on regression |
| Only 5 leagues get tuned params; 7 skipped | Document; global fallback still used — separate coverage task |
| ξ/lookback still grid-noisy | B1 uses 35 combos; future work: shrinkage on ξ/lookback |
| Dataset rebuild runtime / DB dependency | Snapshot inputs; run in maintenance window |
| Accidental overwrite of production JSON | Never `--output` to `classic_dc_league_params.json` until B5 |

---

## Artifacts checklist

| Artifact | HOLD | PROMOTE |
|----------|------|---------|
| `artifacts/dc_rho_mle_comparison.json` | exists | exists |
| `artifacts/dc_rho_mle/params_arm_b_fitted_rho.json` | reference | optional B1 input |
| `artifacts/dc_rho_mle_promotion/dataset.csv` | — | B2 |
| `models/residual_ml/dc_rho_mle_candidate/` | — | B3 |
| `artifacts/dc_rho_mle_promotion/gate_eval.json` | — | B4 |

---

## Decision log (fill when chosen)

| Field | Value |
|-------|--------|
| **Decision** | **PROMOTE (partial)** — DC params + fit_rho; HGB held |
| **Date** | 2026-09-04 |
| **Decided by** | User override (explicit Path B) |
| **B4 best shrink (if PROMOTE)** | **1.0046** (α=0.7) vs baseline 1.0022 — **regression** |
| **Notes** | MLE ρ promoted; grid backup at `classic_dc_league_params_grid_bck.json`. HGB remains Phase 6.1 in `sweep_best/`. Candidate archived at `models/residual_ml/dc_rho_mle_candidate/`. |

---

## Quick answer: “Do we need to rerun ρ optimization?”

| Situation | Rerun? |
|-----------|--------|
| Keep current ship | **No** |
| Re-validate grid vs MLE | **No** — already in `dc_rho_mle_comparison.json` |
| PROMOTE path B1 | **Optional once** — `--fit-rho` to refresh ξ/lookback (or reuse `params_arm_b_fitted_rho.json`) |
| After PROMOTE B2+ | ρ is fitted **inside each DC fit** at scoring time; no separate ρ grid run |
