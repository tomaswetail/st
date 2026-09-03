# Modeling & evaluation wiki

Glossary for **modeling and evaluation jargon** used in residual ML, backtests, and ship gates.

Entity and coupon terminology (ST Match vs Fixture, draw_number, etc.) lives in [`DOMAIN.md`](DOMAIN.md) — this wiki does **not** duplicate that content.

Formulas and pipeline detail: [`product/probability_calculations.md`](product/probability_calculations.md).  
Live config: [`production_profile.md`](production_profile.md). Gate status: [`project_status.md`](project_status.md).

---

## Slice

**Definition:** An evaluation subset of historical match/coupon rows on which metrics (log loss, Brier, etc.) are computed.

**Not the same as:** The full dataset, a training set, or a production live round.

**How used here:**

| Kind | Meaning |
|------|---------|
| **Time-split validation** | Last 20% of rows by `match_date` (`DEFAULT_VALIDATION_FRACTION = 0.20` in `src/utils/time_split.py`) |
| **Exclude-holdout** (default) | Time-split val **excluding** holdout draws (default eval/backtest) |
| **Include-holdout** | Same time-split val **including** holdout draws (`--include-holdout`). Official ship gate: **518 rows** |
| **Holdout-only** | Draws **4951–4960** only (~129 rows). Useful for sanity checks — **not** the ship metric |
| Informal multi-slice | Per-league / per-year breakdowns in multi-slice eval scripts |

Holdout bounds: `HOLDOUT_DRAW_MIN = 4951`, draw window through `4960` (`config/eval_protocol.py`, `config/stryktipset.py`).

**Do not compare numbers across different slices.** A holdout-only LL ≤ 0.99 does not clear the 518-row include-holdout gate.

**Evidence:** [`production_profile.md`](production_profile.md), [`project_status.md`](project_status.md), `src/scripts/backtest_residual_ml.py`, `src/scripts/eval_residual_ml.py`, `config/eval_protocol.py`, `src/utils/time_split.py`

---

## HGB residual

**Definition:** HistGradientBoosting multi-output regressor that predicts **residual logit deltas** versus the **blend baseline**, then maps back to normalized 1X2 probabilities. Model type string: `MODEL_TYPE = "residual_logit_v1"` in `src/calc/residual_ml/trainer.py`.

**Not the same as:** Training a full multinomial classifier from scratch, or replacing the blend with a raw HGB probability vector.

**Pipeline (production order):** market+DC **blend** → optional draw adjust (off) → **HGB residual** → optional **shrink to market**.

Production artifact: `models/residual_ml/sweep_best/`. Package: `src/calc/residual_ml/`.

**Evidence:** `src/calc/residual_ml/trainer.py`, `src/calc/residual_ml/model.py`, [`product/probability_calculations.md`](product/probability_calculations.md) §5, [`production_profile.md`](production_profile.md)

---

## Logit

**Definition:** Binary logit transform \(\mathrm{logit}(p)=\ln(p/(1-p))\), with inverse sigmoid `inv_logit`. Probabilities are clipped before transform (`PROB_EPSILON`).

**Formula / how used here:**

- Draw adjustment shifts in logit space (when enabled)
- HGB residual targets and predictions as per-outcome **logit deltas** vs blend
- Apply: \(q_i=\mathrm{inv\_logit}(\mathrm{logit}(p_i)+\delta_i)\), then **renormalize** over `{1,X,2}`

**Not the same as:** A full multinomial (softmax) logit. Outcomes are adjusted **independently**, then renormalized — see `apply_residual_deltas` (there is no `apply_logit_deltas`).

**Evidence:** `src/calc/residual_ml/baseline.py` — `logit`, `inv_logit`, `apply_residual_deltas`, `target_logit_deltas`; [`product/probability_calculations.md`](product/probability_calculations.md) §5

---

## Shrink / Shrink to market

**Aliases:** **Shrink**, **Shrink to market** (same step).

**Definition:** After ML, mix ML probabilities toward the **market baseline** in **probability space**, then renormalize over `{1,X,2}`.

**Formula:**

\[
p^{\text{final}}_i = (1-\alpha)\,p^{\text{ml}}_i + \alpha\,p^{\text{market}}_i
\quad\text{(then normalize)}
\]

- \(\alpha=0\) → keep ML
- \(\alpha=1\) → market only

**Config:** `RESIDUAL_ML_FINAL_SHRINK_TO_MARKET` / `DataSourceConfig.residual_ml_final_shrink_to_market`.

**Current production α = 0.5** (Phase 6.1). Do not use older report wording that cites α=0.9 as current production.

**Not the same as:** [Blend](#blend) (market+DC **before** ML). Shrink is ML+market **after** ML.

**Evidence:** `shrink_toward_market` in `src/calc/residual_ml/baseline.py`; [`production_profile.md`](production_profile.md); [`product/probability_calculations.md`](product/probability_calculations.md) §6

---

## Blend

**Definition:** Weighted mix of **market baseline** and **engine (DC) baseline** in **probability space** (not logits). Result is the input baseline for residual ML.

**Formula / how used here:** Default / production effective weights **0.7 market / 0.3 DC** when conditional JSON is disabled (`config/blend_weights.json` `enabled: false` → `DataSourceConfig` / `blend_baselines` defaults).

**Not the same as:** [Shrink / Shrink to market](#shrink--shrink-to-market) (post-ML mix toward market only).

**Evidence:** `blend_baselines` in `src/calc/residual_ml/baseline.py`; `config/blend_weights.json`; [`product/probability_calculations.md`](product/probability_calculations.md) §3; [`DOMAIN.md`](DOMAIN.md) (Blend Baseline)

---

## Brier

**Definition:** Proper scoring rule — mean squared error of a predicted probability versus a binary outcome indicator.

**Formula / how used here:** Often **draw Brier** (also home/away one-vs-rest): \(y=\mathbf{1}[\text{label}=X]\), \(p=p_X\); mean \((p-y)^2\).

**Not the same as:** Multiclass log loss. Brier is a complement for calibration / draw focus; the **primary ship gate remains multiclass log loss**, not Brier.

**Evidence:** `_binary_brier`, `score_outcome_metrics` in `src/calc/residual_ml/evaluation.py`

---

## See also (short)

### Market baseline

Odds-implied normalized 1X2 from Svenska Spel. See [`DOMAIN.md`](DOMAIN.md) (Market Baseline) and [`product/probability_calculations.md`](product/probability_calculations.md) §1.

### Engine / DC baseline

Dixon–Coles (or strength) model probabilities. See [`DOMAIN.md`](DOMAIN.md) (Engine Baseline, Classic Dixon–Coles) and [`product/probability_calculations.md`](product/probability_calculations.md) §2.

### Log loss (gate metric)

Mean \(-\ln p_{\text{observed}}\) over `{1,X,2}` (clipped/renormalized). Official gate: pooled LL ≤ 0.99 on the **518-row include-holdout** [slice](#slice). See [`product/probability_calculations.md`](product/probability_calculations.md) §7, [`project_status.md`](project_status.md).

### Draw adjustment

Optional logit-space shift of blend draw probability before HGB; **off** in production (`config/draw_adjustment.json` `enabled: false`). See [`production_profile.md`](production_profile.md), [`product/probability_calculations.md`](product/probability_calculations.md) §4.
