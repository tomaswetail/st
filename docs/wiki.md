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

## Chronological year folds

**Definition:** Expanding and rolling validation folds keyed by the **calendar year** of `match_date` (`src/calc/residual_ml/folds.py`). Expanding trains on all years `< Y` and validates year `Y`; rolling trains on `[Y - window, Y)` (default window 2) and validates `Y`. The primary metric is multiclass log loss; **market-only** 1X2 must be scored on every fold before comparing models. This is **not** an English football season key — those seasons span two calendar years (August–May), and a later helper can add season-start-year. Year folds do **not** replace the 518-row include-holdout ship gate.

**Evidence:** `src/calc/residual_ml/folds.py`, `src/scripts/eval_market_folds.py`, [`reports/2026-09-12-phase4-folds-metrics.md`](reports/2026-09-12-phase4-folds-metrics.md)

---

## HGB residual

**Definition:** HistGradientBoosting multi-output regressor that predicts **residual logit deltas** versus the **market baseline**, then maps back to normalized 1X2 probabilities. Model type string: `MODEL_TYPE = "residual_logit_v1"` in `src/calc/residual_ml/trainer.py`.

**Not the same as:** Training a full multinomial classifier from scratch, or replacing the market baseline with a raw HGB probability vector.

**Pipeline (production order):** `market_baseline` → **HGB residual delta** → optional **shrink to market**.

Package: `src/calc/residual_ml/`. Pre-pivot artifact backup: `models/residual_ml/sweep_best_pre_market_pivot_bck/` (see DEC-015).

**Evidence:** `src/calc/residual_ml/trainer.py`, `src/calc/residual_ml/model.py`, [`product/probability_calculations.md`](product/probability_calculations.md) §2, [`production_profile.md`](production_profile.md)

---

## Model registry (family A vs B)

**Definition:** Offline factory `get_model(backend, family)` in `src/calc/residual_ml/models/` with a shared seeded `predict_proba` → clip-normalized `{1,X,2}`. **Family A (`direct`)** is a multinomial classifier trained with multiclass log loss; its feature matrix **includes** `p_*_market_norm`. **Family B (`residual`)** predicts 3 logit deltas with MSE (same mismatch as live HGB) and applies `apply_residual_deltas`; market columns stay out of the regressor and enter only via the baseline. `market_baseline` is a no-op passthrough. Live scoring still uses `ResidualMLTrainer` / `residual_logit_v1` only — this registry is not wired to `ProbabilityManager`.

**Evidence:** `src/calc/residual_ml/models/`, `src/scripts/eval_registry_smoke.py`, [`reports/2026-09-12-phase5-model-registry.md`](reports/2026-09-12-phase5-model-registry.md)

---

## Post-hoc calibration (temperature / isotonic)

**Definition:** Recalibrate already-produced 1X2 probabilities without refitting the underlying model. **Temperature** applies \(p'_k=\mathrm{softmax}(\log p_k / T)\) then clip+normalize. **Isotonic** is one-vs-rest `IsotonicRegression` on each class probability, then clip+normalize. Fit only on the earlier chronological half of each expanding-year validate fold; score the later half (odd extra row → score). Accept a method only if the unweighted mean OOS log loss is strictly better than raw. Not wired to live scoring.

**Evidence:** `src/calc/residual_ml/calibration.py`, `src/scripts/eval_calibration.py`, [`reports/2026-09-12-phase6-calibration.md`](reports/2026-09-12-phase6-calibration.md)

---

## Compare script / coupon sources / archive odds

**Definition:** Offline evaluation only. `python -m src.scripts.compare_probability_models` scores every expanding-year fold (then an unweighted mean) for market-only plus each available registry backend; default tables are **uncalibrated**. P13@N packs fixture validate rows into consecutive groups of 13 and calls `CouponOptimizer` PREDICTION / MAX_P13. Real-coupon backtest `python -m src.scripts.backtest_probability_sources` compares **`st_market`** (Svenska Spel vig-free Pm already on the coupon) with **`archive_market`** (football-data.co.uk Avg closing via `fixture_odds`). Those two odds sources are **not** the same line: archive closing is a research/eval join (DEC-016), not the live ST market (DEC-002). Live coupon scoring stays Svenska Spel. This does **not** move the 518-row ship gate and does **not** enable residual ML.

**Evidence:** `src/scripts/compare_probability_models.py`, `src/scripts/backtest_probability_sources.py`, `src/scripts/report_train_serve_odds_shift.py`, [`reports/2026-09-12-phase7-compare-backtest.md`](reports/2026-09-12-phase7-compare-backtest.md)

---

## Dual-price fixture dataset / ABC opening–ST residual gate

**Definition:** Research-only residual evaluation on two **universes**, not the official 518-row include-holdout ship gate.

**Dual-price fixture CSV** (`data/residual_ml/fixtures_dataset.csv`): one row per finished fixture. Features are assembled **once** at kickoff date. Two vig-free 1X2 triples and two market-shape blocks are stored with `_opening` / `_closing` suffixes. Missing opening does **not** fill from closing (DEC-016); the row is kept if the other price is usable. Rebuild CLI: `python -m src.scripts.build_residual_ml_dataset --source fixtures` (omit `--price-type` for dual; `--price-type` is the legacy single-triple export).

**ST CSV** (`data/residual_ml/dataset.csv`): same feature schema; `match_id` is `stryktipset_matches.id`; vig-free Svenska Spel in `p_*_market_norm`; opening/closing columns present and null. No DC/blend leftover columns.

**ABC gate** (`python -m src.scripts.eval_abc_residual_gate`): family B residual only (CatBoost + HGB). Opening universe trains vs opening and scores outcome log loss on the Phase 6 calib/score split of each expanding calendar year (primary vs opening baseline; vs closing is printed only). ST universe trains vs ST vig-free and scores the **full** validate year; α + large-move threshold are fit on the last train year; years with n_val < 80 are printed and excluded; fewer than three eligible years is underpowered REJECT. Shrink α ∈ {0.7, 0.8, 0.9}; large-move threshold ∈ {0.02, 0.04, 0.06, 0.08} (below threshold → exact market identity). **ACCEPT** iff unweighted mean model LL is strictly below the named baseline **and** no eligible year has model LL ≥ baseline + 0.01. Equal mean → REJECT. Artifacts: `artifacts/abc_opening_gate.json`, `artifacts/abc_st_gate.json`. Does **not** enable live ML or write `models/residual_ml/sweep_best/`.

**Evidence:** `src/calc/residual_ml/dual_price.py`, `src/calc/residual_ml/abc_gate.py`, `src/scripts/build_residual_ml_dataset.py`, `src/scripts/eval_abc_residual_gate.py`, [`reports/2026-09-12-abc-opening-st-residual.md`](reports/2026-09-12-abc-opening-st-residual.md)

---

## Logit

**Definition:** Binary logit transform \(\mathrm{logit}(p)=\ln(p/(1-p))\), with inverse sigmoid `inv_logit`. Probabilities are clipped before transform (`PROB_EPSILON`).

**Formula / how used here:**

- HGB residual targets and predictions as per-outcome **logit deltas** vs market baseline
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

**Default α = 0.7** when ML is enabled. See [`production_profile.md`](production_profile.md).

**Not the same as:** Applying the residual delta. Shrink is a linear mix of ML output with the market baseline **after** the ML delta was applied.

**Evidence:** `shrink_toward_market` in `src/calc/residual_ml/baseline.py`; [`production_profile.md`](production_profile.md); [`product/probability_calculations.md`](product/probability_calculations.md) §3

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

### Log loss (gate metric)

Mean \(-\ln p_{\text{observed}}\) over `{1,X,2}` (clipped/renormalized). See [`product/probability_calculations.md`](product/probability_calculations.md) §4, [`project_status.md`](project_status.md).
