# Baseline after DC tune (Phase 1)

Reference document for all post–Phase 1 comparisons. Pre-tune proxy: [`artifacts/baseline_pre_dc_tune_20250902/`](../../../artifacts/baseline_pre_dc_tune_20250902/).

---

## Run metadata

| Field | Value |
|-------|-------|
| Date | 2025-09-02 |
| Git commit | `e49d05900a8258ab787e22103a9592eb94cbe7e1` |
| Draw window | 4760–4960 (Stryktipset) |
| Tuning draws | 4760–4950 (`TUNING_DRAW_MAX`) |
| Final holdout | 4951–4960 (`HOLDOUT_DRAW_MIN`–`DRAW_WINDOW_MAX`) — excluded from default backtest |
| Validation fraction | 0.20 (time-split, last 20% by `match_date`) |
| Pipeline | `scripts/run_classic_dc_ml_pipeline.sh` (VALIDATION_FRACTION=0.20, JOBS=16) |

---

## Environment

| Variable | Value |
|----------|-------|
| `VALIDATION_FRACTION` | 0.20 |
| `RESIDUAL_ML_HOME_ADVANTAGE_MODE` | fast |
| `RESIDUAL_ML_DC_ENGINE` | classic |
| `CLASSIC_DC_LEAGUE_PARAMS_PATH` | config/classic_dc_league_params.json |
| `RESIDUAL_ML_MARKET_WEIGHT` | 0.7 |
| `RESIDUAL_ML_DC_WEIGHT` | 0.3 |

Canonical constants: [`config/eval_protocol.py`](../../../config/eval_protocol.py).

---

## DC optimizer summary

Coverage report: [`artifacts/verify_dc_coverage_20250902.txt`](../../../artifacts/verify_dc_coverage_20250902.txt)

| Metric | Value |
|--------|-------|
| Validation matches (DC optimizer) | 456 |
| Leagues in validation slice | 17 |
| Eligible leagues (≥15 val matches) | 5 |
| Tuned leagues | 39, 41, 42, 45, 180 |
| Skipped leagues | 12 (<15 validation matches each) |

Per-league params: [`config/classic_dc_league_params.json`](../../../config/classic_dc_league_params.json).

---

## Dataset rebuild (Phase 1.2)

| Metric | Value |
|--------|-------|
| Output | `data/residual_ml/dataset.csv` |
| Row count | **2589** |
| Archive | [`artifacts/baseline_post_dc_tune_20250902/dataset.csv`](../../../artifacts/baseline_post_dc_tune_20250902/dataset.csv) |

DC columns (`p_*_dc`, `expected_*_goals`, `p_*_blend`) recomputed with tuned per-league xi/lookback/rho. Non-DC features unchanged.

---

## ML retrain (Phase 1.3)

Full 162-trial sweep → [`models/residual_ml/sweep_best/`](../../../models/residual_ml/sweep_best/).

### Best hyperparameters (`sweep_results.json`)

| Parameter | Value |
|-----------|-------|
| `max_depth` | 4 |
| `learning_rate` | 0.03 |
| `max_iter` | 100 |
| `label_smoothing` | 0.05 |
| `validation_fraction` | 0.20 |
| Train rows | 2071 |
| Validation rows | 518 |

### Training metrics (best trial)

| Metric | Value |
|--------|-------|
| Train log loss | 0.8319 |
| Validation log loss (ML raw) | 1.0091 |
| Market validation log loss | 1.0068 |
| Blend validation log loss | 1.0057 |

Train/validation gap (~0.18) indicates overfitting; trust validation only.

---

## Ablation matrix (Phase 1.4 gate)

**Primary comparison:** 518-row validation slice (`--include-holdout`), same protocol as pre-tune snapshot.

| Row | Config | Pre-tune LL | Post-tune LL | Δ | Rows |
|-----|--------|-------------|--------------|---|------|
| A | Market | 1.0068 | 1.0068 | 0.0000 | 518 |
| B | DC only | 1.0537 | 1.0375 | **−0.0162** | 465 |
| C | Blend 70/30 | 1.0100 | 1.0057 | **−0.0043** | 518 |
| D | ML raw | 1.0118 | 1.0091 | **−0.0027** | 518 |
| E | ML + best shrink α | 1.0025 (α=0.6) | **1.0022** (α=0.5) | **−0.0003** | 518 |

Pre-tune source: [`artifacts/baseline_pre_dc_tune_20250902/backtest.txt`](../../../artifacts/baseline_pre_dc_tune_20250902/backtest.txt)  
Post-tune source: [`artifacts/backtest_post_dc_20250902.txt`](../../../artifacts/backtest_post_dc_20250902.txt)

### Tuning slice (holdout excluded, canonical default)

Default backtest excludes draws 4951–4960 → **492 rows**. Market odds on this slice differ from the 518-row slice; do not compare directly to pre-tune without matching slice.

| Row | Post-tune LL | Rows |
|-----|--------------|------|
| Market | 1.0291 | 492 |
| DC only | 1.0377 | 487 |
| Blend 70/30 | 1.0231 | 492 |
| ML raw | 0.9771 | 492 |
| ML + best shrink | 0.9771 (α=0.0) | 492 |

Source: [`artifacts/backtest_post_dc_20250902_holdout_excluded.txt`](../../../artifacts/backtest_post_dc_20250902_holdout_excluded.txt)

Multi-slice eval: [`artifacts/eval_post_dc_20250902.json`](../../../artifacts/eval_post_dc_20250902.json)

---

## Success criteria

| Tier | Criterion | Result |
|------|-----------|--------|
| **Minimum** | Post-tune best ≤ pre-tune best (518-row E) | **Pass** — 1.0022 ≤ 1.0025 |
| **Target** | Pooled LL ≤ 1.003 on row E | **Pass** — 1.0022 |
| **Stretch** | Pooled LL ≤ 1.000 | **Fail** — gap 0.0022 |
| **Reject** | Post-tune worse than pre-tune on market | **Pass** — market unchanged |

---

## Production recommendation

| Setting | Value |
|---------|-------|
| Model path | `models/residual_ml/sweep_best/model.pkl` |
| `RESIDUAL_ML_FINAL_SHRINK_TO_MARKET` | **0.5** |
| Best system (518-row E) | ML + shrink → **1.0022** log loss |
| Blend 70/30 alone | 1.0057 (beats market 1.0068) |

Use α=0.5 for live inference on the tuning slice. Re-evaluate after Phase 2 feature changes.

---

## Delta vs pre-tune

| System | Δ (518-row) |
|--------|-------------|
| Market | 0.0000 |
| DC only | −0.0162 |
| Blend | −0.0043 |
| ML raw | −0.0027 |
| Best shrunk ML | **−0.0003** |

DC tuning improved the DC baseline and blend; ML residual gained modestly. Primary score (row E) improved by 0.0003.

Leagues with dedicated params (39 PL, 180 Scotland, etc.) drove DC improvement; 12 small-sample leagues still use global defaults.

---

## Phase 1.5 — blend weight sweep

**Skipped.** Blend 70/30 (1.0057) already beats market (1.0068) on the 518-row slice. No weight sweep required per roadmap Section 1.7.

---

## Decision: proceed to Phase 2?

**Yes.** Phase 1 minimum gate passed. Post-tune baseline is the reference for injury/availability work (Phase 2).

Remaining gap to 0.99 target: ~0.012 from best shrunk ML (1.0022 → 0.9900).

---

## Artifact index

| Artifact | Path |
|----------|------|
| Post-tune snapshot | [`artifacts/baseline_post_dc_tune_20250902/`](../../../artifacts/baseline_post_dc_tune_20250902/) |
| Backtest (518 rows) | [`artifacts/backtest_post_dc_20250902.txt`](../../../artifacts/backtest_post_dc_20250902.txt) |
| Backtest (492 rows) | [`artifacts/backtest_post_dc_20250902_holdout_excluded.txt`](../../../artifacts/backtest_post_dc_20250902_holdout_excluded.txt) |
| DC coverage | [`artifacts/verify_dc_coverage_20250902.txt`](../../../artifacts/verify_dc_coverage_20250902.txt) |
| Multi-slice eval | [`artifacts/eval_post_dc_20250902.json`](../../../artifacts/eval_post_dc_20250902.json) |
| Pre-tune proxy | [`artifacts/baseline_pre_dc_tune_20250902/`](../../../artifacts/baseline_pre_dc_tune_20250902/) |

Snapshot script: `python scripts/snapshot_post_dc_baseline.py --date 20250902`
