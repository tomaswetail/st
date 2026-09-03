# Phase 6.1 — Restore Phase 1 residual quality

Diagnostic retrain of **one** HGB residual on the archived Phase 1 CSV with pinned Phase 1 hyperparameters.

**Promoted to production (2025-09-03):** copy of this restore is `models/residual_ml/sweep_best/`. Live shrink default **α=0.5**. Previous injury-correction model: `models/residual_ml/sweep_best_injury_correction/`.

References: [`docs/reports/ml_residual/baseline_after_dc_tune.md`](baseline_after_dc_tune.md), [`docs/production_profile.md`](../../production_profile.md).

---

## Decision: **PASS**

Best shrink on the official 518-row include-holdout gate is **1.0022 (α=0.5)**, matching canonical Phase 1 within ~0.001 (exact JSON: 1.0021925884473784 vs Phase 1 1.0022).

Phase 1 residual quality **is reproducible** from remaining artifacts (archived CSV + pinned hyperparams). Later injury rebuild + sweep (depth 3, smoothing 0.1) is what drifted production to **1.0067**.

This does **not** claim pooled LL ≤ 0.99. Restored best shrink is still **1.0022**.

**Production promoted (2025-09-03):** `models/residual_ml/sweep_best/` is this restore. `config/blend_weights.json` and `config/draw_adjustment.json` remain `"enabled": false`. Shrink default **α=0.5**. Injury-correction backup: `models/residual_ml/sweep_best_injury_correction/`.

---

## Commands used

From repo root (`PYTHONPATH=src`). No `--sweep`, no `--market-only-baseline`, no custom `--validation-fraction` (default 0.20). Time-split is first 80% of the archived CSV by `match_date`/`match_id`; no extra holdout-draw exclusion at train time.

### Train

```bash
export PYTHONPATH=src
python -u scripts/train_residual_ml.py \
  --dataset artifacts/baseline_post_dc_tune_20250902/dataset.csv \
  --output-dir models/residual_ml/phase1_restore \
  --version phase1_restore \
  --max-depth 4 \
  --learning-rate 0.03 \
  --max-iter 100 \
  --label-smoothing 0.05 \
  --exclude-injury-features
```

`--exclude-injury-features` is a no-op here (archived CSV has no injury columns). Trainer `random_state` default is 42.

### Eval (official gate)

```bash
python src/scripts/eval_residual_ml.py \
  --dataset artifacts/baseline_post_dc_tune_20250902/dataset.csv \
  --model models/residual_ml/phase1_restore/model.pkl \
  --include-holdout \
  --json artifacts/phase6_restore_phase1_eval.json
```

Eval slice: **518 rows** (time-split validation fraction=0.2, holdout included). Matches Phase 1 protocol.

---

## Training stdout summary

| Field | Value |
|-------|-------|
| Dataset rows | 2589 |
| Train rows | 2071 |
| Validation rows | 518 |
| Train log loss | 0.8319 |
| Validation log loss (ML raw) | 1.0091 |
| Market validation log loss | 1.0068 |
| Blend validation log loss | 1.0057 |
| Model path | `models/residual_ml/phase1_restore/model.pkl` |

These match Phase 1 best-trial training metrics in [`docs/reports/ml_residual/baseline_after_dc_tune.md`](baseline_after_dc_tune.md) (train 0.8319, val ML 1.0091, market 1.0068, blend 1.0057; 2071 / 518).

---

## Official gate: pooled metrics vs Phase 1

Source: [`artifacts/phase6_restore_phase1_eval.json`](../../../artifacts/phase6_restore_phase1_eval.json) `slices.pooled`. Phase 1: [`docs/reports/ml_residual/baseline_after_dc_tune.md`](baseline_after_dc_tune.md), [`artifacts/phase5_holdout_eval.json`](../../../artifacts/phase5_holdout_eval.json).

| Metric | Phase 1 | 6.1 restore |
|--------|---------|-------------|
| Rows | 518 | 518 |
| Market LL | 1.0068 | 1.0068 |
| Blend LL | 1.0057 | 1.0057 |
| ML raw | 1.0091 | 1.0091 |
| Best shrink | 1.0022 (α=0.5) | **1.0022 (α=0.5)** |

Exact JSON (`slices.pooled`):

| Field | Value |
|-------|-------|
| `row_count` | 518 |
| `baselines.market.log_loss` | 1.0067856568672326 |
| `baselines.blend.log_loss` | 1.0056569644642361 |
| `ml_log_loss` | 1.0090960766726624 |
| `best_shrink_log_loss` | 1.0021925884473784 |
| `best_shrink_alpha` | 0.5 |

---

## Secondary check: current production CSV (not the official gate)

Current `data/residual_ml/dataset.csv` contains every Phase 1 feature name from [`artifacts/baseline_post_dc_tune_20250902/sweep_best/feature_schema.json`](../../../artifacts/baseline_post_dc_tune_20250902/sweep_best/feature_schema.json) (66 names). Extra injury columns are present and ignored at predict.

```bash
python src/scripts/eval_residual_ml.py \
  --dataset data/residual_ml/dataset.csv \
  --model models/residual_ml/phase1_restore/model.pkl \
  --include-holdout \
  --json artifacts/phase6_restore_phase1_current_dataset_eval.json
```

| Metric | Value |
|--------|-------|
| Rows | 518 |
| Market LL | 1.0068 |
| Blend LL | 1.0068 |
| ML raw | 1.0089 |
| Best shrink | 1.0017 (α=0.5) |

Blend here equals market to 4 decimals because the current CSV is reblend-patched with conditional blend **disabled** (70/30 via config fallback; production blend LL is 1.0068). This is a compatibility check only — official restore gate remains the archived Phase 1 CSV.

---

## What was not done

- No overwrite of `models/residual_ml/sweep_best/` *at restore time* (promotion happened later; see top of this doc)
- No change to `config/blend_weights.json` or `config/draw_adjustment.json`
- No change to `data/residual_ml/dataset.csv`
- No `build_residual_ml_dataset.py`
- No hyperparameter sweep
- Injuries not re-enabled in HGB, conditional blend, or draw adjustment
- Phase 6.2 (market-anchored HGB) ran afterward and **FAIL**ed Gate B — see [`docs/reports/ml_residual/phase6_market_anchored.md`](phase6_market_anchored.md)
- No git commit
- No assembler / trainer code changes at restore time
