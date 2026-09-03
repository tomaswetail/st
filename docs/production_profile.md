# Production profile (Phase 6.1 shipped)

Shipped configuration for live Stryktipset 1X2 scoring after promoting the Phase 6.1 restore model.
See [`docs/reports/ml_residual/phase6_restore_phase1.md`](reports/ml_residual/phase6_restore_phase1.md).

## Gate status (honest)

| Slice | Rows | Market | Blend | ML (raw) | Best shrink | ≤0.99? |
|-------|------|--------|-------|----------|-------------|--------|
| Time-split val **with holdout** (`--include-holdout`, 20%) | 518 | 1.0068 | 1.0057 | 1.0091 | **1.0022** (α=0.5) | **MISSED** |
| Holdout draws **4951–4960 only** | 129 | **0.9826** | 0.9908 | — | market ≈ 0.983 (α→1.0) | Market alone ≤0.99 |

Artifacts: [`artifacts/phase6_restore_phase1_eval.json`](../artifacts/phase6_restore_phase1_eval.json), [`artifacts/phase5_holdout_eval.json`](../artifacts/phase5_holdout_eval.json).

**Primary Phase 5 gate (pooled LL ≤ 0.99 on include-holdout): MISSED** (best shrink **1.0022**).

**Phase 2 exit (pooled gain from injuries): FAIL** — injury columns kept in CSV for monitoring; not in HGB.

Previous injury-correction production (~1.0067, α=0.9) is backed up at `models/residual_ml/sweep_best_injury_correction/`.

## Pipeline order

1. Fixed blend 70/30 (`config/blend_weights.json` **`enabled: false`** → DataSourceConfig fallback)
2. Draw adjustment (`config/draw_adjustment.json` **`enabled: false`**)
3. HGB residual (`models/residual_ml/sweep_best/model.pkl`) — Phase 6.1 restore; **no injury columns**
4. Shrink toward market (`RESIDUAL_ML_FINAL_SHRINK_TO_MARKET`, default **0.5**)

Cutoff / injury / odds timing: [`docs/production_cutoff_alignment.md`](production_cutoff_alignment.md).
Formulas: [`docs/probability_calculations.md`](product/probability_calculations.md).

Live must load **`models/residual_ml/sweep_best/model.pkl`** (`RESIDUAL_ML_MODEL_PATH`). The code default path is `models/residual_ml/v1/model.pkl` — do not rely on that default.

## Components

### Dixon–Coles

- Params: `config/classic_dc_league_params.json`
- Engine: `RESIDUAL_ML_DC_ENGINE=classic`
- Home advantage: `RESIDUAL_ML_HOME_ADVANTAGE_MODE=fast`

### Blend

- Config: `config/blend_weights.json` — **`enabled: false`**
- Effective weights: market **0.7** / DC **0.3**
- Conditional rules (85/15, injury uncertainty) **disabled**

### Draw adjustment

- Config: `config/draw_adjustment.json` — **`enabled: false`**
- Discovery artifacts retained; not applied in production path

### Residual ML (HGB)

- Model: `models/residual_ml/sweep_best/model.pkl` (copy of `models/residual_ml/phase1_restore/`)
- Trained on `artifacts/baseline_post_dc_tune_20250902/dataset.csv`
- Pinned Phase 1 hyperparams (not an 81-trial sweep):
  - `max_depth=4`, `learning_rate=0.03`, `max_iter=100`, `label_smoothing=0.05`
  - Train/val 2071 / 518; train LL ≈ 0.832; val ML raw ≈ **1.0091**
- Baseline weights: market 0.7 / DC 0.3

### Shrink

- Recommended live: **`RESIDUAL_ML_FINAL_SHRINK_TO_MARKET=0.5`**
- Code default in `DataSourceConfig` is **0.5** (override with env if needed)
- Applied in `ProbabilityManager` after ML predict
- Include-holdout: α=0.5 → **1.0022**

### Injuries

- Snapshots + backfill CLI unchanged (`src/scripts/backfill_injury_snapshots.py`)
- Count-proxy `missing_value_*` treated as missing in the calculator
- **Not fed to production HGB** (restore model has no injury feature names)

## Environment variables (production-oriented)

| Variable | Typical / recommended |
|----------|------------------------|
| `RESIDUAL_ML_ENABLED` | `true` |
| `RESIDUAL_ML_MODEL_PATH` | `models/residual_ml/sweep_best/model.pkl` |
| `RESIDUAL_ML_MARKET_WEIGHT` | `0.7` |
| `RESIDUAL_ML_DC_WEIGHT` | `0.3` |
| `RESIDUAL_ML_BLEND_WEIGHTS_PATH` | `config/blend_weights.json` |
| `RESIDUAL_ML_FINAL_SHRINK_TO_MARKET` | **`0.5`** |
| `RESIDUAL_ML_HOME_ADVANTAGE_MODE` | `fast` |
| `RESIDUAL_ML_DC_ENGINE` | `classic` |

## Monitoring

```bash
PYTHONPATH=src python src/scripts/eval_residual_ml.py \
  --dataset artifacts/baseline_post_dc_tune_20250902/dataset.csv \
  --model models/residual_ml/sweep_best/model.pkl \
  --include-holdout \
  --json artifacts/phase6_ship_eval.json
PYTHONPATH=src python src/scripts/monitor_round_ll.py --draw-number <N>
```

## Caveats

1. **≤0.99 gate MISSED** — best shrink **1.0022**.
2. Conditional blend and draw adjustment remain off.
3. ST odds are stored match odds, not closing lines.
4. Production HGB was trained on the archived Phase 1 CSV, not a post-injury full rebuild.
