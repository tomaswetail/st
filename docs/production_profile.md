# Production profile (MLE-ρ DC + pinned MLE HGB)

Shipped configuration for live Stryktipset 1X2 scoring: **MLE-ρ Dixon–Coles** plus the **pinned MLE HGB** in `models/residual_ml/sweep_best/`. Live scoring and official eval are the same system.

Phase 6.1 (grid-DC + old HGB, best shrink **1.0022** at α=0.5) is **historical**. Backup: `models/residual_ml/sweep_best_phase61_grid_dc/`.

## Gate status (honest)

| Slice | Rows | Market | Blend | ML (raw) | Best shrink | ≤0.99? |
|-------|------|--------|-------|----------|-------------|--------|
| Time-split val **with holdout** (`--include-holdout`, 20%) | 518 | 1.0068 | 1.0083 | 1.0149 | **1.0046** (α=0.7) | **MISSED** |

Artifacts: [`artifacts/dc_rho_mle_promotion/gate_eval_retrain.json`](../artifacts/dc_rho_mle_promotion/gate_eval_retrain.json), [`artifacts/dc_rho_mle_promotion/gate_eval_retrain.txt`](../artifacts/dc_rho_mle_promotion/gate_eval_retrain.txt). Official eval CSV: [`artifacts/dc_rho_mle_promotion/dataset.csv`](../artifacts/dc_rho_mle_promotion/dataset.csv) (2589 rows; do not rebuild).

**Primary Phase 5 gate (pooled LL ≤ 0.99 on include-holdout): MISSED** (best shrink **1.0046**). Stretch 0.99 remains missed.

**Phase 2 exit (pooled gain from injuries): FAIL** — injury columns kept in CSV for monitoring; not in HGB.

Previous injury-correction production (~1.0067, α=0.9) is backed up at `models/residual_ml/sweep_best_injury_correction/`.

## Pipeline order

1. Fixed blend 70/30 (`config/blend_weights.json` **`enabled: false`** → DataSourceConfig fallback)
2. Draw adjustment (`config/draw_adjustment.json` **`enabled: false`**)
3. HGB residual (`models/residual_ml/sweep_best/model.pkl`) — pinned MLE retrain; **no injury columns**
4. Shrink toward market (`RESIDUAL_ML_FINAL_SHRINK_TO_MARKET`, default **0.7**)

Cutoff / injury / odds timing: [`docs/production_cutoff_alignment.md`](production_cutoff_alignment.md).
Formulas: [`docs/probability_calculations.md`](product/probability_calculations.md).

Live must load **`models/residual_ml/sweep_best/model.pkl`** (`RESIDUAL_ML_MODEL_PATH`). The code default path is `models/residual_ml/v1/model.pkl` — do not rely on that default.

## Components

### Dixon–Coles

- Params: `config/classic_dc_league_params.json` — **MLE-fitted ρ** (promoted 2026-09-04; grid backup at `config/classic_dc_league_params_grid_bck.json`)
- **MLE ρ at fit time:** `CLASSIC_DC_FIT_RHO` default **on** (`1`); set `=0` to disable
- Engine: `RESIDUAL_ML_DC_ENGINE=classic`
- Home advantage: `RESIDUAL_ML_HOME_ADVANTAGE_MODE=fast`

Per-league ρ (MLE walk-forward median, 5 leagues tuned):

| League | ξ | Lookback | ρ |
|--------|---|----------|---|
| 39 | 0.00025 | 1460 | −0.0145 |
| 41 | 0.0001 | 1460 | +0.0063 |
| 42 | 0.0001 | 1095 | −0.0172 |
| 45 | 0.005 | 730 | −0.1406 |
| 180 | 0.0001 | 1460 | −0.1086 |

### Blend

- Config: `config/blend_weights.json` — **`enabled: false`**
- Effective weights: market **0.7** / DC **0.3**
- Conditional rules (85/15, injury uncertainty) **disabled**

### Draw adjustment

- Config: `config/draw_adjustment.json` — **`enabled: false`**
- Discovery artifacts retained; not applied in production path

### Residual ML (HGB)

- Model: `models/residual_ml/sweep_best/model.pkl` (copy of `models/residual_ml/dc_rho_mle_retrain/`)
- Trained on `artifacts/dc_rho_mle_promotion/dataset.csv` (2589 rows; not rebuilt)
- Phase 6.1 grid-DC HGB backup: `models/residual_ml/sweep_best_phase61_grid_dc/`
- Pinned Phase 1 hyperparams (not the 81-trial sweep — that gate was **1.0063**, not promoted):
  - `max_depth=4`, `learning_rate=0.03`, `max_iter=100`, `label_smoothing=0.05`
  - Train/val 2071 / 518; train LL ≈ **0.8372**; val ML raw ≈ **1.0149**
- Baseline weights: market 0.7 / DC 0.3

### Shrink

- Recommended live: **`RESIDUAL_ML_FINAL_SHRINK_TO_MARKET=0.7`**
- Code default in `DataSourceConfig` is **0.7** (override with env if needed)
- Applied in `ProbabilityManager` after ML predict
- Include-holdout: α=0.7 → **1.0046**

### Injuries

- Snapshots + backfill CLI unchanged (`src/scripts/backfill_injury_snapshots.py`)
- Count-proxy `missing_value_*` treated as missing in the calculator
- **Not fed to production HGB** (restore/retrain models have no injury feature names)

## Environment variables (production-oriented)

| Variable | Typical / recommended |
|----------|------------------------|
| `RESIDUAL_ML_ENABLED` | `true` |
| `RESIDUAL_ML_MODEL_PATH` | `models/residual_ml/sweep_best/model.pkl` |
| `RESIDUAL_ML_MARKET_WEIGHT` | `0.7` |
| `RESIDUAL_ML_DC_WEIGHT` | `0.3` |
| `RESIDUAL_ML_BLEND_WEIGHTS_PATH` | `config/blend_weights.json` |
| `RESIDUAL_ML_FINAL_SHRINK_TO_MARKET` | **`0.7`** |
| `RESIDUAL_ML_HOME_ADVANTAGE_MODE` | `fast` |
| `RESIDUAL_ML_DC_ENGINE` | `classic` |
| `CLASSIC_DC_FIT_RHO` | **`1`** (default on; set `0` to disable MLE ρ) |

## Monitoring

```bash
python -m src.scripts.eval_residual_ml \
  --dataset artifacts/dc_rho_mle_promotion/dataset.csv \
  --model models/residual_ml/sweep_best/model.pkl \
  --include-holdout \
  --json artifacts/phase6_ship_eval.json
python -m src.scripts.monitor_round_ll --draw-number <N>
```

## Caveats

1. **≤0.99 gate MISSED** — best shrink **1.0046**.
2. Conditional blend and draw adjustment remain off.
3. ST odds are stored match odds, not closing lines.
4. Production HGB was trained on the MLE-ρ promotion CSV (2589 rows), not a post-injury full rebuild.
