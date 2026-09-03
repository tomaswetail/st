# Post-DC-tune baseline snapshot (20250902)

Git commit at snapshot: `e49d05900a8258ab787e22103a9592eb94cbe7e1`

Dataset rows: 2589 (expect ~2589)

Full ablation and production recommendation: [`docs/reports/ml_residual/baseline_after_dc_tune.md`](../../docs/reports/ml_residual/baseline_after_dc_tune.md)

## Copied sweep metrics (`sweep_best/sweep_results.json` best trial)

| System | Log loss | Rows |
|--------|----------|------|
| Market | 1.0068 | 518 |
| Blend 70/30 | 1.0057 | 518 |
| ML (no shrink) | 1.0091 | 518 |

## Files in this snapshot

| File | Source | Notes |
|------|--------|-------|
| `dataset.csv` | `data/residual_ml/dataset.csv` | Post-tune DC features |
| `classic_dc_league_params.json` | `config/classic_dc_league_params.json` | Per-league tuned params |
| `sweep_best/` | `models/residual_ml/sweep_best/` | Best ML model + sweep results |
| `backtest.txt` | generated | 518-row validation (`--include-holdout`) |
| `backtest_holdout_excluded.txt` | generated | 492-row tuning slice (default) |
| `verify_dc_coverage.txt` | copied if present | League coverage report |
| `eval_post_dc.json` | copied if present | Multi-slice eval JSON |

Reproduce `backtest.txt`:

```bash
export PYTHONPATH=.
python scripts/backtest_residual_ml.py \
  --dataset artifacts/baseline_post_dc_tune_20250902/dataset.csv \
  --model artifacts/baseline_post_dc_tune_20250902/sweep_best/model.pkl \
  --validation-fraction 0.20 \
  --include-holdout
```
