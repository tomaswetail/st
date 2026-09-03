# Pre-DC-tune baseline snapshot (20250902)

**Post-hoc proxy snapshot:** the full DC pipeline had already run in this repo when
this folder was created. Only `dataset.csv` (from `dataset_classic.csv`) preserves
pre-tune DC feature columns reliably.

## Copied sweep metrics (`sweep_best/sweep_results.json` best trial)

These numbers reflect the model copied at snapshot time (20250902), not
recomputed on `dataset.csv` in this folder.

| System | Log loss | Rows |
|--------|----------|------|
| Market | 1.0068 | 518 |
| Blend 70/30 | 1.0057 | 518 |
| ML (no shrink) | 1.0091 | 518 |

## Files in this snapshot

| File | Source | Notes |
|------|--------|-------|
| `dataset.csv` | `data/residual_ml/dataset_classic.csv` | **Pre-tune proxy** — classic DC before full grid tune |
| `sweep_best/` | `models/residual_ml/sweep_best/` | Copied at snapshot time (may be post-pipeline model) |
| `backtest.txt` | generated | Backtest on `dataset.csv` @ 20% validation (`--include-holdout`) |
| `classic_dc_league_params.pre_tune_note.txt` | generated | Why league params are omitted |

`classic_dc_league_params.json` is **not** copied — see the note file above.

Reproduce `backtest.txt`:

```bash
export PYTHONPATH=.
python scripts/backtest_residual_ml.py \
  --dataset artifacts/baseline_pre_dc_tune_20250902/dataset.csv \
  --model artifacts/baseline_pre_dc_tune_20250902/sweep_best/model.pkl \
  --validation-fraction 0.20 \
  --include-holdout
```
