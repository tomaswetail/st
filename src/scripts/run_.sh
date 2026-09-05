#!/usr/bin/env bash
# Dataset rebuild → train sweep → eval → backtest.
#
# Usage (from repo root):
#   ./src/scripts/run_.sh
#
# Or:
#   bash src/scripts/run_.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "==> 1/4 Building dataset"
python -u -m src.scripts.build_residual_ml_dataset


echo "==> 2/4 Retrain + sweep"
python -u -m src.scripts.train_residual_ml --dataset data/residual_ml/dataset.csv --sweep \
  --output-dir models/residual_ml/sweep_best
echo "==> 3/4  Re-eval Phase 5 gate (do not retune on holdout)"
python -m src.scripts.eval_residual_ml --include-holdout --ablation \
  --json artifacts/phase5_holdout_eval_after_rebuild.json
echo "==> 4/4  Backtesting"
python -m src.scripts.backtest_residual_ml --validation-fraction 0.20 --include-holdout


echo "Done."
