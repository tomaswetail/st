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

export PYTHONPATH=src

echo "==> 1/4 Building dataset"
python -u src/scripts/build_residual_ml_dataset.py


echo "==> 2/4 Retrain + sweep"
python -u src/scripts/train_residual_ml.py --dataset data/residual_ml/dataset.csv --sweep \
  --output-dir models/residual_ml/sweep_best
echo "==> 3/4  Re-eval Phase 5 gate (do not retune on holdout)"
python src/scripts/eval_residual_ml.py --include-holdout --ablation \
  --json artifacts/phase5_holdout_eval_after_rebuild.json
echo "==> 4/4  Backtesting"
python src/scripts/backtest_residual_ml.py --validation-fraction 0.20 --include-holdout


echo "Done."
