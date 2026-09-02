#!/usr/bin/env bash
# Per-league DC optimization → dataset rebuild → train sweep → backtest.
#
# Usage (from repo root):
#   ./scripts/run_classic_dc_ml_pipeline.sh
#
# Faster Phase 1 (smaller grid, 90 combos/league vs 210):
#   GRID_CONFIG=config/classic_dc_optimization_grid_fast.json ./scripts/run_classic_dc_ml_pipeline.sh
#
# Or:
#   bash scripts/run_classic_dc_ml_pipeline.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH=.
export RESIDUAL_ML_HOME_ADVANTAGE_MODE=fast
export RESIDUAL_ML_DC_ENGINE=classic

VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.20}"
JOBS="${JOBS:-16}"
LEAGUE_PARAMS="${LEAGUE_PARAMS:-config/classic_dc_league_params.json}"
GRID_CONFIG="${GRID_CONFIG:-config/classic_dc_optimization_grid.json}"
DATASET="${DATASET:-data/residual_ml/dataset.csv}"
MODEL="${MODEL:-models/residual_ml/sweep_best/model.pkl}"

echo "==> 1/4 Optimize per-league Dixon–Coles (${VALIDATION_FRACTION} validation, jobs=${JOBS})"
python -u scripts/optimize_classic_dixon_coles.py \
  --validation-fraction "$VALIDATION_FRACTION" \
  --grid-config "$GRID_CONFIG" \
  --output "$LEAGUE_PARAMS" \
  --jobs "$JOBS"

echo "==> 2/4 Build residual ML dataset (fast HA + classic DC)"
python scripts/build_residual_ml_dataset.py

echo "==> 3/4 Hyperparameter sweep (${VALIDATION_FRACTION} validation)"
python scripts/train_residual_ml.py \
  --dataset "$DATASET" \
  --sweep \
  --validation-fraction "$VALIDATION_FRACTION" \
  --output-dir "$(dirname "$MODEL")"

echo "==> 4/4 Backtest on validation slice (${VALIDATION_FRACTION})"
python scripts/backtest_residual_ml.py \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --validation-fraction "$VALIDATION_FRACTION"

echo "Done."
