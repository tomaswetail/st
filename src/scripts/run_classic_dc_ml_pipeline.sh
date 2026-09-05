#!/usr/bin/env bash
# Per-league DC optimization → dataset rebuild → train sweep → backtest.
#
# Usage (from repo root):
#   ./src/scripts/run_classic_dc_ml_pipeline.sh
#
# Faster Phase 1 (smaller grid, 90 combos/league vs 210):
#   GRID_CONFIG=config/classic_dc_optimization_grid_fast.json ./src/scripts/run_classic_dc_ml_pipeline.sh
#
# Or:
#   bash src/scripts/run_classic_dc_ml_pipeline.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export RESIDUAL_ML_HOME_ADVANTAGE_MODE=fast
export RESIDUAL_ML_DC_ENGINE=classic

VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.20}"
JOBS="${JOBS:-16}"
LEAGUE_PARAMS="${LEAGUE_PARAMS:-config/classic_dc_league_params.json}"
GRID_CONFIG="${GRID_CONFIG:-config/classic_dc_optimization_grid.json}"
DATASET="${DATASET:-data/residual_ml/dataset.csv}"
MODEL="${MODEL:-models/residual_ml/sweep_best/model.pkl}"

echo "==> 1/4 Optimize per-league Dixon–Coles (${VALIDATION_FRACTION} validation, jobs=${JOBS})"
python -u -m src.scripts.optimize_classic_dixon_coles \
  --validation-fraction "$VALIDATION_FRACTION" \
  --grid-config "$GRID_CONFIG" \
  --output "$LEAGUE_PARAMS" \
  --jobs "$JOBS"

echo "==> 2/4 Build residual ML dataset (fast HA + classic DC)"
python -m src.scripts.build_residual_ml_dataset

echo "==> 3/4 Hyperparameter sweep (${VALIDATION_FRACTION} validation)"
python -m src.scripts.train_residual_ml \
  --dataset "$DATASET" \
  --sweep \
  --validation-fraction "$VALIDATION_FRACTION" \
  --output-dir "$(dirname "$MODEL")"

echo "==> 4/4 Backtest on validation slice (${VALIDATION_FRACTION})"
python -m src.scripts.backtest_residual_ml \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --validation-fraction "$VALIDATION_FRACTION"

echo "Done."
