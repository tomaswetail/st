"""Canonical evaluation protocol for residual ML and DC tuning.

Draw window bounds come from ``config/stryktipset.py`` (4760–4960 inclusive).
Holdout draws (4951–4960) are reserved for Phase 5 final evaluation only.
"""

from __future__ import annotations

from config.stryktipset import STRYKETIPSET_DRAW_MAX, STRYKETIPSET_DRAW_MIN
from utils.time_split import DEFAULT_VALIDATION_FRACTION

VALIDATION_FRACTION = DEFAULT_VALIDATION_FRACTION

# Last ~5% of the draw window (10 of 201 draws); excluded from tuning/backtest.
HOLDOUT_DRAW_MIN = 4951
TUNING_DRAW_MAX = 4950

DRAW_WINDOW_MIN = STRYKETIPSET_DRAW_MIN
DRAW_WINDOW_MAX = STRYKETIPSET_DRAW_MAX
