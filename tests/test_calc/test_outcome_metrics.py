"""Tests for draw/home/away outcome metrics and ablation helpers."""

from __future__ import annotations

import pytest

from calc.draw_adjustment import DrawAdjustmentConfig
from calc.residual_ml.evaluation import (
    apply_draw_adjustment_to_rows,
    score_outcome_metrics,
)


def test_score_outcome_metrics_draw_and_home_away():
    rows = [
        {
            "label": "X",
            "p_home_blend": 0.3,
            "p_draw_blend": 0.5,
            "p_away_blend": 0.2,
        },
        {
            "label": "1",
            "p_home_blend": 0.6,
            "p_draw_blend": 0.2,
            "p_away_blend": 0.2,
        },
        {
            "label": "2",
            "p_home_blend": 0.2,
            "p_draw_blend": 0.2,
            "p_away_blend": 0.6,
        },
    ]
    metrics = score_outcome_metrics(
        rows, ("p_home_blend", "p_draw_blend", "p_away_blend")
    )
    assert metrics["row_count"] == 3
    assert metrics["draw_log_loss"] is not None
    assert metrics["draw_brier"] is not None
    assert metrics["home_log_loss"] is not None
    assert metrics["away_log_loss"] is not None
    assert metrics["pooled_multiclass_log_loss"] is not None
    assert 0.0 < metrics["draw_brier"] < 1.0


def test_apply_draw_adjustment_to_rows_changes_draw_when_enabled():
    rows = [
        {
            "label": "1",
            "p_home_blend": 0.45,
            "p_draw_blend": 0.25,
            "p_away_blend": 0.30,
            "away_npxg_for": 1.0,
        }
    ]
    config = DrawAdjustmentConfig(
        enabled=True,
        intercept=0.5,
        features={"away_npxg_for": -0.2},
    )
    adjusted = apply_draw_adjustment_to_rows(rows, draw_config=config)
    assert len(adjusted) == 1
    assert sum(
        adjusted[0][key] for key in ("p_home_blend", "p_draw_blend", "p_away_blend")
    ) == pytest.approx(1.0)
    assert adjusted[0]["p_draw_blend"] != pytest.approx(0.25)
