"""Tests for draw/home/away outcome metrics (market baseline only)."""

from __future__ import annotations

from src.calc.residual_ml.evaluation import score_outcome_metrics


def test_score_outcome_metrics_draw_and_home_away():
    rows = [
        {
            "label": "X",
            "p_home_market_norm": 0.3,
            "p_draw_market_norm": 0.5,
            "p_away_market_norm": 0.2,
        },
        {
            "label": "1",
            "p_home_market_norm": 0.6,
            "p_draw_market_norm": 0.2,
            "p_away_market_norm": 0.2,
        },
        {
            "label": "2",
            "p_home_market_norm": 0.2,
            "p_draw_market_norm": 0.2,
            "p_away_market_norm": 0.6,
        },
    ]
    metrics = score_outcome_metrics(
        rows,
        ("p_home_market_norm", "p_draw_market_norm", "p_away_market_norm"),
    )
    assert metrics["row_count"] == 3
    assert metrics["draw_log_loss"] is not None
    assert metrics["draw_brier"] is not None
    assert metrics["home_log_loss"] is not None
    assert metrics["away_log_loss"] is not None
    assert metrics["pooled_multiclass_log_loss"] is not None
    assert 0.0 < metrics["draw_brier"] < 1.0
