"""Unit tests for draw-driver discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.calc.draw_driver_analysis import (
    build_analysis_rows,
    filter_tuning_rows,
    render_draw_driver_markdown,
    run_draw_discovery,
    write_discovery_artifacts,
)


def _row(
    *,
    match_id: int,
    match_date: str,
    label: str,
    p_draw_blend: float,
    league_draw_rate: float,
    expected_goal_total: float,
    draw_number: int = 4800,
) -> dict:
    return {
        "match_id": match_id,
        "match_date": match_date,
        "draw_number": draw_number,
        "label": label,
        "p_draw_blend": p_draw_blend,
        "p_draw_market_norm": p_draw_blend,
        "league_draw_rate": league_draw_rate,
        "combined_draw_rate": league_draw_rate,
        "combined_close_match_rate": 0.5,
        "combined_low_scoring_rate": 0.4,
        "expected_goal_total": expected_goal_total,
        "market_vs_dc_draw": 0.0,
        "market_balance": 0.1,
    }


def _synthetic_rows(n: int = 120) -> list[dict]:
    rows: list[dict] = []
    for index in range(n):
        # Higher league_draw_rate → more draws; higher EG total → fewer draws
        league_draw = 0.15 + 0.4 * (index % 10) / 9.0
        eg_total = 3.5 - 2.0 * (index % 7) / 6.0
        # Soft rule for label
        draw_score = league_draw - 0.08 * eg_total + 0.02 * ((index % 5) - 2)
        label = "X" if draw_score > 0.12 else ("1" if index % 2 == 0 else "2")
        p_blend = max(0.05, min(0.45, 0.2 + 0.3 * league_draw - 0.03 * eg_total))
        year = 2023 + index // 60
        month = 1 + (index % 12)
        day = 1 + (index % 28)
        rows.append(
            _row(
                match_id=index + 1,
                match_date=f"{year:04d}-{month:02d}-{day:02d}",
                label=label,
                p_draw_blend=p_blend,
                league_draw_rate=league_draw,
                expected_goal_total=eg_total,
                draw_number=4800 + index // 10,
            )
        )
    return rows


def test_filter_excludes_holdout_and_missing_blend():
    rows = [
        _row(
            match_id=1,
            match_date="2025-01-01",
            label="X",
            p_draw_blend=0.3,
            league_draw_rate=0.25,
            expected_goal_total=2.5,
            draw_number=4955,
        ),
        _row(
            match_id=2,
            match_date="2025-01-02",
            label="1",
            p_draw_blend=0.25,
            league_draw_rate=0.2,
            expected_goal_total=2.8,
            draw_number=4900,
        ),
        {
            "match_id": 3,
            "match_date": "2025-01-03",
            "draw_number": 4901,
            "label": "X",
            "p_draw_blend": "",
        },
    ]
    filtered = filter_tuning_rows(rows, exclude_holdout=True)
    assert len(filtered) == 1
    assert filtered[0]["match_id"] == 2


def test_build_analysis_rows_sets_surprise():
    rows = [
        _row(
            match_id=1,
            match_date="2024-06-01",
            label="X",
            p_draw_blend=0.25,
            league_draw_rate=0.3,
            expected_goal_total=2.0,
        )
    ]
    analysis = build_analysis_rows(rows)
    assert analysis[0]["is_draw"] == 1
    assert analysis[0]["draw_surprise"] == pytest.approx(0.75)


def test_run_draw_discovery_synthetic(tmp_path: Path):
    result = run_draw_discovery(_synthetic_rows(150), validation_fraction=0.2)
    assert result.n_train > 0
    assert result.n_validation > 0
    assert "p_draw_blend" in result.feature_names
    assert result.l1_coefficients
    paths = write_discovery_artifacts(result, tmp_path)
    assert paths["l1"].exists()
    assert paths["importance"].exists()
    markdown = render_draw_driver_markdown(result)
    assert "Draw driver analysis" in markdown
    assert "Stable L1 drivers" in markdown
