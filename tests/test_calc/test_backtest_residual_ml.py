"""Tests for residual ML backtest row filters."""

from __future__ import annotations

import pytest

from calc.residual_ml.filters import filter_rows_by_draw, select_backtest_rows


def _row(match_id: int, match_date: str, draw_number: int, label: str = "1") -> dict:
    return {
        "match_id": match_id,
        "match_date": match_date,
        "draw_number": draw_number,
        "label": label,
    }


def test_filter_rows_by_draw_inclusive_range():
    rows = [
        _row(1, "2024-01-01", 100),
        _row(2, "2024-01-02", 101),
        _row(3, "2024-01-03", 102),
        _row(4, "2024-01-04", 103),
    ]
    assert [row["draw_number"] for row in filter_rows_by_draw(rows, min_draw=101)] == [
        101,
        102,
        103,
    ]
    assert [row["draw_number"] for row in filter_rows_by_draw(rows, max_draw=101)] == [
        100,
        101,
    ]
    assert [
        row["draw_number"]
        for row in filter_rows_by_draw(rows, min_draw=101, max_draw=102)
    ] == [101, 102]


def test_filter_rows_by_draw_accepts_string_draw_numbers():
    rows = [{"draw_number": "4960", "match_id": 1, "match_date": "2024-01-01"}]
    filtered = filter_rows_by_draw(rows, min_draw=4960, max_draw=4960)
    assert len(filtered) == 1


def test_select_backtest_rows_default_uses_validation_split():
    rows = [_row(index, f"2024-01-{index:02d}", 100 + index) for index in range(1, 11)]
    selected, description = select_backtest_rows(rows, validation_fraction=0.2)
    assert len(selected) == 2
    assert "time-split validation fraction=0.2" in description
    assert selected[0]["match_id"] == 9
    assert selected[1]["match_id"] == 10


def test_select_backtest_rows_all_rows_keeps_full_set():
    rows = [_row(index, f"2024-01-{index:02d}", 100 + index) for index in range(1, 6)]
    selected, description = select_backtest_rows(rows, all_rows=True, min_draw=102)
    assert len(selected) == 4
    assert "all rows" in description
    assert "optimistic" in description


def test_select_backtest_rows_rejects_invalid_fraction():
    with pytest.raises(ValueError, match="validation_fraction"):
        select_backtest_rows([_row(1, "2024-01-01", 1)], validation_fraction=1.0)


def test_select_backtest_rows_max_draw_excludes_holdout():
    rows = [_row(index, f"2024-01-{index:02d}", draw_number) for index, draw_number in [
        (1, 4948),
        (2, 4949),
        (3, 4950),
        (4, 4951),
        (5, 4960),
    ]]
    selected, description = select_backtest_rows(
        rows,
        all_rows=True,
        max_draw=4950,
    )
    assert [row["draw_number"] for row in selected] == [4948, 4949, 4950]
    assert "4951" not in description
    assert "4960" not in description
