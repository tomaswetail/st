"""Tests for shared time-split helpers."""

from __future__ import annotations

from src.utils.time_split import (
    DEFAULT_VALIDATION_FRACTION,
    split_index_for_fraction,
    time_split_dataset_rows,
)


def _row(match_id: int, match_date: str) -> dict:
    return {"match_id": match_id, "match_date": match_date}


def test_default_validation_fraction_is_point_two():
    assert DEFAULT_VALIDATION_FRACTION == 0.2


def test_time_split_dataset_rows_uses_last_twenty_percent():
    rows = [_row(index, f"2024-01-{index:02d}") for index in range(1, 11)]
    train_rows, validation_rows = time_split_dataset_rows(rows, validation_fraction=0.2)
    assert len(train_rows) == 8
    assert len(validation_rows) == 2
    assert validation_rows[0]["match_id"] == 9
    assert validation_rows[1]["match_id"] == 10


def test_split_index_for_fraction_matches_dataset_rows():
    rows = [_row(index, f"2024-01-{index:02d}") for index in range(1, 101)]
    split_index = split_index_for_fraction(len(rows), 0.2)
    train_rows, validation_rows = time_split_dataset_rows(rows, validation_fraction=0.2)
    assert split_index == len(train_rows)
    assert len(validation_rows) == len(rows) - split_index


def test_time_split_dataset_rows_rejects_empty_input():
    train_rows, validation_rows = time_split_dataset_rows([], validation_fraction=0.2)
    assert train_rows == []
    assert validation_rows == []


def test_time_split_dataset_rows_sorts_unordered_input():
    rows = [_row(3, "2024-01-03"), _row(1, "2024-01-01"), _row(2, "2024-01-02")]
    _, validation_rows = time_split_dataset_rows(rows, validation_fraction=0.34)
    assert [row["match_id"] for row in validation_rows] == [2, 3]
