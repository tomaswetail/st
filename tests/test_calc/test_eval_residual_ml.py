"""Tests for multi-slice residual ML evaluation helpers."""

from __future__ import annotations

from src.calc.residual_ml.evaluation import (
    build_multi_slice_report,
    draw_quarter_ranges,
    run_backtest_scoring,
    slice_rows_by_availability,
    slice_rows_by_draw_quarter,
    slice_rows_by_league,
    slice_rows_by_year,
    top_pick_accuracy,
)

MARKET_KEYS = ("p_home_market_norm", "p_draw_market_norm", "p_away_market_norm")


def _row(
    match_id: int,
    match_date: str,
    draw_number: int,
    label: str = "1",
    *,
    league_external_id: int | None = None,
    market_probs: tuple[float, float, float] | None = None,
) -> dict:
    row = {
        "match_id": match_id,
        "match_date": match_date,
        "draw_number": draw_number,
        "label": label,
    }
    if league_external_id is not None:
        row["league_external_id"] = league_external_id
    if market_probs is not None:
        row["p_home_market_norm"] = market_probs[0]
        row["p_draw_market_norm"] = market_probs[1]
        row["p_away_market_norm"] = market_probs[2]
    return row


def test_draw_quarter_ranges_cover_full_window():
    ranges = draw_quarter_ranges(draw_min=4760, draw_max=4960)
    assert [label for label, _, _ in ranges] == ["Q1", "Q2", "Q3", "Q4"]
    assert ranges[0][1] == 4760
    assert ranges[-1][2] == 4960
    assert sum(end - start + 1 for _, start, end in ranges) == 201


def test_slice_rows_by_year_groups_match_dates():
    rows = [
        _row(1, "2024-03-01", 4800),
        _row(2, "2024-08-01", 4810),
        _row(3, "2025-01-01", 4900),
    ]
    grouped = slice_rows_by_year(rows)
    assert set(grouped.keys()) == {"2024", "2025"}
    assert len(grouped["2024"]) == 2
    assert len(grouped["2025"]) == 1


def test_slice_rows_by_draw_quarter_assigns_rows():
    rows = [
        _row(1, "2024-01-01", 4760),
        _row(2, "2024-01-02", 4810),
        _row(3, "2024-01-03", 4860),
        _row(4, "2024-01-04", 4910),
    ]
    grouped = slice_rows_by_draw_quarter(rows, draw_min=4760, draw_max=4960)
    assert [row["match_id"] for row in grouped["Q1"]] == [1]
    assert [row["match_id"] for row in grouped["Q2"]] == [2]
    assert [row["match_id"] for row in grouped["Q3"]] == [3]
    assert [row["match_id"] for row in grouped["Q4"]] == [4]


def test_slice_rows_by_league_returns_none_without_column():
    rows = [_row(1, "2024-01-01", 4800)]
    assert slice_rows_by_league(rows) is None


def test_slice_rows_by_league_groups_present_column():
    rows = [
        _row(1, "2024-01-01", 4800, league_external_id=39),
        _row(2, "2024-01-02", 4801, league_external_id=39),
        _row(3, "2024-01-03", 4802, league_external_id=41),
    ]
    grouped = slice_rows_by_league(rows)
    assert grouped is not None
    assert len(grouped["39"]) == 2
    assert len(grouped["41"]) == 1


def test_top_pick_accuracy_counts_argmax_matches():
    rows = [
        _row(1, "2024-01-01", 4800, label="1", market_probs=(0.7, 0.2, 0.1)),
        _row(2, "2024-01-02", 4801, label="X", market_probs=(0.7, 0.2, 0.1)),
        _row(3, "2024-01-03", 4802, label="2", market_probs=(0.1, 0.2, 0.7)),
    ]
    accuracy, count = top_pick_accuracy(rows, MARKET_KEYS)
    assert count == 3
    assert accuracy == 2 / 3


def test_slice_rows_by_availability_groups_flag():
    rows = [
        {**_row(1, "2024-01-01", 4800), "has_availability": 1},
        {**_row(2, "2024-01-02", 4801), "has_availability": 0},
        {**_row(3, "2024-01-03", 4802), "has_availability": 1},
    ]
    grouped = slice_rows_by_availability(rows)
    assert grouped is not None
    assert len(grouped["availability:1"]) == 2
    assert len(grouped["availability:0"]) == 1


class _FakeTrainer:
    def predict_match_proba(self, row):
        return {"1": 0.50, "X": 0.30, "2": 0.20}


def _scored_row(
    match_id: int,
    match_date: str,
    draw_number: int,
    label: str,
    *,
    league_external_id: int,
    market_probs: tuple[float, float, float],
) -> dict:
    return _row(
        match_id,
        match_date,
        draw_number,
        label,
        league_external_id=league_external_id,
        market_probs=market_probs,
    )


def _scored_fixture_rows() -> list[dict]:
    return [
        _scored_row(
            1, "2025-03-01", 4900, "1", league_external_id=39, market_probs=(0.60, 0.25, 0.15)
        ),
        _scored_row(
            2, "2025-03-08", 4901, "X", league_external_id=39, market_probs=(0.45, 0.30, 0.25)
        ),
        _scored_row(
            3, "2026-01-10", 4950, "2", league_external_id=180, market_probs=(0.35, 0.30, 0.35)
        ),
        _scored_row(
            4, "2026-01-17", 4951, "1", league_external_id=180, market_probs=(0.50, 0.28, 0.22)
        ),
    ]


def test_build_multi_slice_report_fills_ml_on_year_and_league():
    rows = _scored_fixture_rows()
    trainer = _FakeTrainer()
    report = build_multi_slice_report(
        rows,
        trainer=trainer,
        production_shrink_alpha=0.7,
    )

    for slice_name in ("pooled", "year:2025", "year:2026", "league:39", "league:180"):
        metrics = report[slice_name]
        assert metrics.ml_log_loss is not None, slice_name
        assert metrics.best_shrink_log_loss is not None, slice_name
        assert metrics.best_shrink_alpha is not None, slice_name
        assert metrics.production_shrink_alpha == 0.7
        assert metrics.production_shrink_log_loss is not None, slice_name


def test_build_multi_slice_report_pooled_matches_direct_scoring():
    rows = _scored_fixture_rows()
    trainer = _FakeTrainer()
    scoring = run_backtest_scoring(rows, trainer)
    report = build_multi_slice_report(rows, trainer=trainer)

    raw_ml = next(loss for alpha, loss in scoring.shrink_results if alpha <= 0)
    assert report["pooled"].ml_log_loss == raw_ml
    assert report["pooled"].best_shrink_log_loss == scoring.best_loss
    assert report["pooled"].best_shrink_alpha == scoring.best_alpha
    assert report["pooled"].row_count == len(rows)


def test_build_multi_slice_report_without_trainer_keeps_ml_null():
    rows = _scored_fixture_rows()
    report = build_multi_slice_report(rows)

    assert report["pooled"].baselines["market"][0] is not None
    assert report["year:2025"].baselines["market"][0] is not None
    assert report["league:39"].baselines["market"][0] is not None
    assert report["pooled"].ml_log_loss is None
    assert report["year:2025"].ml_log_loss is None
    assert report["year:2025"].best_shrink_log_loss is None
    assert report["league:39"].ml_log_loss is None
    assert report["league:39"].best_shrink_log_loss is None
