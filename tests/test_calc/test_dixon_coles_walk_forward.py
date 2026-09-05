"""Unit tests for classic Dixon–Coles walk-forward (no DB)."""

from __future__ import annotations

from datetime import date

import pytest

from src.calc.dixon_coles.metrics import log_loss_one, ranked_probability_score
from src.calc.dixon_coles.model import DixonColesModel, DixonColesPrediction
from src.calc.dixon_coles.walk_forward import EvalMatch, run_walk_forward, run_walk_forward_per_league


def _eval(
    match_date: date,
    *,
    label: str = "1",
    home: int = 1,
    away: int = 2,
    market: tuple[float, float, float] = (0.5, 0.3, 0.2),
    league_external_id: int = 39,
) -> EvalMatch:
    return EvalMatch(
        match_date=match_date,
        home_team_external_id=home,
        away_team_external_id=away,
        label=label,
        p_home_market=market[0],
        p_draw_market=market[1],
        p_away_market=market[2],
        league_external_id=league_external_id,
    )


def _stub_model(p_home: float, p_draw: float, p_away: float) -> DixonColesModel:
    model = DixonColesModel()
    model._fitted = True

    def predict(home_team_id: int, away_team_id: int) -> DixonColesPrediction:
        del home_team_id, away_team_id
        return DixonColesPrediction(
            lambda_home=1.0,
            lambda_away=1.0,
            p_home=p_home,
            p_draw=p_draw,
            p_away=p_away,
        )

    model.predict = predict  # type: ignore[method-assign]
    return model


def test_walk_forward_requests_fit_on_match_days_only_chronologically():
    matches = [
        _eval(date(2024, 1, 10)),
        _eval(date(2024, 1, 10), home=3, away=4),
        _eval(date(2024, 1, 12), label="X"),
        _eval(date(2024, 1, 15), label="2"),
    ]
    requested: list[date] = []

    def fit_for_date(day: date) -> DixonColesModel:
        requested.append(day)
        return _stub_model(0.4, 0.3, 0.3)

    result = run_walk_forward(matches, fit_for_date)
    assert requested == [date(2024, 1, 10), date(2024, 1, 12), date(2024, 1, 15)]
    assert result.n_scored == 4
    assert result.n_skipped == 0
    # No future day relative to earlier requests.
    assert requested == sorted(requested)


def test_walk_forward_known_probs_match_metrics():
    matches = [
        _eval(date(2024, 2, 1), label="1", market=(0.5, 0.25, 0.25)),
        _eval(date(2024, 2, 2), label="X", market=(0.2, 0.5, 0.3)),
    ]
    dc_probs = (0.6, 0.2, 0.2)

    def fit_for_date(_day: date) -> DixonColesModel:
        return _stub_model(*dc_probs)

    result = run_walk_forward(matches, fit_for_date)
    expected_dc = (
        log_loss_one("1", *dc_probs) + log_loss_one("X", *dc_probs)
    ) / 2
    expected_market = (
        log_loss_one("1", 0.5, 0.25, 0.25) + log_loss_one("X", 0.2, 0.5, 0.3)
    ) / 2
    expected_dc_rps = (
        ranked_probability_score("1", *dc_probs)
        + ranked_probability_score("X", *dc_probs)
    ) / 2
    assert result.dc_log_loss == pytest.approx(expected_dc)
    assert result.market_log_loss == pytest.approx(expected_market)
    assert result.dc_rps == pytest.approx(expected_dc_rps)


def test_walk_forward_skips_when_fit_returns_none():
    matches = [
        _eval(date(2024, 3, 1)),
        _eval(date(2024, 3, 2)),
    ]

    def fit_for_date(day: date) -> DixonColesModel | None:
        if day == date(2024, 3, 1):
            return None
        return _stub_model(0.4, 0.3, 0.3)

    result = run_walk_forward(matches, fit_for_date)
    assert result.n_scored == 1
    assert result.n_skipped == 1
    assert result.skip_reasons["no_fit"] == 1


def test_walk_forward_is_deterministic():
    matches = [
        _eval(date(2024, 4, 1), label="1"),
        _eval(date(2024, 4, 2), label="2"),
    ]

    def fit_for_date(_day: date) -> DixonColesModel:
        return _stub_model(0.45, 0.25, 0.3)

    first = run_walk_forward(matches, fit_for_date)
    second = run_walk_forward(matches, fit_for_date)
    assert first.dc_log_loss == second.dc_log_loss
    assert first.market_log_loss == second.market_log_loss
    assert first.n_scored == second.n_scored
    assert first.beats_market == second.beats_market


def test_walk_forward_per_league_fits_per_league_and_day():
    matches = [
        _eval(date(2024, 5, 1), league_external_id=39),
        _eval(date(2024, 5, 1), home=3, away=4, league_external_id=140),
        _eval(date(2024, 5, 2), league_external_id=39),
    ]
    requested: list[tuple[int, date]] = []

    def fit_for_league_and_date(league_id: int, day: date) -> DixonColesModel:
        requested.append((league_id, day))
        return _stub_model(0.4, 0.3, 0.3)

    result = run_walk_forward_per_league(matches, fit_for_league_and_date)
    assert requested == [
        (39, date(2024, 5, 1)),
        (39, date(2024, 5, 2)),
        (140, date(2024, 5, 1)),
    ]
    assert result.n_scored == 3
    assert result.n_skipped == 0
