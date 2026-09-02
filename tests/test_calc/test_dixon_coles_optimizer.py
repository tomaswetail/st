"""Unit tests for classic Dixon–Coles hyperparameter optimizer."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from calc.dixon_coles.metrics import log_loss_one
from calc.dixon_coles.model import DixonColesModel, filter_matches_by_lookback
from calc.dixon_coles.optimizer import (
    DixonColesOptimizer,
    group_eval_matches_by_league,
    optimize_single_league,
)
from calc.dixon_coles.types import DixonColesMatch
from calc.dixon_coles.walk_forward import EvalMatch


def _fixture_match(
    day_offset: int,
    home_team_id: int,
    away_team_id: int,
    goals_home: int,
    goals_away: int,
    *,
    base: date = date(2024, 6, 1),
) -> DixonColesMatch:
    return DixonColesMatch(
        match_date=base + timedelta(days=day_offset),
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        goals_home=goals_home,
        goals_away=goals_away,
    )


def _eval_match(
    match_date: date,
    *,
    label: str = "1",
    league_external_id: int = 1,
    home: int = 10,
    away: int = 20,
) -> EvalMatch:
    return EvalMatch(
        match_date=match_date,
        home_team_external_id=home,
        away_team_external_id=away,
        label=label,
        p_home_market=0.5,
        p_draw_market=0.25,
        p_away_market=0.25,
        league_external_id=league_external_id,
    )


def _synthetic_league_history(
    *,
    as_of: date,
    home_bias: int,
) -> list[DixonColesMatch]:
    teams = [10, 20, 30, 40]
    matches: list[DixonColesMatch] = []
    day = 0
    for _round in range(10):
        for index, home in enumerate(teams):
            away = teams[(index + 1) % len(teams)]
            matches.append(
                _fixture_match(
                    day,
                    home,
                    away,
                    goals_home=home_bias,
                    goals_away=0,
                    base=as_of - timedelta(days=200),
                )
            )
            day += 2
    return matches


def test_group_eval_matches_by_league():
    matches = [
        _eval_match(date(2024, 1, 1), league_external_id=39),
        _eval_match(date(2024, 1, 2), league_external_id=140),
        _eval_match(date(2024, 1, 3), league_external_id=39),
    ]
    grouped = group_eval_matches_by_league(matches)
    assert len(grouped[39]) == 2
    assert len(grouped[140]) == 1


def test_optimizer_skips_leagues_with_insufficient_validation_matches():
    validation_start = date(2024, 8, 1)
    validation_end = date(2024, 8, 3)
    eval_by_league = {
        39: [_eval_match(validation_start, league_external_id=39)],
        140: [
            _eval_match(validation_start, league_external_id=140),
            _eval_match(validation_end, league_external_id=140),
        ],
    }
    fixtures_by_league = {
        39: _synthetic_league_history(as_of=validation_end, home_bias=2),
        140: _synthetic_league_history(as_of=validation_end, home_bias=2),
    }
    optimizer = DixonColesOptimizer()
    result = optimizer.optimize(
        eval_matches_by_league=eval_by_league,
        fixtures_by_league=fixtures_by_league,
        validation_start=validation_start,
        validation_end=validation_end,
        xi_values=[0.001],
        lookback_values=[365],
        rho_values=[-0.1],
        min_training_matches=20,
        min_team_matches=3,
        min_eval_matches_per_league=2,
    )
    assert 39 not in result.league_results
    assert 140 in result.league_results
    assert "too few validation matches" in result.skipped_leagues[39]


def test_optimizer_prefers_better_parameter_region_on_synthetic_data():
    validation_start = date(2024, 9, 1)
    validation_end = date(2024, 9, 3)
    eval_matches = [
        _eval_match(validation_start, label="1", home=10, away=20, league_external_id=39),
        _eval_match(validation_end, label="1", home=10, away=20, league_external_id=39),
        _eval_match(validation_end, label="1", home=30, away=40, league_external_id=39),
    ]
    fixtures = _synthetic_league_history(as_of=validation_end, home_bias=3)
    optimizer = DixonColesOptimizer()
    result = optimizer.optimize(
        eval_matches_by_league={39: eval_matches},
        fixtures_by_league={39: fixtures},
        validation_start=validation_start,
        validation_end=validation_end,
        xi_values=[0.0005, 0.005],
        lookback_values=[365, 730],
        rho_values=[-0.1, 0.0],
        min_training_matches=20,
        min_team_matches=3,
        min_eval_matches_per_league=2,
    )
    league_result = result.league_results[39]
    assert league_result.evaluated_matches > 0
    best = min(league_result.parameter_results, key=lambda row: row.log_loss)
    assert league_result.best_xi == best.xi
    assert league_result.best_lookback == best.lookback
    assert league_result.best_rho == best.rho


def test_optimizer_is_deterministic():
    validation_start = date(2024, 10, 1)
    validation_end = date(2024, 10, 2)
    eval_matches = [
        _eval_match(validation_start, label="1", league_external_id=39),
        _eval_match(validation_end, label="X", league_external_id=39),
    ]
    fixtures = _synthetic_league_history(as_of=validation_end, home_bias=2)
    kwargs = dict(
        eval_matches_by_league={39: eval_matches},
        fixtures_by_league={39: fixtures},
        validation_start=validation_start,
        validation_end=validation_end,
        xi_values=[0.001],
        lookback_values=[365],
        rho_values=[-0.1],
        min_training_matches=20,
        min_team_matches=3,
        min_eval_matches_per_league=2,
    )
    optimizer = DixonColesOptimizer()
    first = optimizer.optimize(**kwargs)
    second = optimizer.optimize(**kwargs)
    assert first.league_results[39].best_xi == second.league_results[39].best_xi
    assert first.league_results[39].log_loss == second.league_results[39].log_loss


def test_filter_matches_by_lookback_excludes_outside_window():
    as_of = date(2024, 6, 1)
    matches = [
        _fixture_match(-400, 1, 2, 1, 0, base=as_of),
        _fixture_match(-30, 1, 2, 1, 0, base=as_of),
    ]
    filtered = filter_matches_by_lookback(matches, as_of=as_of, lookback_days=60)
    assert len(filtered) == 1
    assert filtered[0].match_date == as_of - timedelta(days=30)


def test_different_rho_changes_low_score_probabilities():
    as_of = date(2024, 7, 1)
    matches = _synthetic_league_history(as_of=as_of, home_bias=1)
    model_negative = DixonColesModel(
        xi=0.001,
        rho=-0.2,
        lookback_days=365,
        min_team_matches=3,
        as_of=as_of,
    ).fit(matches, as_of=as_of)
    model_zero = DixonColesModel(
        xi=0.001,
        rho=0.0,
        lookback_days=365,
        min_team_matches=3,
        as_of=as_of,
    ).fit(matches, as_of=as_of)
    negative = model_negative.predict(10, 20)
    zero = model_zero.predict(10, 20)
    assert negative.p_draw != pytest.approx(zero.p_draw)
    assert abs(
        (negative.p_home + negative.p_draw + negative.p_away) - 1.0
    ) < 1e-6


def test_log_loss_on_known_probability():
    loss = log_loss_one("1", 0.6, 0.2, 0.2)
    assert loss == pytest.approx(-math.log(0.6), rel=1e-6)


def _optimizer_kwargs(
    *,
    eval_by_league: dict[int, list[EvalMatch]],
    fixtures_by_league: dict[int, list[DixonColesMatch]],
    validation_start: date,
    validation_end: date,
) -> dict:
    return dict(
        eval_matches_by_league=eval_by_league,
        fixtures_by_league=fixtures_by_league,
        validation_start=validation_start,
        validation_end=validation_end,
        xi_values=[0.001, 0.002],
        lookback_values=[365, 730],
        rho_values=[-0.1, 0.0],
        min_training_matches=20,
        min_team_matches=3,
        min_eval_matches_per_league=2,
    )


def test_optimizer_avg_training_cached_per_lookback():
    validation_start = date(2024, 10, 1)
    validation_end = date(2024, 10, 2)
    eval_matches = [
        _eval_match(validation_start, label="1", league_external_id=39),
        _eval_match(validation_end, label="X", league_external_id=39),
    ]
    fixtures = _synthetic_league_history(as_of=validation_end, home_bias=2)
    optimizer = DixonColesOptimizer()
    result = optimizer.optimize(
        **_optimizer_kwargs(
            eval_by_league={39: eval_matches},
            fixtures_by_league={39: fixtures},
            validation_start=validation_start,
            validation_end=validation_end,
        ),
    )
    league_result = result.league_results[39]
    by_lookback: dict[int, list[float]] = {}
    for row in league_result.parameter_results:
        by_lookback.setdefault(row.lookback, []).append(row.avg_training_matches)
    for lookback, values in by_lookback.items():
        assert len(values) > 1
        assert values == pytest.approx([values[0]] * len(values))


def test_optimizer_parallel_matches_sequential():
    validation_start = date(2024, 9, 1)
    validation_end = date(2024, 9, 3)
    eval_by_league = {
        39: [
            _eval_match(validation_start, label="1", league_external_id=39),
            _eval_match(validation_end, label="1", league_external_id=39),
        ],
        140: [
            _eval_match(validation_start, label="X", league_external_id=140),
            _eval_match(validation_end, label="2", league_external_id=140),
        ],
    }
    fixtures_by_league = {
        39: _synthetic_league_history(as_of=validation_end, home_bias=3),
        140: _synthetic_league_history(as_of=validation_end, home_bias=1),
    }
    kwargs = _optimizer_kwargs(
        eval_by_league=eval_by_league,
        fixtures_by_league=fixtures_by_league,
        validation_start=validation_start,
        validation_end=validation_end,
    )
    optimizer = DixonColesOptimizer()
    sequential = optimizer.optimize(**kwargs, jobs=1)
    parallel = optimizer.optimize(**kwargs, jobs=2)
    for league_id in (39, 140):
        seq = sequential.league_results[league_id]
        par = parallel.league_results[league_id]
        assert seq.best_xi == par.best_xi
        assert seq.best_lookback == par.best_lookback
        assert seq.best_rho == par.best_rho
        assert seq.log_loss == pytest.approx(par.log_loss)


def test_optimizer_warm_start_matches_cold_start():
    validation_start = date(2024, 9, 1)
    validation_end = date(2024, 9, 3)
    eval_matches = [
        _eval_match(validation_start, label="1", league_external_id=39),
        _eval_match(validation_end, label="1", league_external_id=39),
    ]
    fixtures = _synthetic_league_history(as_of=validation_end, home_bias=3)
    kwargs = dict(
        validation_start=validation_start,
        validation_end=validation_end,
        xi_values=[0.001, 0.002],
        lookback_values=[365, 730],
        rho_values=[-0.1, 0.0],
        min_training_matches=20,
        min_team_matches=3,
        max_goals=10,
        log_progress=False,
    )
    warm = optimize_single_league(
        39,
        eval_matches,
        fixtures,
        use_warm_start=True,
        **kwargs,
    )
    cold = optimize_single_league(
        39,
        eval_matches,
        fixtures,
        use_warm_start=False,
        **kwargs,
    )
    assert warm.best_xi == cold.best_xi
    assert warm.best_lookback == cold.best_lookback
    assert warm.best_rho == cold.best_rho
    assert warm.log_loss == pytest.approx(cold.log_loss, abs=1e-5)
