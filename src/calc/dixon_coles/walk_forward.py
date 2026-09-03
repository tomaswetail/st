"""Chronological walk-forward evaluation for classic Dixon–Coles."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from calc.dixon_coles.metrics import mean_log_loss, mean_rps
from calc.dixon_coles.model import DixonColesModel


@dataclass(frozen=True)
class EvalMatch:
    """One labeled Stryktipset match for walk-forward scoring."""

    match_date: date
    home_team_external_id: int
    away_team_external_id: int
    label: str
    p_home_market: float
    p_draw_market: float
    p_away_market: float
    league_external_id: int


@dataclass
class WalkForwardResult:
    """Aggregated DC vs market metrics on scored walk-forward rows."""

    dc_log_loss: float
    market_log_loss: float
    dc_rps: float
    market_rps: float
    n_scored: int
    n_skipped: int
    skip_reasons: dict[str, int] = field(default_factory=dict)
    validation_start: date | None = None
    validation_end: date | None = None

    @property
    def beats_market(self) -> bool:
        return self.n_scored > 0 and self.dc_log_loss < self.market_log_loss


FitForDate = Callable[[date], DixonColesModel | None]
FitForLeagueAndDate = Callable[[int, date], DixonColesModel | None]


def _score_matches(
    by_group: dict[tuple[int, date], list[EvalMatch]],
    fit_fn: Callable[[int, date], DixonColesModel | None],
) -> WalkForwardResult:
    labels: list[str] = []
    dc_rows: list[tuple[float, float, float]] = []
    market_rows: list[tuple[float, float, float]] = []
    skip_reasons: dict[str, int] = defaultdict(int)
    n_skipped = 0

    for league_id, day in sorted(by_group):
        model = fit_fn(league_id, day)
        if model is None:
            skipped = len(by_group[(league_id, day)])
            n_skipped += skipped
            skip_reasons["no_fit"] += skipped
            continue
        for match in by_group[(league_id, day)]:
            try:
                prediction = model.predict(
                    match.home_team_external_id,
                    match.away_team_external_id,
                )
            except Exception:
                n_skipped += 1
                skip_reasons["predict_error"] += 1
                continue
            labels.append(match.label)
            dc_rows.append(
                (prediction.p_home, prediction.p_draw, prediction.p_away)
            )
            market_rows.append(
                (
                    match.p_home_market,
                    match.p_draw_market,
                    match.p_away_market,
                )
            )

    n_scored = len(labels)
    dates = sorted({match_date for _, match_date in by_group})
    return WalkForwardResult(
        dc_log_loss=mean_log_loss(labels, dc_rows) if n_scored else 0.0,
        market_log_loss=mean_log_loss(labels, market_rows) if n_scored else 0.0,
        dc_rps=mean_rps(labels, dc_rows) if n_scored else 0.0,
        market_rps=mean_rps(labels, market_rows) if n_scored else 0.0,
        n_scored=n_scored,
        n_skipped=n_skipped,
        skip_reasons=dict(skip_reasons),
        validation_start=dates[0] if dates else None,
        validation_end=dates[-1] if dates else None,
    )


def run_walk_forward(
    eval_matches: list[EvalMatch],
    fit_for_date: FitForDate,
) -> WalkForwardResult:
    """Fit once per calendar day and score all matches that day.

    ``fit_for_date(day)`` must use only history strictly before ``day``.
    For multi-league data use ``run_walk_forward_per_league`` instead.
    """
    if not eval_matches:
        return WalkForwardResult(
            dc_log_loss=0.0,
            market_log_loss=0.0,
            dc_rps=0.0,
            market_rps=0.0,
            n_scored=0,
            n_skipped=0,
        )

    by_day: dict[date, list[EvalMatch]] = defaultdict(list)
    for match in eval_matches:
        by_day[match.match_date].append(match)

    def fit_fn(_league_id: int, day: date) -> DixonColesModel | None:
        return fit_for_date(day)

    by_group: dict[tuple[int, date], list[EvalMatch]] = defaultdict(list)
    for day, matches in by_day.items():
        league_id = matches[0].league_external_id
        by_group[(league_id, day)] = matches

    return _score_matches(by_group, fit_fn)


def run_walk_forward_per_league(
    eval_matches: list[EvalMatch],
    fit_for_league_and_date: FitForLeagueAndDate,
) -> WalkForwardResult:
    """Fit once per (league, calendar day) and score all matches in that group."""
    if not eval_matches:
        return WalkForwardResult(
            dc_log_loss=0.0,
            market_log_loss=0.0,
            dc_rps=0.0,
            market_rps=0.0,
            n_scored=0,
            n_skipped=0,
        )

    by_group: dict[tuple[int, date], list[EvalMatch]] = defaultdict(list)
    for match in eval_matches:
        by_group[(match.league_external_id, match.match_date)].append(match)

    return _score_matches(by_group, fit_for_league_and_date)
