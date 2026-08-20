"""Weighted recent form calculation from historical matches."""

from __future__ import annotations

from datetime import date

from objects.schema.data_classes.recent_form_stats import RecentFormStats
from objects.schema.data_classes.team_rest_days import MatchRecentFormFeatures
from objects.schema.db.fixture import Fixture as HistoricalMatch
from utils.fixture_fields import fixture_away_name, fixture_goals_away, fixture_goals_home, fixture_home_name, fixture_match_date, fixture_outcome


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp a float value to the inclusive range [low, high]."""
    return max(low, min(high, value))


def _points_for_team(is_home: bool, result: str) -> int:
    """Return league points earned by a team from a 1/X/2 result."""
    if result == "X":
        return 1
    if result == "1":
        return 3 if is_home else 0
    if result == "2":
        return 0 if is_home else 3
    return 0


def _result_for_team(is_home: bool, result: str) -> str:
    """Map a 1/X/2 result to W/D/L from the given team's perspective."""
    if result == "X":
        return "D"
    if result == "1":
        return "W" if is_home else "L"
    if result == "2":
        return "L" if is_home else "W"
    return "D"


def calculate_recent_form(
    matches: list[HistoricalMatch],
    team: str,
    before_date: date,
    lookback: int = 5,
) -> RecentFormStats:
    """Compute recency-weighted form stats from matches strictly before before_date."""
    relevant: list[tuple[HistoricalMatch, bool]] = []
    for m in matches:
        if fixture_match_date(m) >= before_date:
            continue
        if fixture_home_name(m) == team:
            relevant.append((m, True))
        elif fixture_away_name(m) == team:
            relevant.append((m, False))

    relevant.sort(key=lambda x: fixture_match_date(x[0]))
    selected = relevant[-lookback:]

    if not selected:
        return RecentFormStats(
            team=team,
            before_date=before_date,
            lookback=lookback,
            matches_used=0,
            points=0,
            goals_for=0,
            goals_against=0,
            goal_diff=0,
            wins=0,
            draws=0,
            losses=0,
            weighted_points=0.0,
            max_weighted_points=0.0,
            weighted_form_score=None,
            points_per_match=None,
            goals_for_per_match=None,
            goals_against_per_match=None,
            goal_diff_per_match=None,
        )

    points = 0
    goals_for = 0
    goals_against = 0
    wins = draws = losses = 0
    weighted_points = 0.0
    weights: list[int] = []

    for idx, (match, is_home) in enumerate(selected):
        weight = idx + 1
        weights.append(weight)
        if is_home:
            gf, ga = fixture_goals_home(match) or 0, fixture_goals_away(match) or 0
        else:
            gf, ga = fixture_goals_away(match) or 0, fixture_goals_home(match) or 0
        match_points = _points_for_team(is_home, fixture_outcome(match) or "X")
        points += match_points
        goals_for += gf
        goals_against += ga
        weighted_points += match_points * weight
        outcome = _result_for_team(is_home, fixture_outcome(match) or "X")
        if outcome == "W":
            wins += 1
        elif outcome == "D":
            draws += 1
        else:
            losses += 1

    matches_used = len(selected)
    goal_diff = goals_for - goals_against
    max_weighted_points = 3.0 * sum(weights)
    weighted_form_score = (
        weighted_points / max_weighted_points if max_weighted_points > 0 else None
    )

    return RecentFormStats(
        team=team,
        before_date=before_date,
        lookback=lookback,
        matches_used=matches_used,
        points=points,
        goals_for=goals_for,
        goals_against=goals_against,
        goal_diff=goal_diff,
        wins=wins,
        draws=draws,
        losses=losses,
        weighted_points=weighted_points,
        max_weighted_points=max_weighted_points,
        weighted_form_score=weighted_form_score,
        points_per_match=points / matches_used,
        goals_for_per_match=goals_for / matches_used,
        goals_against_per_match=goals_against / matches_used,
        goal_diff_per_match=goal_diff / matches_used,
    )


def build_match_recent_form_features(
    matches: list[HistoricalMatch],
    home_team: str,
    away_team: str,
    match_date: date,
    lookback: int = 5,
) -> MatchRecentFormFeatures:
    """Compare home and away recent form and compute a form advantage score."""
    home_form = calculate_recent_form(matches, home_team, match_date, lookback)
    away_form = calculate_recent_form(matches, away_team, match_date, lookback)

    notes: list[str] = []

    if home_form.matches_used == 0 and away_form.matches_used == 0:
        notes.append("No recent form data for either team; neutral form adjustment.")
    else:
        if home_form.matches_used < lookback:
            notes.append(
                f"Partial home form data: {home_form.matches_used}/{lookback} matches."
            )
        if away_form.matches_used < lookback:
            notes.append(
                f"Partial away form data: {away_form.matches_used}/{lookback} matches."
            )

    if home_form.weighted_form_score is None or away_form.weighted_form_score is None:
        weighted_form_diff = 0.0
    else:
        weighted_form_diff = home_form.weighted_form_score - away_form.weighted_form_score

    if home_form.points_per_match is None or away_form.points_per_match is None:
        points_diff = 0.0
    else:
        points_diff = home_form.points_per_match - away_form.points_per_match

    if home_form.goal_diff_per_match is None or away_form.goal_diff_per_match is None:
        goal_diff_per_match_diff = 0.0
    else:
        goal_diff_per_match_diff = (
            home_form.goal_diff_per_match - away_form.goal_diff_per_match
        )

    if home_form.goals_for_per_match is None or away_form.goals_for_per_match is None:
        goals_for_per_match_diff = 0.0
    else:
        goals_for_per_match_diff = (
            home_form.goals_for_per_match - away_form.goals_for_per_match
        )

    if home_form.goals_against_per_match is None or away_form.goals_against_per_match is None:
        goals_against_per_match_diff = 0.0
    else:
        goals_against_per_match_diff = (
            away_form.goals_against_per_match - home_form.goals_against_per_match
        )

    confidence = _clamp(
        min(home_form.matches_used, away_form.matches_used) / lookback,
        0.0,
        1.0,
    )

    normalized_points_diff = points_diff / 3.0
    normalized_goal_diff = _clamp(goal_diff_per_match_diff / 3.0, -1.0, 1.0)
    normalized_defensive_diff = _clamp(goals_against_per_match_diff / 3.0, -1.0, 1.0)

    raw_score = (
        0.45 * weighted_form_diff
        + 0.25 * normalized_points_diff
        + 0.20 * normalized_goal_diff
        + 0.10 * normalized_defensive_diff
    )
    form_advantage_score = _clamp(raw_score, -1.0, 1.0) * confidence

    return MatchRecentFormFeatures(
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
        lookback=lookback,
        home_form=home_form,
        away_form=away_form,
        points_diff=points_diff,
        weighted_form_diff=weighted_form_diff,
        goal_diff_per_match_diff=goal_diff_per_match_diff,
        goals_for_per_match_diff=goals_for_per_match_diff,
        goals_against_per_match_diff=goals_against_per_match_diff,
        form_advantage_score=form_advantage_score,
        confidence=confidence,
        notes=notes,
    )
