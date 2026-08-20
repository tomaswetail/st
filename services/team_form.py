"""Recent form features from match history."""

from __future__ import annotations

from collections import defaultdict
from datetime import date

from objects.schema.db.fixture import Fixture
from utils.fixture_fields import (
    fixture_away_name,
    fixture_goals_away,
    fixture_goals_home,
    fixture_home_name,
    fixture_match_date,
    fixture_outcome,
)


def _points_for_team(is_home: bool, match_result: str) -> int:
    if match_result == "X":
        return 1
    if match_result == "1":
        return 3 if is_home else 0
    if match_result == "2":
        return 0 if is_home else 3
    return 0


class TeamFormCalculator:
    """Compute last-N match form per team."""

    def __init__(self, n_matches: int = 5) -> None:
        self.n = n_matches
        self._history: dict[str, list[tuple[date, bool, int, int, str]]] = defaultdict(
            list
        )

    def ingest(self, matches: list[Fixture]) -> None:
        for match in sorted(matches, key=fixture_match_date):
            result = fixture_outcome(match) or "X"
            goals_home = fixture_goals_home(match) or 0
            goals_away = fixture_goals_away(match) or 0
            self._history[fixture_home_name(match)].append(
                (fixture_match_date(match), True, goals_home, goals_away, result)
            )
            self._history[fixture_away_name(match)].append(
                (fixture_match_date(match), False, goals_away, goals_home, result)
            )

    def form_before(
        self, team: str, before: date
    ) -> tuple[int | None, int | None, int | None]:
        """Points, goals for, goals against in last N before date."""
        entries = [e for e in self._history.get(team, []) if e[0] < before]
        entries = entries[-self.n :]
        if not entries:
            return None, None, None
        points = sum(_points_for_team(h, r) for _, h, gf, ga, r in entries)
        gf_total = sum(e[2] for e in entries)
        ga_total = sum(e[3] for e in entries)
        return points, gf_total, ga_total
