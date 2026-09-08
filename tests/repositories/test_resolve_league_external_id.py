"""Tests for league_external_id resolution via fixtures."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.objects.repositories.fixture_repository import FixtureRepository


def _repo() -> FixtureRepository:
    session = MagicMock()
    repo = FixtureRepository(session)
    repo.league_repo = MagicMock()
    return repo


def _team(*, team_id: int, external_id: int, name: str = "Team") -> SimpleNamespace:
    return SimpleNamespace(id=team_id, external_id=external_id, name=name)


def _match(
    *,
    league_name: str | None = "Unknown Cup",
    league_country_name: str | None = "England",
    home: SimpleNamespace | None = None,
    away: SimpleNamespace | None = None,
    start_time: datetime | None = datetime(2026, 5, 30, 15, 0, tzinfo=timezone.utc),
) -> SimpleNamespace:
    return SimpleNamespace(
        league_name=league_name,
        league_country_name=league_country_name,
        home_team=home or _team(team_id=1, external_id=100),
        away_team=away or _team(team_id=2, external_id=200),
        start_time=start_time,
    )


def test_name_hit_returns_external_id_without_fixture_lookup():
    repo = _repo()
    repo.league_repo.get_by_name_and_country = MagicMock(
        return_value=SimpleNamespace(external_id=39)
    )
    repo.find_by_date_range_and_teams = MagicMock(return_value=[])

    resolved = repo.resolve_league_external_id_for_match(_match(league_name="Premier League"))

    assert resolved == 39
    repo.find_by_date_range_and_teams.assert_not_called()


def test_name_miss_uses_this_match_fixture():
    repo = _repo()
    repo.league_repo.get_by_name_and_country = MagicMock(return_value=None)
    repo.league_repo.get_by_name = MagicMock(return_value=None)
    fixture = SimpleNamespace(
        league_id=39,
        fixture_date=datetime(2026, 5, 30, 15, 0, tzinfo=timezone.utc),
    )
    repo.find_by_date_range_and_teams = MagicMock(return_value=[fixture])

    resolved = repo.resolve_league_external_id_for_match(_match())

    assert resolved == 39
    repo.find_by_date_range_and_teams.assert_called_once()


def test_this_match_fixture_wins_over_team_history():
    repo = _repo()
    repo.league_repo.get_by_name_and_country = MagicMock(return_value=None)
    repo.league_repo.get_by_name = MagicMock(return_value=None)
    fixture = SimpleNamespace(
        league_id=39,
        fixture_date=datetime(2026, 5, 30, 15, 0, tzinfo=timezone.utc),
    )
    repo.find_by_date_range_and_teams = MagicMock(return_value=[fixture])
    repo.resolve_league_external_id_for_team = MagicMock(return_value=2)

    resolved = repo.resolve_league_external_id_for_match(_match())

    assert resolved == 39
    repo.resolve_league_external_id_for_team.assert_not_called()


def test_home_then_away_history_when_no_fixture():
    repo = _repo()
    repo.league_repo.get_by_name_and_country = MagicMock(return_value=None)
    repo.league_repo.get_by_name = MagicMock(return_value=None)
    repo.find_by_date_range_and_teams = MagicMock(return_value=[])
    home = _team(team_id=1, external_id=100)
    away = _team(team_id=2, external_id=200)

    def _history(team: SimpleNamespace) -> int | None:
        if team.external_id == 100:
            return 39
        return None

    repo.resolve_league_external_id_for_team = MagicMock(side_effect=_history)

    resolved = repo.resolve_league_external_id_for_match(
        _match(home=home, away=away)
    )

    assert resolved == 39


def test_home_away_history_disagreement_returns_none():
    repo = _repo()
    repo.league_repo.get_by_name_and_country = MagicMock(return_value=None)
    repo.league_repo.get_by_name = MagicMock(return_value=None)
    repo.find_by_date_range_and_teams = MagicMock(return_value=[])
    home = _team(team_id=1, external_id=100)
    away = _team(team_id=2, external_id=200)

    def _history(team: SimpleNamespace) -> int | None:
        return 39 if team.external_id == 100 else 45

    repo.resolve_league_external_id_for_team = MagicMock(side_effect=_history)

    resolved = repo.resolve_league_external_id_for_match(
        _match(home=home, away=away)
    )

    assert resolved is None


def test_skip_name_goes_straight_to_fixture_path():
    repo = _repo()
    repo._league_external_id_from_match_name = MagicMock(return_value=39)
    fixture = SimpleNamespace(
        league_id=180,
        fixture_date=datetime(2026, 5, 30, 15, 0, tzinfo=timezone.utc),
    )
    repo.find_by_date_range_and_teams = MagicMock(return_value=[fixture])

    resolved = repo.resolve_league_external_id_for_match(
        _match(), skip_name=True
    )

    assert resolved == 180
    repo._league_external_id_from_match_name.assert_not_called()
