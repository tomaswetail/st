"""Unit tests for EntityResolver name variants and match linking."""

from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from data_sources.football_data.entity_resolver import EntityResolver
from objects.schema.data_classes.provider_dtos import ProviderMatch


def _resolver() -> EntityResolver:
    session = MagicMock()
    resolver = EntityResolver(session, provider="fotmob")
    resolver.mapping_repo.get_by_external = MagicMock(return_value=None)
    return resolver


def _provider_match(
    *,
    home_team_name: str = "Nottingham Forest",
    away_team_name: str = "Arsenal",
    kickoff_at: datetime = datetime(2025, 8, 15, 19, 0, tzinfo=timezone.utc),
) -> ProviderMatch:
    return ProviderMatch(
        provider_match_id="4000001",
        provider_league_id="47",
        provider_season_id="2025",
        home_team_id="1",
        away_team_id="2",
        home_team_name=home_team_name,
        away_team_name=away_team_name,
        kickoff_at=kickoff_at,
        status="finished",
    )


def test_names_for_team_includes_name_short_medium_and_mapped():
    resolver = _resolver()
    team = SimpleNamespace(
        name="Nottingham Forest",
        short_name="Forest",
        medium_name="Nott'm Forest",
    )
    resolver.team_repo.to_football_data_name = MagicMock(
        side_effect=lambda name: "Nott'm Forest"
    )
    resolver.team_repo.team_name_wide_search = MagicMock(return_value=None)
    resolver._aliases = {}

    names = resolver._names_for_team(team, "Nottingham Forest", league_id=1)

    assert "Nottingham Forest" in names
    assert "Forest" in names
    assert "Nott'm Forest" in names


def test_names_for_team_uses_provider_alias_and_wide_search():
    resolver = _resolver()
    team = SimpleNamespace(name="Arsenal", short_name=None, medium_name=None)
    resolver.team_repo.to_football_data_name = MagicMock(
        side_effect=lambda name: name if name == "Arsenal" else None
    )
    resolver.team_repo.team_name_wide_search = MagicMock(
        return_value=SimpleNamespace(name="Arsenal")
    )
    resolver._aliases = {"Arsenal FC": "Arsenal"}

    names = resolver._names_for_team(team, "Arsenal FC", league_id=1)

    assert "Arsenal FC" in names
    assert "Arsenal" in names


def test_resolve_match_uses_name_variants_for_date_team_lookup():
    resolver = _resolver()
    historical = SimpleNamespace(
        id=99,
        match_date=date(2025, 8, 15),
        home_team="Nott'm Forest",
        away_team="Arsenal",
    )
    home = SimpleNamespace(
        id=1,
        name="Nottingham Forest",
        short_name="Forest",
        medium_name=None,
    )
    away = SimpleNamespace(
        id=2,
        name="Arsenal",
        short_name=None,
        medium_name=None,
    )
    resolver._names_for_team = MagicMock(
        side_effect=[
            ["Nottingham Forest", "Nott'm Forest", "Forest"],
            ["Arsenal"],
        ]
    )
    resolver.historical_repo.find_by_date_range_and_teams = MagicMock(
        return_value=[historical]
    )

    result = resolver.resolve_match(
        _provider_match(),
        league_code="E0",
        league_id=1,
        home_team=home,
        away_team=away,
    )

    assert result.method == "date_teams"
    assert result.match.id == 99
    resolver.historical_repo.find_by_date_range_and_teams.assert_called_once()
    call_kwargs = resolver.historical_repo.find_by_date_range_and_teams.call_args.kwargs
    assert "Nott'm Forest" in call_kwargs["home_names"]
    assert "Arsenal" in call_kwargs["away_names"]


def test_resolve_match_picks_closest_when_ambiguous():
    resolver = _resolver()
    farther = SimpleNamespace(id=1, match_date=date(2025, 8, 14))
    closer = SimpleNamespace(id=2, match_date=date(2025, 8, 15))
    home = SimpleNamespace(id=1, name="Chelsea", short_name=None, medium_name=None)
    away = SimpleNamespace(id=2, name="Liverpool", short_name=None, medium_name=None)
    resolver._names_for_team = MagicMock(side_effect=[["Chelsea"], ["Liverpool"]])
    resolver.historical_repo.find_by_date_range_and_teams = MagicMock(
        return_value=[farther, closer]
    )

    result = resolver.resolve_match(
        _provider_match(
            home_team_name="Chelsea",
            away_team_name="Liverpool",
            kickoff_at=datetime(2025, 8, 15, 15, 0, tzinfo=timezone.utc),
        ),
        league_code="E0",
        league_id=1,
        home_team=home,
        away_team=away,
    )

    assert result.method == "date_teams_ambiguous"
    assert result.match.id == 2
    assert result.warnings


def test_resolve_match_unresolved_when_no_candidates():
    resolver = _resolver()
    home = SimpleNamespace(id=1, name="Chelsea", short_name=None, medium_name=None)
    away = SimpleNamespace(id=2, name="Liverpool", short_name=None, medium_name=None)
    resolver._names_for_team = MagicMock(side_effect=[["Chelsea"], ["Liverpool"]])
    resolver.historical_repo.find_by_date_range_and_teams = MagicMock(return_value=[])
    resolver.historical_repo.find_by_season_and_teams = MagicMock(return_value=[])

    result = resolver.resolve_match(
        _provider_match(
            home_team_name="Chelsea",
            away_team_name="Liverpool",
        ),
        league_code="E0",
        league_id=1,
        home_team=home,
        away_team=away,
        season="2025",
    )

    assert result.match is None
    assert result.method == "unresolved"
