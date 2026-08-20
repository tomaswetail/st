"""Unit tests for EntityResolver name variants and match linking."""

from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from data_sources.entity_resolver import EntityResolver
from objects.schema.data_classes.provider_dtos import ProviderMatch


def _resolver() -> EntityResolver:
    session = MagicMock()
    resolver = EntityResolver(session, provider="api-football")
    resolver.mapping_repo.get_by_external = MagicMock(return_value=None)
    resolver._append_unresolved_match = MagicMock()
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
        code="Forest",
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
    team = SimpleNamespace(name="Arsenal", code=None, country=None)
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
        fixture_date=date(2025, 8, 15),
        home_team="Nott'm Forest",
        away_team="Arsenal",
    )
    home = SimpleNamespace(
        id=1,
        name="Nottingham Forest",
        external_id=101,
        code="Forest",
        country=None,
    )
    away = SimpleNamespace(
        id=2,
        name="Arsenal",
        external_id=102,
        code=None,
        country=None,
    )
    resolver._names_for_team = MagicMock(
        side_effect=[
            ["Nottingham Forest", "Nott'm Forest", "Forest"],
            ["Arsenal"],
        ]
    )
    resolver.fixture_repo.find_by_date_range_and_teams = MagicMock(
        return_value=[historical]
    )

    result = resolver.resolve_match(
        _provider_match(),
        league_external_id=39,
        league_id=1,
        home_team=home,
        away_team=away,
    )

    assert result.method == "date_teams"
    assert result.match.id == 99
    resolver.fixture_repo.find_by_date_range_and_teams.assert_called_once()
    call_kwargs = resolver.fixture_repo.find_by_date_range_and_teams.call_args.kwargs
    assert call_kwargs["home_team_ids"] == [101]
    assert call_kwargs["away_team_ids"] == [102]
    assert call_kwargs["home_names"] is None
    assert call_kwargs["away_names"] is None


def test_resolve_match_picks_closest_when_ambiguous():
    resolver = _resolver()
    farther = SimpleNamespace(id=1, fixture_date=date(2025, 8, 14))
    closer = SimpleNamespace(id=2, fixture_date=date(2025, 8, 15))
    home = SimpleNamespace(id=1, name="Chelsea", external_id=101, code=None, country=None)
    away = SimpleNamespace(id=2, name="Liverpool", external_id=102, code=None, country=None)
    resolver._names_for_team = MagicMock(side_effect=[["Chelsea"], ["Liverpool"]])
    resolver.fixture_repo.find_by_date_range_and_teams = MagicMock(
        return_value=[farther, closer]
    )

    result = resolver.resolve_match(
        _provider_match(
            home_team_name="Chelsea",
            away_team_name="Liverpool",
            kickoff_at=datetime(2025, 8, 15, 15, 0, tzinfo=timezone.utc),
        ),
        league_external_id=39,
        league_id=1,
        home_team=home,
        away_team=away,
    )

    assert result.method == "date_teams_ambiguous"
    assert result.match.id == 2
    assert result.warnings


def test_resolve_match_unresolved_when_no_candidates():
    resolver = _resolver()
    home = SimpleNamespace(id=1, name="Chelsea", external_id=101, code=None, country=None)
    away = SimpleNamespace(id=2, name="Liverpool", external_id=102, code=None, country=None)
    resolver._names_for_team = MagicMock(side_effect=[["Chelsea"], ["Liverpool"]])
    resolver.fixture_repo.find_by_date_range_and_teams = MagicMock(return_value=[])
    resolver.fixture_repo.find_by_season_and_teams = MagicMock(return_value=[])

    result = resolver.resolve_match(
        _provider_match(
            home_team_name="Chelsea",
            away_team_name="Liverpool",
        ),
        league_external_id=39,
        league_id=1,
        home_team=home,
        away_team=away,
        season="2025",
    )

    assert result.match is None
    assert result.method == "unresolved"


def test_resolve_team_create_if_missing_creates_and_maps():
    resolver = _resolver()
    created = SimpleNamespace(id=42, name="New FC", external_id=999)
    resolver.team_repo.get = MagicMock(return_value=None)
    resolver.team_repo.create_from_provider_team = MagicMock(return_value=created)
    resolver.team_repo.flush = MagicMock()
    resolver.team_repo.get_by_name_and_league = MagicMock(return_value=None)
    resolver.team_repo.get_by_name = MagicMock(return_value=None)
    resolver.team_repo.team_name_wide_search = MagicMock(return_value=None)
    resolver.team_repo.team_likely_name_wide_search = MagicMock(return_value=None)
    resolver._candidate_team_names = MagicMock(return_value=[])
    resolver._aliases = {}
    resolver.ensure_mapping = MagicMock()
    resolver.team_repo.find_exact_normalized = MagicMock(return_value=None)
    resolver.team_repo.find_by_club_affix = MagicMock(return_value=None)
    resolver.team_repo.find_fuzzy_duplicate = MagicMock(return_value=None)
    resolver.team_repo.find_substring_duplicate = MagicMock(return_value=None)

    result = resolver.resolve_team(
        provider_team_id="999",
        provider_team_name="New FC",
        league_id=1,
        create_if_missing=True,
    )

    assert result.method == "created"
    assert result.team is created
    resolver.team_repo.create_from_provider_team.assert_called_once_with(
        external_id=999,
        name="New FC",
    )
    resolver.ensure_mapping.assert_called_once()
    assert resolver.ensure_mapping.call_args.kwargs["internal_entity_id"] == 42


def test_resolve_team_reuses_normalized_duplicate_instead_of_creating():
    resolver = _resolver()
    existing = SimpleNamespace(id=7, name="Franke")
    resolver.team_repo.get = MagicMock(return_value=None)
    resolver.team_repo.create = MagicMock()
    resolver._candidate_team_names = MagicMock(return_value=[])
    resolver._aliases = {}
    resolver.ensure_mapping = MagicMock()
    resolver.team_repo.find_exact_normalized = MagicMock(return_value=existing)
    resolver.team_repo.find_by_club_affix = MagicMock(return_value=None)
    resolver.team_repo.find_fuzzy_duplicate = MagicMock(return_value=None)
    resolver.team_repo.find_substring_duplicate = MagicMock(return_value=None)

    result = resolver.resolve_team(
        provider_team_id="IK Franke",
        provider_team_name="IK Franke",
        league_id=1,
        create_if_missing=True,
    )

    assert result.method == "normalized"
    assert result.team is existing
    resolver.team_repo.create.assert_not_called()
    resolver.ensure_mapping.assert_called_once()
    assert resolver.ensure_mapping.call_args.kwargs["internal_entity_id"] == 7


def test_resolve_team_does_not_fuzzy_match_angers_to_rangers():
    resolver = _resolver()
    created = SimpleNamespace(id=42, name="Angers", external_id=501)
    resolver.team_repo.get = MagicMock(return_value=None)
    resolver.team_repo.create_from_provider_team = MagicMock(return_value=created)
    resolver.team_repo.flush = MagicMock()
    resolver._candidate_team_names = MagicMock(return_value=["Rangers"])
    resolver._aliases = {}
    resolver.ensure_mapping = MagicMock()
    resolver.team_repo.find_exact_normalized = MagicMock(return_value=None)
    resolver.team_repo.find_by_club_affix = MagicMock(return_value=None)
    resolver.team_repo.find_fuzzy_duplicate = MagicMock(return_value=None)
    resolver.team_repo.find_substring_duplicate = MagicMock(return_value=None)

    result = resolver.resolve_team(
        provider_team_id="501",
        provider_team_name="Angers",
        league_id=1,
        create_if_missing=True,
    )

    assert result.method == "created"
    assert result.team is created
    resolver.team_repo.create_from_provider_team.assert_called_once()


def test_resolve_team_does_not_alias_manchester_city_to_man_united():
    resolver = _resolver()
    created = SimpleNamespace(id=90, name="Manchester City", external_id=502)
    man_united = SimpleNamespace(id=809, name="Man United", external_id=1)
    resolver.team_repo.get = MagicMock(return_value=None)
    resolver.team_repo.create_from_provider_team = MagicMock(return_value=created)
    resolver.team_repo.flush = MagicMock()
    resolver.team_repo.get_by_name_and_league = MagicMock(return_value=None)
    resolver.team_repo.get_by_name = MagicMock(
        side_effect=lambda name, *args, **kwargs: (
            man_united if name == "Man United" else None
        )
    )
    resolver._candidate_team_names = MagicMock(return_value=["Man United"])
    resolver._aliases = {"Manchester City": "Man City"}
    resolver.ensure_mapping = MagicMock()
    resolver.team_repo.find_exact_normalized = MagicMock(return_value=None)
    resolver.team_repo.find_by_club_affix = MagicMock(return_value=None)
    resolver.team_repo.find_fuzzy_duplicate = MagicMock(return_value=None)
    resolver.team_repo.find_substring_duplicate = MagicMock(return_value=None)

    result = resolver.resolve_team(
        provider_team_id="502",
        provider_team_name="Manchester City",
        league_id=1,
        create_if_missing=True,
    )

    assert result.method == "created"
    assert result.team is created
    resolver.team_repo.create_from_provider_team.assert_called_once()


def test_resolve_team_still_aliases_manchester_city_to_man_city():
    resolver = _resolver()
    man_city = SimpleNamespace(id=90, name="Man City", external_id=1)
    resolver.team_repo.get = MagicMock(return_value=None)
    resolver.team_repo.create = MagicMock()
    resolver.team_repo.get_by_name_and_league = MagicMock(return_value=None)
    resolver.team_repo.get_by_name = MagicMock(
        side_effect=lambda name, *args, **kwargs: (
            man_city if name == "Man City" else None
        )
    )
    resolver._candidate_team_names = MagicMock(return_value=[])
    resolver._aliases = {"Manchester City": "Man City"}
    resolver.ensure_mapping = MagicMock()

    result = resolver.resolve_team(
        provider_team_id="502",
        provider_team_name="Manchester City",
        league_id=1,
        create_if_missing=True,
    )

    assert result.method == "alias"
    assert result.team is man_city
    resolver.team_repo.create.assert_not_called()


def test_resolve_team_does_not_match_villarreal_to_villarreal_b_duplicate():
    resolver = _resolver()
    created = SimpleNamespace(id=51, name="Villarreal", external_id=503)
    duplicate = SimpleNamespace(id=9, name="Villarreal B", external_id=1)
    resolver.team_repo.get = MagicMock(return_value=None)
    resolver.team_repo.create_from_provider_team = MagicMock(return_value=created)
    resolver.team_repo.flush = MagicMock()
    resolver._candidate_team_names = MagicMock(return_value=[])
    resolver._aliases = {}
    resolver.ensure_mapping = MagicMock()
    resolver.team_repo.find_exact_normalized = MagicMock(return_value=None)
    resolver.team_repo.find_by_club_affix = MagicMock(return_value=None)
    resolver.team_repo.find_fuzzy_duplicate = MagicMock(return_value=duplicate)
    resolver.team_repo.find_substring_duplicate = MagicMock(return_value=None)

    result = resolver.resolve_team(
        provider_team_id="503",
        provider_team_name="Villarreal",
        league_id=1,
        create_if_missing=True,
    )

    assert result.method == "created"
    assert result.team is created
    resolver.team_repo.create_from_provider_team.assert_called_once()


def test_resolve_team_does_not_exact_match_oxford_city_to_oxford():
    resolver = _resolver()
    created = SimpleNamespace(id=70, name="Oxford City", external_id=504)
    oxford = SimpleNamespace(id=11, name="Oxford", external_id=1)
    resolver.team_repo.get = MagicMock(return_value=None)
    resolver.team_repo.create_from_provider_team = MagicMock(return_value=created)
    resolver.team_repo.flush = MagicMock()
    resolver.team_repo.get_by_name_and_league = MagicMock(return_value=oxford)
    resolver.team_repo.get_by_name = MagicMock(return_value=oxford)
    resolver._candidate_team_names = MagicMock(return_value=["Oxford"])
    resolver._aliases = {}
    resolver.ensure_mapping = MagicMock()
    resolver.team_repo.find_exact_normalized = MagicMock(return_value=None)
    resolver.team_repo.find_by_club_affix = MagicMock(return_value=None)
    resolver.team_repo.find_fuzzy_duplicate = MagicMock(return_value=None)
    resolver.team_repo.find_substring_duplicate = MagicMock(return_value=None)

    result = resolver.resolve_team(
        provider_team_id="504",
        provider_team_name="Oxford City",
        league_id=1,
        create_if_missing=True,
    )

    assert result.method == "created"
    assert result.team is created
    resolver.team_repo.create_from_provider_team.assert_called_once()


def test_resolve_match_appends_unresolved_row_to_csv(tmp_path):
    csv_path = tmp_path / "unresolved_matches.csv"
    resolver = EntityResolver(MagicMock(), provider="api-football")
    resolver.config.unresolved_matches_csv_path = csv_path
    resolver.mapping_repo.get_by_external = MagicMock(return_value=None)
    home = SimpleNamespace(id=1, name="Chelsea", external_id=101, code=None, country=None)
    away = SimpleNamespace(id=2, name="Liverpool", external_id=102, code=None, country=None)
    resolver._names_for_team = MagicMock(side_effect=[["Chelsea"], ["Liverpool"]])
    resolver.fixture_repo.find_by_date_range_and_teams = MagicMock(return_value=[])
    resolver.fixture_repo.find_by_season_and_teams = MagicMock(return_value=[])

    result = resolver.resolve_match(
        _provider_match(
            home_team_name="Chelsea",
            away_team_name="Liverpool",
        ),
        league_external_id=39,
        league_id=1,
        home_team=home,
        away_team=away,
        season="2025",
    )

    assert result.method == "unresolved"
    text = csv_path.read_text(encoding="utf-8")
    assert "provider_match_id" in text
    assert "4000001" in text
    assert "Chelsea" in text
    assert "Liverpool" in text

    resolver._append_unresolved_match(
        _provider_match(
            home_team_name="Chelsea",
            away_team_name="Liverpool",
        ),
        league_external_id=39,
        league_id=1,
        home_team=home,
        away_team=away,
        season="2025",
    )
    rows = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 3
