"""Parser, metrics, HTTP, resolution, and service tests for football data ingestion."""

from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from data_sources.entity_resolver import EntityResolver, MatchResolution, TeamResolution
from data_sources.football_data.http_client import (
    NotFoundError,
    ThrottledHttpClient,
)
from data_sources.football_data.metrics import (
    calculate_derived_metrics,
    shot_fingerprint,
)
from data_sources.football_data.providers.sofascore import (
    parse_sofascore_match_details,
    parse_sofascore_matches,
)
from data_sources.football_data.results import MatchImportResult
from data_sources.football_data.service import ExtendedMatchDataService
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.data_classes.provider_dtos import (
    ProviderMatch,
    ProviderMatchDetails,
    ProviderShot,
    ProviderTeam,
)
from calc.strength_calculator import StrengthCalculator
from tests.football_data.conftest import load_fixture


# ---------------------------------------------------------------------------
# 1. SofaScore response parsing
# ---------------------------------------------------------------------------


def test_sofascore_match_details_parsing():
    payload = load_fixture("sofascore", "match_details.json")
    details = parse_sofascore_match_details(payload)
    assert details.match.provider_match_id == "1200001"
    assert details.home_xg == pytest.approx(1.7)
    assert len(details.shots) == 3
    assert details.shots[0].team_id == details.match.home_team_id
    assert details.shots[1].is_penalty is True


def test_sofascore_season_matches_parsing():
    payload = load_fixture("sofascore", "season_matches.json")
    matches = parse_sofascore_matches(payload, "17", "76986")
    assert len(matches) == 1
    assert matches[0].home_team_name == "Manchester City"


# ---------------------------------------------------------------------------
# 2. Team resolution
# ---------------------------------------------------------------------------


def test_team_resolution_order_mapping_then_exact():
    session = MagicMock()
    resolver = EntityResolver(session, provider="sofascore")
    mapped_team = SimpleNamespace(id=10, name="Man City", external_id=10, code=None, country=None)
    resolver.mapping_repo.get_by_external = MagicMock(
        return_value=SimpleNamespace(internal_entity_id=10)
    )
    resolver.team_repo.get = MagicMock(return_value=mapped_team)
    result = resolver.resolve_team(
        provider_team_id="8456",
        provider_team_name="Manchester City",
    )
    assert result.method == "mapping"
    assert result.team.id == 10

    resolver.mapping_repo.get_by_external = MagicMock(return_value=None)
    resolver._team_name_cache = ["Manchester City", "Arsenal"]
    resolver._aliases = {}
    resolver.team_repo.get_by_name = MagicMock(
        return_value=SimpleNamespace(id=11, name="Arsenal")
    )
    resolver.team_repo.get_by_name_and_league = MagicMock(return_value=None)
    resolver.team_repo.team_name_wide_search = MagicMock(return_value=None)
    resolver.team_repo.team_likely_name_wide_search = MagicMock(return_value=None)
    result = resolver.resolve_team(
        provider_team_id="999",
        provider_team_name="Arsenal FC",
    )
    assert result.method == "exact_name"
    assert result.team.id == 11


def test_team_resolution_unresolved_low_confidence(caplog):
    session = MagicMock()
    config = DataSourceConfig(fuzzy_match_threshold=95)
    resolver = EntityResolver(session, config=config, provider="sofascore")
    resolver.mapping_repo.get_by_external = MagicMock(return_value=None)
    resolver._team_name_cache = ["Arsenal"]
    resolver._aliases = {}
    resolver.team_repo.get_by_name = MagicMock(return_value=None)
    resolver.team_repo.get_by_name_and_league = MagicMock(return_value=None)
    resolver.team_repo.team_name_wide_search = MagicMock(return_value=None)
    resolver.team_repo.team_likely_name_wide_search = MagicMock(return_value=None)
    resolver.team_repo.find_exact_normalized = MagicMock(return_value=None)
    resolver.team_repo.find_by_club_affix = MagicMock(return_value=None)
    resolver.team_repo.find_fuzzy_duplicate = MagicMock(return_value=None)
    resolver.team_repo.find_substring_duplicate = MagicMock(return_value=None)
    result = resolver.resolve_team(
        provider_team_id="1",
        provider_team_name="Completely Unknown United",
    )
    assert result.team is None
    assert result.method == "unresolved"


# ---------------------------------------------------------------------------
# 3–5. Match resolution / postponed / mappings
# ---------------------------------------------------------------------------


def test_match_resolution_by_mapping():
    session = MagicMock()
    resolver = EntityResolver(session, provider="sofascore")
    historical = SimpleNamespace(id=55)
    resolver.mapping_repo.get_by_external = MagicMock(
        return_value=SimpleNamespace(internal_entity_id=55)
    )
    resolver.fixture_repo.get = MagicMock(return_value=historical)
    provider_match = ProviderMatch(
        provider_match_id="4000001",
        provider_league_id="47",
        provider_season_id="2025",
        home_team_id="1",
        away_team_id="2",
        home_team_name="A",
        away_team_name="B",
        kickoff_at=datetime(2025, 8, 15, 18, 0, tzinfo=timezone.utc),
        status="finished",
    )
    result = resolver.resolve_match(
        provider_match, league_external_id=39, home_team=None, away_team=None, league_id=47
    )
    assert result.method == "mapping"
    assert result.match.id == 55


def test_postponed_fixture_matching_via_season():
    session = MagicMock()
    resolver = EntityResolver(session, provider="sofascore")
    resolver.mapping_repo.get_by_external = MagicMock(return_value=None)

    postponed = SimpleNamespace(
        id=77,
        fixture_date=date(2025, 9, 1),
        home_team="Chelsea",
        away_team="Liverpool",
    )
    resolver.fixture_repo.find_by_date_range_and_teams = MagicMock(return_value=[])
    resolver.fixture_repo.find_by_season_and_teams = MagicMock(
        return_value=[postponed]
    )

    provider_match = ProviderMatch(
        provider_match_id="4000002",
        provider_league_id="47",
        provider_season_id="2526",
        home_team_id="1",
        away_team_id="2",
        home_team_name="Chelsea",
        away_team_name="Liverpool",
        kickoff_at=datetime(2025, 8, 20, 19, 0, tzinfo=timezone.utc),
        status="Postponed",
    )
    home = SimpleNamespace(id=1, name="Chelsea", external_id=10, code=None, country=None)
    away = SimpleNamespace(id=2, name="Liverpool", external_id=10, code=None, country=None)
    resolver.team_repo.to_football_data_name = MagicMock(side_effect=lambda n: n)
    resolver.team_repo.team_name_wide_search = MagicMock(return_value=None)
    result = resolver.resolve_match(
        provider_match,
        league_external_id=39,
        home_team=home,
        away_team=away,
        season="2526",
        league_id=47,
    )
    assert result.match is not None
    assert result.match.id == 77
    assert "postponed" in result.method or any(
        "postponed" in warning.lower() for warning in result.warnings
    )


def test_provider_id_mapping_upsert_unique_keys():
    session = MagicMock()
    from objects.repositories.external_entity_mapping_repository import (
        ExternalEntityMappingRepository,
    )

    repo = ExternalEntityMappingRepository(session)
    fake_row = SimpleNamespace(id=1, external_entity_id="47")
    session.scalar.return_value = None
    session.scalars.return_value.one.return_value = fake_row
    with patch(
        "objects.repositories.external_entity_mapping_repository.pg_insert"
    ) as insert_mock:
        statement = MagicMock()
        insert_mock.return_value.values.return_value.on_conflict_do_update.return_value.returning.return_value = (
            statement
        )
        row = repo.upsert(
            provider="sofascore",
            entity_type="league",
            internal_entity_id=3,
            external_entity_id="47",
            external_name="Premier League",
        )
        assert row.external_entity_id == "47"
        insert_mock.assert_called()


def test_provider_id_mapping_upsert_keeps_existing_internal_link():
    session = MagicMock()
    from objects.repositories.external_entity_mapping_repository import (
        ExternalEntityMappingRepository,
    )

    repo = ExternalEntityMappingRepository(session)
    existing = SimpleNamespace(
        id=1,
        external_entity_id="St. Pauli",
        internal_entity_id=17,
        external_name="St. Pauli",
    )
    # No row for the new external id; existing row for internal id 17.
    repo.get_by_external = MagicMock(return_value=None)
    repo.get_by_internal = MagicMock(return_value=existing)

    with patch(
        "objects.repositories.external_entity_mapping_repository.pg_insert"
    ) as insert_mock:
        row = repo.upsert(
            provider="api-football",
            entity_type="team",
            internal_entity_id=17,
            external_entity_id="St Pauli",
            external_name="St Pauli",
        )
        assert row is existing
        assert row.external_entity_id == "St. Pauli"
        insert_mock.assert_not_called()


# ---------------------------------------------------------------------------
# 6–9. Metrics / missing xG / shot dedupe
# ---------------------------------------------------------------------------


def _sample_details() -> ProviderMatchDetails:
    payload = load_fixture("sofascore", "match_details.json")
    return parse_sofascore_match_details(payload)


def _metrics_details() -> ProviderMatchDetails:
    match = ProviderMatch(
        provider_match_id="m1",
        provider_league_id="17",
        provider_season_id="2024/2025",
        home_team_id="h",
        away_team_id="a",
        home_team_name="Home",
        away_team_name="Away",
        kickoff_at=datetime(2024, 8, 17, 15, 0, tzinfo=timezone.utc),
        status="finished",
    )
    shots = [
        ProviderShot(
            provider_shot_id="1", team_id="h", player_id=None, minute=10, second=0,
            xg=0.4, xgot=None, outcome="Miss", situation="OpenPlay",
            body_part=None, shot_type=None, is_penalty=False, is_own_goal=False,
            coordinates=None,
        ),
        ProviderShot(
            provider_shot_id="2", team_id="h", player_id=None, minute=40, second=0,
            xg=0.75, xgot=None, outcome="Goal", situation="Penalty",
            body_part=None, shot_type=None, is_penalty=True, is_own_goal=False,
            coordinates=None,
        ),
        ProviderShot(
            provider_shot_id="3", team_id="h", player_id=None, minute=70, second=0,
            xg=0.2, xgot=None, outcome="Saved", situation="corner",
            body_part=None, shot_type=None, is_penalty=False, is_own_goal=False,
            coordinates=None,
        ),
        ProviderShot(
            provider_shot_id="4", team_id="a", player_id=None, minute=20, second=0,
            xg=0.5, xgot=None, outcome="Miss", situation="OpenPlay",
            body_part=None, shot_type=None, is_penalty=False, is_own_goal=False,
            coordinates=None,
        ),
        ProviderShot(
            provider_shot_id="5", team_id="a", player_id=None, minute=55, second=0,
            xg=0.15, xgot=None, outcome="Miss", situation="freekick",
            body_part=None, shot_type=None, is_penalty=False, is_own_goal=False,
            coordinates=None,
        ),
    ]
    return ProviderMatchDetails(
        match=match, shots=shots, statistics={}, home_xg=1.85, away_xg=0.9
    )


def test_non_penalty_xg_calculation():
    details = _metrics_details()
    metrics = calculate_derived_metrics(
        details,
        home_team_external_id=details.match.home_team_id,
        away_team_external_id=details.match.away_team_id,
    )
    # Home shots xG: 0.4 + 0.75 + 0.2 = 1.35; npxG excludes penalty 0.75 => 0.6
    assert metrics.home_non_penalty_xg == pytest.approx(0.6)
    assert metrics.away_non_penalty_xg == pytest.approx(0.65)


def test_set_piece_xg_calculation():
    details = _metrics_details()
    metrics = calculate_derived_metrics(
        details,
        home_team_external_id=details.match.home_team_id,
        away_team_external_id=details.match.away_team_id,
    )
    # Home set piece: penalty 0.75 + corner 0.2 = 0.95
    assert metrics.home_set_piece_xg == pytest.approx(0.95)
    # Away set piece: free kick 0.15
    assert metrics.away_set_piece_xg == pytest.approx(0.15)


def test_missing_xg_not_treated_as_zero():
    match = ProviderMatch(
        provider_match_id="1",
        provider_league_id="1",
        provider_season_id="1",
        home_team_id="h",
        away_team_id="a",
        home_team_name="H",
        away_team_name="A",
        kickoff_at=datetime.now(tz=timezone.utc),
        status="finished",
    )
    shots = [
        ProviderShot(
            provider_shot_id="1",
            team_id="h",
            player_id=None,
            minute=10,
            second=0,
            xg=None,
            xgot=None,
            outcome="Miss",
            situation="OpenPlay",
            body_part=None,
            shot_type=None,
            is_penalty=False,
            is_own_goal=False,
            coordinates=None,
        )
    ]
    details = ProviderMatchDetails(match=match, shots=shots, statistics={})
    metrics = calculate_derived_metrics(
        details, home_team_external_id="h", away_team_external_id="a"
    )
    assert metrics.home_xg is None
    assert metrics.home_xg_from_shots is None
    assert metrics.home_non_penalty_xg is None


def test_shot_deduplication_fingerprint_stable():
    shot = ProviderShot(
        provider_shot_id=None,
        team_id="h",
        player_id="1",
        minute=10,
        second=5,
        xg=0.2,
        xgot=None,
        outcome="Miss",
        situation="OpenPlay",
        body_part=None,
        shot_type=None,
        is_penalty=False,
        is_own_goal=False,
        coordinates={"x": 1, "y": 2},
    )
    first = shot_fingerprint(match_id=9, provider="sofascore", shot=shot, team_internal_id=3)
    second = shot_fingerprint(match_id=9, provider="sofascore", shot=shot, team_internal_id=3)
    assert first == second
    other = shot_fingerprint(match_id=9, provider="sofascore", shot=shot, team_internal_id=4)
    assert first != other


# ---------------------------------------------------------------------------
# 11–14. Service behaviour: idempotent, retry, fallback, isolation
# ---------------------------------------------------------------------------


def test_idempotent_repeated_imports_skip_without_force():
    session = MagicMock()
    provider = MagicMock()
    provider.name = "sofascore"
    service = ExtendedMatchDataService(
        provider=provider, session=session, dry_run=False
    )
    service.fixture_repo.get = MagicMock(
        return_value=SimpleNamespace(id=1, home_team="A", away_team="B")
    )
    service.stats_repo.get_by_match_and_provider = MagicMock(
        return_value=SimpleNamespace(id=99)
    )
    result = service.fetch_and_store_match(1, force_refresh=False)
    assert result.status == "skipped"
    provider.fetch_match_details.assert_not_called()


def test_retry_and_rate_limit_handling(tmp_path):
    client = ThrottledHttpClient(
        base_url="https://example.test",
        max_retries=2,
        request_delay_ms=0,
        cache_ttl_seconds=0,
        cache_dir=tmp_path,
        enable_cache=False,
    )
    responses = [
        httpx.Response(429, text="slow down"),
        httpx.Response(500, text="error"),
        httpx.Response(200, json={"ok": True}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    transport = httpx.MockTransport(handler)
    client._client = httpx.Client(transport=transport, base_url="https://example.test")
    with patch("data_sources.football_data.http_client.time.sleep"):
        data = client.get_json("/path")
    assert data == {"ok": True}


def test_http_404_raises_not_found(tmp_path):
    client = ThrottledHttpClient(
        base_url="https://example.test",
        max_retries=0,
        request_delay_ms=0,
        cache_dir=tmp_path,
        enable_cache=False,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="missing")

    client._client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    )
    with pytest.raises(NotFoundError):
        client.get_json("/missing")


def test_fetch_match_details_uses_seed_when_provider_missing():
    session = MagicMock()
    provider = MagicMock()
    provider.name = "sofascore"
    provider.fetch_match_details.side_effect = NotFoundError("missing")

    service = ExtendedMatchDataService(
        provider=provider,
        session=session,
        dry_run=True,
    )
    historical = SimpleNamespace(id=5, home_team="Manchester City", away_team="Arsenal")
    seed = ProviderMatch(
        provider_match_id="fb-1",
        provider_league_id="17",
        provider_season_id="2024/2025",
        home_team_id="1",
        away_team_id="2",
        home_team_name="Home",
        away_team_name="Away",
        kickoff_at=datetime(2024, 8, 17, 15, 0, tzinfo=timezone.utc),
        status="finished",
    )
    fetched, used = service._fetch_match_details(
        "fb-1", historical, seed_match=seed
    )
    assert used == "sofascore"
    assert fetched is not None
    assert fetched.match.provider_match_id == "fb-1"
    assert fetched.shots == []


def test_transaction_rollback_for_one_failed_match_keeps_others():
    session = MagicMock()
    # begin_nested context manager
    session.begin_nested.return_value.__enter__ = MagicMock(return_value=None)
    session.begin_nested.return_value.__exit__ = MagicMock(return_value=False)

    provider = MagicMock()
    provider.name = "sofascore"
    service = ExtendedMatchDataService(provider=provider, session=session)

    results = [
        MatchImportResult(internal_match_id=1, provider_match_id="a", status="imported"),
        MatchImportResult(
            internal_match_id=2, provider_match_id="b", status="failed", error="boom"
        ),
        MatchImportResult(internal_match_id=3, provider_match_id="c", status="imported"),
    ]

    def side_effect(match_id, force_refresh=False):
        if match_id == 2:
            raise RuntimeError("boom")
        return results[match_id - 1]

    service.fetch_and_store_match = MagicMock(side_effect=side_effect)
    batch = service.fetch_and_store_matches([1, 2, 3])
    assert batch.imported == 2
    assert batch.failed == 1
    assert batch.requested == 3


def test_unresolved_teams_created_from_sofascore():
    session = MagicMock()
    provider = MagicMock()
    provider.name = "sofascore"
    provider.fetch_team.side_effect = lambda team_id: ProviderTeam(
        provider_team_id=str(team_id),
        name=f"Team {team_id}",
        short_name=f"T{team_id}",
        country_code="ENG",
        country_name="England",
    )

    service = ExtendedMatchDataService(
        provider=provider, session=session, dry_run=False
    )
    service.resolver.resolve_team = MagicMock(
        return_value=TeamResolution(
            team=None, confidence=0.0, method="unresolved", unresolved_name="x"
        )
    )
    created_home = SimpleNamespace(id=101, name="Team 8456")
    created_away = SimpleNamespace(id=102, name="Team 9825")
    service.resolver.team_repo.create_from_provider_team = MagicMock(
        side_effect=[created_home, created_away]
    )
    historical = SimpleNamespace(id=50, home_team="Team 8456", away_team="Team 9825")
    service.resolver.resolve_match = MagicMock(
        return_value=MatchResolution(
            match=historical, method="date_teams", warnings=[]
        )
    )
    service.resolver.ensure_mapping = MagicMock()
    service._fetch_match_details = MagicMock(
        return_value=(_sample_details(), "sofascore")
    )
    service._persist_match_details = MagicMock(
        return_value=MatchImportResult(
            internal_match_id=50,
            provider_match_id="m1",
            status="imported",
            shots_imported=0,
        )
    )

    fixture = ProviderMatch(
        provider_match_id="m1",
        provider_league_id="47",
        provider_season_id="2025/2026",
        home_team_id="8456",
        away_team_id="9825",
        home_team_name="Man City",
        away_team_name="Arsenal",
        kickoff_at=datetime(2025, 8, 1, 15, 0, tzinfo=timezone.utc),
        status="FT",
    )
    result = service._import_provider_fixture(
        fixture=fixture,
        league_id=1,
        league_external_id=39,
        season="2526",
        force_refresh=False,
    )

    assert result.status == "imported"
    assert service.resolver.team_repo.create_from_provider_team.call_count == 2
    service.resolver.team_repo.create_from_provider_team.assert_any_call(
        external_id=8456,
        name="Team 8456",
        code="ENG",
        country="England",
    )
    provider.fetch_team.assert_any_call("8456")
    provider.fetch_team.assert_any_call("9825")
    assert any(
        call.kwargs.get("entity_type") == "team"
        and call.kwargs.get("internal_entity_id") == 101
        for call in service.resolver.ensure_mapping.call_args_list
    )


# ---------------------------------------------------------------------------
# 15. No future-data leakage in team features
# ---------------------------------------------------------------------------


def test_team_features_no_future_data_leakage():
    session = MagicMock()
    calculator = StrengthCalculator(session=session, provider="sofascore")
    team = SimpleNamespace(id=1, name="Manchester City", league_id=1)
    calculator.team_repo.get = MagicMock(return_value=team)

    past = SimpleNamespace(
        id=10,
        fixture_date=date(2025, 8, 1),
        home_team=SimpleNamespace(name="Manchester City"),
        away_team=SimpleNamespace(name="Arsenal"),
        home_goals=2,
        away_goals=1,
    )
    future = SimpleNamespace(
        id=11,
        fixture_date=date(2025, 8, 20),
        home_team=SimpleNamespace(name="Manchester City"),
        away_team=SimpleNamespace(name="Chelsea"),
        home_goals=3,
        away_goals=0,
    )
    # Repository query already filters match_date < before; simulate only past returned.
    calculator._load_team_match_stats = MagicMock(
        return_value=[
            (
                past,
                SimpleNamespace(
                    home_non_penalty_xg=1.0,
                    away_non_penalty_xg=0.5,
                    home_shots=10,
                    away_shots=8,
                    home_shots_on_target=4,
                    away_shots_on_target=3,
                    average_home_shot_xg=0.2,
                    average_away_shot_xg=0.1,
                    home_set_piece_xg=0.3,
                    away_set_piece_xg=0.1,
                    home_xg=1.2,
                    away_xg=0.6,
                    home_xg_from_shots=1.1,
                    away_xg_from_shots=0.5,
                    home_xgot=0.8,
                    away_xgot=0.4,
                ),
                True,
            )
        ]
    )
    calculator.league_averages = MagicMock(
        return_value={"attack": 1.0, "defence": 1.0, "npxg_for": 1.0, "npxg_against": 1.0}
    )
    cutoff = datetime(2025, 8, 10, tzinfo=timezone.utc)
    features = calculator.get_team_features(1, before=cutoff, lookback_matches=20)
    assert features.sample_size == 1
    assert features.non_penalty_xg_for is not None
    # Ensure loader was asked with cutoff (no future match id 11).
    args, kwargs = calculator._load_team_match_stats.call_args
    assert kwargs["before_date"] == cutoff.date()
    loaded_ids = [row[0].id for row in calculator._load_team_match_stats.return_value]
    assert 11 not in loaded_ids
    assert future.id not in loaded_ids


def test_xg_disagreement_warning():
    details = _metrics_details()
    # Provider home_xg 1.85 vs shot sum 1.35
    metrics = calculate_derived_metrics(
        details,
        home_team_external_id=details.match.home_team_id,
        away_team_external_id=details.match.away_team_id,
        xg_aggregate_tolerance=0.1,
    )
    assert any("Home xG disagreement" in warning for warning in metrics.warnings)
    assert metrics.home_xg == pytest.approx(1.85)
    assert metrics.home_xg_from_shots == pytest.approx(1.35)
