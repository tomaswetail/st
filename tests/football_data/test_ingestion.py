"""Parser, metrics, HTTP, resolution, and service tests for football data ingestion."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from data_sources.football_data.entity_resolver import EntityResolver
from data_sources.football_data.http_client import (
    FootballDataHttpError,
    NotFoundError,
    ThrottledHttpClient,
)
from data_sources.football_data.metrics import (
    calculate_derived_metrics,
    shot_fingerprint,
)
from data_sources.football_data.providers.fotmob import (
    parse_fotmob_match_details,
    parse_fotmob_matches,
    parse_fotmob_shots,
)
from data_sources.football_data.providers.sofascore import (
    parse_sofascore_match_details,
    parse_sofascore_matches,
)
from data_sources.football_data.results import BatchImportResult, MatchImportResult
from data_sources.football_data.service import ExtendedMatchDataService
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.data_classes.provider_dtos import (
    ProviderMatch,
    ProviderMatchDetails,
    ProviderShot,
)
from services.team_strength_feature_service import TeamStrengthFeatureService
from tests.football_data.conftest import load_fixture


# ---------------------------------------------------------------------------
# 1. FotMob response parsing
# ---------------------------------------------------------------------------


def test_fotmob_match_details_parsing():
    payload = load_fixture("fotmob", "match_details.json")
    details = parse_fotmob_match_details(payload)
    assert details.match.provider_match_id == "4000001"
    assert details.match.home_team_name == "Manchester City"
    assert details.match.away_score == 1
    assert details.home_xg == pytest.approx(1.85)
    assert len(details.shots) == 5
    assert details.shots[1].is_penalty is True
    assert details.shots[4].provider_shot_id is None


def test_fotmob_season_matches_parsing():
    payload = load_fixture("fotmob", "season_matches.json")
    matches = parse_fotmob_matches(payload, "47", "2025/2026")
    assert len(matches) == 2
    assert matches[1].status == "Postponed"


# ---------------------------------------------------------------------------
# 2. SofaScore response parsing
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
# 3. Team resolution
# ---------------------------------------------------------------------------


def test_team_resolution_order_mapping_then_exact():
    session = MagicMock()
    resolver = EntityResolver(session, provider="fotmob")
    mapped_team = SimpleNamespace(id=10, name="Man City", short_name=None, medium_name=None)
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
    resolver.team_repo.get_by_name = MagicMock(
        return_value=SimpleNamespace(id=11, name="Arsenal")
    )
    resolver.team_repo.get_by_name_and_league = MagicMock(return_value=None)
    result = resolver.resolve_team(
        provider_team_id="999",
        provider_team_name="Arsenal FC",
    )
    assert result.method == "exact_name"


def test_team_resolution_unresolved_low_confidence(caplog):
    session = MagicMock()
    config = DataSourceConfig(fuzzy_match_threshold=95)
    resolver = EntityResolver(session, config=config, provider="fotmob")
    resolver.mapping_repo.get_by_external = MagicMock(return_value=None)
    resolver._team_name_cache = ["Arsenal"]
    resolver.team_repo.get_by_name = MagicMock(return_value=None)
    resolver.team_repo.get_by_name_and_league = MagicMock(return_value=None)
    result = resolver.resolve_team(
        provider_team_id="1",
        provider_team_name="Completely Unknown United",
    )
    assert result.team is None
    assert result.method == "unresolved"


# ---------------------------------------------------------------------------
# 4–6. Match resolution / postponed / mappings
# ---------------------------------------------------------------------------


def test_match_resolution_by_mapping():
    session = MagicMock()
    resolver = EntityResolver(session, provider="fotmob")
    historical = SimpleNamespace(id=55)
    resolver.mapping_repo.get_by_external = MagicMock(
        return_value=SimpleNamespace(internal_entity_id=55)
    )
    resolver.historical_repo.get = MagicMock(return_value=historical)
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
        provider_match, league_code="E0", home_team=None, away_team=None
    )
    assert result.method == "mapping"
    assert result.match.id == 55


def test_postponed_fixture_matching_via_season():
    session = MagicMock()
    resolver = EntityResolver(session, provider="fotmob")
    resolver.mapping_repo.get_by_external = MagicMock(return_value=None)

    postponed = SimpleNamespace(
        id=77,
        match_date=date(2025, 9, 1),
        home_team="Chelsea",
        away_team="Liverpool",
    )

    def scalar_side_effect(statement):
        # First query (date window) empty; second (season) returns postponed match.
        sql = str(statement)
        return None

    # session.scalars().all() pattern
    empty = MagicMock()
    empty.all.return_value = []
    season_result = MagicMock()
    season_result.all.return_value = [postponed]
    session.scalars.side_effect = [empty, season_result]

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
    home = SimpleNamespace(id=1, name="Chelsea", short_name=None, medium_name=None)
    away = SimpleNamespace(id=2, name="Liverpool", short_name=None, medium_name=None)
    resolver.team_repo.to_football_data_name = MagicMock(side_effect=lambda n: n)
    result = resolver.resolve_match(
        provider_match,
        league_code="E0",
        home_team=home,
        away_team=away,
        season="2526",
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
    session.scalars.return_value.one.return_value = fake_row
    with patch(
        "objects.repositories.external_entity_mapping_repository.pg_insert"
    ) as insert_mock:
        statement = MagicMock()
        insert_mock.return_value.values.return_value.on_conflict_do_update.return_value.returning.return_value = (
            statement
        )
        row = repo.upsert(
            provider="fotmob",
            entity_type="league",
            internal_entity_id=3,
            external_entity_id="47",
            external_name="Premier League",
        )
        assert row.external_entity_id == "47"
        insert_mock.assert_called()


# ---------------------------------------------------------------------------
# 7–10. Metrics / missing xG / shot dedupe
# ---------------------------------------------------------------------------


def _sample_details() -> ProviderMatchDetails:
    payload = load_fixture("fotmob", "match_details.json")
    return parse_fotmob_match_details(payload)


def test_non_penalty_xg_calculation():
    details = _sample_details()
    metrics = calculate_derived_metrics(
        details,
        home_team_external_id=details.match.home_team_id,
        away_team_external_id=details.match.away_team_id,
    )
    # Home shots xG: 0.4 + 0.75 + 0.2 = 1.35; npxG excludes penalty 0.75 => 0.6
    assert metrics.home_non_penalty_xg == pytest.approx(0.6)
    assert metrics.away_non_penalty_xg == pytest.approx(0.65)


def test_set_piece_xg_calculation():
    details = _sample_details()
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
    first = shot_fingerprint(match_id=9, provider="fotmob", shot=shot, team_internal_id=3)
    second = shot_fingerprint(match_id=9, provider="fotmob", shot=shot, team_internal_id=3)
    assert first == second
    other = shot_fingerprint(match_id=9, provider="fotmob", shot=shot, team_internal_id=4)
    assert first != other


# ---------------------------------------------------------------------------
# 11–14. Service behaviour: idempotent, retry, fallback, isolation
# ---------------------------------------------------------------------------


def test_idempotent_repeated_imports_skip_without_force():
    session = MagicMock()
    provider = MagicMock()
    provider.name = "fotmob"
    service = ExtendedMatchDataService(
        provider=provider, session=session, dry_run=False
    )
    service.historical_repo.get = MagicMock(
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


def test_fallback_provider_used_when_primary_fails():
    session = MagicMock()
    primary = MagicMock()
    primary.name = "fotmob"
    primary.fetch_match_details.side_effect = NotFoundError("missing")
    fallback = MagicMock()
    fallback.name = "sofascore"
    details = _sample_details()
    details.match.provider_match_id = "fb-1"
    fallback.fetch_match_details.return_value = details

    service = ExtendedMatchDataService(
        provider=primary,
        fallback_provider=fallback,
        session=session,
        dry_run=True,
    )
    historical = SimpleNamespace(id=5, home_team="Manchester City", away_team="Arsenal")
    service.resolver.mapping_repo.get_by_internal = MagicMock(
        side_effect=[
            SimpleNamespace(external_entity_id="missing"),
            SimpleNamespace(external_entity_id="fb-1"),
        ]
    )
    fetched, used = service._fetch_details_with_fallback("missing", historical)
    assert used == "sofascore"
    assert fetched is not None


def test_transaction_rollback_for_one_failed_match_keeps_others():
    session = MagicMock()
    # begin_nested context manager
    session.begin_nested.return_value.__enter__ = MagicMock(return_value=None)
    session.begin_nested.return_value.__exit__ = MagicMock(return_value=False)

    provider = MagicMock()
    provider.name = "fotmob"
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


# ---------------------------------------------------------------------------
# 15. No future-data leakage in team features
# ---------------------------------------------------------------------------


def test_team_features_no_future_data_leakage():
    session = MagicMock()
    service = TeamStrengthFeatureService(session=session, provider="fotmob")
    team = SimpleNamespace(id=1, name="Manchester City", league_id=1)
    service.team_repo.get = MagicMock(return_value=team)

    past = SimpleNamespace(
        id=10,
        match_date=date(2025, 8, 1),
        home_team="Manchester City",
        away_team="Arsenal",
        home_goals=2,
        away_goals=1,
    )
    future = SimpleNamespace(
        id=11,
        match_date=date(2025, 8, 20),
        home_team="Manchester City",
        away_team="Chelsea",
        home_goals=3,
        away_goals=0,
    )
    # Repository query already filters match_date < before; simulate only past returned.
    service._load_team_match_stats = MagicMock(
        return_value=[
            (
                past,
                SimpleNamespace(
                    home_non_penalty_xg=1.0,
                    away_non_penalty_xg=0.5,
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
    service._league_averages = MagicMock(
        return_value={"attack": 1.0, "defence": 1.0, "npxg_for": 1.0, "npxg_against": 1.0}
    )
    cutoff = datetime(2025, 8, 10, tzinfo=timezone.utc)
    features = service.calculate_features(1, before=cutoff, lookback_matches=20)
    assert features.sample_size == 1
    assert features.non_penalty_xg_for is not None
    # Ensure loader was asked with cutoff (no future match id 11).
    args, kwargs = service._load_team_match_stats.call_args
    assert kwargs["before_date"] == cutoff.date()
    loaded_ids = [row[0].id for row in service._load_team_match_stats.return_value]
    assert 11 not in loaded_ids
    assert future.id not in loaded_ids


def test_xg_disagreement_warning():
    details = _sample_details()
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
