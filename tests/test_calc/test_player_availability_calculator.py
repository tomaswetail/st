"""Unit tests for PlayerAvailabilityCalculator."""

from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.calc.player_availability_calculator import PlayerAvailabilityCalculator
from src.objects.schema.data_classes.data_sources import DataSourceConfig


def _match(
    *,
    start_time: datetime = datetime(2025, 8, 15, 18, 0, tzinfo=timezone.utc),
) -> SimpleNamespace:
    return SimpleNamespace(
        id=99,
        start_time=start_time,
        home_team=SimpleNamespace(id=1, name="Arsenal", external_id=42),
        away_team=SimpleNamespace(id=2, name="Chelsea", external_id=43),
    )


def _snapshot(
    *,
    home_missing_value: float | None = 2_000_000.0,
    away_missing_value: float | None = 500_000.0,
    home_unavailable_count: int = 2,
    away_unavailable_count: int = 1,
    coverage_level: str = "full_lineup",
    players_json: list[dict] | None = None,
    snapshot_at: datetime | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=1,
        fixture_id=10,
        provider="fotmob",
        source="fotmob_lineup",
        snapshot_at=snapshot_at
        or datetime(2025, 8, 15, 17, 0, tzinfo=timezone.utc),
        home_missing_value=home_missing_value,
        away_missing_value=away_missing_value,
        home_unavailable_count=home_unavailable_count,
        away_unavailable_count=away_unavailable_count,
        coverage_level=coverage_level,
        players_json=players_json
        or [
            {
                "external_id": "1",
                "team_side": "home",
                "is_starter": True,
                "status": "starter",
            },
            {
                "external_id": "2",
                "team_side": "home",
                "is_starter": True,
                "status": "starter",
            },
            {
                "external_id": "10",
                "team_side": "away",
                "is_starter": True,
                "status": "starter",
            },
        ],
    )


def test_calculate_returns_empty_when_fixture_unresolved():
    session = MagicMock()
    calculator = PlayerAvailabilityCalculator(session, config=DataSourceConfig())
    calculator.fixture_repo.find_by_date_range_and_teams = MagicMock(return_value=[])

    features = calculator.calculate(_match())

    assert features.has_availability == 0
    assert features.home_missing_player_value is None
    assert features.missing_value_difference is None


def test_calculate_scales_missing_value_and_interactions():
    session = MagicMock()
    calculator = PlayerAvailabilityCalculator(session, config=DataSourceConfig())
    calculator.fixture_repo.find_by_date_range_and_teams = MagicMock(
        return_value=[
            SimpleNamespace(
                id=10,
                fixture_date=datetime(2025, 8, 15, tzinfo=timezone.utc),
            )
        ]
    )
    calculator.availability_repo.list_before_cutoff = MagicMock(
        return_value=[_snapshot()]
    )

    features = calculator.calculate(
        _match(),
        favourite_strength=0.5,
        home_short_rest=2,
        away_short_rest=0,
    )

    assert features.has_availability == 1
    assert features.home_missing_player_value == 2.0
    assert features.away_missing_player_value == 0.5
    assert features.missing_value_difference == 1.5
    assert features.home_unavailable_count == 2
    assert features.away_unavailable_count == 1
    assert features.missing_value_x_favourite == 0.75
    assert features.short_rest_x_missing_value == 3.0


def test_lineup_changes_vs_previous_fixture():
    session = MagicMock()
    calculator = PlayerAvailabilityCalculator(session, config=DataSourceConfig())
    calculator.fixture_repo.find_by_date_range_and_teams = MagicMock(
        return_value=[
            SimpleNamespace(
                id=10,
                fixture_date=datetime(2025, 8, 15, tzinfo=timezone.utc),
            )
        ]
    )

    current = _snapshot(
        players_json=[
            {
                "external_id": "1",
                "team_side": "home",
                "is_starter": True,
                "status": "starter",
            },
            {
                "external_id": "3",
                "team_side": "home",
                "is_starter": True,
                "status": "starter",
            },
            {
                "external_id": "10",
                "team_side": "away",
                "is_starter": True,
                "status": "starter",
            },
        ]
    )
    previous = _snapshot(
        players_json=[
            {
                "external_id": "1",
                "team_side": "home",
                "is_starter": True,
                "status": "starter",
            },
            {
                "external_id": "2",
                "team_side": "home",
                "is_starter": True,
                "status": "starter",
            },
            {
                "external_id": "10",
                "team_side": "away",
                "is_starter": True,
                "status": "starter",
            },
        ]
    )

    def _list_before(fixture_id, *, cutoff, provider=None):
        if fixture_id == 10:
            return [current]
        if fixture_id == 9:
            return [previous]
        return []

    calculator.availability_repo.list_before_cutoff = MagicMock(
        side_effect=_list_before
    )

    features = calculator.calculate(
        _match(),
        home_previous_fixture=SimpleNamespace(id=9),
        away_previous_fixture=None,
    )

    # home: {1,3} vs {1,2} → symmetric diff size 2
    assert features.home_lineup_changes == 2
    assert features.away_lineup_changes is None


def test_ignores_post_cutoff_snapshots_via_repo_filter():
    session = MagicMock()
    calculator = PlayerAvailabilityCalculator(session, config=DataSourceConfig())
    calculator.fixture_repo.find_by_date_range_and_teams = MagicMock(
        return_value=[SimpleNamespace(id=10, fixture_date=datetime(2025, 8, 15))]
    )
    calculator.availability_repo.list_before_cutoff = MagicMock(return_value=[])

    features = calculator.calculate(_match())

    assert features.has_availability == 0
    calculator.availability_repo.list_before_cutoff.assert_called()


def test_count_proxy_missing_values_are_treated_as_none():
    session = MagicMock()
    calculator = PlayerAvailabilityCalculator(session, config=DataSourceConfig())
    calculator.fixture_repo.find_by_date_range_and_teams = MagicMock(
        return_value=[SimpleNamespace(id=10, fixture_date=datetime(2025, 8, 15))]
    )
    calculator.availability_repo.list_before_cutoff = MagicMock(
        return_value=[
            _snapshot(
                home_missing_value=3.0,
                away_missing_value=1.0,
                coverage_level="partial",
            )
        ]
    )

    features = calculator.calculate(_match())

    assert features.has_availability == 1
    assert features.home_missing_player_value is None
    assert features.away_missing_player_value is None
    assert features.missing_value_difference is None
    assert features.home_unavailable_count == 2


def test_fixture_shaped_uses_fixture_pk_directly():
    session = MagicMock()
    calculator = PlayerAvailabilityCalculator(session, config=DataSourceConfig())
    calculator.fixture_repo.find_by_date_range_and_teams = MagicMock(return_value=[])
    calculator.availability_repo.list_before_cutoff = MagicMock(return_value=[])
    fixture = SimpleNamespace(
        id=55,
        fixture_date=datetime(2025, 8, 15, 18, 0, tzinfo=timezone.utc),
        home_team=SimpleNamespace(id=1, name="Arsenal", external_id=42),
        away_team=SimpleNamespace(id=2, name="Chelsea", external_id=43),
    )

    features = calculator.calculate(fixture)

    assert features.has_availability == 0
    calculator.fixture_repo.find_by_date_range_and_teams.assert_not_called()
    assert calculator.availability_repo.list_before_cutoff.call_args.args[0] == 55


def test_before_date_override_is_passed_to_snapshot_cutoff():
    session = MagicMock()
    calculator = PlayerAvailabilityCalculator(session, config=DataSourceConfig())
    calculator.fixture_repo.find_by_date_range_and_teams = MagicMock(
        return_value=[SimpleNamespace(id=10, fixture_date=datetime(2025, 8, 15))]
    )
    calculator.availability_repo.list_before_cutoff = MagicMock(return_value=[])

    calculator.calculate(_match(), before_date=date(2025, 6, 1))

    cutoff = calculator.availability_repo.list_before_cutoff.call_args.kwargs["cutoff"]
    assert cutoff.date() == date(2025, 6, 1)
