"""Cutoff-safe player availability features from match_availability snapshots."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from objects.models.fixture import FixtureModel
from objects.models.match_availability import MatchAvailabilityModel
from objects.models.st_match import STMatchModel
from objects.repositories.fixture_repository import FixtureRepository
from objects.repositories.match_availability_repository import MatchAvailabilityRepository
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.data_classes.player_availability_features import (
    PlayerAvailabilityFeatures,
)
from utils.fixture_fields import fixture_match_date

# Market values in provider payloads are often absolute currency; scale to millions.
_MARKET_VALUE_SCALE = 1_000_000.0
_UNAVAILABLE_STATUSES = frozenset({"injured", "suspended", "unavailable", "doubtful"})
_PROVIDER_PREFERENCE = ("api-football",)


def _empty_features() -> PlayerAvailabilityFeatures:
    return PlayerAvailabilityFeatures(
        home_missing_player_value=None,
        away_missing_player_value=None,
        missing_value_difference=None,
        home_unavailable_count=None,
        away_unavailable_count=None,
        home_lineup_changes=None,
        away_lineup_changes=None,
        missing_value_x_favourite=None,
        short_rest_x_missing_value=None,
        has_availability=0,
        coverage_level=None,
    )


def _scale_market_value(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) / _MARKET_VALUE_SCALE


# Count-only proxies from API-Football are stored as raw unavailable counts, not currency.
_COUNT_PROXY_MAX_SCALED = 1e-4


def _missing_value_from_snapshot(value: float | None) -> float | None:
    scaled = _scale_market_value(value)
    if scaled is None:
        return None
    if abs(scaled) <= _COUNT_PROXY_MAX_SCALED:
        return None
    return scaled


def _starter_external_ids(
    players_json: list[dict[str, Any]] | None,
    *,
    team_side: str,
) -> set[str]:
    if not players_json:
        return set()
    return {
        str(player["external_id"])
        for player in players_json
        if player.get("team_side") == team_side
        and player.get("is_starter")
        and player.get("external_id") is not None
    }


def _lineup_change_count(
    current_starters: set[str],
    previous_starters: set[str],
) -> int | None:
    if not current_starters or not previous_starters:
        return None
    return len(current_starters.symmetric_difference(previous_starters))


class PlayerAvailabilityCalculator:
    """Derive injury/availability ML features from persisted snapshots only.

    Resolves the ST coupon match to a historical fixture via team external ids
    and kickoff window, then loads the latest ``match_availability`` row with
    ``snapshot_at <= match.start_time`` (no future leakage).
    """

    def __init__(
        self,
        session: Session,
        config: DataSourceConfig | None = None,
    ) -> None:
        self.session = session
        self.config = config or DataSourceConfig()
        self.fixture_repo = FixtureRepository(session)
        self.availability_repo = MatchAvailabilityRepository(session)
        self._fixture_cache: dict[tuple[int, int, date], FixtureModel | None] = {}
        self._availability_cache: dict[int, MatchAvailabilityModel | None] = {}

    def clear_caches(self) -> None:
        self._fixture_cache.clear()
        self._availability_cache.clear()

    def calculate(
        self,
        match: STMatchModel,
        *,
        favourite_strength: float | None = None,
        home_short_rest: int | None = None,
        away_short_rest: int | None = None,
        home_previous_fixture: FixtureModel | None = None,
        away_previous_fixture: FixtureModel | None = None,
    ) -> PlayerAvailabilityFeatures:
        if match.home_team is None or match.away_team is None:
            raise ValueError(f"Missing team on match id={match.id}")
        if match.start_time is None:
            raise ValueError(f"Missing start_time on match id={match.id}")

        cutoff_dt = (
            match.start_time
            if isinstance(match.start_time, datetime)
            else datetime.combine(match.start_time, datetime.min.time())
        )
        fixture = self._resolve_fixture(match)
        if fixture is None:
            return _empty_features()

        snapshot = self._latest_snapshot(fixture.id, cutoff_dt)
        if snapshot is None or snapshot.coverage_level == "none":
            return _empty_features()

        home_missing = _missing_value_from_snapshot(snapshot.home_missing_value)
        away_missing = _missing_value_from_snapshot(snapshot.away_missing_value)
        if home_missing is None and away_missing is None:
            home_missing, away_missing = self._missing_values_from_players(
                snapshot.players_json
            )

        missing_diff = None
        if home_missing is not None and away_missing is not None:
            missing_diff = home_missing - away_missing
        elif home_missing is not None:
            missing_diff = home_missing
        elif away_missing is not None:
            missing_diff = -away_missing

        home_unavailable = snapshot.home_unavailable_count
        away_unavailable = snapshot.away_unavailable_count

        home_lineup_changes = self._lineup_changes_for_side(
            snapshot,
            team_side="home",
            previous_fixture=home_previous_fixture,
            cutoff_dt=cutoff_dt,
        )
        away_lineup_changes = self._lineup_changes_for_side(
            snapshot,
            team_side="away",
            previous_fixture=away_previous_fixture,
            cutoff_dt=cutoff_dt,
        )

        missing_value_x_favourite = None
        if missing_diff is not None and favourite_strength is not None:
            missing_value_x_favourite = missing_diff * float(favourite_strength)

        short_rest_x_missing_value = None
        if missing_diff is not None:
            home_sr = int(home_short_rest or 0)
            away_sr = int(away_short_rest or 0)
            short_rest_x_missing_value = missing_diff * float(home_sr - away_sr)

        return PlayerAvailabilityFeatures(
            home_missing_player_value=home_missing,
            away_missing_player_value=away_missing,
            missing_value_difference=missing_diff,
            home_unavailable_count=home_unavailable,
            away_unavailable_count=away_unavailable,
            home_lineup_changes=home_lineup_changes,
            away_lineup_changes=away_lineup_changes,
            missing_value_x_favourite=missing_value_x_favourite,
            short_rest_x_missing_value=short_rest_x_missing_value,
            has_availability=1,
            coverage_level=snapshot.coverage_level,
        )

    def _resolve_fixture(self, match: STMatchModel) -> FixtureModel | None:
        home_external_id = getattr(match.home_team, "external_id", None)
        away_external_id = getattr(match.away_team, "external_id", None)
        if home_external_id is None or away_external_id is None:
            return None

        kickoff = match.start_time
        if isinstance(kickoff, datetime):
            kickoff_date = kickoff.date()
        else:
            kickoff_date = kickoff

        cache_key = (int(home_external_id), int(away_external_id), kickoff_date)
        if cache_key in self._fixture_cache:
            return self._fixture_cache[cache_key]

        tolerance_days = max(
            1,
            int(self.config.kickoff_match_tolerance_minutes / (24 * 60)) or 1,
        )
        candidates = self.fixture_repo.find_by_date_range_and_teams(
            date_from=kickoff_date - timedelta(days=tolerance_days),
            date_to=kickoff_date + timedelta(days=tolerance_days),
            home_team_ids=[int(home_external_id)],
            away_team_ids=[int(away_external_id)],
        )
        if not candidates:
            self._fixture_cache[cache_key] = None
            return None

        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            chosen = min(
                candidates,
                key=lambda fixture: abs(
                    (fixture_match_date(fixture) - kickoff_date).days
                ),
            )
        self._fixture_cache[cache_key] = chosen
        return chosen

    def _latest_snapshot(
        self,
        fixture_id: int,
        cutoff_dt: datetime,
    ) -> MatchAvailabilityModel | None:
        if fixture_id in self._availability_cache:
            cached = self._availability_cache[fixture_id]
            if cached is None:
                return None
            if cached.snapshot_at <= cutoff_dt:
                return cached

        preferred: MatchAvailabilityModel | None = None
        for provider in _PROVIDER_PREFERENCE:
            rows = self.availability_repo.list_before_cutoff(
                fixture_id,
                cutoff=cutoff_dt,
                provider=provider,
            )
            if rows:
                preferred = rows[0]
                break
        if preferred is None:
            rows = self.availability_repo.list_before_cutoff(
                fixture_id,
                cutoff=cutoff_dt,
            )
            preferred = rows[0] if rows else None

        self._availability_cache[fixture_id] = preferred
        return preferred

    def _lineup_changes_for_side(
        self,
        current: MatchAvailabilityModel,
        *,
        team_side: str,
        previous_fixture: FixtureModel | None,
        cutoff_dt: datetime,
    ) -> int | None:
        if previous_fixture is None:
            return None
        previous = self._latest_snapshot(previous_fixture.id, cutoff_dt)
        if previous is None:
            return None
        return _lineup_change_count(
            _starter_external_ids(current.players_json, team_side=team_side),
            _starter_external_ids(previous.players_json, team_side=team_side),
        )

    @staticmethod
    def _missing_values_from_players(
        players_json: list[dict[str, Any]] | None,
    ) -> tuple[float | None, float | None]:
        if not players_json:
            return None, None
        home_values: list[float] = []
        away_values: list[float] = []
        for player in players_json:
            status = str(player.get("status") or "").lower()
            if status not in _UNAVAILABLE_STATUSES:
                continue
            scaled = _scale_market_value(
                float(player["market_value"])
                if player.get("market_value") is not None
                else None
            )
            if scaled is None:
                continue
            if player.get("team_side") == "home":
                home_values.append(scaled)
            elif player.get("team_side") == "away":
                away_values.append(scaled)
        return (
            float(sum(home_values)) if home_values else None,
            float(sum(away_values)) if away_values else None,
        )
