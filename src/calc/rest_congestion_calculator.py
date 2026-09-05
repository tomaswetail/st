"""Rest, fixture congestion, and extra-time fatigue features (no future leakage)."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Sequence

from sqlalchemy.orm import Session

from src.objects.models.fixture import FixtureModel
from src.objects.models.st_match import STMatchModel
from src.objects.repositories.fixture_repository import FixtureRepository
from src.objects.schema.data_classes.data_sources import DataSourceConfig
from src.objects.schema.data_classes.player_availability_features import (
    PlayerAvailabilityFeatures,
)
from src.objects.schema.data_classes.rest_congestion_features import RestCongestionFeatures
from src.utils.fixture_fields import fixture_match_date, fixture_went_to_extra_time


class RestCongestionCalculator:
    """Derive rest/congestion ML features from prior matches only."""

    def __init__(
        self,
        session: Session,
        config: DataSourceConfig | None = None,
    ) -> None:
        self.session = session
        self.config = config or DataSourceConfig()
        self.fixture_repo = FixtureRepository(session)
        self._team_history_cache: dict[
            tuple[str, date, int], list[FixtureModel]
        ] = {}

    def clear_caches(self) -> None:
        """Drop team history lookback cache."""
        self._team_history_cache.clear()

    def calculate(
        self,
        match: STMatchModel,
        *,
        availability: PlayerAvailabilityFeatures | None = None,
    ) -> RestCongestionFeatures:
        """Compute rest and congestion features for one ST fixture.

        When ``availability`` is provided with ``has_availability=1``, fills
        lineup/squad-depth stubs from those signals.
        """
        if match.home_team is None or match.away_team is None:
            raise ValueError(f"Missing team on match id={match.id}")
        if match.start_time is None:
            raise ValueError(f"Missing start_time on match id={match.id}")

        cutoff = (
            match.start_time.date()
            if isinstance(match.start_time, datetime)
            else match.start_time
        )
        lookback = self.config.rest_congestion_lookback_matches
        home_history = self._load_team_history(match.home_team.name, cutoff, lookback)
        away_history = self._load_team_history(match.away_team.name, cutoff, lookback)

        home_previous = home_history[0] if home_history else None
        away_previous = away_history[0] if away_history else None

        home_rest_days = self._rest_days(cutoff, home_previous)
        away_rest_days = self._rest_days(cutoff, away_previous)
        rest_day_difference = (
            home_rest_days - away_rest_days
            if home_rest_days is not None and away_rest_days is not None
            else None
        )

        window_start = cutoff - timedelta(days=self.config.rest_congestion_window_days)
        home_matches_last_14_days = self._count_in_window(
            home_history, window_start=window_start, cutoff=cutoff
        )
        away_matches_last_14_days = self._count_in_window(
            away_history, window_start=window_start, cutoff=cutoff
        )

        home_short_rest = self._short_rest(home_rest_days)
        away_short_rest = self._short_rest(away_rest_days)
        home_congestion = self._congestion(home_matches_last_14_days)
        away_congestion = self._congestion(away_matches_last_14_days)

        home_extra_time = fixture_went_to_extra_time(home_previous) if home_previous else False
        away_extra_time = fixture_went_to_extra_time(away_previous) if away_previous else False
        home_extra_time_short_rest = int(home_extra_time) * home_short_rest
        away_extra_time_short_rest = int(away_extra_time) * away_short_rest

        home_lineup_changes = None
        away_lineup_changes = None
        congestion_x_squad_depth = None
        short_rest_x_rotation = None
        if availability is not None and availability.has_availability:
            home_lineup_changes = availability.home_lineup_changes
            away_lineup_changes = availability.away_lineup_changes
            unavailable_total = (
                int(availability.home_unavailable_count or 0)
                + int(availability.away_unavailable_count or 0)
            )
            congestion_x_squad_depth = float(
                (home_congestion + away_congestion) * unavailable_total
            )
            home_rotation = int(home_lineup_changes or 0)
            away_rotation = int(away_lineup_changes or 0)
            short_rest_x_rotation = float(
                home_short_rest * home_rotation + away_short_rest * away_rotation
            )

        return RestCongestionFeatures(
            home_rest_days=home_rest_days,
            away_rest_days=away_rest_days,
            rest_day_difference=rest_day_difference,
            home_matches_last_14_days=home_matches_last_14_days,
            away_matches_last_14_days=away_matches_last_14_days,
            home_short_rest=home_short_rest,
            away_short_rest=away_short_rest,
            home_congestion=home_congestion,
            away_congestion=away_congestion,
            home_extra_time_in_previous_match=home_extra_time,
            away_extra_time_in_previous_match=away_extra_time,
            home_extra_time_short_rest=home_extra_time_short_rest,
            away_extra_time_short_rest=away_extra_time_short_rest,
            extra_time_x_short_rest=(
                home_extra_time_short_rest + away_extra_time_short_rest
            ),
            home_lineup_changes=home_lineup_changes,
            away_lineup_changes=away_lineup_changes,
            congestion_x_squad_depth=congestion_x_squad_depth,
            short_rest_x_rotation=short_rest_x_rotation,
        )

    def previous_fixtures(
        self, match: STMatchModel
    ) -> tuple[FixtureModel | None, FixtureModel | None]:
        """Return (home_previous, away_previous) fixtures strictly before kickoff."""
        if match.home_team is None or match.away_team is None:
            raise ValueError(f"Missing team on match id={match.id}")
        if match.start_time is None:
            raise ValueError(f"Missing start_time on match id={match.id}")
        cutoff = (
            match.start_time.date()
            if isinstance(match.start_time, datetime)
            else match.start_time
        )
        lookback = self.config.rest_congestion_lookback_matches
        home_history = self._load_team_history(match.home_team.name, cutoff, lookback)
        away_history = self._load_team_history(match.away_team.name, cutoff, lookback)
        return (
            home_history[0] if home_history else None,
            away_history[0] if away_history else None,
        )

    def _load_team_history(
        self,
        team_name: str,
        cutoff: date,
        lookback: int,
    ) -> list[FixtureModel]:
        cache_key = (team_name, cutoff, lookback)
        cached = self._team_history_cache.get(cache_key)
        if cached is not None:
            return cached
        matches = self.fixture_repo.find_before_date_by_team(
            team_name=team_name,
            before_date=cutoff,
            venue=None,
            limit=lookback,
        )
        history = [
            fixture
            for fixture in matches
            if fixture_match_date(fixture) < cutoff
        ]
        self._team_history_cache[cache_key] = history
        return history

    @staticmethod
    def _rest_days(cutoff: date, previous: FixtureModel | None) -> int | None:
        if previous is None:
            return None
        return (cutoff - fixture_match_date(previous)).days

    @staticmethod
    def _count_in_window(
        history: Sequence[FixtureModel],
        *,
        window_start: date,
        cutoff: date,
    ) -> int:
        return sum(
            1
            for fixture in history
            if window_start <= fixture_match_date(fixture) < cutoff
        )

    def _short_rest(self, rest_days: int | None) -> int:
        if rest_days is None:
            return 0
        return max(0, self.config.rest_short_rest_threshold_days - rest_days)

    def _congestion(self, matches_in_window: int) -> int:
        return max(0, matches_in_window - self.config.rest_congestion_match_threshold)
