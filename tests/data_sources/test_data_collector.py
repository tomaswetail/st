"""DataCollector API-Football-only import tests."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from src.data_sources.api_football_leagues import ApiFootballLeagueEntry
from src.data_sources.data_collector import DataCollector
from src.objects.schema.data_classes.fixture import Fixture
from src.objects.schema.db.fixture import FixtureCreate


def _api_fixture() -> Fixture:
    kickoff = datetime(2024, 8, 17, 15, 0, tzinfo=timezone.utc)
    return Fixture(
        fixture_id=100,
        fixture_referee=None,
        fixture_timezone="UTC",
        fixture_date=kickoff,
        fixture_timestamp=int(kickoff.timestamp()),
        period_first=None,
        period_second=None,
        venue_id=None,
        venue_name=None,
        venue_city=None,
        status_long="Match Finished",
        status_short="FT",
        league_id=39,
        league_name="Premier League",
        league_country="England",
        league_flag=None,
        league_season=2024,
        league_round=None,
        home_team_id=42,
        home_team_name="Arsenal",
        home_team_winner=True,
        away_team_id=39,
        away_team_name="Wolves",
        away_team_winner=False,
        goals_home=2,
        goals_away=0,
        score_halftime_home=None,
        score_halftime_away=None,
        score_fulltime_home=2,
        score_fulltime_away=0,
        score_extratime_home=None,
        score_extratime_away=None,
        score_penalty_home=None,
        score_penalty_away=None,
    )
