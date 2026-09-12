"""Historical 1X2 odds from football-data.co.uk attached to existing fixtures."""

from src.data_sources.football_data_odds.service import (
    FootballDataOddsIngestService,
    IngestTotals,
    LeagueSeasonStats,
)

__all__ = [
    "FootballDataOddsIngestService",
    "IngestTotals",
    "LeagueSeasonStats",
]
