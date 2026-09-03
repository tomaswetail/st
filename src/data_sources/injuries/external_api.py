"""API-Football client wrapper used as the injury/availability data provider."""

from __future__ import annotations

from datetime import datetime

from data_sources.api_football_client import APIFootballClient
from data_sources.injuries.dtos import PlayerAvailabilityRecord
from data_sources.injuries.parsers.api_football import (
    parse_api_football_availability,
)
from objects.schema.data_classes.data_sources import DataSourceConfig


class ApiFootballInjuryProvider:
    """Fetch fixture injuries + lineups from API-Football.

    This is the sole Phase 2 injury/lineup source.
    """

    provider_name = "api-football"

    def __init__(
        self,
        client: APIFootballClient | None = None,
        *,
        config: DataSourceConfig | None = None,
    ) -> None:
        self.client = client or APIFootballClient(config=config or DataSourceConfig())

    def fetch_fixture_availability(
        self,
        *,
        internal_fixture_id: int,
        api_fixture_id: int,
        home_team_external_id: int | str | None,
        away_team_external_id: int | str | None,
        kickoff_at: datetime,
    ):
        lineups = self.client.get_fixture_lineups(int(api_fixture_id))
        injuries = self.client.get_fixture_injuries(int(api_fixture_id))
        return parse_api_football_availability(
            fixture_id=internal_fixture_id,
            home_team_external_id=home_team_external_id,
            away_team_external_id=away_team_external_id,
            kickoff_at=kickoff_at,
            lineups_payload=lineups,
            injuries_payload=injuries,
        )
