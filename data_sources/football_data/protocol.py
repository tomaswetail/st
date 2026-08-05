"""Provider protocol for historical football data adapters."""

from __future__ import annotations

from typing import Protocol

from objects.schema.data_classes.provider_dtos import (
    ProviderLeague,
    ProviderMatch,
    ProviderMatchDetails,
    ProviderSeason,
    ProviderShot,
)


class FootballDataProvider(Protocol):
    """Sync interface implemented by FotMob and SofaScore adapters."""

    name: str

    def fetch_available_leagues(self) -> list[ProviderLeague]:
        """Return the provider's full league catalogue."""
        ...

    def fetch_league_seasons(self, provider_league_id: str) -> list[ProviderSeason]:
        """List seasons available for a provider league."""
        ...

    def fetch_season_matches(
        self,
        provider_league_id: str,
        provider_season_id: str,
        *,
        country_code: str | None = None,
    ) -> list[ProviderMatch]:
        """List fixtures for a league season (ccode3 for FotMob)."""
        ...

    def fetch_match_details(self, provider_match_id: str) -> ProviderMatchDetails:
        """Fetch match metadata, stats, and shots."""
        ...

    def fetch_match_shots(self, provider_match_id: str) -> list[ProviderShot]:
        """Fetch shot events for a provider match."""
        ...
