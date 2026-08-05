"""Provider-agnostic historical football data ingestion (FotMob / SofaScore)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_sources.football_data.league_catalogue import LeagueCatalogueService
    from data_sources.football_data.service import ExtendedMatchDataService

__all__ = ["ExtendedMatchDataService", "LeagueCatalogueService"]


def __getattr__(name: str):
    """Lazy-load public service classes on attribute access."""
    if name == "ExtendedMatchDataService":
        from data_sources.football_data.service import ExtendedMatchDataService as cls

        return cls
    if name == "LeagueCatalogueService":
        from data_sources.football_data.league_catalogue import LeagueCatalogueService as cls

        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
