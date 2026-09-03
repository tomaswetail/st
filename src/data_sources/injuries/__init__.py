"""Injury and player availability ingestion (API-Football only)."""

from data_sources.injuries.backfill import InjuryBackfillResult, InjuryBackfillService
from data_sources.injuries.dtos import MatchAvailabilitySnapshot, PlayerAvailabilityRecord

__all__ = [
    "InjuryBackfillResult",
    "InjuryBackfillService",
    "MatchAvailabilitySnapshot",
    "PlayerAvailabilityRecord",
]
