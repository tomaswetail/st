"""Import result DTOs for historical football data ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ImportStatus = Literal["imported", "updated", "skipped", "unresolved", "failed"]


@dataclass
class MatchImportResult:
    """Outcome of importing one historical match."""
    internal_match_id: int
    provider_match_id: str | None
    status: ImportStatus
    shots_imported: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class BatchImportResult:
    """Aggregated counters and per-match results for a batch import."""
    requested: int
    imported: int = 0
    updated: int = 0
    skipped: int = 0
    unresolved: int = 0
    failed: int = 0
    results: list[MatchImportResult] = field(default_factory=list)

    def add(self, result: MatchImportResult) -> None:
        """Append a result and bump the matching status counter."""
        self.results.append(result)
        if result.status == "imported":
            self.imported += 1
        elif result.status == "updated":
            self.updated += 1
        elif result.status == "skipped":
            self.skipped += 1
        elif result.status == "unresolved":
            self.unresolved += 1
        elif result.status == "failed":
            self.failed += 1
