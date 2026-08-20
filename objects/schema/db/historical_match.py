"""Prefer objects.schema.db.fixture.

Legacy aliases keep older imports loading during the HistoricalMatch → Fixture migration.
"""

from objects.schema.db.fixture import Fixture, FixtureCreate

HistoricalMatch = Fixture
HistoricalMatchCreate = FixtureCreate
HistoricalMatchDraft = FixtureCreate

__all__ = [
    "Fixture",
    "FixtureCreate",
    "HistoricalMatch",
    "HistoricalMatchCreate",
    "HistoricalMatchDraft",
]
