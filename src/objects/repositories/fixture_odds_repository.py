"""Historical fixture odds persistence."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.objects.models.fixture_odds import FixtureOddsModel
from src.objects.repositories.base import BaseRepository
from src.objects.repositories.utils import json_safe


class FixtureOddsRepository(BaseRepository[FixtureOddsModel]):
    model = FixtureOddsModel

    def list_for_fixture(self, fixture_id: int) -> list[FixtureOddsModel]:
        query = select(self.model).where(self.model.fixture_id == fixture_id)
        return list(self.session.scalars(query).all())

    def get_for_fixture(
        self,
        fixture_id: int,
        *,
        bookmaker: str,
        price_type: str,
        provider: str | None = None,
    ) -> FixtureOddsModel | None:
        query = select(self.model).where(
            self.model.fixture_id == fixture_id,
            self.model.bookmaker == bookmaker,
            self.model.price_type == price_type,
        )
        if provider is not None:
            query = query.where(self.model.provider == provider)
        return self.session.scalar(query)

    def upsert(
        self,
        *,
        fixture_id: int,
        provider: str,
        source: str,
        bookmaker: str,
        price_type: str,
        odds_home: float | None,
        odds_draw: float | None,
        odds_away: float | None,
        snapshot_at: datetime,
        kickoff_at: datetime,
        raw_payload: dict[str, Any] | None = None,
    ) -> FixtureOddsModel:
        payload = {
            "fixture_id": fixture_id,
            "provider": provider,
            "source": source,
            "bookmaker": bookmaker,
            "price_type": price_type,
            "odds_home": odds_home,
            "odds_draw": odds_draw,
            "odds_away": odds_away,
            "snapshot_at": snapshot_at,
            "kickoff_at": kickoff_at,
            "raw_payload": json_safe(raw_payload) if raw_payload is not None else None,
        }
        statement = (
            pg_insert(self.model)
            .values(**payload)
            .on_conflict_do_update(
                constraint="uq_fixture_odds_fixture_provider_bookmaker_price",
                set_={
                    key: value
                    for key, value in payload.items()
                    if key not in ("fixture_id", "provider", "bookmaker", "price_type")
                },
            )
            .returning(self.model)
        )
        return self.session.scalars(statement).one()

    def fixture_ids_with_odds(
        self,
        fixture_ids: list[int],
        *,
        provider: str | None = None,
    ) -> set[int]:
        if not fixture_ids:
            return set()
        query = select(self.model.fixture_id).where(
            self.model.fixture_id.in_(fixture_ids)
        )
        if provider:
            query = query.where(self.model.provider == provider)
        return set(self.session.scalars(query).all())
