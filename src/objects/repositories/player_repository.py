"""Player registry persistence."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.objects.models.player import PlayerModel
from src.objects.repositories.base import BaseRepository


class PlayerRepository(BaseRepository[PlayerModel]):
    model = PlayerModel

    def get_by_provider_external_id(
        self,
        provider: str,
        external_id: str,
    ) -> PlayerModel | None:
        return self.session.scalar(
            select(self.model).where(
                self.model.provider == provider,
                self.model.external_id == external_id,
            )
        )

    def upsert(
        self,
        *,
        provider: str,
        external_id: str,
        name: str | None = None,
        team_external_id: str | None = None,
        market_value: float | None = None,
    ) -> PlayerModel:
        payload = {
            "provider": provider,
            "external_id": external_id,
            "name": name,
            "team_external_id": team_external_id,
            "market_value": market_value,
        }
        statement = (
            pg_insert(self.model)
            .values(**payload)
            .on_conflict_do_update(
                constraint="uq_players_provider_external_id",
                set_={
                    "name": name,
                    "team_external_id": team_external_id,
                    "market_value": market_value,
                },
            )
            .returning(self.model)
        )
        return self.session.scalars(statement).one()
