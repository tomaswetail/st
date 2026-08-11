from __future__ import annotations

from typing import Any

from sqlalchemy import select

from objects.models.st_round import STRoundModel

from objects.repositories.base import BaseRepository
from objects.schema.db.st_round import STRound


class STRoundRepository(BaseRepository[STRoundModel]):
    model = STRoundModel

    def get_by_draw_number(self, draw_number: int) -> list[STRoundModel]:
        return self.session.scalars(
                select(self.model)
                .where(self.model.draw_number == draw_number)
                .order_by(self.model.event_number)
            ).first()


    def get_by_product_draw_event(
        self,
        product_id: int,
        draw_number: int,
        event_number: int,
    ) -> STRoundModel | None:
        return self.session.scalar(
            select(self.model).where(
                self.model.product_id == product_id,
                self.model.draw_number == draw_number,
                self.model.event_number == event_number,
            )
        )

    def upsert(
        self,
        *,
        product_id: int,
        product_name: str,
        draw_number: int,
    ) -> STRoundModel:
        round_model = self.get_by_draw_number(
            draw_number
        )
        if round_model is None:
            round_model = self.create(
                product_id=product_id,
                draw_number=draw_number,
            )

        round_model.product_name = product_name
        return round_model

    def to_schema(self, model: STRoundModel) -> STRound:
        return STRound(
            id=model.id,
            product_id=model.product_id,
            product_name=model.product_name,
            draw_number=model.draw_number,
            event_number=model.event_number,
            description=model.description or "",
            comment=model.comment or "",
            cancelled=model.cancelled,
        )
