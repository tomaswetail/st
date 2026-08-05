from typing import Any

from sqlalchemy import select

from objects.models.st_match import STMatchModel
from objects.models.st_match_bet import STMatchBetModel

from objects.repositories.base import BaseRepository
from objects.repositories.utils import distribution_by_outcome


class STMatchBetRepository(BaseRepository[STMatchBetModel]):
    model = STMatchBetModel

    def get_by_stryktipset_match_id(
        self, stryktipset_match_id: int
    ) -> STMatchBetModel | None:
        return self.session.scalar(
            select(self.model).where(
                self.model.stryktipset_match_id == stryktipset_match_id
            )
        )

    def upsert_from_draw_event(
        self, match: STMatchModel, draw_event: dict[str, Any]
    ) -> STMatchBetModel | None:
        bet_metrics = draw_event.get("betMetrics")
        distribution_1 = distribution_by_outcome(bet_metrics, "1")
        distribution_X = distribution_by_outcome(bet_metrics, "X")
        distribution_2 = distribution_by_outcome(bet_metrics, "2")

        if (
            distribution_1 is None
            or distribution_X is None
            or distribution_2 is None
        ):
            return None

        match_bet = self.get_by_stryktipset_match_id(match.id)
        if match_bet is None:
            match_bet = self.create(stryktipset_match_id=match.id)

        match_bet.distribution_1 = distribution_1
        match_bet.distribution_X = distribution_X
        match_bet.distribution_2 = distribution_2
        return match_bet
