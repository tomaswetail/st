from typing import Any

from sqlalchemy import select

from objects.models.st_match import STMatchModel
from objects.models.st_match_odds import STMatchOddsModel

from objects.repositories.base import BaseRepository
from utils.common import parse_swedish_decimal


class STMatchOddsRepository(BaseRepository[STMatchOddsModel]):
    model = STMatchOddsModel

    def get_by_stryktipset_match_id(
        self, stryktipset_match_id: int
    ) -> STMatchOddsModel | None:
        return self.session.scalar(
            select(self.model).where(
                self.model.stryktipset_match_id == stryktipset_match_id
            )
        )

    def upsert_from_draw_event(
        self, match: STMatchModel, draw_event: dict[str, Any]
    ) -> STMatchOddsModel | None:
        _odds = draw_event.get("startOdds")
        odds_1 = parse_swedish_decimal(_odds['one'])
        odds_X = parse_swedish_decimal(_odds['x'])
        odds_2 = parse_swedish_decimal(_odds['two'])

        if (
            odds_1 is None
            or odds_X is None
            or odds_2 is None
        ):
            return None

        match_odds = self.get_by_stryktipset_match_id(match.id)
        if match_odds is None:
            match_odds = self.create(stryktipset_match_id=match.id)

        match_odds.odds_1 = odds_1
        match_odds.odds_X = odds_X
        match_odds.odds_2 = odds_2
        return match_odds
