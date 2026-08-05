from typing import Any

from sqlalchemy import select

from objects.models.st_match import STMatchModel
from objects.models.st_round import STRoundModel
from objects.models.team import TeamModel

from objects.repositories.base import BaseRepository
from objects.repositories.utils import (
    parse_datetime,
    result_by_type,
    stryktipset_result,
    to_int,
)

RESULT_TYPE_HALFTIME = 1
RESULT_TYPE_FULLTIME = 2


class STMatchRepository(BaseRepository[STMatchModel]):
    model = STMatchModel

    def get_by_external_id(self, external_id: int) -> STMatchModel | None:
        return self.session.scalar(
            select(self.model).where(self.model.external_id == external_id)
        )

    def get_by_stryktipset_round_id(
        self, stryktipset_round_id: int
    ) -> list[STMatchModel]:
        return list(
            self.session.scalars(
                select(self.model).where(
                    self.model.stryktipset_round_id == stryktipset_round_id
                )
            ).all()
        )

    def upsert_from_draw(
        self,
        match_data: dict[str, Any],
        *,
        round_model: STRoundModel,
        home_team: TeamModel,
        away_team: TeamModel,
        home_participant: dict[str, Any],
        away_participant: dict[str, Any],
    ) -> STMatchModel:
        external_id = match_data["matchId"]
        match = self.get_by_external_id(external_id)
        if match is None:
            match = self.create(external_id=external_id)

        league = match_data.get("league") or {}
        league_country = league.get("country") or {}
        results = match_data.get("result")

        fulltime = result_by_type(results, RESULT_TYPE_FULLTIME)
        halftime = result_by_type(results, RESULT_TYPE_HALFTIME)

        match.stryktipset_round_id = round_model.draw_number
        match.start_time = parse_datetime(match_data.get("matchStart"))
        match.status = match_data.get("status")
        match.status_id = match_data.get("statusId")
        match.league_name = league.get("name")
        match.league_country_name = league_country.get("name")
        match.home_team_id = home_team.id
        match.away_team_id = away_team.id

        if fulltime is not None:
            match.home_score = to_int(fulltime.get("home"))
            match.away_score = to_int(fulltime.get("away"))
        else:
            match.home_score = to_int(home_participant.get("result"))
            match.away_score = to_int(away_participant.get("result"))

        if halftime is not None:
            match.halftime_home_score = to_int(halftime.get("home"))
            match.halftime_away_score = to_int(halftime.get("away"))

        match.stryktipset_result = stryktipset_result(
            match.home_score,
            match.away_score,
        )

        return match
