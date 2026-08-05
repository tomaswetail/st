from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from data_sources.svenskaspel_api_client import SvenskaSpelClient
from objects.repositories.league_repository import LeagueRepository
from objects.repositories.st_match_bet_repository import STMatchBetRepository
from objects.repositories.st_match_repository import STMatchRepository
from objects.repositories.st_round_repository import STRoundRepository
from objects.repositories.team_repository import TeamRepository
from objects.schema.db.st_round import STRound

from utils.common import LEAGUES


def _participant_by_type(
    participants: list[dict[str, Any]], role: str
) -> dict[str, Any]:
    for participant in participants:
        if participant.get("type") == role:
            return participant
    raise ValueError(f"Missing participant with type={role!r}")

def _league_data(match_data):

    league_data = {"name": match_data['league']['name']}
    country = match_data['league'].get('country')
    if country:
        league_data["country"] = country['isoCode']
    return league_data


class STDrawManager:
    """Fetch Stryktipset draws and upsert Team, Match, and Round records."""

    def __init__(
        self,
        session: Session,
        client: SvenskaSpelClient | None = None,
    ) -> None:
        self.client = client or SvenskaSpelClient()
        self.teams_repo = TeamRepository(session)
        self.leagues_repo = LeagueRepository(session)
        self.rounds_repo = STRoundRepository(session)
        self.matches_repo = STMatchRepository(session)
        self.match_bets_repo = STMatchBetRepository(session)

    def import_draw(self, draw_number: int) -> list[STRound]:
        payload = self.client.fetch_draw(draw_number)
        draw = payload["draw"]

        product_id = draw["productId"]
        product_name = draw["productName"]
        draw_num = draw["drawNumber"]

        imported_rounds: list[STRound] = []

        round_model = self.rounds_repo.upsert(
            product_id=product_id,
            product_name=product_name,
            draw_number=draw_num,
        )
        self.rounds_repo.flush()

        for draw_event in draw.get("drawEvents") or []:
            match_data = draw_event.get("match") or {}
            participants = match_data.get("participants") or []

            league_data = _league_data(match_data)
            league = None
            if league_data['name'] in LEAGUES:

                league = self.leagues_repo.upsert_from_match(league_data)
                self.leagues_repo.flush()

            home_participant = _participant_by_type(participants, "home")
            away_participant = _participant_by_type(participants, "away")

            if league:
                home_participant["league_id"] = league.id
                away_participant["league_id"] = league.id

            home_team = self.teams_repo.upsert_from_participant(home_participant)
            away_team = self.teams_repo.upsert_from_participant(away_participant)
            self.teams_repo.flush()

            match = self.matches_repo.upsert_from_draw(
                match_data,
                round_model=round_model,
                home_team=home_team,
                away_participant=away_participant,
                home_participant=home_participant,
                away_team=away_team,
            )
            self.matches_repo.flush()
            self.match_bets_repo.upsert_from_draw_event(match, draw_event)
            imported_rounds.append(self.rounds_repo.to_schema(round_model))

        self.rounds_repo.commit()
        return imported_rounds
