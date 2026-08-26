from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from data_sources.entity_resolver import EntityResolver
from data_sources.svenskaspel_api_client import SvenskaSpelClient
from objects.repositories.league_repository import LeagueRepository
from objects.repositories.st_match_bet_repository import STMatchBetRepository
from objects.repositories.st_match_odds_repository import STMatchOddsRepository
from objects.repositories.st_match_repository import STMatchRepository
from objects.repositories.st_round_repository import STRoundRepository
from objects.repositories.team_repository import TeamRepository
from objects.schema.db.st_round import STRound




def _participant_by_type(
    participants: list[dict[str, Any]], role: str
) -> dict[str, Any]:
    for participant in participants:
        if participant.get("type") == role:
            return participant
    raise ValueError(f"Missing participant with type={role!r}")


class STDrawManager:
    """Fetch Stryktipset draws and upsert Team, Match, and Round records."""

    def __init__(
        self,
        session: Session,
    ) -> None:
        self.client = SvenskaSpelClient()
        self.teams_repo = TeamRepository(session)
        self.leagues_repo = LeagueRepository(session)
        self.rounds_repo = STRoundRepository(session)
        self.matches_repo = STMatchRepository(session)
        self.match_bets_repo = STMatchBetRepository(session)
        self.match_odds_repo = STMatchOddsRepository(session)
        self.entity_resolver = EntityResolver(session, provider="svenska-spel")
        self.team_mappings = {}

    @staticmethod
    def _team_from_resolution(resolution) -> Any | None:
        if resolution.team is None or resolution.method == "unresolved":
            return None
        return resolution.team

    def import_all_draws_and_name_check(self):
        missed_teams = []#4760
        for draw_number in range(4760, 4959):
            print(f"***************************************{draw_number}********************************")
            payload = self.client.fetch_draw(draw_number)
            draw = payload["draw"]

            draw_num = draw["drawNumber"]

            for draw_event in draw.get("drawEvents") or []:
                match_data = draw_event.get("match") or {}
                participants = match_data.get("participants") or []
                home_participant = _participant_by_type(participants, "home")
                away_participant = _participant_by_type(participants, "away")

                if away_participant['name'] != 'Skottland' and away_participant['name'] != 'Skottland':
                    continue

                home_resolved = self.entity_resolver.resolve_team(provider_team_id=home_participant['id'],
                                                                  provider_team_name=home_participant['name'])

                if home_resolved.method == 'unresolved':
                    missed_teams.append(home_participant['name'])
                if not home_resolved.team:
                    missed_teams.append(away_participant['name'])
                    continue

                away_resolved = self.entity_resolver.resolve_team(provider_team_id=away_participant['id'],
                                                                  provider_team_name=away_participant['name'])
                if away_resolved.method == 'unresolved':
                    missed_teams.append(away_participant['name'])
                if not away_resolved.team:
                    missed_teams.append(away_participant['name'])
                    continue
                self.team_mappings[away_participant['name']] = away_resolved.team.name
                self.team_mappings[home_participant['name']] = home_resolved.team.name
        print(self.team_mappings)
        return list(set(missed_teams))



    def import_all_draws(self):

        for draw_number in range(4760, 4959):
            self.import_draw(draw_number)

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
        missed_teams = []
        for draw_event in draw.get("drawEvents") or []:
            match_data = draw_event.get("match") or {}
            participants = match_data.get("participants") or []
            home_participant = _participant_by_type(participants, "home")
            away_participant = _participant_by_type(participants, "away")

            home_resolved = self.entity_resolver.resolve_team(
                provider_team_id=home_participant["id"],
                provider_team_name=home_participant["name"],
            )
            home_team = self._team_from_resolution(home_resolved)
            if home_team is None:
                missed_teams.append(home_participant["name"])

            away_resolved = self.entity_resolver.resolve_team(
                provider_team_id=away_participant["id"],
                provider_team_name=away_participant["name"],
            )
            away_team = self._team_from_resolution(away_resolved)
            if away_team is None:
                missed_teams.append(away_participant["name"])

            if home_team is None or away_team is None:
                continue

            match = self.matches_repo.upsert_from_draw(
                match_data,
                round_model=round_model,
                home_team=home_team,
                away_participant=away_participant,
                home_participant=home_participant,
                away_team=away_team,
            )
            self.matches_repo.flush()
            _odds = draw_event.get("startOdds")
            self.match_bets_repo.upsert_from_draw_event(match, draw_event)
            self.match_odds_repo.upsert_from_draw_event(match, draw_event)
            imported_rounds.append(self.rounds_repo.to_schema(round_model))

        self.rounds_repo.commit()
        return imported_rounds
