from typing import Any

from sqlalchemy import select

from objects.models.team import TeamModel

from utils.team_name_matcher import to_football_data_name

from objects.repositories.base import BaseRepository


class TeamRepository(BaseRepository[TeamModel]):
    model = TeamModel

    def get_by_external_id(self, external_id: int) -> TeamModel | None:
        return self.session.scalar(
            select(self.model).where(self.model.external_id == external_id)
        )

    def get_by_name_and_league(
        self, name: str, league_id: int
    ) -> TeamModel | None:
        return self.session.scalar(
            select(self.model).where(
                self.model.name == name,
                self.model.league_id == league_id,
            )
        )

    def get_by_likely_name(
        self, name: str
    ) -> TeamModel | None:
        return self.session.scalar(
            select(self.model).where(
                self.model.name.ilike(f'%{name}%')
            )
        )


    def get_by_name(
        self, name: str
    ) -> TeamModel | None:
        return self.session.scalar(
            select(self.model).where(
                self.model.name == name
            )
        )

    def get_all_names(self) -> list[str]:
        return list(
            self.session.scalars(
                select(self.model.name).distinct().order_by(self.model.name)
            ).all()
        )

    def to_football_data_name(self, svenska_spel_name: str) -> str | None:
        if not hasattr(self, "_football_data_names"):
            from objects.repositories.historical_match_repository import HistoricalMatchRepository

            historical_repo = HistoricalMatchRepository(self.session)
            self._football_data_names = historical_repo.get_distinct_home_teams()
        return to_football_data_name(svenska_spel_name, self._football_data_names)

    def team_name_wide_search(self, team_name):
        if not team_name:
            return
        if len(team_name.split(' ')) > 1:
            team_name = team_name.split(' ')[0]
            team = self.get_by_likely_name(team_name)
            return team
        return self.get_by_likely_name(team_name)


    def ensure_from_historical(self, name: str, league_id: int) -> TeamModel:
        team = self.get_by_name_and_league(name, league_id)
        if team is None:
            team = self.get_by_likely_name(name)
        if team is None:
            if len(name.split(' ')) > 1:
                name = name.split(' ')[0]
                team = self.get_by_likely_name(name)
        if team is None:
            team = self.create(name=name, league_id=league_id)
        return team

    def upsert_from_participant(self, participant: dict[str, Any]) -> TeamModel:
        external_id = participant["id"]
        team = self.get_by_external_id(external_id)
        if team is None:
            team = self.create(external_id=external_id)

        team.name = participant["name"]
        team.short_name = participant.get("shortName")
        team.medium_name = participant.get("mediumName")
        team.country_name = participant.get("countryName")
        team.iso_code = participant.get("isoCode")
        if participant.get("league_id") is not None:
            team.league_id = participant["league_id"]
        return team
