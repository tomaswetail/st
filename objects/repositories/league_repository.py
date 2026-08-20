import logging


from sqlalchemy import select


from objects.models.league import LeagueModel
from objects.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class LeagueRepository(BaseRepository[LeagueModel]):
    model = LeagueModel

    def get_by_external_id(self, external_id: int) -> LeagueModel | None:
        return self.session.scalar(
            select(self.model).where(self.model.external_id == external_id)
        )

    def get_by_name(self, name: str) -> LeagueModel | None:
        return self.session.scalar(
            select(self.model).where(self.model.league_name == name)
        )

    def get_by_name_and_country(
        self, name: str, country: str
    ) -> LeagueModel | None:
        return self.session.scalar(
            select(self.model).where(
                self.model.league_name == name,
                self.model.country_name == country,
            )
        )

    def get_by_likely_name_and_country(
        self, name: str, country: str
    ) -> LeagueModel | None:
        matches = self.session.scalars(
            select(self.model).where(
                self.model.league_name.ilike(f"%{name}%"),
                self.model.country_name == country,
            )
        ).all()
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return matches[0]
        return None

    def upsert_from_api(
        self,
        *,
        external_id: int,
        league_name: str,
        league_type: str,
        country_name: str,
        country_code: str | None = None,
    ) -> LeagueModel:
        league = self.get_by_external_id(external_id)
        if league is None:
            league = self.create(
                external_id=external_id,
                league_name=league_name,
                league_type=league_type,
                country_name=country_name,
                country_code=country_code,
            )
        else:
            league.league_name = league_name
            league.league_type = league_type
            league.country_name = country_name
            league.country_code = country_code
        return league
