from sqlalchemy.orm import Session

from objects.repositories.st_match_repository import STMatchRepository
from objects.repositories.st_round_repository import STRoundRepository


class ProbabilityManager:

    def __init__(
        self,
        session: Session,
    ) -> None:
        self.rounds_repo = STRoundRepository(session)
        self.matches_repo = STMatchRepository(session)
    def process(self, draw_number: int):
        round = self.rounds_repo.get_by_draw_number(draw_number)
        if not round:
            raise ValueError(f"No Stryktipset round found for draw_number={draw_number}")

        matches = self.matches_repo.get_by_stryktipset_round_id(draw_number)
        if not matches:
            raise ValueError(f"No matches found for draw_number={draw_number}")

        for match in matches:
            if match.home_team is None or match.away_team is None:
                raise ValueError(f"Missing team on match id={match.id}")
            if match.start_time is None:
                raise ValueError(f"Missing start_time on match id={match.id}")
