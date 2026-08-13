from sqlalchemy.orm import Session

from calc.market_probabilities import MarketProbabilities
from calc.strength_calculator import StrengthCalculator
from objects.repositories.st_match_repository import STMatchRepository
from objects.repositories.st_round_repository import STRoundRepository
from objects.repositories.st_match_odds_repository import STMatchOddsRepository


class ProbabilityManager:

    def __init__(
        self,
        session: Session,
    ) -> None:
        self.session = session
        self.rounds_repo = STRoundRepository(session)
        self.matches_repo = STMatchRepository(session)
        self.match_odds_repo = STMatchOddsRepository(session)

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
            base_probs = MarketProbabilities(match.match_odds).get_probs()
            features = StrengthCalculator(self.session).get_match_features(match_id)

            result = home_advantage_calculator.process(team, current_date)

            coefficient = result.home_advantage


