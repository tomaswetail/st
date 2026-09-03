from objects.schema.db.st_match_odds import STMatchOdds
from utils.common import odds_to_probabilities


class MarketProbabilities:

    def __init__(self, match_odds: STMatchOdds):
        self.match_odds = match_odds

    def get_probs(self):
        return odds_to_probabilities(self.match_odds.odds_1, self.match_odds.odds_X, self.match_odds.odds_2)
