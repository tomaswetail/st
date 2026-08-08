from objects.schema.db.st_match_bet import STMatchBet


class MarketProbabilities:

    def __init__(self, match_bet: STMatchBet):
        self.match_bet = match_bet

    def get_probs(self):
        return {
            '1': self.match_bet.distribution_1,
            'X': self.match_bet.distribution_X,
            '2': self.match_bet.distribution_2
        }

