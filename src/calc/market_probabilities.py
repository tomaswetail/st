from __future__ import annotations

import math

from sqlalchemy.orm import Session

from src.objects.repositories.fixture_odds_repository import FixtureOddsRepository
from src.objects.schema.data_classes.data_sources import DataSourceConfig
from src.objects.schema.data_classes.market_probability import MarketProbabilityBreakdown
from src.objects.schema.db.st_match_odds import STMatchOdds
from src.utils.common import validate_decimal_odds

FIXTURE_ODDS_FALLBACK_BOOKMAKER = "Avg"
_ENTROPY_PROBABILITY_CLIP = 1e-15


class MarketProbabilities:

    def __init__(self, match_odds: STMatchOdds):
        self.match_odds = match_odds

    def get_probs(self):
        return self.describe().as_probs()

    def describe(self) -> MarketProbabilityBreakdown:
        return self.from_decimal_odds(
            self.match_odds.odds_1,
            self.match_odds.odds_X,
            self.match_odds.odds_2,
        )

    @classmethod
    def from_decimal_odds(
        cls,
        odds_home: object,
        odds_draw: object,
        odds_away: object,
        *,
        bookmaker: str | None = None,
        price_type: str | None = None,
    ) -> MarketProbabilityBreakdown:
        """Convert a decimal 1X2 triple into implied, vig-free, and shape features.

        ``q_i = 1 / odds_i``, ``overround = sum(q)``, ``p_i = q_i / overround``.
        Overround < 1 is valid and flagged via ``overround_below_one``; the match
        is not dropped. Entropy is ``-sum(p_i * ln(p_i))`` (natural log); a
        probability of 0 is clipped to 1e-15 for entropy only.
        """
        home = validate_decimal_odds("home", odds_home)
        draw = validate_decimal_odds("draw", odds_draw)
        away = validate_decimal_odds("away", odds_away)

        implied_home = 1.0 / home
        implied_draw = 1.0 / draw
        implied_away = 1.0 / away
        overround = implied_home + implied_draw + implied_away
        p_home = implied_home / overround
        p_draw = implied_draw / overround
        p_away = implied_away / overround

        ranked = sorted((p_home, p_draw, p_away), reverse=True)
        top_probability = ranked[0]
        second_probability = ranked[1]
        entropy = -sum(
            _entropy_term(probability) for probability in (p_home, p_draw, p_away)
        )

        return MarketProbabilityBreakdown(
            odds_home=home,
            odds_draw=draw,
            odds_away=away,
            implied_home=implied_home,
            implied_draw=implied_draw,
            implied_away=implied_away,
            p_home=p_home,
            p_draw=p_draw,
            p_away=p_away,
            overround=overround,
            overround_below_one=overround < 1.0,
            market_top_probability=top_probability,
            market_second_probability=second_probability,
            market_probability_gap=top_probability - second_probability,
            market_entropy=entropy,
            bookmaker=bookmaker,
            price_type=price_type,
        )


def _entropy_term(probability: float) -> float:
    clipped = probability if probability > 0 else _ENTROPY_PROBABILITY_CLIP
    return clipped * math.log(clipped)


def load_fixture_market_probabilities(
    session: Session,
    fixture_id: int,
    *,
    config: DataSourceConfig | None = None,
) -> MarketProbabilityBreakdown | None:
    """Load vig-free market probabilities from a stored ``fixture_odds`` row.

    Selects the configured ``(bookmaker, price_type)`` for ``fixtures.id``
    (defaults: bookmaker ``Avg``, price_type ``closing``). This is choosing a
    stored bookmaker row, not re-averaging 1/odds across books.

    Fallback: if the preferred row is missing and the preferred bookmaker is
    not ``Avg``, load ``Avg`` at the **same** ``price_type``. If preferred is
    already ``Avg`` and missing, or fallback ``Avg`` is also missing, return
    ``None``. Never fall back across ``opening`` vs ``closing``. Invalid odds
    on a found row raise; do not skip to another book.
    """
    resolved_config = config or DataSourceConfig()
    repository = FixtureOddsRepository(session)
    preferred_bookmaker = resolved_config.fixture_odds_bookmaker
    price_type = resolved_config.fixture_odds_price_type
    provider = resolved_config.fixture_odds_provider

    row = repository.get_for_fixture(
        fixture_id,
        bookmaker=preferred_bookmaker,
        price_type=price_type,
        provider=provider,
    )
    if row is None and preferred_bookmaker != FIXTURE_ODDS_FALLBACK_BOOKMAKER:
        row = repository.get_for_fixture(
            fixture_id,
            bookmaker=FIXTURE_ODDS_FALLBACK_BOOKMAKER,
            price_type=price_type,
            provider=provider,
        )
    if row is None:
        return None
    return MarketProbabilities.from_decimal_odds(
        row.odds_home,
        row.odds_draw,
        row.odds_away,
        bookmaker=row.bookmaker,
        price_type=row.price_type,
    )
