"""Load Stryktipset coupon matches into optimizer structs.

Supports in-memory fixtures (unit tests) and DB load via ST repositories.

Leakage UNKNOWN: odds typically come from startOdds on import with no
timestamp; public shares on STMatchBetModel have no timestamp;
regCloseTime is in the API but not on STRoundModel.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

from sqlalchemy.orm import Session, selectinload

from src.calc.stryktipset_optimizer.fair_probs import (
    InvalidOddsError,
    fair_probabilities_from_odds,
)
from src.calc.stryktipset_optimizer.public_probs import (
    InvalidPublicShareError,
    normalize_public_shares,
)
from src.objects.models.st_match import STMatchModel
from src.objects.models.st_round import STRoundModel
from src.objects.repositories.st_round_repository import STRoundRepository
from src.utils.common import OUTCOMES, Outcome


LEAKAGE_LIMITATIONS = (
    "UNKNOWN temporal leakage: stored odds have no timestamp (typically "
    "startOdds on import); public streckprocent has no timestamp "
    "(STMatchBetModel ints, usually 0–100); regCloseTime exists in the "
    "Svenska Spel API but is not persisted on STRoundModel. Treat the "
    "stored snapshot as operable coupon-time data; do not claim "
    "closing-line safety."
)


class CouponDataError(ValueError):
    """Raised when coupon matches cannot be loaded or validated."""


@dataclass(frozen=True)
class CouponMatchInput:
    """In-memory ST Match line for optimization (not a Fixture)."""

    odds_1: float
    odds_x: float
    odds_2: float
    public_1: float | int
    public_x: float | int
    public_2: float | int
    match_index: int | None = None
    external_id: int | None = None
    label: str | None = None
    start_time: datetime | None = None
    result: Outcome | None = None


@dataclass(frozen=True)
class PreparedMatch:
    """Validated match with Pm and Pp ready for scoring."""

    match_index: int
    market_probs: dict[Outcome, float]
    public_probs: dict[Outcome, float]
    label: str | None
    external_id: int | None
    start_time: datetime | None
    result: Outcome | None


@dataclass(frozen=True)
class PreparedCoupon:
    """Validated full coupon (typically 13 ST Matches)."""

    draw_number: int | None
    matches: tuple[PreparedMatch, ...]
    limitations: str = LEAKAGE_LIMITATIONS


def prepare_matches(
    match_inputs: Sequence[CouponMatchInput],
    *,
    public_epsilon: float = 1e-6,
    expected_count: int | None = 13,
    draw_number: int | None = None,
) -> PreparedCoupon:
    """Validate and convert raw match inputs to Pm/Pp structs."""
    if expected_count is not None and len(match_inputs) != expected_count:
        raise CouponDataError(
            f"expected {expected_count} matches, got {len(match_inputs)}"
        )
    if not match_inputs:
        raise CouponDataError("coupon has no matches")

    prepared: list[PreparedMatch] = []
    for position, raw in enumerate(match_inputs):
        match_index = raw.match_index if raw.match_index is not None else position
        try:
            pm = fair_probabilities_from_odds(raw.odds_1, raw.odds_x, raw.odds_2)
        except InvalidOddsError as exc:
            raise CouponDataError(
                f"match index {match_index}: invalid odds ({exc})"
            ) from exc
        try:
            pp = normalize_public_shares(
                raw.public_1,
                raw.public_x,
                raw.public_2,
                public_epsilon=public_epsilon,
            )
        except InvalidPublicShareError as exc:
            raise CouponDataError(
                f"match index {match_index}: invalid public shares ({exc})"
            ) from exc

        result = raw.result
        if result is not None and result not in OUTCOMES:
            raise CouponDataError(
                f"match index {match_index}: invalid result {result!r}"
            )

        prepared.append(
            PreparedMatch(
                match_index=match_index,
                market_probs=pm,
                public_probs=pp,
                label=raw.label,
                external_id=raw.external_id,
                start_time=raw.start_time,
                result=result,
            )
        )

    return PreparedCoupon(
        draw_number=draw_number,
        matches=tuple(prepared),
        limitations=LEAKAGE_LIMITATIONS,
    )


def _label_from_st_match(match: STMatchModel) -> str:
    home = getattr(match.home_team, "name", None) or "?"
    away = getattr(match.away_team, "name", None) or "?"
    return f"{home}–{away}"


def coupon_inputs_from_st_matches(
    matches: Sequence[STMatchModel],
) -> list[CouponMatchInput]:
    """Map ORM STMatchModel rows (with odds+bet) to CouponMatchInput."""
    inputs: list[CouponMatchInput] = []
    ordered = sorted(
        matches,
        key=lambda m: (
            m.start_time is None,
            m.start_time or datetime.min,
            m.external_id or 0,
            m.id or 0,
        ),
    )
    for position, match in enumerate(ordered):
        odds = match.match_odds
        bet = match.match_bet
        if odds is None:
            raise CouponDataError(
                f"ST Match external_id={match.external_id} missing match_odds"
            )
        if bet is None:
            raise CouponDataError(
                f"ST Match external_id={match.external_id} missing match_bet"
            )
        result: Outcome | None = None
        if match.stryktipset_result in OUTCOMES:
            result = match.stryktipset_result  # type: ignore[assignment]
        inputs.append(
            CouponMatchInput(
                odds_1=float(odds.odds_1),
                odds_x=float(odds.odds_X),
                odds_2=float(odds.odds_2),
                public_1=bet.distribution_1,
                public_x=bet.distribution_X,
                public_2=bet.distribution_2,
                match_index=position,
                external_id=match.external_id,
                label=_label_from_st_match(match),
                start_time=match.start_time,
                result=result,
            )
        )
    return inputs


def load_coupon_from_session(
    session: Session,
    draw_number: int,
    *,
    public_epsilon: float = 1e-6,
    expected_count: int | None = 13,
) -> PreparedCoupon:
    """Load a draw's ST Matches with odds+bet via repositories."""
    from sqlalchemy import select

    rounds_repo = STRoundRepository(session)
    round_model = rounds_repo.get_by_draw_number(draw_number)
    if round_model is None:
        raise CouponDataError(f"no STRound for draw_number={draw_number}")

    stmt = (
        select(STMatchModel)
        .where(STMatchModel.stryktipset_round_id == round_model.id)
        .options(
            selectinload(STMatchModel.match_odds),
            selectinload(STMatchModel.match_bet),
            selectinload(STMatchModel.home_team),
            selectinload(STMatchModel.away_team),
        )
    )
    matches = list(session.scalars(stmt).all())
    if not matches:
        raise CouponDataError(
            f"draw_number={draw_number} has no ST Matches"
        )
    inputs = coupon_inputs_from_st_matches(matches)
    return prepare_matches(
        inputs,
        public_epsilon=public_epsilon,
        expected_count=expected_count,
        draw_number=draw_number,
    )


def load_rounds_for_backtest(
    session: Session,
    *,
    min_draw_number: int | None = None,
    max_draw_number: int | None = None,
    expected_count: int = 13,
    public_epsilon: float = 1e-6,
) -> list[PreparedCoupon]:
    """Load complete settled coupons ordered for chronological OOS.

    Ordering key: min(start_time) of matches, else draw_number.
    Incomplete rounds (≠ expected_count, missing odds/bet/result) are skipped.
    """
    from sqlalchemy import select

    stmt = select(STRoundModel).options(
        selectinload(STRoundModel.matches)
        .selectinload(STMatchModel.match_odds),
        selectinload(STRoundModel.matches)
        .selectinload(STMatchModel.match_bet),
        selectinload(STRoundModel.matches)
        .selectinload(STMatchModel.home_team),
        selectinload(STRoundModel.matches)
        .selectinload(STMatchModel.away_team),
    )
    if min_draw_number is not None:
        stmt = stmt.where(STRoundModel.draw_number >= min_draw_number)
    if max_draw_number is not None:
        stmt = stmt.where(STRoundModel.draw_number <= max_draw_number)

    rounds = list(session.scalars(stmt).all())
    prepared_list: list[tuple[Any, PreparedCoupon]] = []

    for round_model in rounds:
        try:
            inputs = coupon_inputs_from_st_matches(round_model.matches)
            coupon = prepare_matches(
                inputs,
                public_epsilon=public_epsilon,
                expected_count=expected_count,
                draw_number=round_model.draw_number,
            )
        except CouponDataError:
            continue
        if any(match.result is None for match in coupon.matches):
            continue
        start_times = [
            match.start_time
            for match in coupon.matches
            if match.start_time is not None
        ]
        sort_key: Any
        if start_times:
            sort_key = (0, min(start_times), round_model.draw_number)
        else:
            sort_key = (1, round_model.draw_number)
        prepared_list.append((sort_key, coupon))

    prepared_list.sort(key=lambda item: item[0])
    return [coupon for _, coupon in prepared_list]
