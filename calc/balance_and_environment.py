"""Match balance and expected-goal environment feature calculator."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import Any

from sqlalchemy.orm import Session

from calc.strength_calculator import StrengthCalculator
from objects.models.historical_match import HistoricalMatchModel
from objects.models.st_match import STMatchModel
from objects.schema.data_classes.balance_and_environment_features import (
    BalanceAndEnvironmentFeatures,
)
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.db.historical_match import HistoricalMatch
from utils.common import ensure_unit_probabilities


def _match_side_name(side: Any) -> str:
    """Return team name from a pydantic string or ORM TeamModel relationship."""
    if side is None:
        return ""
    if isinstance(side, str):
        return side
    return str(side.name)


class BalanceAndEnvironment:
    """Derive draw-oriented balance and environment features from strength + history."""

    def __init__(
        self,
        session: Session,
        config: DataSourceConfig | None = None,
        strength_calculator: StrengthCalculator | None = None,
    ) -> None:
        self.session = session
        self.config = config or DataSourceConfig()
        self.strength_calculator = strength_calculator or StrengthCalculator(
            session=session,
            config=self.config,
        )

    def calculate(
        self,
        match: STMatchModel,
        historical_matches: Sequence[HistoricalMatch | HistoricalMatchModel],
        market_probabilities: dict[str, float | None],
    ) -> BalanceAndEnvironmentFeatures:
        """Compute balance/environment features for one ST fixture."""
        if match.home_team is None or match.away_team is None:
            raise ValueError(f"Missing team on match id={match.id}")
        if match.start_time is None:
            raise ValueError(f"Missing start_time on match id={match.id}")

        cutoff = (
            match.start_time.date()
            if isinstance(match.start_time, datetime)
            else match.start_time
        )
        strength = self.strength_calculator.get_fixture_features(
            match.home_team_id,
            match.away_team_id,
            match.start_time,
            match_id=match.id,
        )

        home_attack_strength = strength.home_attack_strength
        away_attack_strength = strength.away_attack_strength
        home_defence = strength.home_defence_strength
        away_defence = strength.away_defence_strength
        home_xg = strength.expected_home_goals
        away_xg = strength.expected_away_goals

        unit_market_probabilities = ensure_unit_probabilities(market_probabilities)
        p_home = unit_market_probabilities.get("1")
        p_away = unit_market_probabilities.get("2")

        home_rates = self._team_recent_rates(
            historical_matches,
            team_name=match.home_team.name,
            before_date=cutoff,
        )
        away_rates = self._team_recent_rates(
            historical_matches,
            team_name=match.away_team.name,
            before_date=cutoff,
        )

        return BalanceAndEnvironmentFeatures(
            attack_strength_difference=self._abs_diff(
                home_attack_strength, away_attack_strength
            ),
            expected_goal_difference=self._abs_diff(home_xg, away_xg),
            expected_goal_total=self._sum_or_none(home_xg, away_xg),
            market_balance=self._abs_diff(p_home, p_away),
            defence_strength_difference=self._abs_diff(home_defence, away_defence),
            favourite_strength=self._favourite_strength(p_home, p_away),
            home_recent_draw_rate=home_rates["draw_rate"],
            away_recent_draw_rate=away_rates["draw_rate"],
            home_one_goal_match_rate=home_rates["one_goal_rate"],
            away_one_goal_match_rate=away_rates["one_goal_rate"],
            home_close_match_rate=home_rates["close_rate"],
            away_close_match_rate=away_rates["close_rate"],
            home_low_scoring_rate=home_rates["low_scoring_rate"],
            away_low_scoring_rate=away_rates["low_scoring_rate"],
            combined_draw_rate=self._combined_mean(
                home_rates["draw_rate"], away_rates["draw_rate"]
            ),
            combined_one_goal_match_rate=self._combined_mean(
                home_rates["one_goal_rate"], away_rates["one_goal_rate"]
            ),
            combined_close_match_rate=self._combined_mean(
                home_rates["close_rate"], away_rates["close_rate"]
            ),
            combined_low_scoring_rate=self._combined_mean(
                home_rates["low_scoring_rate"], away_rates["low_scoring_rate"]
            ),
            home_recent_sample_size=home_rates["sample_size"],
            away_recent_sample_size=away_rates["sample_size"],
        )

    def _team_recent_rates(
        self,
        historical_matches: Sequence[HistoricalMatch | HistoricalMatchModel],
        *,
        team_name: str,
        before_date: date,
    ) -> dict[str, Any]:
        recent = self._recent_team_matches(
            historical_matches, team_name=team_name, before_date=before_date
        )
        sample_size = len(recent)
        if sample_size == 0:
            return {
                "draw_rate": None,
                "one_goal_rate": None,
                "close_rate": None,
                "low_scoring_rate": None,
                "sample_size": 0,
            }

        draw_count = 0
        one_goal_count = 0
        close_count = 0
        low_scoring_count = 0
        low_scoring_threshold = self.config.balance_low_scoring_goal_threshold

        for historical_match in recent:
            goals_for, goals_against = self._goals_for_team(
                historical_match, team_name
            )
            goal_diff = abs(goals_for - goals_against)
            total_goals = goals_for + goals_against

            if self._is_draw(historical_match):
                draw_count += 1
            if goal_diff == 1:
                one_goal_count += 1
            if goal_diff <= 1:
                close_count += 1
            if total_goals <= low_scoring_threshold:
                low_scoring_count += 1

        return {
            "draw_rate": draw_count / sample_size,
            "one_goal_rate": one_goal_count / sample_size,
            "close_rate": close_count / sample_size,
            "low_scoring_rate": low_scoring_count / sample_size,
            "sample_size": sample_size,
        }

    def _recent_team_matches(
        self,
        historical_matches: Sequence[HistoricalMatch | HistoricalMatchModel],
        *,
        team_name: str,
        before_date: date,
    ) -> list[HistoricalMatch | HistoricalMatchModel]:
        relevant = [
            historical_match
            for historical_match in historical_matches
            if historical_match.match_date < before_date
            and (
                _match_side_name(historical_match.home_team) == team_name
                or _match_side_name(historical_match.away_team) == team_name
            )
        ]
        relevant.sort(key=lambda row: row.match_date, reverse=True)
        return relevant[: self.config.balance_recent_matches]

    @staticmethod
    def _goals_for_team(
        historical_match: HistoricalMatch | HistoricalMatchModel,
        team_name: str,
    ) -> tuple[int, int]:
        if _match_side_name(historical_match.home_team) == team_name:
            return historical_match.home_goals, historical_match.away_goals
        return historical_match.away_goals, historical_match.home_goals

    @staticmethod
    def _is_draw(historical_match: HistoricalMatch | HistoricalMatchModel) -> bool:
        result = str(historical_match.result).upper()
        if result in {"X", "D"}:
            return True
        return historical_match.home_goals == historical_match.away_goals

    @staticmethod
    def _abs_diff(left: float | None, right: float | None) -> float | None:
        if left is None or right is None:
            return None
        return abs(left - right)

    @staticmethod
    def _sum_or_none(left: float | None, right: float | None) -> float | None:
        if left is None or right is None:
            return None
        return left + right

    @staticmethod
    def _favourite_strength(
        p_home: float | None, p_away: float | None
    ) -> float | None:
        if p_home is None or p_away is None:
            return None
        return max(p_home, p_away)

    @staticmethod
    def _combined_mean(
        home_rate: float | None, away_rate: float | None
    ) -> float | None:
        if home_rate is None and away_rate is None:
            return None
        if home_rate is None:
            return away_rate
        if away_rate is None:
            return home_rate
        return (home_rate + away_rate) / 2.0
