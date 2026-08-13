"""League-level behaviour features from historical matches (no future leakage)."""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Sequence

from sqlalchemy.orm import Session

from calc.strength_helpers import shrink
from objects.models.historical_match import HistoricalMatchModel
from objects.models.st_match import STMatchModel
from objects.repositories.historical_match_repository import HistoricalMatchRepository
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.data_classes.league_behavior_features import LeagueBehaviorFeatures
from utils.common import odds_to_probabilities


@dataclass(frozen=True)
class _RawLeagueStats:
    sample_size: int
    draw_rate: float | None
    home_win_rate: float | None
    away_win_rate: float | None
    avg_goals: float | None
    goal_std: float | None
    favourite_win_rate: float | None
    market_sample_size: int
    competitive_balance: float | None
    result_completeness: float
    market_completeness: float
    balance_completeness: float


class LeagueBehaviorCalculator:
    """Derive league context features for residual ML from prior matches only."""

    def __init__(
        self,
        session: Session,
        config: DataSourceConfig | None = None,
    ) -> None:
        self.session = session
        self.config = config or DataSourceConfig()
        self.historical_repo = HistoricalMatchRepository(session)
        self._features_cache: dict[tuple[int, date], LeagueBehaviorFeatures] = {}
        self._global_stats_cache: dict[date, _RawLeagueStats] = {}

    def calculate(self, match: STMatchModel) -> LeagueBehaviorFeatures:
        """Compute league behaviour features for one ST fixture."""
        if match.home_team is None:
            raise ValueError(f"Missing home_team on match id={match.id}")
        if match.start_time is None:
            raise ValueError(f"Missing start_time on match id={match.id}")
        if match.home_team.league_id is None:
            raise ValueError(f"Missing league_id on home_team for match id={match.id}")

        cutoff = (
            match.start_time.date()
            if isinstance(match.start_time, datetime)
            else match.start_time
        )
        league_id = match.home_team.league_id
        cache_key = (league_id, cutoff)
        cached = self._features_cache.get(cache_key)
        if cached is not None:
            return cached

        lookback = self.config.league_behavior_lookback_matches
        league_matches = self.historical_repo.find_before_date_by_league_id(
            league_id=league_id,
            before_date=cutoff,
            limit=lookback,
        )
        league_stats = self._compute_raw_stats(league_matches)
        global_stats = self._global_stats(cutoff)
        features = self._assemble_features(league_stats, global_stats)
        self._features_cache[cache_key] = features
        return features

    def _global_stats(self, cutoff: date) -> _RawLeagueStats:
        cached = self._global_stats_cache.get(cutoff)
        if cached is not None:
            return cached

        lookback = self.config.league_behavior_lookback_matches
        all_before = self.historical_repo.get_filtered(before_date=cutoff)
        # get_filtered is oldest-first; take newest lookback window.
        sample = all_before[-lookback:] if len(all_before) > lookback else all_before
        stats = self._compute_raw_stats(sample)
        self._global_stats_cache[cutoff] = stats
        return stats

    def _assemble_features(
        self,
        league_stats: _RawLeagueStats,
        global_stats: _RawLeagueStats,
    ) -> LeagueBehaviorFeatures:
        prior_strength = self.config.league_behavior_shrinkage_matches
        sample_size = league_stats.sample_size
        prior_weight = (
            sample_size / (sample_size + prior_strength)
            if (sample_size + prior_strength) > 0
            else 0.0
        )

        return LeagueBehaviorFeatures(
            league_draw_rate=self._shrink_toward(
                league_stats.draw_rate,
                sample_size,
                global_stats.draw_rate if global_stats.draw_rate is not None else 0.25,
                prior_strength,
            ),
            league_home_win_rate=self._shrink_toward(
                league_stats.home_win_rate,
                sample_size,
                (
                    global_stats.home_win_rate
                    if global_stats.home_win_rate is not None
                    else 0.45
                ),
                prior_strength,
            ),
            league_away_win_rate=self._shrink_toward(
                league_stats.away_win_rate,
                sample_size,
                (
                    global_stats.away_win_rate
                    if global_stats.away_win_rate is not None
                    else 0.30
                ),
                prior_strength,
            ),
            league_avg_goals=self._shrink_toward(
                league_stats.avg_goals,
                sample_size,
                global_stats.avg_goals if global_stats.avg_goals is not None else 2.5,
                prior_strength,
            ),
            league_goal_std=self._shrink_toward(
                league_stats.goal_std,
                sample_size if league_stats.goal_std is not None else 0,
                global_stats.goal_std if global_stats.goal_std is not None else 1.5,
                prior_strength,
            ),
            league_favourite_win_rate=self._shrink_toward(
                league_stats.favourite_win_rate,
                league_stats.market_sample_size,
                (
                    global_stats.favourite_win_rate
                    if global_stats.favourite_win_rate is not None
                    else 0.55
                ),
                prior_strength,
            ),
            league_competitive_balance=self._shrink_toward(
                league_stats.competitive_balance,
                sample_size if league_stats.competitive_balance is not None else 0,
                (
                    global_stats.competitive_balance
                    if global_stats.competitive_balance is not None
                    else 0.5
                ),
                prior_strength,
            ),
            league_promoted_team_effect=0.0,
            league_sample_size=sample_size,
            league_data_quality=self._data_quality(league_stats),
            league_prior_weight=prior_weight,
        )

    @staticmethod
    def _shrink_toward(
        observed: float | None,
        sample_size: int,
        prior: float,
        prior_strength: int,
    ) -> float:
        """Empirical-Bayes shrink; missing observed uses prior (equivalent to n=0)."""
        if observed is None:
            return prior
        shrunk = shrink(
            observed,
            sample_size,
            prior_rating=prior,
            prior_strength=prior_strength,
        )
        return prior if shrunk is None else shrunk

    def _data_quality(self, stats: _RawLeagueStats) -> float:
        reference = self.config.league_behavior_quality_reference_matches
        sample_term = min(1.0, stats.sample_size / reference) if reference > 0 else 0.0
        return max(
            0.0,
            min(
                1.0,
                (
                    sample_term
                    + stats.result_completeness
                    + stats.market_completeness
                    + stats.balance_completeness
                )
                / 4.0,
            ),
        )

    def _compute_raw_stats(
        self, matches: Sequence[HistoricalMatchModel]
    ) -> _RawLeagueStats:
        sample_size = len(matches)
        if sample_size == 0:
            return _RawLeagueStats(
                sample_size=0,
                draw_rate=None,
                home_win_rate=None,
                away_win_rate=None,
                avg_goals=None,
                goal_std=None,
                favourite_win_rate=None,
                market_sample_size=0,
                competitive_balance=None,
                result_completeness=0.0,
                market_completeness=0.0,
                balance_completeness=0.0,
            )

        draw_count = 0
        home_win_count = 0
        away_win_count = 0
        usable_result_count = 0
        goal_totals: list[float] = []
        favourite_wins = 0
        market_sample_size = 0
        team_goal_diffs: dict[str, list[float]] = defaultdict(list)

        for historical_match in matches:
            outcome = self._match_outcome(historical_match)
            if outcome is not None:
                usable_result_count += 1
                if outcome == "X":
                    draw_count += 1
                elif outcome == "1":
                    home_win_count += 1
                else:
                    away_win_count += 1

            if (
                historical_match.home_goals is not None
                and historical_match.away_goals is not None
            ):
                total_goals = float(
                    historical_match.home_goals + historical_match.away_goals
                )
                goal_totals.append(total_goals)
                goal_diff = float(
                    historical_match.home_goals - historical_match.away_goals
                )
                team_goal_diffs[historical_match.home_team].append(goal_diff)
                team_goal_diffs[historical_match.away_team].append(-goal_diff)

            favourite_side = self._market_favourite_side(historical_match)
            if favourite_side is not None and outcome is not None:
                market_sample_size += 1
                if favourite_side == outcome:
                    favourite_wins += 1

        min_team_matches = self.config.league_behavior_min_team_matches_for_balance
        team_means = [
            statistics.fmean(diffs)
            for diffs in team_goal_diffs.values()
            if len(diffs) >= min_team_matches
        ]
        teams_seen = len(team_goal_diffs)
        balance_completeness = (
            len(team_means) / teams_seen if teams_seen > 0 else 0.0
        )
        competitive_balance = None
        if len(team_means) >= 2:
            strength_std = statistics.pstdev(team_means)
            competitive_balance = 1.0 / (1.0 + strength_std)

        return _RawLeagueStats(
            sample_size=sample_size,
            draw_rate=draw_count / sample_size if usable_result_count else None,
            home_win_rate=home_win_count / sample_size if usable_result_count else None,
            away_win_rate=away_win_count / sample_size if usable_result_count else None,
            avg_goals=statistics.fmean(goal_totals) if goal_totals else None,
            goal_std=(
                statistics.pstdev(goal_totals) if len(goal_totals) >= 2 else None
            ),
            favourite_win_rate=(
                favourite_wins / market_sample_size if market_sample_size else None
            ),
            market_sample_size=market_sample_size,
            competitive_balance=competitive_balance,
            result_completeness=usable_result_count / sample_size,
            market_completeness=market_sample_size / sample_size,
            balance_completeness=balance_completeness,
        )

    @staticmethod
    def _match_outcome(historical_match: HistoricalMatchModel) -> str | None:
        result = historical_match.result
        if result is not None:
            normalized = str(result).strip().upper()
            if normalized in {"1", "X", "2"}:
                return normalized
            if normalized == "D":
                return "X"
        if (
            historical_match.home_goals is None
            or historical_match.away_goals is None
        ):
            return None
        if historical_match.home_goals > historical_match.away_goals:
            return "1"
        if historical_match.home_goals < historical_match.away_goals:
            return "2"
        return "X"

    @staticmethod
    def _market_favourite_side(historical_match: HistoricalMatchModel) -> str | None:
        odds_home = historical_match.odds_home
        odds_draw = historical_match.odds_draw
        odds_away = historical_match.odds_away
        if (
            odds_home is None
            or odds_draw is None
            or odds_away is None
            or odds_home <= 0
            or odds_draw <= 0
            or odds_away <= 0
        ):
            return None
        try:
            probabilities = odds_to_probabilities(odds_home, odds_draw, odds_away)
        except ValueError:
            return None
        p_home = probabilities["1"]
        p_away = probabilities["2"]
        if p_home > p_away:
            return "1"
        if p_away > p_home:
            return "2"
        return None
