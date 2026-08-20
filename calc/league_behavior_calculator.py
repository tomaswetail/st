"""League-level behaviour features from historical matches (no future leakage)."""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Sequence

from sqlalchemy.orm import Session

from calc.strength_helpers import shrink
from objects.models.fixture import FixtureModel
from objects.models.st_match import STMatchModel
from objects.repositories.fixture_repository import FixtureRepository
from objects.repositories.league_repository import LeagueRepository
from utils.fixture_fields import fixture_goals_away, fixture_goals_home, fixture_home_name, fixture_away_name, fixture_match_date, fixture_outcome
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.data_classes.league_behavior_features import LeagueBehaviorFeatures


@dataclass(frozen=True)
class _RawLeagueStats:
    sample_size: int
    result_sample_size: int
    goal_sample_size: int
    market_data_sample_size: int
    favourite_result_sample_size: int
    balance_sample_size: int
    promotion_sample_size: int
    draw_rate: float | None
    home_win_rate: float | None
    away_win_rate: float | None
    avg_goals: float | None
    goal_std: float | None
    favourite_win_rate: float | None
    competitive_balance: float | None
    promoted_team_effect: float | None
    result_completeness: float
    goal_completeness: float
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
        self.fixture_repo = FixtureRepository(session)
        self._features_cache: dict[tuple[int, date], LeagueBehaviorFeatures] = {}
        self._global_stats_cache: dict[date, _RawLeagueStats] = {}
        self.league_repo = LeagueRepository(session)

    def _resolve_league_id(self, match: STMatchModel) -> int | None:
        if match.league_name:
            league = self.league_repo.get_by_name(match.league_name)
            if league is not None:
                return league.id
        return self.fixture_repo.resolve_internal_league_id_for_team(match.home_team)

    def calculate(self, match: STMatchModel) -> LeagueBehaviorFeatures:
        """Compute league behaviour features for one ST fixture."""
        if match.home_team is None:
            raise ValueError(f"Missing home_team on match id={match.id}")
        if match.start_time is None:
            raise ValueError(f"Missing start_time on match id={match.id}")
        # Historical matches store calendar dates only, so the cutoff is a date.
        # Strict `<` excludes same-day fixtures (no datetime kickoff on history rows).
        cutoff = (
            match.start_time.date()
            if isinstance(match.start_time, datetime)
            else match.start_time
        )
        league_id = self._resolve_league_id(match)
        if league_id is None:
            raise ValueError(f"Missing league for home_team on match id={match.id}")
        cache_key = (league_id, cutoff)
        cached = self._features_cache.get(cache_key)
        if cached is not None:
            return cached

        lookback = self.config.league_behavior_lookback_matches
        league_matches = self._newest_lookback(
            self._matches_strictly_before(
                self.fixture_repo.find_before_date_by_league_id(
                    league_id=league_id,
                    before_date=cutoff,
                    limit=lookback,
                ),
                cutoff=cutoff,
            ),
            lookback=lookback,
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
        sample = self._newest_lookback(
            self._matches_strictly_before(
                self.fixture_repo.get_filtered(
                    before_date=cutoff,
                    limit=lookback,
                ),
                cutoff=cutoff,
            ),
            lookback=lookback,
        )
        stats = self._compute_raw_stats(sample)
        self._global_stats_cache[cutoff] = stats
        return stats

    @staticmethod
    def _matches_strictly_before(
        matches: Sequence[FixtureModel],
        *,
        cutoff: date,
    ) -> list[FixtureModel]:
        """Defense-in-depth: keep only match_date < cutoff (never <=)."""
        return [
            fixture
            for fixture in matches
            if fixture_match_date(fixture) < cutoff
        ]

    @staticmethod
    def _newest_lookback(
        matches: Sequence[FixtureModel],
        *,
        lookback: int,
    ) -> list[FixtureModel]:
        """Keep the newest ``lookback`` matches after a strict-before filter."""
        newest_first = sorted(
            matches,
            key=lambda fixture: fixture_match_date(fixture),
            reverse=True,
        )
        return newest_first[:lookback]

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
        global_promotion = (
            global_stats.promoted_team_effect
            if global_stats.promoted_team_effect is not None
            else 0.0
        )

        return LeagueBehaviorFeatures(
            league_draw_rate=self._shrink_toward(
                league_stats.draw_rate,
                league_stats.result_sample_size,
                global_stats.draw_rate if global_stats.draw_rate is not None else 0.25,
                prior_strength,
            ),
            league_home_win_rate=self._shrink_toward(
                league_stats.home_win_rate,
                league_stats.result_sample_size,
                (
                    global_stats.home_win_rate
                    if global_stats.home_win_rate is not None
                    else 0.45
                ),
                prior_strength,
            ),
            league_away_win_rate=self._shrink_toward(
                league_stats.away_win_rate,
                league_stats.result_sample_size,
                (
                    global_stats.away_win_rate
                    if global_stats.away_win_rate is not None
                    else 0.30
                ),
                prior_strength,
            ),
            league_avg_goals=self._shrink_toward(
                league_stats.avg_goals,
                league_stats.goal_sample_size,
                global_stats.avg_goals if global_stats.avg_goals is not None else 2.5,
                prior_strength,
            ),
            league_goal_std=self._shrink_toward(
                league_stats.goal_std,
                league_stats.goal_sample_size if league_stats.goal_std is not None else 0,
                global_stats.goal_std if global_stats.goal_std is not None else 1.5,
                prior_strength,
            ),
            league_favourite_win_rate=self._shrink_toward(
                league_stats.favourite_win_rate,
                league_stats.favourite_result_sample_size,
                (
                    global_stats.favourite_win_rate
                    if global_stats.favourite_win_rate is not None
                    else 0.55
                ),
                prior_strength,
            ),
            league_competitive_balance=self._shrink_toward(
                league_stats.competitive_balance,
                league_stats.balance_sample_size,
                (
                    global_stats.competitive_balance
                    if global_stats.competitive_balance is not None
                    else 0.5
                ),
                prior_strength,
            ),
            league_promoted_team_effect=self._shrink_toward(
                league_stats.promoted_team_effect,
                league_stats.promotion_sample_size,
                global_promotion,
                prior_strength,
            ),
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
                    + stats.goal_completeness
                    + stats.market_completeness
                    + stats.balance_completeness
                )
                / 5.0,
            ),
        )

    def _compute_raw_stats(
        self, matches: Sequence[FixtureModel]
    ) -> _RawLeagueStats:
        sample_size = len(matches)
        if sample_size == 0:
            return _RawLeagueStats(
                sample_size=0,
                result_sample_size=0,
                goal_sample_size=0,
                market_data_sample_size=0,
                favourite_result_sample_size=0,
                balance_sample_size=0,
                promotion_sample_size=0,
                draw_rate=None,
                home_win_rate=None,
                away_win_rate=None,
                avg_goals=None,
                goal_std=None,
                favourite_win_rate=None,
                competitive_balance=None,
                promoted_team_effect=None,
                result_completeness=0.0,
                goal_completeness=0.0,
                market_completeness=0.0,
                balance_completeness=0.0,
            )

        draw_count = 0
        home_win_count = 0
        away_win_count = 0
        result_sample_size = 0
        goal_totals: list[float] = []
        favourite_wins = 0
        market_data_sample_size = 0
        favourite_result_sample_size = 0
        team_goal_diffs: dict[str, list[float]] = defaultdict(list)
        promoted_vs_established_diffs: list[float] = []

        for fixture in matches:
            outcome = self._match_outcome(fixture)
            if outcome is not None:
                result_sample_size += 1
                if outcome == "X":
                    draw_count += 1
                elif outcome == "1":
                    home_win_count += 1
                else:
                    away_win_count += 1

            if (
                fixture_goals_home(fixture) is not None
                and fixture_goals_away(fixture) is not None
            ):
                total_goals = float(
                    fixture_goals_home(fixture) + fixture_goals_away(fixture)
                )
                goal_totals.append(total_goals)
                goal_diff = float(
                    fixture_goals_home(fixture) - fixture_goals_away(fixture)
                )
                team_goal_diffs[fixture_home_name(fixture)].append(goal_diff)
                team_goal_diffs[fixture_away_name(fixture)].append(-goal_diff)

                promoted_goal_diff = self._promoted_vs_established_goal_diff(
                    fixture
                )
                if promoted_goal_diff is not None:
                    promoted_vs_established_diffs.append(promoted_goal_diff)

            favourite_side = self._market_favourite_side(fixture)
            if favourite_side is not None:
                market_data_sample_size += 1
                if outcome is not None:
                    favourite_result_sample_size += 1
                    if favourite_side == outcome:
                        favourite_wins += 1

        goal_sample_size = len(goal_totals)
        min_team_matches = self.config.league_behavior_min_team_matches_for_balance
        qualifying_diffs = {
            team: diffs
            for team, diffs in team_goal_diffs.items()
            if len(diffs) >= min_team_matches
        }
        team_means = [statistics.fmean(diffs) for diffs in qualifying_diffs.values()]
        teams_seen = len(team_goal_diffs)
        balance_completeness = (
            len(team_means) / teams_seen if teams_seen > 0 else 0.0
        )
        balance_sample_size = sum(len(diffs) for diffs in qualifying_diffs.values())
        competitive_balance = None
        if len(team_means) >= 2:
            strength_std = statistics.pstdev(team_means)
            competitive_balance = 1.0 / (1.0 + strength_std)
        else:
            balance_sample_size = 0

        promotion_sample_size = len(promoted_vs_established_diffs)
        promoted_team_effect = (
            statistics.fmean(promoted_vs_established_diffs)
            if promoted_vs_established_diffs
            else None
        )

        return _RawLeagueStats(
            sample_size=sample_size,
            result_sample_size=result_sample_size,
            goal_sample_size=goal_sample_size,
            market_data_sample_size=market_data_sample_size,
            favourite_result_sample_size=favourite_result_sample_size,
            balance_sample_size=balance_sample_size,
            promotion_sample_size=promotion_sample_size,
            draw_rate=(
                draw_count / result_sample_size if result_sample_size else None
            ),
            home_win_rate=(
                home_win_count / result_sample_size if result_sample_size else None
            ),
            away_win_rate=(
                away_win_count / result_sample_size if result_sample_size else None
            ),
            avg_goals=statistics.fmean(goal_totals) if goal_totals else None,
            goal_std=(
                statistics.pstdev(goal_totals) if len(goal_totals) >= 2 else None
            ),
            favourite_win_rate=(
                favourite_wins / favourite_result_sample_size
                if favourite_result_sample_size
                else None
            ),
            competitive_balance=competitive_balance,
            promoted_team_effect=promoted_team_effect,
            result_completeness=result_sample_size / sample_size,
            goal_completeness=goal_sample_size / sample_size,
            market_completeness=market_data_sample_size / sample_size,
            balance_completeness=balance_completeness,
        )

    def _promoted_vs_established_goal_diff(
        self, fixture: FixtureModel
    ) -> float | None:
        """One observation per promoted-vs-established match (non-overlapping)."""
        if (
            fixture_goals_home(fixture) is None
            or fixture_goals_away(fixture) is None
        ):
            return None
        home_promoted, away_promoted = self._promotion_flags(fixture)
        if home_promoted is True and away_promoted is False:
            return float(fixture_goals_home(fixture) - fixture_goals_away(fixture))
        if away_promoted is True and home_promoted is False:
            return float(fixture_goals_away(fixture) - fixture_goals_home(fixture))
        return None

    @staticmethod
    def _promotion_flags(
        fixture: FixtureModel,
    ) -> tuple[bool | None, bool | None]:
        """Read optional promotion status; None means unavailable (do not invent)."""
        home = LeagueBehaviorCalculator._coerce_promotion_flag(
            getattr(fixture, "home_promoted", None)
        )
        away = LeagueBehaviorCalculator._coerce_promotion_flag(
            getattr(fixture, "away_promoted", None)
        )
        raw_data = getattr(fixture, "raw_data", None)
        if isinstance(raw_data, dict):
            if home is None:
                home = LeagueBehaviorCalculator._coerce_promotion_flag(
                    raw_data.get("home_promoted")
                )
            if away is None:
                away = LeagueBehaviorCalculator._coerce_promotion_flag(
                    raw_data.get("away_promoted")
                )
        return home, away

    @staticmethod
    def _coerce_promotion_flag(value: Any) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "promoted"}:
                return True
            if normalized in {"0", "false", "no", "established"}:
                return False
        return None

    @staticmethod
    def _match_outcome(fixture: FixtureModel) -> str | None:
        return fixture_outcome(fixture)

    @staticmethod
    def _market_favourite_side(fixture: FixtureModel) -> str | None:
        """Odds are no longer stored on fixtures; market favourite is unavailable."""
        del fixture
        return None

