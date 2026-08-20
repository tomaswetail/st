"""League and team-specific home advantage on a log-goal / log-npxG scale."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import exp, log
from typing import NamedTuple

from sqlalchemy.orm import Session

from calc.strength_calculator import StrengthCalculator
from calc.strength_helpers import normalize_strength, npxg_or_xg, weighted_mean_from_pairs
from utils.seasons import season_code_to_start_year, start_year_to_season_code
from objects.models.fixture import FixtureModel
from data_sources.api_football_leagues import code_for_api_football_league_id
from utils.fixture_fields import fixture_away_name, fixture_home_name, fixture_match_date
from objects.models.match_advanced_stats import MatchAdvancedStatsModel
from objects.models.team import TeamModel
from objects.repositories.fixture_repository import FixtureRepository
from objects.repositories.league_repository import LeagueRepository
from objects.repositories.team_repository import TeamRepository
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.data_classes.team_strength_features import TeamStrengthFeatures
from objects.schema.db.team import Team
from utils.common import LEAGUE_NAMES_REV
from utils.competition_type import competition_type_flags, is_league_match


@dataclass
class HomeAdvantageResult:
    home_advantage: float
    league_season_home_advantage: float
    team_home_advantage: float
    raw_team_home_advantage: float
    team_home_advantage_shrinkage_weight: float
    home_attack_residual: float
    away_attack_residual: float
    home_defence_residual: float
    away_defence_residual: float
    home_performance: float
    away_performance: float
    home_match_count: int
    away_match_count: int
    raw_league_season_home_advantage: float = 0.0
    league_home_advantage_shrinkage_weight: float = 0.0
    league_home_advantage_sample_size: int = 0
    competition_home_advantage: float = 0.0
    raw_competition_home_advantage: float = 0.0
    competition_home_advantage_shrinkage_weight: float = 0.0
    competition_home_advantage_sample_size: int = 0


class _MatchNpxg(NamedTuple):
    match: FixtureModel
    played_at_home: bool
    xg_for: float
    xg_against: float
    opponent_name: str
    league_code: str
    season: str


class HomeAdvantageCalculator:
    """Estimate league HA plus additional team-specific home advantage."""

    def __init__(
        self,
        session: Session,
        config: DataSourceConfig | None = None,
        strength_calculator: StrengthCalculator | None = None,
    ) -> None:
        self.session = session
        self.config = config or DataSourceConfig()
        self.fixture_repo = FixtureRepository(session)
        self.team_repo = TeamRepository(session)
        self.league_repo = LeagueRepository(session)
        self.strength_calculator = strength_calculator or StrengthCalculator(
            session=session,
            config=self.config,
        )
        self._features_cache: dict[tuple[int, date], TeamStrengthFeatures] = {}
        self._league_baselines_cache: dict[
            tuple[int, str, date], dict[str, float]
        ] = {}
        self._competition_beta_cache: dict[
            date, dict[str, tuple[float, float, float, int]]
        ] = {}

    def _team_league_id(self, team: Team | TeamModel) -> int | None:
        team_model = self.team_repo.get(team.id)
        if team_model is None and getattr(team, "external_id", None):
            team_model = self.team_repo.get_by_external_id(int(team.external_id))
        if team_model is None:
            team_model = self.team_repo.get_by_name(team.name)
        if team_model is None:
            return None
        return self.fixture_repo.resolve_internal_league_id_for_team(
            team_model
        )

    def process(
        self,
        team: Team,
        current_date: date,
        *,
        target_league_code: str | None = None,
    ) -> HomeAdvantageResult:
        """Return combined league + team + competition home advantage before cutoff."""
        competition = self._calc_competition_home_advantage(
            target_league_code, current_date
        )
        competition_home_advantage = float(
            competition["competition_home_advantage"]
        )

        league_id = self._team_league_id(team)
        if league_id is None:
            return self._empty_result(
                0.0,
                competition_home_advantage=competition_home_advantage,
                raw_competition_home_advantage=float(
                    competition["raw_competition_home_advantage"]
                ),
                competition_home_advantage_shrinkage_weight=float(
                    competition["competition_home_advantage_shrinkage_weight"]
                ),
                competition_home_advantage_sample_size=int(
                    competition["competition_home_advantage_sample_size"]
                ),
            )

        season = self._resolve_season_from_team_history(team, current_date)
        if season is None:
            league_ha_result = {
                "league_season_home_advantage": 0.0,
                "raw_league_season_home_advantage": 0.0,
                "league_home_advantage_shrinkage_weight": 0.0,
                "league_home_advantage_sample_size": 0,
            }
            league_ha = 0.0
        else:
            league_ha_result = self.calc_league_season_home_advantage(
                league_id, season, current_date, return_diagnostics=True
            )
            league_ha = league_ha_result["league_season_home_advantage"]

        team_result = self.calculate_team_home_advantage(
            team=team,
            league_id=league_id,
            season=season or "",
            target_date=current_date,
            league_season_home_advantage=league_ha,
        )
        return HomeAdvantageResult(
            home_advantage=(
                league_ha
                + team_result.team_home_advantage
                + competition_home_advantage
            ),
            league_season_home_advantage=league_ha,
            team_home_advantage=team_result.team_home_advantage,
            raw_team_home_advantage=team_result.raw_team_home_advantage,
            team_home_advantage_shrinkage_weight=(
                team_result.team_home_advantage_shrinkage_weight
            ),
            home_attack_residual=team_result.home_attack_residual,
            away_attack_residual=team_result.away_attack_residual,
            home_defence_residual=team_result.home_defence_residual,
            away_defence_residual=team_result.away_defence_residual,
            home_performance=team_result.home_performance,
            away_performance=team_result.away_performance,
            home_match_count=team_result.home_match_count,
            away_match_count=team_result.away_match_count,
            raw_league_season_home_advantage=league_ha_result[
                "raw_league_season_home_advantage"
            ],
            league_home_advantage_shrinkage_weight=league_ha_result[
                "league_home_advantage_shrinkage_weight"
            ],
            league_home_advantage_sample_size=league_ha_result[
                "league_home_advantage_sample_size"
            ],
            competition_home_advantage=competition_home_advantage,
            raw_competition_home_advantage=float(
                competition["raw_competition_home_advantage"]
            ),
            competition_home_advantage_shrinkage_weight=float(
                competition["competition_home_advantage_shrinkage_weight"]
            ),
            competition_home_advantage_sample_size=int(
                competition["competition_home_advantage_sample_size"]
            ),
        )

    def _calc_competition_home_advantage(
        self,
        target_league_code: str | None,
        before_date: date,
    ) -> dict[str, float | int]:
        """Learn shrunk competition-type effects and apply to the target fixture."""
        zero = {
            "competition_home_advantage": 0.0,
            "raw_competition_home_advantage": 0.0,
            "competition_home_advantage_shrinkage_weight": 0.0,
            "competition_home_advantage_sample_size": 0,
        }
        if target_league_code is None or is_league_match(target_league_code):
            return zero

        betas = self._learn_competition_betas(before_date)
        is_domestic, is_international, is_friendly = competition_type_flags(
            target_league_code
        )
        competition_home_advantage = (
            betas["domestic"][0] * float(is_domestic)
            + betas["international"][0] * float(is_international)
            + betas["friendly"][0] * float(is_friendly)
        )
        max_ha = self.config.max_competition_home_advantage
        competition_home_advantage = max(
            -max_ha, min(max_ha, competition_home_advantage)
        )

        if is_domestic:
            active = betas["domestic"]
        elif is_international:
            active = betas["international"]
        elif is_friendly:
            active = betas["friendly"]
        else:
            return zero

        _beta, raw_effect, weight, sample_size = active
        return {
            "competition_home_advantage": competition_home_advantage,
            "raw_competition_home_advantage": raw_effect,
            "competition_home_advantage_shrinkage_weight": weight,
            "competition_home_advantage_sample_size": sample_size,
        }

    def _learn_competition_betas(
        self, before_date: date
    ) -> dict[str, tuple[float, float, float, int]]:
        """Return per-type (beta, raw_effect, weight, sample_size) before cutoff."""
        cached = self._competition_beta_cache.get(before_date)
        if cached is not None:
            return cached

        goal_sums = self.fixture_repo.get_goal_sums_by_league_before_date(
            before_date
        )
        buckets = self._bucket_goal_sums_by_competition_type(goal_sums)
        league_reference_ha = self._log_home_advantage_from_bucket(
            buckets["league"]
        )
        shrinkage_matches = self.config.competition_ha_shrinkage_matches

        betas: dict[str, tuple[float, float, float, int]] = {}
        for bucket_name in ("domestic", "international", "friendly"):
            sum_home, home_count, sum_away, away_count = buckets[bucket_name]
            sample_size = min(home_count, away_count)
            if sample_size <= 0 or home_count <= 0 or away_count <= 0:
                betas[bucket_name] = (0.0, 0.0, 0.0, 0)
                continue

            home_rate = sum_home / home_count
            away_rate = sum_away / away_count
            if home_rate <= 0 or away_rate <= 0:
                betas[bucket_name] = (0.0, 0.0, 0.0, sample_size)
                continue

            raw_effect = log(home_rate / away_rate) - league_reference_ha
            if shrinkage_matches <= 0:
                weight = 1.0
            else:
                weight = sample_size / (sample_size + shrinkage_matches)
            beta = raw_effect * weight
            betas[bucket_name] = (beta, raw_effect, weight, sample_size)

        self._competition_beta_cache[before_date] = betas
        return betas

    @staticmethod
    def _bucket_goal_sums_by_competition_type(
        goal_sums: dict[str, tuple[int, int, int, int]],
    ) -> dict[str, tuple[int, int, int, int]]:
        buckets = {
            "league": (0, 0, 0, 0),
            "domestic": (0, 0, 0, 0),
            "international": (0, 0, 0, 0),
            "friendly": (0, 0, 0, 0),
        }
        for league_code, totals in goal_sums.items():
            is_domestic, is_international, is_friendly = competition_type_flags(
                league_code
            )
            if is_domestic:
                bucket_key = "domestic"
            elif is_international:
                bucket_key = "international"
            elif is_friendly:
                bucket_key = "friendly"
            else:
                bucket_key = "league"
            buckets[bucket_key] = HomeAdvantageCalculator._merge_goal_bucket(
                buckets[bucket_key], totals
            )
        return buckets

    @staticmethod
    def _merge_goal_bucket(
        left: tuple[int, int, int, int],
        right: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        return (
            left[0] + right[0],
            left[1] + right[1],
            left[2] + right[2],
            left[3] + right[3],
        )

    @staticmethod
    def _log_home_advantage_from_bucket(
        bucket: tuple[int, int, int, int],
    ) -> float:
        sum_home, home_count, sum_away, away_count = bucket
        if home_count <= 0 or away_count <= 0:
            return 0.0
        home_rate = sum_home / home_count
        away_rate = sum_away / away_count
        if home_rate <= 0 or away_rate <= 0:
            return 0.0
        return log(home_rate / away_rate)

    def calc_league_season_home_advantage(
        self,
        league_id: int,
        season: str,
        before_date: date,
        *,
        return_diagnostics: bool = False,
    ) -> float | dict[str, float | int]:
        """Shrunk log ratio of home vs away goal rates for a league/season."""
        raw, sample_size = self._raw_league_season_home_advantage(
            league_id, season, before_date
        )
        prior = self._league_home_advantage_prior(league_id, season, before_date)
        shrinkage_matches = self.config.league_home_advantage_shrinkage_matches
        if sample_size <= 0:
            weight = 0.0
            shrunk = prior
        elif shrinkage_matches <= 0:
            weight = 1.0
            shrunk = raw
        else:
            weight = sample_size / (sample_size + shrinkage_matches)
            shrunk = weight * raw + (1.0 - weight) * prior

        if return_diagnostics:
            return {
                "league_season_home_advantage": shrunk,
                "raw_league_season_home_advantage": raw,
                "league_home_advantage_shrinkage_weight": weight,
                "league_home_advantage_sample_size": sample_size,
            }
        return shrunk

    def _raw_league_season_home_advantage(
        self, league_id: int, season: str, before_date: date
    ) -> tuple[float, int]:
        """Return (raw log HA, league match count) with strict date cutoff."""
        sum_home_goals, home_matches = (
            self.fixture_repo.get_home_goals_sum_by_league(
                league_id, season, before_date
            )
        )
        sum_away_goals, away_matches = (
            self.fixture_repo.get_away_goals_sum_by_league(
                league_id, season, before_date
            )
        )
        sample_size = min(home_matches, away_matches)
        if sample_size == 0:
            return 0.0, 0
        home_goal_rate = sum_home_goals / home_matches
        away_goal_rate = sum_away_goals / away_matches
        if home_goal_rate <= 0 or away_goal_rate <= 0:
            return 0.0, sample_size
        return log(home_goal_rate / away_goal_rate), sample_size

    def _league_home_advantage_prior(
        self, league_id: int, season: str, before_date: date
    ) -> float:
        """Prior HA: previous season for same league, else global config prior."""
        previous_season = self._previous_season_arg(season)
        if previous_season is not None:
            prior_raw, prior_sample = self._raw_league_season_home_advantage(
                league_id, previous_season, before_date
            )
            if prior_sample > 0:
                return prior_raw
        return self.config.league_home_advantage_global_prior

    @staticmethod
    def _previous_season_arg(season: str) -> str | None:
        """Return previous start-year season arg, or None if it cannot be derived."""
        try:
            start_year = int(season)
        except ValueError:
            try:
                start_year = season_code_to_start_year(season)
            except (TypeError, ValueError):
                return None
        return str(start_year - 1)

    def calculate_team_home_advantage(
        self,
        team: Team,
        league_id: int,
        season: str,
        target_date: date,
        league_season_home_advantage: float | None = None,
    ) -> HomeAdvantageResult:
        """Estimate additional team-specific HA from opponent-adjusted npxG residuals."""
        self._features_cache.clear()
        self._league_baselines_cache.clear()
        league_diagnostics = {
            "raw_league_season_home_advantage": 0.0,
            "league_home_advantage_shrinkage_weight": 0.0,
            "league_home_advantage_sample_size": 0,
        }
        if league_season_home_advantage is None:
            if not season:
                league_season_home_advantage = 0.0
            else:
                result = self.calc_league_season_home_advantage(
                    league_id, season, target_date, return_diagnostics=True
                )
                league_season_home_advantage = float(
                    result["league_season_home_advantage"]
                )
                league_diagnostics = {
                    "raw_league_season_home_advantage": float(
                        result["raw_league_season_home_advantage"]
                    ),
                    "league_home_advantage_shrinkage_weight": float(
                        result["league_home_advantage_shrinkage_weight"]
                    ),
                    "league_home_advantage_sample_size": int(
                        result["league_home_advantage_sample_size"]
                    ),
                }
        team_model = self.team_repo.get(team.id)
        if team_model is None:
            return self._empty_result(
                league_season_home_advantage, **league_diagnostics
            )

        match_rows = self._load_team_npxg_matches(team_model, target_date)
        if not match_rows:
            return self._empty_result(
                league_season_home_advantage, **league_diagnostics
            )

        epsilon = self.config.home_advantage_epsilon
        decay_rate = self.config.home_advantage_recency_decay_rate

        home_attack: list[tuple[float, float]] = []
        home_defence: list[tuple[float, float]] = []
        away_attack: list[tuple[float, float]] = []
        away_defence: list[tuple[float, float]] = []

        for row in match_rows:
            if not row.season:
                continue
            match_league = self.league_repo.get_by_code(row.league_code)
            if match_league is None:
                continue

            baselines = self._league_baselines_before(
                match_league.id,
                row.season,
                fixture_match_date(row.match),
            )
            league_home_npxg = baselines.get("home_npxg")
            league_away_npxg = baselines.get("away_npxg")
            league_overall_npxg = baselines.get("npxg")
            if (
                league_home_npxg is None
                or league_away_npxg is None
                or league_overall_npxg is None
                or league_home_npxg <= 0
                or league_away_npxg <= 0
                or league_overall_npxg <= 0
            ):
                continue

            expected_for, expected_against = self._expected_npxg(
                team_id=team.id,
                opponent_name=row.opponent_name,
                match_date=fixture_match_date(row.match),
                played_at_home=row.played_at_home,
                league_home_npxg=league_home_npxg,
                league_away_npxg=league_away_npxg,
                league_overall_npxg=league_overall_npxg,
            )
            if expected_for is None or expected_against is None:
                continue

            attack_residual = log(
                (row.xg_for + epsilon) / (expected_for + epsilon)
            )
            defence_residual = log(
                (expected_against + epsilon) / (row.xg_against + epsilon)
            )
            days_before = (target_date - fixture_match_date(row.match)).days
            weight = exp(-decay_rate * days_before)

            if row.played_at_home:
                home_attack.append((attack_residual, weight))
                home_defence.append((defence_residual, weight))
            else:
                away_attack.append((attack_residual, weight))
                away_defence.append((defence_residual, weight))

        usable_home_match_count = len(home_attack)
        usable_away_match_count = len(away_attack)

        if usable_home_match_count == 0 or usable_away_match_count == 0:
            return self._empty_result(
                league_season_home_advantage,
                home_match_count=usable_home_match_count,
                away_match_count=usable_away_match_count,
                **league_diagnostics,
            )

        mean_home_attack = weighted_mean_from_pairs(home_attack) or 0.0
        mean_away_attack = weighted_mean_from_pairs(away_attack) or 0.0
        mean_home_defence = weighted_mean_from_pairs(home_defence) or 0.0
        mean_away_defence = weighted_mean_from_pairs(away_defence) or 0.0

        home_performance = (mean_home_attack + mean_home_defence)/2
        away_performance = (mean_away_attack + mean_away_defence)/2
        raw_team_home_advantage = home_performance - away_performance

        effective_sample_size = min(usable_home_match_count, usable_away_match_count)
        shrinkage_matches = self.config.home_advantage_shrinkage_matches
        if shrinkage_matches <= 0:
            shrinkage_weight = 1.0
        else:
            shrinkage_weight = effective_sample_size / (
                effective_sample_size + shrinkage_matches
            )
        team_home_advantage = raw_team_home_advantage * shrinkage_weight
        max_ha = self.config.max_team_home_advantage
        team_home_advantage = max(-max_ha, min(max_ha, team_home_advantage))

        return HomeAdvantageResult(
            home_advantage=league_season_home_advantage + team_home_advantage,
            league_season_home_advantage=league_season_home_advantage,
            team_home_advantage=team_home_advantage,
            raw_team_home_advantage=raw_team_home_advantage,
            team_home_advantage_shrinkage_weight=shrinkage_weight,
            home_attack_residual=mean_home_attack,
            away_attack_residual=mean_away_attack,
            home_defence_residual=mean_home_defence,
            away_defence_residual=mean_away_defence,
            home_performance=home_performance,
            away_performance=away_performance,
            home_match_count=usable_home_match_count,
            away_match_count=usable_away_match_count,
            raw_league_season_home_advantage=float(
                league_diagnostics["raw_league_season_home_advantage"]
            ),
            league_home_advantage_shrinkage_weight=float(
                league_diagnostics["league_home_advantage_shrinkage_weight"]
            ),
            league_home_advantage_sample_size=int(
                league_diagnostics["league_home_advantage_sample_size"]
            ),
        )

    def _expected_npxg(
        self,
        *,
        team_id: int,
        opponent_name: str,
        match_date: date,
        played_at_home: bool,
        league_home_npxg: float,
        league_away_npxg: float,
        league_overall_npxg: float,
    ) -> tuple[float | None, float | None]:
        """Opponent-adjusted expected npxG for/against, excluding team HA.

        Defence values are weakness multipliers: >1.0 means concedes more than average.
        """
        opponent = self.team_repo.get_by_name(opponent_name)
        if opponent is None:
            return None, None

        team_features = self._team_features_before(team_id, match_date)
        opponent_features = self._team_features_before(opponent.id, match_date)

        team_attack = self._overall_attack_strength(
            team_features, league_overall_npxg
        )
        team_defence_weakness = self._overall_defence_weakness(
            team_features, league_overall_npxg
        )
        opponent_attack = self._overall_attack_strength(
            opponent_features, league_overall_npxg
        )
        opponent_defence_weakness = self._overall_defence_weakness(
            opponent_features, league_overall_npxg
        )
        if None in (
            team_attack,
            team_defence_weakness,
            opponent_attack,
            opponent_defence_weakness,
        ):
            return None, None
        assert team_attack is not None
        assert team_defence_weakness is not None
        assert opponent_attack is not None
        assert opponent_defence_weakness is not None

        if played_at_home:
            expected_for = (
                league_home_npxg * team_attack * opponent_defence_weakness
            )
            expected_against = (
                league_away_npxg * opponent_attack * team_defence_weakness
            )
        else:
            expected_for = (
                league_away_npxg * team_attack * opponent_defence_weakness
            )
            expected_against = (
                league_home_npxg * opponent_attack * team_defence_weakness
            )
        return expected_for, expected_against

    def _league_baselines_before(
        self, league_id: int, season: str, before_date: date
    ) -> dict[str, float]:
        cache_key = (league_id, season, before_date)
        cached = self._league_baselines_cache.get(cache_key)
        if cached is not None:
            return cached
        baselines = self.strength_calculator.league_averages_by_league_id(
            league_id, before_date, season=season
        )
        self._league_baselines_cache[cache_key] = baselines
        return baselines

    def _team_features_before(
        self, team_id: int, before_date: date
    ) -> TeamStrengthFeatures:
        cache_key = (team_id, before_date)
        cached = self._features_cache.get(cache_key)
        if cached is not None:
            return cached
        features = self.strength_calculator.get_team_features(
            team_id,
            before=datetime.combine(before_date, datetime.min.time()),
        )
        self._features_cache[cache_key] = features
        return features

    @staticmethod
    def _overall_attack_strength(
        features: TeamStrengthFeatures, league_overall_npxg: float
    ) -> float | None:
        if features.opponent_adjusted_attack_strength is not None:
            return features.opponent_adjusted_attack_strength
        return normalize_strength(
            features.recency_weighted_attack_rating, league_overall_npxg
        )

    @staticmethod
    def _overall_defence_weakness(
        features: TeamStrengthFeatures, league_overall_npxg: float
    ) -> float | None:
        """Defence weakness: >1.0 means concedes more npxG than league average."""
        if features.opponent_adjusted_defence_strength is not None:
            return features.opponent_adjusted_defence_strength
        return normalize_strength(
            features.recency_weighted_defence_rating, league_overall_npxg
        )

    def _load_team_npxg_matches(
        self, team: TeamModel, before_date: date
    ) -> list[_MatchNpxg]:
        lookback = max(self.config.team_strength_lookback_matches * 3, 60)
        fixtures = self.fixture_repo.find_before_date_by_team(
            team_name=team.name,
            before_date=before_date,
            venue=None,
            limit=lookback,
        )
        if not fixtures:
            return []

        match_stat_rows = self.strength_calculator.attach_advanced_stats(
            fixtures, team.name
        )
        results: list[_MatchNpxg] = []
        for fixture, advanced_stats, played_at_home in match_stat_rows:
            xg_for, xg_against = self._npxg_pair(
                advanced_stats, played_at_home=played_at_home
            )
            if xg_for is None or xg_against is None:
                continue
            league_code = code_for_api_football_league_id(fixture.league_id)
            if not league_code or not fixture.league_season:
                continue
            opponent_name = (
                fixture_away_name(fixture)
                if played_at_home
                else fixture_home_name(fixture)
            )
            results.append(
                _MatchNpxg(
                    match=fixture,
                    played_at_home=played_at_home,
                    xg_for=xg_for,
                    xg_against=xg_against,
                    opponent_name=opponent_name,
                    league_code=league_code,
                    season=str(fixture.league_season),
                )
            )
        return results

    @staticmethod
    def _npxg_pair(
        advanced_stats: MatchAdvancedStatsModel, *, played_at_home: bool
    ) -> tuple[float | None, float | None]:
        if played_at_home:
            xg_for = npxg_or_xg(
                advanced_stats.home_non_penalty_xg, advanced_stats.home_xg
            )
            xg_against = npxg_or_xg(
                advanced_stats.away_non_penalty_xg, advanced_stats.away_xg
            )
        else:
            xg_for = npxg_or_xg(
                advanced_stats.away_non_penalty_xg, advanced_stats.away_xg
            )
            xg_against = npxg_or_xg(
                advanced_stats.home_non_penalty_xg, advanced_stats.home_xg
            )
        return xg_for, xg_against

    def _resolve_season_from_team_history(
        self, team: Team, before_date: date
    ) -> str | None:
        """Return season start-year for the team's current league only (no fallback)."""
        league_id = self._team_league_id(team)
        if league_id is None:
            return None
        league = self.league_repo.get(league_id)
        if league is None:
            return None
        league_code = LEAGUE_NAMES_REV.get(league.league_name) or code_for_api_football_league_id(
            league.external_id
        )
        if not league_code:
            return None

        lookback = max(self.config.team_strength_lookback_matches * 3, 60)
        fixtures = self.fixture_repo.find_before_date_by_team(
            team_name=team.name,
            before_date=before_date,
            venue=None,
            limit=lookback,
        )
        for fixture in fixtures:
            code = code_for_api_football_league_id(fixture.league_id)
            if code != league_code:
                continue
            if not fixture.league_season:
                continue
            return self._season_to_repo_arg(str(fixture.league_season))
        return None

    @staticmethod
    def _season_to_repo_arg(season: str) -> str:
        """Convert stored season codes to the start-year form used by goal-sum queries."""
        text = str(season).strip()
        if len(text) == 4 and text.isdigit():
            start = int(text[:2])
            end = int(text[2:])
            if end == (start + 1) % 100:
                return str(season_code_to_start_year(text))
            if int(text) >= 1900:
                return text
        try:
            return str(season_code_to_start_year(text))
        except (TypeError, ValueError):
            return text

    @staticmethod
    def _empty_result(
        league_season_home_advantage: float,
        *,
        home_match_count: int = 0,
        away_match_count: int = 0,
        raw_league_season_home_advantage: float = 0.0,
        league_home_advantage_shrinkage_weight: float = 0.0,
        league_home_advantage_sample_size: int = 0,
        competition_home_advantage: float = 0.0,
        raw_competition_home_advantage: float = 0.0,
        competition_home_advantage_shrinkage_weight: float = 0.0,
        competition_home_advantage_sample_size: int = 0,
    ) -> HomeAdvantageResult:
        return HomeAdvantageResult(
            home_advantage=league_season_home_advantage + competition_home_advantage,
            league_season_home_advantage=league_season_home_advantage,
            team_home_advantage=0.0,
            raw_team_home_advantage=0.0,
            team_home_advantage_shrinkage_weight=0.0,
            home_attack_residual=0.0,
            away_attack_residual=0.0,
            home_defence_residual=0.0,
            away_defence_residual=0.0,
            home_performance=0.0,
            away_performance=0.0,
            home_match_count=home_match_count,
            away_match_count=away_match_count,
            raw_league_season_home_advantage=raw_league_season_home_advantage,
            league_home_advantage_shrinkage_weight=league_home_advantage_shrinkage_weight,
            league_home_advantage_sample_size=league_home_advantage_sample_size,
            competition_home_advantage=competition_home_advantage,
            raw_competition_home_advantage=raw_competition_home_advantage,
            competition_home_advantage_shrinkage_weight=(
                competition_home_advantage_shrinkage_weight
            ),
            competition_home_advantage_sample_size=competition_home_advantage_sample_size,
        )
