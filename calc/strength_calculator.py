"""Team strength features and Dixon-Coles 1/X/2 probabilities.

Uses only persisted ``match_advanced_stats`` joined to ``historical_matches``.
History is always filtered with ``match_date < before`` so the target match
cannot leak into ratings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

from sqlalchemy.orm import Session

from database import SessionLocal
from objects.models.historical_match import HistoricalMatchModel
from objects.models.match_advanced_stats import MatchAdvancedStatsModel
from objects.models.team import TeamModel
from objects.repositories.historical_match_repository import HistoricalMatchRepository
from objects.repositories.match_advanced_stats_repository import (
    MatchAdvancedStatsRepository,
)
from objects.repositories.team_repository import TeamRepository
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.data_classes.team_strength_features import (
    MatchStrengthFeatures,
    TeamStrengthFeatures,
)

# (metric_value, recency_weight) pairs, newest match first.
WeightedObservation = tuple[float, float]
MatchStatRow = tuple[HistoricalMatchModel, MatchAdvancedStatsModel, bool]


@dataclass
class _MatchSideMetrics:
    """Team-perspective metrics extracted from one historical match."""

    non_penalty_xg_for: float | None
    non_penalty_xg_against: float | None
    shot_quality_for: float | None
    shot_quality_against: float | None
    set_piece_xg_for: float | None
    set_piece_xg_against: float | None
    goalkeeper_xgot_faced: float | None
    goals_conceded: float
    attack_xg: float | None
    defence_xg: float | None
    played_at_home: bool


@dataclass
class _ObservationBuckets:
    """Accumulators for weighted team metrics across lookback matches."""

    non_penalty_xg_for: list[WeightedObservation] = field(default_factory=list)
    non_penalty_xg_against: list[WeightedObservation] = field(default_factory=list)
    shot_quality_for: list[WeightedObservation] = field(default_factory=list)
    shot_quality_against: list[WeightedObservation] = field(default_factory=list)
    set_piece_xg_for: list[WeightedObservation] = field(default_factory=list)
    set_piece_xg_against: list[WeightedObservation] = field(default_factory=list)
    attack_xg: list[WeightedObservation] = field(default_factory=list)
    defence_xg: list[WeightedObservation] = field(default_factory=list)
    opponent_adjusted_attack: list[WeightedObservation] = field(default_factory=list)
    opponent_adjusted_defence: list[WeightedObservation] = field(default_factory=list)
    home_attack_values: list[float] = field(default_factory=list)
    home_defence_values: list[float] = field(default_factory=list)
    away_attack_values: list[float] = field(default_factory=list)
    away_defence_values: list[float] = field(default_factory=list)
    xgot_faced_values: list[float] = field(default_factory=list)
    goals_conceded_values: list[float] = field(default_factory=list)


def weighted_mean(
    values: list[float],
    weights: list[float],
) -> float | None:
    """Return the weighted arithmetic mean, or None if inputs are unusable."""
    if not values or not weights or len(values) != len(weights):
        return None
    total_weight = sum(weights)
    if total_weight <= 0:
        return None
    return sum(value * weight for value, weight in zip(values, weights)) / total_weight


def shrink(
    observed: float | None,
    sample_size: int,
    *,
    prior_rating: float = 1.0,
    prior_strength: int = 8,
) -> float | None:
    """Shrink ``observed`` toward ``prior_rating`` using sample vs prior strength."""
    if observed is None:
        return None
    if prior_strength <= 0:
        return observed
    return (
        sample_size * observed + prior_strength * prior_rating
    ) / (sample_size + prior_strength)


def recency_weights(match_count: int, decay: float = 0.9) -> list[float]:
    """Newest-first exponential weights: ``decay ** matches_ago``."""
    if match_count <= 0:
        return []
    return [decay**matches_ago for matches_ago in range(match_count)]


def normalize_strength(
    team_rate: float | None,
    league_rate: float | None,
) -> float | None:
    """Return team rate relative to league rate (1.0 = average)."""
    if team_rate is None or league_rate is None or league_rate <= 0:
        return None
    return team_rate / league_rate


def expected_goals_from_strengths(
    home_attack_strength: float | None,
    away_defence_strength: float | None,
    away_attack_strength: float | None,
    home_defence_strength: float | None,
    league_home_goal_rate: float,
    league_away_goal_rate: float,
) -> tuple[float | None, float | None]:
    """Build Poisson λ_home / λ_away from relative strengths and league rates."""
    if None in (
        home_attack_strength,
        away_defence_strength,
        away_attack_strength,
        home_defence_strength,
    ):
        return None, None
    assert home_attack_strength is not None
    assert away_defence_strength is not None
    assert away_attack_strength is not None
    assert home_defence_strength is not None
    return (
        league_home_goal_rate * home_attack_strength * away_defence_strength,
        league_away_goal_rate * away_attack_strength * home_defence_strength,
    )


def _poisson_pmf(goals: int, expected_goals: float) -> float:
    """Poisson probability mass for scoring ``goals`` when λ=``expected_goals``."""
    if expected_goals <= 0:
        return 1.0 if goals == 0 else 0.0
    return math.exp(-expected_goals) * (expected_goals**goals) / math.factorial(goals)


def _dixon_coles_tau(
    home_goals: int,
    away_goals: int,
    lambda_home: float,
    lambda_away: float,
    rho: float,
) -> float:
    """Dixon–Coles low-score dependence factor τ(home, away)."""
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lambda_home * lambda_away * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lambda_home * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + lambda_away * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def _scoreline_probability(
    home_goals: int,
    away_goals: int,
    lambda_home: float,
    lambda_away: float,
    rho: float,
) -> float:
    """Independent Poisson scoreline probability with Dixon–Coles τ."""
    independent = _poisson_pmf(home_goals, lambda_home) * _poisson_pmf(
        away_goals, lambda_away
    )
    return max(
        0.0,
        independent
        * _dixon_coles_tau(home_goals, away_goals, lambda_home, lambda_away, rho),
    )


def dixon_coles_matrix(
    lambda_home: float,
    lambda_away: float,
    rho: float = -0.13,
    max_goals: int = 10,
) -> tuple[list[list[float]], float, float, float]:
    """Build a scoreline matrix and renormalized 1/X/2 probabilities."""
    size = max_goals + 1
    matrix = [[0.0] * size for _ in range(size)]
    home_win = draw = away_win = total = 0.0
    for home_goals in range(size):
        for away_goals in range(size):
            probability = _scoreline_probability(
                home_goals, away_goals, lambda_home, lambda_away, rho
            )
            matrix[home_goals][away_goals] = probability
            total += probability
            if home_goals > away_goals:
                home_win += probability
            elif home_goals == away_goals:
                draw += probability
            else:
                away_win += probability
    return _renormalize_dixon_coles(matrix, home_win, draw, away_win, total)


def _renormalize_dixon_coles(
    matrix: list[list[float]],
    home_win: float,
    draw: float,
    away_win: float,
    total: float,
) -> tuple[list[list[float]], float, float, float]:
    """Scale matrix and 1/X/2 probs so they sum to 1."""
    if total <= 0:
        return matrix, home_win, draw, away_win
    scale = 1.0 / total
    for home_goals, row in enumerate(matrix):
        for away_goals, probability in enumerate(row):
            matrix[home_goals][away_goals] = probability * scale
    return matrix, home_win * scale, draw * scale, away_win * scale


def _mean(values: list[float]) -> float | None:
    """Simple arithmetic mean, or None for an empty list."""
    if not values:
        return None
    return sum(values) / len(values)


def _weighted_mean_from_pairs(observations: list[WeightedObservation]) -> float | None:
    """Weighted mean from (value, weight) pairs."""
    if not observations:
        return None
    return weighted_mean(
        [value for value, _weight in observations],
        [weight for _value, weight in observations],
    )


def _extract_side_metrics(
    historical_match: HistoricalMatchModel,
    advanced_stats: MatchAdvancedStatsModel,
    played_at_home: bool,
) -> _MatchSideMetrics:
    """Map home/away advanced-stats columns onto the team's perspective."""
    if played_at_home:
        return _MatchSideMetrics(
            non_penalty_xg_for=advanced_stats.home_non_penalty_xg,
            non_penalty_xg_against=advanced_stats.away_non_penalty_xg,
            shot_quality_for=advanced_stats.average_home_shot_xg,
            shot_quality_against=advanced_stats.average_away_shot_xg,
            set_piece_xg_for=advanced_stats.home_set_piece_xg,
            set_piece_xg_against=advanced_stats.away_set_piece_xg,
            goalkeeper_xgot_faced=advanced_stats.away_xgot,
            goals_conceded=float(historical_match.away_goals),
            attack_xg=(
                advanced_stats.home_xg_from_shots
                if advanced_stats.home_xg_from_shots is not None
                else advanced_stats.home_xg
            ),
            defence_xg=(
                advanced_stats.away_xg_from_shots
                if advanced_stats.away_xg_from_shots is not None
                else advanced_stats.away_xg
            ),
            played_at_home=True,
        )
    return _MatchSideMetrics(
        non_penalty_xg_for=advanced_stats.away_non_penalty_xg,
        non_penalty_xg_against=advanced_stats.home_non_penalty_xg,
        shot_quality_for=advanced_stats.average_away_shot_xg,
        shot_quality_against=advanced_stats.average_home_shot_xg,
        set_piece_xg_for=advanced_stats.away_set_piece_xg,
        set_piece_xg_against=advanced_stats.home_set_piece_xg,
        goalkeeper_xgot_faced=advanced_stats.home_xgot,
        goals_conceded=float(historical_match.home_goals),
        attack_xg=(
            advanced_stats.away_xg_from_shots
            if advanced_stats.away_xg_from_shots is not None
            else advanced_stats.away_xg
        ),
        defence_xg=(
            advanced_stats.home_xg_from_shots
            if advanced_stats.home_xg_from_shots is not None
            else advanced_stats.home_xg
        ),
        played_at_home=False,
    )


def _append_observation(
    bucket: list[WeightedObservation],
    value: float | None,
    weight: float,
) -> None:
    """Append a weighted observation when the metric is present."""
    if value is not None:
        bucket.append((value, weight))


class StrengthCalculator:
    """Compute leakage-safe team and match strength features from stored xG stats."""

    def __init__(
        self,
        session: Session | None = None,
        config: DataSourceConfig | None = None,
        provider: str | None = None,
    ) -> None:
        """Wire DB session, config, and repositories used for history loads."""
        self._owns_session = session is None
        self.session = session or SessionLocal()
        self.config = config or DataSourceConfig()
        self.provider = provider or self.config.football_data_provider
        self.stats_repo = MatchAdvancedStatsRepository(self.session)
        self.team_repo = TeamRepository(self.session)
        self.historical_repo = HistoricalMatchRepository(self.session)

    def close(self) -> None:
        """Close the session when this calculator created it."""
        if self._owns_session:
            self.session.close()

    def get_team_features(
        self,
        team_id: int,
        before: datetime,
        venue: Literal["home", "away"] | None = None,
        lookback_matches: int | None = None,
    ) -> TeamStrengthFeatures:
        """Return recency-weighted strength features for one team before a cutoff."""
        lookback = lookback_matches or self.config.team_strength_lookback_matches
        team = self.team_repo.get(team_id)
        if team is None:
            return self._empty_team_features(team_id, before, venue, lookback)

        before_date = before.date() if isinstance(before, datetime) else before
        match_stat_rows = self._load_team_match_stats(
            team=team,
            before_date=before_date,
            venue=venue,
            lookback_matches=lookback,
        )
        if not match_stat_rows:
            return self._empty_team_features(team_id, before, venue, lookback)

        return self._build_team_features_from_rows(
            team_id=team_id,
            before=before,
            venue=venue,
            lookback_matches=lookback,
            match_stat_rows=match_stat_rows,
            league_baselines=self._league_averages(team, before_date),
        )

    def get_match_features(
        self,
        match_id: int,
        lookback_matches: int | None = None,
    ) -> MatchStrengthFeatures:
        """Combine home/away team features and Dixon–Coles 1/X/2 for one match."""
        historical_match = self.historical_repo.get(match_id)
        if historical_match is None:
            return MatchStrengthFeatures(
                match_id=match_id,
                home_team_id=None,
                away_team_id=None,
                home=None,
                away=None,
            )

        home_team = self.team_repo.get_by_name(historical_match.home_team)
        away_team = self.team_repo.get_by_name(historical_match.away_team)
        feature_cutoff = datetime.combine(
            historical_match.match_date, datetime.min.time()
        )
        home_features = self._team_features_for_side(
            home_team, feature_cutoff, "home", lookback_matches
        )
        away_features = self._team_features_for_side(
            away_team, feature_cutoff, "away", lookback_matches
        )
        return self._assemble_match_features(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            home_features=home_features,
            away_features=away_features,
        )

    def explain(
        self,
        features: TeamStrengthFeatures | MatchStrengthFeatures,
    ) -> str:
        """Human-readable dump of team or match strength features for debugging."""
        if isinstance(features, MatchStrengthFeatures):
            return self._explain_match_features(features)
        return self._explain_team_features(features)

    def _team_features_for_side(
        self,
        team: TeamModel | None,
        before: datetime,
        venue: Literal["home", "away"],
        lookback_matches: int | None,
    ) -> TeamStrengthFeatures | None:
        """Load team features for one side of a fixture, or None if team missing."""
        if team is None:
            return None
        return self.get_team_features(
            team.id,
            before=before,
            venue=venue,
            lookback_matches=lookback_matches,
        )

    def _assemble_match_features(
        self,
        *,
        match_id: int,
        home_team: TeamModel | None,
        away_team: TeamModel | None,
        home_features: TeamStrengthFeatures | None,
        away_features: TeamStrengthFeatures | None,
    ) -> MatchStrengthFeatures:
        """Build MatchStrengthFeatures from side features and Dixon–Coles."""
        league_home, league_away = self._league_goal_rates(home_team)
        expected_home, expected_away = self._match_expected_goals(
            home_features, away_features, league_home, league_away
        )
        home_win, draw, away_win = self._dixon_coles_probs(expected_home, expected_away)
        return self._match_features_payload(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            home_features=home_features,
            away_features=away_features,
            expected_home_goals=expected_home,
            expected_away_goals=expected_away,
            home_win_probability=home_win,
            draw_probability=draw,
            away_win_probability=away_win,
        )

    def _league_goal_rates(
        self, home_team: TeamModel | None
    ) -> tuple[float, float]:
        """League home/away goal baselines with safe defaults."""
        league_home_goal_rate = 1.35
        league_away_goal_rate = 1.20
        if home_team is None or home_team.league_id is None:
            return league_home_goal_rate, league_away_goal_rate
        try:
            league_home_goal_rate = float(
                self.historical_repo.get_home_goal_average_by_league(
                    home_team.league_id
                )
            )
            league_away_goal_rate = float(
                self.historical_repo.get_away_goal_average_by_league(
                    home_team.league_id
                )
            )
        except (TypeError, ZeroDivisionError, KeyError):
            pass
        return league_home_goal_rate, league_away_goal_rate

    def _match_expected_goals(
        self,
        home_features: TeamStrengthFeatures | None,
        away_features: TeamStrengthFeatures | None,
        league_home_goal_rate: float,
        league_away_goal_rate: float,
    ) -> tuple[float | None, float | None]:
        """Resolve side strengths then compute λ_home / λ_away."""
        return expected_goals_from_strengths(
            self._side_attack_strength(home_features, "home", league_home_goal_rate),
            self._side_defence_strength(away_features, "away", league_home_goal_rate),
            self._side_attack_strength(away_features, "away", league_away_goal_rate),
            self._side_defence_strength(home_features, "home", league_away_goal_rate),
            league_home_goal_rate,
            league_away_goal_rate,
        )

    def _side_attack_strength(
        self,
        features: TeamStrengthFeatures | None,
        venue: Literal["home", "away"],
        league_rate: float,
    ) -> float | None:
        """Venue attack strength with recency-xG fallback."""
        if features is None:
            return None
        venue_strength = (
            features.home_attack_strength
            if venue == "home"
            else features.away_attack_strength
        )
        return self._prefer_venue_or_recency_strength(
            venue_strength=venue_strength,
            recency_rating=features.recency_weighted_attack_rating,
            league_rate=league_rate,
        )

    def _side_defence_strength(
        self,
        features: TeamStrengthFeatures | None,
        venue: Literal["home", "away"],
        league_rate: float,
    ) -> float | None:
        """Venue defence strength with recency-xG fallback."""
        if features is None:
            return None
        venue_strength = (
            features.home_defence_strength
            if venue == "home"
            else features.away_defence_strength
        )
        return self._prefer_venue_or_recency_strength(
            venue_strength=venue_strength,
            recency_rating=features.recency_weighted_defence_rating,
            league_rate=league_rate,
        )

    def _dixon_coles_probs(
        self,
        expected_home_goals: float | None,
        expected_away_goals: float | None,
    ) -> tuple[float | None, float | None, float | None]:
        """Return 1/X/2 probs from expected goals, or Nones if λ missing."""
        if expected_home_goals is None or expected_away_goals is None:
            return None, None, None
        _matrix, home_win, draw, away_win = dixon_coles_matrix(
            expected_home_goals,
            expected_away_goals,
            rho=self.config.dixon_coles_rho,
            max_goals=self.config.dixon_coles_max_goals,
        )
        return home_win, draw, away_win

    @staticmethod
    def _match_features_payload(
        *,
        match_id: int,
        home_team: TeamModel | None,
        away_team: TeamModel | None,
        home_features: TeamStrengthFeatures | None,
        away_features: TeamStrengthFeatures | None,
        expected_home_goals: float | None,
        expected_away_goals: float | None,
        home_win_probability: float | None,
        draw_probability: float | None,
        away_win_probability: float | None,
    ) -> MatchStrengthFeatures:
        """Flatten side features into the match-level dataclass."""
        return MatchStrengthFeatures(
            match_id=match_id,
            home_team_id=home_team.id if home_team else None,
            away_team_id=away_team.id if away_team else None,
            home=home_features,
            away=away_features,
            **_side_feature_fields(home_features, away_features),
            expected_home_goals=expected_home_goals,
            expected_away_goals=expected_away_goals,
            dixon_coles_home_probability=home_win_probability,
            dixon_coles_draw_probability=draw_probability,
            dixon_coles_away_probability=away_win_probability,
        )

    def _explain_match_features(self, features: MatchStrengthFeatures) -> str:
        """Format match-level explain text including nested team dumps."""
        lines = [
            f"Match {features.match_id}",
            f"λ home={features.expected_home_goals} λ away={features.expected_away_goals}",
            (
                f"Dixon-Coles 1={features.dixon_coles_home_probability} "
                f"X={features.dixon_coles_draw_probability} "
                f"2={features.dixon_coles_away_probability}"
            ),
        ]
        if features.home is not None:
            lines.extend(["--- home ---", self._explain_team_features(features.home)])
        if features.away is not None:
            lines.extend(["--- away ---", self._explain_team_features(features.away)])
        return "\n".join(lines)

    @staticmethod
    def _explain_team_features(features: TeamStrengthFeatures) -> str:
        """Format team-level explain text."""
        return "\n".join(
            [
                f"Team {features.team_id} before {features.before} venue={features.venue}",
                f"Matches used: {features.sample_size}",
                f"Weighted npxG for: {features.non_penalty_xg_for}",
                f"Weighted npxG against: {features.non_penalty_xg_against}",
                f"Home attack strength: {features.home_attack_strength}",
                f"Home defence strength: {features.home_defence_strength}",
                f"Away attack strength: {features.away_attack_strength}",
                f"Away defence strength: {features.away_defence_strength}",
                f"Opponent-adjusted attack: {features.opponent_adjusted_attack_strength}",
                f"Opponent-adjusted defence: {features.opponent_adjusted_defence_strength}",
                f"Set-piece for/against: {features.set_piece_xg_for} / {features.set_piece_xg_against}",
                f"Goalkeeper prevention: {features.goalkeeper_goals_prevented}",
            ]
        )

    @staticmethod
    def _prefer_venue_or_recency_strength(
        *,
        venue_strength: float | None,
        recency_rating: float | None,
        league_rate: float,
    ) -> float | None:
        """Prefer league-relative venue strength; else normalize raw recency xG."""
        if venue_strength is not None:
            return venue_strength
        if recency_rating is None:
            return None
        return normalize_strength(recency_rating, league_rate) or recency_rating

    def _build_team_features_from_rows(
        self,
        *,
        team_id: int,
        before: datetime,
        venue: Literal["home", "away"] | None,
        lookback_matches: int,
        match_stat_rows: list[MatchStatRow],
        league_baselines: dict[str, float],
    ) -> TeamStrengthFeatures:
        """Aggregate per-match advanced stats into TeamStrengthFeatures."""
        buckets = self._collect_observations(match_stat_rows, league_baselines)
        prior = self.config.team_strength_prior_matches or (
            self.config.football_data_feature_shrinkage_prior_matches
        )
        return self._team_features_from_buckets(
            team_id=team_id,
            before=before,
            venue=venue,
            lookback_matches=lookback_matches,
            sample_size=len(match_stat_rows),
            buckets=buckets,
            league_baselines=league_baselines,
            prior_match_count=prior,
        )

    def _collect_observations(
        self,
        match_stat_rows: list[MatchStatRow],
        league_baselines: dict[str, float],
    ) -> _ObservationBuckets:
        """Walk lookback rows and fill weighted observation buckets."""
        buckets = _ObservationBuckets()
        match_weights = recency_weights(
            len(match_stat_rows), self.config.team_strength_recency_decay
        )
        for match_index, (historical_match, advanced_stats, played_at_home) in enumerate(
            match_stat_rows
        ):
            metrics = _extract_side_metrics(
                historical_match, advanced_stats, played_at_home
            )
            self._accumulate_match_metrics(
                buckets,
                metrics,
                match_weights[match_index],
                league_baselines,
            )
        return buckets

    def _accumulate_match_metrics(
        self,
        buckets: _ObservationBuckets,
        metrics: _MatchSideMetrics,
        match_weight: float,
        league_baselines: dict[str, float],
    ) -> None:
        """Add one match's metrics into the observation buckets."""
        _append_observation(
            buckets.non_penalty_xg_for, metrics.non_penalty_xg_for, match_weight
        )
        _append_observation(
            buckets.non_penalty_xg_against, metrics.non_penalty_xg_against, match_weight
        )
        _append_observation(
            buckets.shot_quality_for, metrics.shot_quality_for, match_weight
        )
        _append_observation(
            buckets.shot_quality_against, metrics.shot_quality_against, match_weight
        )
        _append_observation(
            buckets.set_piece_xg_for, metrics.set_piece_xg_for, match_weight
        )
        _append_observation(
            buckets.set_piece_xg_against, metrics.set_piece_xg_against, match_weight
        )
        self._accumulate_attack_defence(buckets, metrics, match_weight)
        self._accumulate_opponent_adjustment(
            buckets, metrics, match_weight, league_baselines
        )
        if metrics.goalkeeper_xgot_faced is not None:
            buckets.xgot_faced_values.append(metrics.goalkeeper_xgot_faced)
            buckets.goals_conceded_values.append(metrics.goals_conceded)

    @staticmethod
    def _accumulate_attack_defence(
        buckets: _ObservationBuckets,
        metrics: _MatchSideMetrics,
        match_weight: float,
    ) -> None:
        """Accumulate attack/defence xG and venue splits."""
        if metrics.attack_xg is not None:
            buckets.attack_xg.append((metrics.attack_xg, match_weight))
            if metrics.played_at_home:
                buckets.home_attack_values.append(metrics.attack_xg)
            else:
                buckets.away_attack_values.append(metrics.attack_xg)
        if metrics.defence_xg is not None:
            buckets.defence_xg.append((metrics.defence_xg, match_weight))
            if metrics.played_at_home:
                buckets.home_defence_values.append(metrics.defence_xg)
            else:
                buckets.away_defence_values.append(metrics.defence_xg)

    def _accumulate_opponent_adjustment(
        self,
        buckets: _ObservationBuckets,
        metrics: _MatchSideMetrics,
        match_weight: float,
        league_baselines: dict[str, float],
    ) -> None:
        """Optionally accumulate simple opponent-adjusted attack/defence."""
        if self.config.football_data_opponent_adjustment != "simple":
            return
        league_npxg_for = league_baselines.get("npxg_for")
        league_npxg_against = league_baselines.get("npxg_against")
        if (
            metrics.non_penalty_xg_for is not None
            and league_npxg_against
            and league_npxg_against > 0
        ):
            buckets.opponent_adjusted_attack.append(
                (metrics.non_penalty_xg_for / league_npxg_against, match_weight)
            )
        if (
            metrics.non_penalty_xg_against is not None
            and league_npxg_for
            and league_npxg_for > 0
        ):
            buckets.opponent_adjusted_defence.append(
                (metrics.non_penalty_xg_against / league_npxg_for, match_weight)
            )

    def _team_features_from_buckets(
        self,
        *,
        team_id: int,
        before: datetime,
        venue: Literal["home", "away"] | None,
        lookback_matches: int,
        sample_size: int,
        buckets: _ObservationBuckets,
        league_baselines: dict[str, float],
        prior_match_count: int,
    ) -> TeamStrengthFeatures:
        """Reduce observation buckets into the public TeamStrengthFeatures DTO."""
        return TeamStrengthFeatures(
            team_id=team_id,
            before=before,
            venue=venue,
            lookback_matches=lookback_matches,
            sample_size=sample_size,
            **self._core_metrics_from_buckets(
                buckets, league_baselines, sample_size, prior_match_count
            ),
            **self._venue_strengths_from_buckets(
                buckets, league_baselines, prior_match_count
            ),
            **self._advanced_metrics_from_buckets(
                buckets, sample_size, prior_match_count
            ),
        )

    def _advanced_metrics_from_buckets(
        self,
        buckets: _ObservationBuckets,
        sample_size: int,
        prior_match_count: int,
    ) -> dict[str, float | bool | None]:
        """Recency, opponent-adjusted, set-piece, GK, and coverage flags."""
        return {
            "recency_weighted_attack_rating": _weighted_mean_from_pairs(
                buckets.attack_xg
            ),
            "recency_weighted_defence_rating": _weighted_mean_from_pairs(
                buckets.defence_xg
            ),
            "opponent_adjusted_attack_strength": shrink(
                _weighted_mean_from_pairs(buckets.opponent_adjusted_attack),
                sample_size,
                prior_rating=1.0,
                prior_strength=prior_match_count,
            ),
            "opponent_adjusted_defence_strength": shrink(
                _weighted_mean_from_pairs(buckets.opponent_adjusted_defence),
                sample_size,
                prior_rating=1.0,
                prior_strength=prior_match_count,
            ),
            "set_piece_xg_for": _weighted_mean_from_pairs(buckets.set_piece_xg_for),
            "set_piece_xg_against": _weighted_mean_from_pairs(
                buckets.set_piece_xg_against
            ),
            "goalkeeper_goals_prevented": self._goalkeeper_prevention(buckets),
            "has_xg_data": bool(buckets.non_penalty_xg_for or buckets.attack_xg),
            "has_xgot_data": bool(buckets.xgot_faced_values),
            "has_set_piece_data": bool(
                buckets.set_piece_xg_for or buckets.set_piece_xg_against
            ),
        }

    def _core_metrics_from_buckets(
        self,
        buckets: _ObservationBuckets,
        league_baselines: dict[str, float],
        sample_size: int,
        prior_match_count: int,
    ) -> dict[str, float | None]:
        """npxG and shot-quality fields derived from observation buckets."""
        return {
            "non_penalty_xg_for": shrink(
                _weighted_mean_from_pairs(buckets.non_penalty_xg_for),
                sample_size,
                prior_rating=league_baselines.get("npxg_for") or 1.0,
                prior_strength=prior_match_count,
            ),
            "non_penalty_xg_against": shrink(
                _weighted_mean_from_pairs(buckets.non_penalty_xg_against),
                sample_size,
                prior_rating=league_baselines.get("npxg_against") or 1.0,
                prior_strength=prior_match_count,
            ),
            "average_shot_xg_for": _weighted_mean_from_pairs(buckets.shot_quality_for),
            "average_shot_xg_against": _weighted_mean_from_pairs(
                buckets.shot_quality_against
            ),
        }

    def _venue_strengths_from_buckets(
        self,
        buckets: _ObservationBuckets,
        league_baselines: dict[str, float],
        prior_match_count: int,
    ) -> dict[str, float | None]:
        """Home/away attack and defence strengths from venue-split rates."""
        min_venue = self.config.team_strength_min_venue_matches

        home_attack_baseline = league_baselines.get("home_npxg") or 1.0
        home_defence_baseline = league_baselines.get("away_npxg") or 1.0

        away_attack_baseline = league_baselines.get("away_npxg") or 1.0
        away_defence_baseline = league_baselines.get("home_npxg") or 1.0

        overall_attack = _weighted_mean_from_pairs(buckets.attack_xg)
        overall_defence = _weighted_mean_from_pairs(buckets.defence_xg)
        return {
            "home_attack_strength": self._venue_strength(
                buckets.home_attack_values,
                home_attack_baseline,
                overall_rate=overall_attack,
                prior_match_count=prior_match_count,
                minimum_venue_matches=min_venue,
            ),
            "home_defence_strength": self._venue_strength(
                buckets.home_defence_values,
                home_defence_baseline,
                overall_rate=overall_defence,
                prior_match_count=prior_match_count,
                minimum_venue_matches=min_venue,
            ),
            "away_attack_strength": self._venue_strength(
                buckets.away_attack_values,
                away_attack_baseline,
                overall_rate=overall_attack,
                prior_match_count=prior_match_count,
                minimum_venue_matches=min_venue,
            ),
            "away_defence_strength": self._venue_strength(
                buckets.away_defence_values,
                away_defence_baseline,
                overall_rate=overall_defence,
                prior_match_count=prior_match_count,
                minimum_venue_matches=min_venue,
            ),
        }

    def _goalkeeper_prevention(
        self, buckets: _ObservationBuckets
    ) -> float | None:
        """Goals prevented per match from xGOT faced, shrunk toward zero."""
        if not buckets.xgot_faced_values:
            return None
        prevented_per_match = (
            sum(buckets.xgot_faced_values) - sum(buckets.goals_conceded_values)
        ) / len(buckets.xgot_faced_values)
        return shrink(
            prevented_per_match,
            len(buckets.xgot_faced_values),
            prior_rating=0.0,
            prior_strength=max(1, self.config.goalkeeper_prior_shots // 10),
        )

    @staticmethod
    def _venue_strength(
        venue_rates: list[float],
        league_baseline: float,
        *,
        overall_rate: float | None,
        prior_match_count: int,
        minimum_venue_matches: int,
    ) -> float | None:
        """League-relative venue strength; small samples shrink toward overall."""
        venue_mean = _mean(venue_rates)
        if venue_mean is None or league_baseline <= 0:
            return None
        venue_ratio = venue_mean / league_baseline
        if len(venue_rates) < minimum_venue_matches and overall_rate is not None:
            return shrink(
                venue_ratio,
                len(venue_rates),
                prior_rating=overall_rate / league_baseline,
                prior_strength=max(1, minimum_venue_matches - len(venue_rates)),
            )
        return shrink(
            venue_ratio,
            len(venue_rates),
            prior_rating=1.0,
            prior_strength=prior_match_count,
        )

    def _load_team_match_stats(
        self,
        *,
        team: TeamModel,
        before_date: date,
        venue: Literal["home", "away"] | None,
        lookback_matches: int,
    ) -> list[MatchStatRow]:
        """Load newest-first (match, stats, played_at_home) rows before cutoff."""
        historical_matches = self.historical_repo.find_before_date_by_team(
            team_name=team.name,
            before_date=before_date,
            venue=venue,
            limit=lookback_matches,
        )
        if not historical_matches:
            return []
        return self._attach_advanced_stats(historical_matches, team.name)

    def _attach_advanced_stats(
        self,
        historical_matches: list[HistoricalMatchModel],
        team_name: str,
    ) -> list[MatchStatRow]:
        """Join matches to provider advanced stats; skip matches without stats."""
        advanced_stats_by_match_id = {
            row.match_id: row
            for row in self.stats_repo.list_for_matches(
                [match.id for match in historical_matches], provider=self.provider
            )
        }
        match_stat_rows: list[MatchStatRow] = []
        for historical_match in historical_matches:
            advanced_stats = advanced_stats_by_match_id.get(historical_match.id)
            if advanced_stats is None:
                continue
            match_stat_rows.append(
                (
                    historical_match,
                    advanced_stats,
                    historical_match.home_team == team_name,
                )
            )
        return match_stat_rows

    def _league_averages(
        self, team: TeamModel, before_date: date
    ) -> dict[str, float]:
        """League baselines from matches played before the cutoff (no leakage)."""
        if team.league_id is None:
            return {}
        league_team_names = self.team_repo.get_names_by_league_id(team.league_id)
        if not league_team_names:
            return {}
        league_matches = self.historical_repo.find_before_date_by_team_names(
            team_names=league_team_names,
            before_date=before_date,
            limit=500,
        )
        return self._baselines_from_stats(
            self.stats_repo.list_for_matches(
                [match.id for match in league_matches], provider=self.provider
            )
        )

    @staticmethod
    def _baselines_from_stats(
        advanced_stats_rows: list[MatchAdvancedStatsModel],
    ) -> dict[str, float]:
        """Compute attack/defence/npxG league baselines from advanced stats."""
        home_xg_values = [
            row.home_xg for row in advanced_stats_rows if row.home_xg is not None
        ]
        away_xg_values = [
            row.away_xg for row in advanced_stats_rows if row.away_xg is not None
        ]
        home_npxg = [
            row.home_non_penalty_xg
            for row in advanced_stats_rows
            if row.home_non_penalty_xg is not None
        ]

        away_npxg = [
            row.away_non_penalty_xg
            for row in advanced_stats_rows
            if row.away_non_penalty_xg is not None
        ]

        baselines: dict[str, float] = {}
        if home_xg_values:
            baselines["attack"] = sum(home_xg_values) / len(home_xg_values)
        if away_xg_values:
            baselines["defence"] = sum(away_xg_values) / len(away_xg_values)
        if home_npxg:
            baselines["home_npxg"] = sum(home_npxg) / len(home_npxg)
        if away_npxg:
            baselines["away_npxg"] = sum(away_npxg) / len(away_npxg)
        all_npxg = home_npxg + away_npxg
        if all_npxg:
            baselines["npxg"] = sum(all_npxg) / len(all_npxg)
        return baselines

    @staticmethod
    def _empty_team_features(
        team_id: int,
        before: datetime,
        venue: Literal["home", "away"] | None,
        lookback_matches: int,
    ) -> TeamStrengthFeatures:
        """Empty feature payload when the team or history is missing."""
        return TeamStrengthFeatures(
            team_id=team_id,
            before=before,
            venue=venue,
            lookback_matches=lookback_matches,
            sample_size=0,
            non_penalty_xg_for=None,
            non_penalty_xg_against=None,
            average_shot_xg_for=None,
            average_shot_xg_against=None,
            home_attack_strength=None,
            home_defence_strength=None,
            away_attack_strength=None,
            away_defence_strength=None,
            recency_weighted_attack_rating=None,
            recency_weighted_defence_rating=None,
        )


def _attr(features: TeamStrengthFeatures | None, name: str) -> float | None:
    """Read a float attribute from features, or None when features is missing."""
    if features is None:
        return None
    return getattr(features, name)


def _side_feature_fields(
    home_features: TeamStrengthFeatures | None,
    away_features: TeamStrengthFeatures | None,
) -> dict[str, float | None]:
    """Map home/away TeamStrengthFeatures into MatchStrengthFeatures field values."""
    return {
        "home_npxg_for": _attr(home_features, "non_penalty_xg_for"),
        "home_npxg_against": _attr(home_features, "non_penalty_xg_against"),
        "away_npxg_for": _attr(away_features, "non_penalty_xg_for"),
        "away_npxg_against": _attr(away_features, "non_penalty_xg_against"),
        "home_shot_quality_for": _attr(home_features, "average_shot_xg_for"),
        "home_shot_quality_conceded": _attr(home_features, "average_shot_xg_against"),
        "away_shot_quality_for": _attr(away_features, "average_shot_xg_for"),
        "away_shot_quality_conceded": _attr(away_features, "average_shot_xg_against"),
        "home_attack_strength": _attr(home_features, "home_attack_strength"),
        "home_defence_strength": _attr(home_features, "home_defence_strength"),
        "away_attack_strength": _attr(away_features, "away_attack_strength"),
        "away_defence_strength": _attr(away_features, "away_defence_strength"),
        "home_opponent_adjusted_attack": _attr(
            home_features, "opponent_adjusted_attack_strength"
        ),
        "home_opponent_adjusted_defence": _attr(
            home_features, "opponent_adjusted_defence_strength"
        ),
        "away_opponent_adjusted_attack": _attr(
            away_features, "opponent_adjusted_attack_strength"
        ),
        "away_opponent_adjusted_defence": _attr(
            away_features, "opponent_adjusted_defence_strength"
        ),
        "home_set_piece_attack": _attr(home_features, "set_piece_xg_for"),
        "home_set_piece_defence": _attr(home_features, "set_piece_xg_against"),
        "away_set_piece_attack": _attr(away_features, "set_piece_xg_for"),
        "away_set_piece_defence": _attr(away_features, "set_piece_xg_against"),
        "home_goalkeeper_prevention": _attr(
            home_features, "goalkeeper_goals_prevented"
        ),
        "away_goalkeeper_prevention": _attr(
            away_features, "goalkeeper_goals_prevented"
        ),
    }
