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
from calc.strength_helpers import (
    WeightedObservation,
    append_observation,
    baselines_from_stats,
    normalize_strength,
    recency_weights,
    shrink,
    weighted_mean_from_pairs,
)

# (xg, shots, recency_weight) for average shot xG = sum(w*xg) / sum(w*shots).
ShotQualityObservation = tuple[float, float, float]
MatchStatRow = tuple[HistoricalMatchModel, MatchAdvancedStatsModel, bool]


@dataclass
class _MatchSideMetrics:
    """Team-perspective metrics extracted from one historical match."""

    opponent_team_name: str
    match_date: date
    non_penalty_xg_for: float | None
    non_penalty_xg_against: float | None
    shots_for: int | None
    shots_against: int | None
    set_piece_xg_for: float | None
    set_piece_xg_against: float | None
    goalkeeper_xgot_faced: float | None
    goals_conceded: float
    attack_xg: float | None
    defence_xg: float | None
    played_at_home: bool
    shots_on_target_faced: int | None


@dataclass
class _ObservationBuckets:
    """Accumulators for weighted team metrics across lookback matches."""

    non_penalty_xg_for: list[WeightedObservation] = field(default_factory=list)
    non_penalty_xg_against: list[WeightedObservation] = field(default_factory=list)
    shot_quality_for: list[ShotQualityObservation] = field(default_factory=list)
    shot_quality_against: list[ShotQualityObservation] = field(default_factory=list)
    set_piece_xg_for: list[WeightedObservation] = field(default_factory=list)
    set_piece_xg_against: list[WeightedObservation] = field(default_factory=list)
    attack_xg: list[WeightedObservation] = field(default_factory=list)
    defence_xg: list[WeightedObservation] = field(default_factory=list)
    opponent_adjusted_attack: list[WeightedObservation] = field(default_factory=list)
    opponent_adjusted_defence: list[WeightedObservation] = field(default_factory=list)
    home_attack_values: list[WeightedObservation] = field(default_factory=list)
    home_defence_values: list[WeightedObservation] = field(default_factory=list)
    away_attack_values: list[WeightedObservation] = field(default_factory=list)
    away_defence_values: list[WeightedObservation] = field(default_factory=list)
    xgot_faced_values: list[WeightedObservation] = field(default_factory=list)
    goals_conceded_values: list[WeightedObservation] = field(default_factory=list)
    shots_on_target_faced_values: list[WeightedObservation] = field(
        default_factory=list
    )


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


def _weighted_shot_quality(
    observations: list[ShotQualityObservation],
) -> float | None:
    """Average shot xG: sum(weight * xg) / sum(weight * shots)."""
    if not observations:
        return None
    weighted_xg = sum(weight * xg for xg, _shots, weight in observations)
    weighted_shots = sum(weight * shots for _xg, shots, weight in observations)
    if weighted_shots <= 0:
        return None
    return weighted_xg / weighted_shots


def _extract_side_metrics(
    historical_match: HistoricalMatchModel,
    advanced_stats: MatchAdvancedStatsModel,
    played_at_home: bool,
) -> _MatchSideMetrics:
    """Map home/away advanced-stats columns onto the team's perspective."""
    if played_at_home:
        return _MatchSideMetrics(
            opponent_team_name=historical_match.away_team,
            match_date=historical_match.match_date,
            non_penalty_xg_for=advanced_stats.home_non_penalty_xg,
            non_penalty_xg_against=advanced_stats.away_non_penalty_xg,
            shots_for=advanced_stats.home_shots,
            shots_against=advanced_stats.away_shots,
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
            shots_on_target_faced=advanced_stats.away_shots_on_target,
        )
    return _MatchSideMetrics(
        opponent_team_name=historical_match.home_team,
        match_date=historical_match.match_date,
        non_penalty_xg_for=advanced_stats.away_non_penalty_xg,
        non_penalty_xg_against=advanced_stats.home_non_penalty_xg,
        shots_for=advanced_stats.away_shots,
        shots_against=advanced_stats.home_shots,
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
        shots_on_target_faced=advanced_stats.home_shots_on_target,
    )


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
            venue=None,
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
            league_baselines=self.league_averages(team, before_date),
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
            match_date=historical_match.match_date,
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
        match_date: date,
    ) -> MatchStrengthFeatures:
        """Build MatchStrengthFeatures from side features and Dixon–Coles."""
        league_home, league_away = self._league_goal_rates(home_team, match_date)
        league_npxg = (
            self.league_averages(home_team, match_date).get("npxg")
            if home_team is not None
            else None
        )
        expected_home, expected_away = self._match_expected_goals(
            home_features,
            away_features,
            league_home,
            league_away,
            league_npxg,
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
        self,
        home_team: TeamModel | None,
        before_date: date,
    ) -> tuple[float, float]:
        """League home/away goal baselines with safe defaults."""
        league_home_goal_rate = 1.35
        league_away_goal_rate = 1.20
        if home_team is None or home_team.league_id is None:
            return league_home_goal_rate, league_away_goal_rate
        try:
            league_home_goal_rate = float(
                self.historical_repo.get_home_goal_average_by_league_before_date(
                    home_team.league_id,
                    before_date
                )
            )
            league_away_goal_rate = float(
                self.historical_repo.get_away_goal_average_by_league_before_date(
                    home_team.league_id,
                    before_date

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
        league_npxg: float | None,
    ) -> tuple[float | None, float | None]:
        """Resolve side strengths then compute λ_home / λ_away."""
        return expected_goals_from_strengths(
            self._side_attack_strength(home_features, "home", league_npxg),
            self._side_defence_strength(away_features, "away", league_npxg),
            self._side_attack_strength(away_features, "away", league_npxg),
            self._side_defence_strength(home_features, "home", league_npxg),
            league_home_goal_rate,
            league_away_goal_rate,
        )

    def _side_attack_strength(
        self,
        features: TeamStrengthFeatures | None,
        venue: Literal["home", "away"],
        league_npxg: float | None,
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
            league_npxg=league_npxg,
        )

    def _side_defence_strength(
        self,
        features: TeamStrengthFeatures | None,
        venue: Literal["home", "away"],
        league_npxg: float | None,
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
            league_npxg=league_npxg,
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
                f"Set-piece attack/defence: {features.set_piece_attack_strength} / {features.set_piece_defence_strength}",
                f"Goalkeeper prevention rating: {features.goalkeeper_prevention_rating}",
                f"Goalkeeper goals prevented: {features.goalkeeper_goals_prevented}",
            ]
        )

    @staticmethod
    def _prefer_venue_or_recency_strength(
        *,
        venue_strength: float | None,
        recency_rating: float | None,
        league_npxg: float | None,
    ) -> float | None:
        """Prefer league-relative venue strength; else normalize raw recency npxG."""
        if venue_strength is not None:
            return venue_strength
        if recency_rating is None:
            return None
        return normalize_strength(recency_rating, league_npxg)

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
        buckets = self._collect_observations(match_stat_rows)
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
            )
        return buckets

    def _accumulate_match_metrics(
        self,
        buckets: _ObservationBuckets,
        metrics: _MatchSideMetrics,
        match_weight: float,
    ) -> None:
        """Add one match's metrics into the observation buckets."""
        append_observation(
            buckets.non_penalty_xg_for, metrics.non_penalty_xg_for, match_weight
        )
        append_observation(
            buckets.non_penalty_xg_against, metrics.non_penalty_xg_against, match_weight
        )
        append_observation(
            buckets.set_piece_xg_for, metrics.set_piece_xg_for, match_weight
        )
        append_observation(
            buckets.set_piece_xg_against, metrics.set_piece_xg_against, match_weight
        )
        self._accumulate_shot_quality(buckets, metrics, match_weight)
        self._accumulate_attack_defence(buckets, metrics, match_weight)
        self._accumulate_opponent_adjustment(buckets, metrics, match_weight)
        if (
            metrics.goalkeeper_xgot_faced is not None
            and metrics.shots_on_target_faced is not None
        ):
            buckets.xgot_faced_values.append(
                (metrics.goalkeeper_xgot_faced, match_weight)
            )
            buckets.goals_conceded_values.append(
                (metrics.goals_conceded, match_weight)
            )
            buckets.shots_on_target_faced_values.append(
                (float(metrics.shots_on_target_faced), match_weight)
            )

    @staticmethod
    def _accumulate_shot_quality(
        buckets: _ObservationBuckets,
        metrics: _MatchSideMetrics,
        match_weight: float,
    ) -> None:
        """Accumulate (xg, shots, weight) for weighted average shot xG."""
        if metrics.attack_xg is not None and metrics.shots_for is not None:
            buckets.shot_quality_for.append(
                (metrics.attack_xg, float(metrics.shots_for), match_weight)
            )
        if metrics.defence_xg is not None and metrics.shots_against is not None:
            buckets.shot_quality_against.append(
                (metrics.defence_xg, float(metrics.shots_against), match_weight)
            )

    @staticmethod
    def _accumulate_attack_defence(
        buckets: _ObservationBuckets,
        metrics: _MatchSideMetrics,
        match_weight: float,
    ) -> None:
        """Accumulate attack/defence xG and venue splits."""
        if metrics.non_penalty_xg_for is not None:
            buckets.attack_xg.append((metrics.non_penalty_xg_for, match_weight))
            if metrics.played_at_home:
                buckets.home_attack_values.append((metrics.non_penalty_xg_for, match_weight))
            else:
                buckets.away_attack_values.append((metrics.non_penalty_xg_for, match_weight))
        if metrics.non_penalty_xg_against is not None:
            buckets.defence_xg.append((metrics.non_penalty_xg_against, match_weight))
            if metrics.played_at_home:
                buckets.home_defence_values.append((metrics.non_penalty_xg_against, match_weight))
            else:
                buckets.away_defence_values.append((metrics.non_penalty_xg_against, match_weight))

    def _get_opponent_strength_before(
            self,
            opponent_team_name: str,
            before_date: date,
    ) -> tuple[float | None, float | None, float | None]:
        """Return opponent attack/defence and league npxG using only earlier matches."""

        team = self.team_repo.get_by_name(opponent_team_name)
        if team is None:
            return None, None, None

        matches = self.historical_repo.find_before_date_by_team(
            team_name=opponent_team_name,
            before_date=before_date,
            venue=None,
            limit=self.config.team_strength_lookback_matches,
        )
        if not matches:
            return None, None, None

        rows = self.attach_advanced_stats(matches, opponent_team_name)
        if not rows:
            return None, None, None

        weights = recency_weights(
            len(rows),
            self.config.team_strength_recency_decay,
        )

        attack: list[WeightedObservation] = []
        defence: list[WeightedObservation] = []

        for weight, (match, stats, played_at_home) in zip(weights, rows):
            metrics = _extract_side_metrics(match, stats, played_at_home)

            append_observation(
                attack,
                metrics.non_penalty_xg_for,
                weight,
            )
            append_observation(
                defence,
                metrics.non_penalty_xg_against,
                weight,
            )

        attack_npxg = weighted_mean_from_pairs(attack)
        defence_npxg = weighted_mean_from_pairs(defence)

        baselines = self.league_averages(team, before_date)

        league_npxg = baselines.get("npxg")
        if league_npxg is None or league_npxg <= 0:
            return None, None, None

        prior_match_count = self.config.team_strength_prior_matches or (
            self.config.football_data_feature_shrinkage_prior_matches
        )
        return (
            shrink(
                normalize_strength(attack_npxg, league_npxg),
                len(attack),
                prior_rating=1.0,
                prior_strength=prior_match_count,
            ),
            shrink(
                normalize_strength(defence_npxg, league_npxg),
                len(defence),
                prior_rating=1.0,
                prior_strength=prior_match_count,
            ),
            league_npxg,
        )

    def _accumulate_opponent_adjustment(
        self,
        buckets: _ObservationBuckets,
        metrics: _MatchSideMetrics,
        match_weight: float,
    ) -> None:
        """Optionally accumulate simple opponent-adjusted attack/defence."""
        opponent_attack, opponent_defence, league_npxg = (
            self._get_opponent_strength_before(
                metrics.opponent_team_name,
                metrics.match_date,
            )
        )

        if (
                metrics.non_penalty_xg_for is not None
                and opponent_defence is not None
                and league_npxg
        ):
            expected_conceded = league_npxg * opponent_defence

            buckets.opponent_adjusted_attack.append(
                (
                    metrics.non_penalty_xg_for / expected_conceded,
                    match_weight,
                )
            )

        if (
                metrics.non_penalty_xg_against is not None
                and opponent_attack is not None
                and league_npxg
        ):
            expected_created = league_npxg * opponent_attack

            buckets.opponent_adjusted_defence.append(
                (
                    metrics.non_penalty_xg_against / expected_created,
                    match_weight,
                )
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
            **self._core_metrics_from_buckets(buckets),
            **self._venue_strengths_from_buckets(
                buckets, league_baselines, prior_match_count
            ),
            **self._advanced_metrics_from_buckets(
                buckets, league_baselines, prior_match_count
            ),
        )

    def _advanced_metrics_from_buckets(
        self,
        buckets: _ObservationBuckets,
        league_baselines: dict[str, float],
        prior_match_count: int,
    ) -> dict[str, float | bool | None]:
        """Recency, opponent-adjusted, set-piece, GK, and coverage flags."""
        league_set_piece_xg = league_baselines.get("set_piece_xg")
        return {
            "recency_weighted_attack_rating": weighted_mean_from_pairs(
                buckets.attack_xg
            ),
            "recency_weighted_defence_rating": weighted_mean_from_pairs(
                buckets.defence_xg
            ),
            "opponent_adjusted_attack_strength": shrink(
                weighted_mean_from_pairs(buckets.opponent_adjusted_attack),
                len(buckets.opponent_adjusted_attack),
                prior_rating=1.0,
                prior_strength=prior_match_count,
            ),
            "opponent_adjusted_defence_strength": shrink(
                weighted_mean_from_pairs(buckets.opponent_adjusted_defence),
                len(buckets.opponent_adjusted_defence),
                prior_rating=1.0,
                prior_strength=prior_match_count,
            ),
            "set_piece_attack_strength": shrink(
                normalize_strength(
                    weighted_mean_from_pairs(buckets.set_piece_xg_for),
                    league_set_piece_xg,
                ),
                len(buckets.set_piece_xg_for),
                prior_rating=1.0,
                prior_strength=prior_match_count,
            ),
            "set_piece_defence_strength": shrink(
                normalize_strength(
                    weighted_mean_from_pairs(buckets.set_piece_xg_against),
                    league_set_piece_xg,
                ),
                len(buckets.set_piece_xg_against),
                prior_rating=1.0,
                prior_strength=prior_match_count,
            ),
            **self._goalkeeper_feature_fields(buckets),
            "has_xg_data": bool(buckets.non_penalty_xg_for or buckets.attack_xg),
            "has_xgot_data": bool(buckets.xgot_faced_values),
            "has_set_piece_data": bool(
                buckets.set_piece_xg_for or buckets.set_piece_xg_against
            ),
        }

    def _goalkeeper_feature_fields(
        self, buckets: _ObservationBuckets
    ) -> dict[str, float | None]:
        """Build goalkeeper rating and goals-prevented feature fields."""
        prevention_rating, goals_prevented = self._goalkeeper_metrics(buckets)
        return {
            "goalkeeper_prevention_rating": prevention_rating,
            "goalkeeper_goals_prevented": goals_prevented,
        }

    def _core_metrics_from_buckets(
        self,
        buckets: _ObservationBuckets,
    ) -> dict[str, float | None]:
        """npxG and shot-quality fields derived from observation buckets."""
        return {
            "non_penalty_xg_for": weighted_mean_from_pairs(
                buckets.non_penalty_xg_for
            ),
            "non_penalty_xg_against": weighted_mean_from_pairs(
                buckets.non_penalty_xg_against
            ),
            "average_shot_xg_for": _weighted_shot_quality(buckets.shot_quality_for),
            "average_shot_xg_against": _weighted_shot_quality(
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

        overall_attack = weighted_mean_from_pairs(buckets.attack_xg)
        overall_defence = weighted_mean_from_pairs(buckets.defence_xg)
        overall_npxg = league_baselines.get("npxg")
        overall_attack_strength = (
            overall_attack / overall_npxg
            if overall_attack is not None and overall_npxg and overall_npxg > 0
            else None
        )
        overall_defence_strength = (
            overall_defence / overall_npxg
            if overall_defence is not None and overall_npxg and overall_npxg > 0
            else None
        )
        return {
            "home_attack_strength": self._venue_strength(
                buckets.home_attack_values,
                home_attack_baseline,
                overall_strength=overall_attack_strength,
                prior_match_count=prior_match_count,
                minimum_venue_matches=min_venue,
            ),
            "home_defence_strength": self._venue_strength(
                buckets.home_defence_values,
                home_defence_baseline,
                overall_strength=overall_defence_strength,
                prior_match_count=prior_match_count,
                minimum_venue_matches=min_venue,
            ),
            "away_attack_strength": self._venue_strength(
                buckets.away_attack_values,
                away_attack_baseline,
                overall_strength=overall_attack_strength,
                prior_match_count=prior_match_count,
                minimum_venue_matches=min_venue,
            ),
            "away_defence_strength": self._venue_strength(
                buckets.away_defence_values,
                away_defence_baseline,
                overall_strength=overall_defence_strength,
                prior_match_count=prior_match_count,
                minimum_venue_matches=min_venue,
            ),
        }

    def _goalkeeper_metrics(
        self, buckets: _ObservationBuckets
    ) -> tuple[float | None, float | None]:
        """Return (prevention_rating per shot, weighted goals prevented)."""
        if not buckets.xgot_faced_values:
            return None, None

        weighted_xgot = sum(
            value * weight for value, weight in buckets.xgot_faced_values
        )
        weighted_goals = sum(
            value * weight for value, weight in buckets.goals_conceded_values
        )
        weighted_shots = sum(
            value * weight for value, weight in buckets.shots_on_target_faced_values
        )

        if weighted_shots <= 0:
            return None, None

        goals_prevented = weighted_xgot - weighted_goals
        prevention_rating = shrink(
            goals_prevented / weighted_shots,
            max(1, int(round(weighted_shots))),
            prior_rating=0.0,
            prior_strength=self.config.goalkeeper_prior_shots,
        )
        return prevention_rating, goals_prevented

    @staticmethod
    def _venue_strength(
        venue_rates: list[WeightedObservation],
        league_baseline: float,
        *,
        overall_strength: float | None,
        prior_match_count: int,
        minimum_venue_matches: int,
    ) -> float | None:
        """League-relative venue strength; small samples shrink toward overall."""
        venue_mean = weighted_mean_from_pairs(venue_rates)
        if venue_mean is None or league_baseline <= 0:
            return None
        venue_ratio = venue_mean / league_baseline
        prior_rating = (
            overall_strength
            if len(venue_rates) < minimum_venue_matches
            and overall_strength is not None
            else 1.0
        )
        return shrink(
            venue_ratio,
            len(venue_rates),
            prior_rating=prior_rating,
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
        return self.attach_advanced_stats(historical_matches, team.name)

    def attach_advanced_stats(
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

    def league_averages(
        self, team: TeamModel, before_date: date, season: str | None = None
    ) -> dict[str, float]:
        """League baselines from league matches played before the cutoff (no leakage)."""
        if team.league_id is None:
            return {}
        return self.league_averages_by_league_id(
            team.league_id, before_date, season=season
        )

    def league_averages_by_league_id(
        self,
        league_id: int,
        before_date: date,
        season: str | None = None,
    ) -> dict[str, float]:
        """League npxG baselines for ``league_id`` (optionally one season) before cutoff."""
        league_matches = self.historical_repo.find_before_date_by_league_id(
            league_id=league_id,
            before_date=before_date,
            season=season,
            limit=500,
        )
        if not league_matches:
            return {}
        stats_by_match_id = {
            row.match_id: row
            for row in self.stats_repo.list_for_matches(
                [match.id for match in league_matches], provider=self.provider
            )
        }
        # Preserve newest-first order so recency weights align with match dates.
        ordered_stats = [
            stats_by_match_id[match.id]
            for match in league_matches
            if match.id in stats_by_match_id
        ]
        return baselines_from_stats(
            ordered_stats,
            decay=self.config.team_strength_recency_decay,
        )

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
        "home_set_piece_attack": _attr(home_features, "set_piece_attack_strength"),
        "home_set_piece_defence": _attr(home_features, "set_piece_defence_strength"),
        "away_set_piece_attack": _attr(away_features, "set_piece_attack_strength"),
        "away_set_piece_defence": _attr(away_features, "set_piece_defence_strength"),
        "home_goalkeeper_prevention": _attr(
            home_features, "goalkeeper_prevention_rating"
        ),
        "away_goalkeeper_prevention": _attr(
            away_features, "goalkeeper_prevention_rating"
        ),
    }
