"""Historical team strength features from stored advanced stats (no leakage)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from database import SessionLocal
from objects.models.historical_match import HistoricalMatchModel
from objects.models.match_advanced_stats import MatchAdvancedStatsModel
from objects.models.team import TeamModel
from objects.repositories.match_advanced_stats_repository import (
    MatchAdvancedStatsRepository,
)
from objects.repositories.team_repository import TeamRepository
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.data_classes.team_strength_features import TeamStrengthFeatures


class TeamStrengthFeatureService:
    """Calculate pre-cutoff team features from persisted match advanced stats."""

    def __init__(
        self,
        session: Session | None = None,
        config: DataSourceConfig | None = None,
        provider: str | None = None,
    ) -> None:
        self._owns_session = session is None
        self.session = session or SessionLocal()
        self.config = DataSourceConfig()
        self.provider = provider or self.config.football_data_provider
        self.stats_repo = MatchAdvancedStatsRepository(self.session)
        self.team_repo = TeamRepository(self.session)

    def close(self) -> None:
        if self._owns_session:
            self.session.close()

    def calculate_features(
        self,
        team_id: int,
        before: datetime,
        venue: Literal["home", "away"] | None = None,
        lookback_matches: int = 20,
        decay: float = 0.9,
    ) -> TeamStrengthFeatures:
        team = self.team_repo.get(team_id)
        if team is None:
            return self._empty(team_id, before, venue, lookback_matches)

        before_date = before.date() if hasattr(before, "date") else before
        rows = self._load_team_match_stats(
            team=team,
            before_date=before_date,
            venue=venue,
            lookback_matches=lookback_matches,
        )
        if not rows:
            return self._empty(team_id, before, venue, lookback_matches)

        league_avgs = self._league_averages(team, before_date)
        prior = max(1, self.config.football_data_feature_shrinkage_prior_matches)

        npxg_for_vals: list[float] = []
        npxg_against_vals: list[float] = []
        avg_shot_for_vals: list[float] = []
        avg_shot_against_vals: list[float] = []
        set_for_vals: list[float] = []
        set_against_vals: list[float] = []
        attack_weights: list[tuple[float, float]] = []
        defence_weights: list[tuple[float, float]] = []
        xgot_faced_vals: list[float] = []
        goals_conceded_vals: list[float] = []

        home_attack: list[float] = []
        home_defence: list[float] = []
        away_attack: list[float] = []
        away_defence: list[float] = []

        for index, (match, stats, is_home) in enumerate(rows):
            weight = decay**index
            if is_home:
                for_npxg = stats.home_non_penalty_xg
                against_npxg = stats.away_non_penalty_xg
                for_avg = stats.average_home_shot_xg
                against_avg = stats.average_away_shot_xg
                for_set = stats.home_set_piece_xg
                against_set = stats.away_set_piece_xg
                xgot_faced = stats.away_xgot
                goals_conceded = float(match.away_goals)
                attack_raw = stats.home_xg_from_shots if stats.home_xg_from_shots is not None else stats.home_xg
                defence_raw = stats.away_xg_from_shots if stats.away_xg_from_shots is not None else stats.away_xg
            else:
                for_npxg = stats.away_non_penalty_xg
                against_npxg = stats.home_non_penalty_xg
                for_avg = stats.average_away_shot_xg
                against_avg = stats.average_home_shot_xg
                for_set = stats.away_set_piece_xg
                against_set = stats.home_set_piece_xg
                xgot_faced = stats.home_xgot
                goals_conceded = float(match.home_goals)
                attack_raw = stats.away_xg_from_shots if stats.away_xg_from_shots is not None else stats.away_xg
                defence_raw = stats.home_xg_from_shots if stats.home_xg_from_shots is not None else stats.home_xg

            if for_npxg is not None:
                npxg_for_vals.append(for_npxg)
            if against_npxg is not None:
                npxg_against_vals.append(against_npxg)
            if for_avg is not None:
                avg_shot_for_vals.append(for_avg)
            if against_avg is not None:
                avg_shot_against_vals.append(against_avg)
            if for_set is not None:
                set_for_vals.append(for_set)
            if against_set is not None:
                set_against_vals.append(against_set)
            if xgot_faced is not None:
                xgot_faced_vals.append(xgot_faced)
                goals_conceded_vals.append(goals_conceded)
            if attack_raw is not None:
                attack_weights.append((attack_raw, weight))
                if is_home:
                    home_attack.append(attack_raw)
                else:
                    away_attack.append(attack_raw)
            if defence_raw is not None:
                defence_weights.append((defence_raw, weight))
                if is_home:
                    home_defence.append(defence_raw)
                else:
                    away_defence.append(defence_raw)

        sample = len(rows)
        shrink = sample / (sample + prior)

        def _mean(values: list[float]) -> float | None:
            if not values:
                return None
            return sum(values) / len(values)

        def _shrink_to_league(value: float | None, league_key: str) -> float | None:
            if value is None:
                return None
            league_avg = league_avgs.get(league_key)
            if league_avg is None:
                return value
            return shrink * value + (1 - shrink) * league_avg

        def _weighted(pairs: list[tuple[float, float]]) -> float | None:
            if not pairs:
                return None
            total_w = sum(weight for _, weight in pairs)
            if total_w <= 0:
                return None
            return sum(value * weight for value, weight in pairs) / total_w

        goals_prevented = None
        if xgot_faced_vals:
            raw = sum(xgot_faced_vals) - sum(goals_conceded_vals)
            # Per-match average goals prevented, shrunk toward 0 league baseline.
            per_match = raw / len(xgot_faced_vals)
            goals_prevented = shrink * per_match

        league_attack = league_avgs.get("attack") or 1.0
        league_defence = league_avgs.get("defence") or 1.0

        def _strength(values: list[float], league_avg: float) -> float | None:
            mean = _mean(values)
            if mean is None or league_avg <= 0:
                return None
            return _shrink_to_league(mean / league_avg, "attack" if league_avg == league_attack else "defence")

        # Opponent adjustment reserved; configurable method currently "none" or "simple".
        if self.config.football_data_opponent_adjustment == "simple":
            # Simple: divide by opponent average when available (same league avg proxy).
            pass

        return TeamStrengthFeatures(
            team_id=team_id,
            before=before,
            venue=venue,
            lookback_matches=lookback_matches,
            sample_size=sample,
            non_penalty_xg_for=_shrink_to_league(_mean(npxg_for_vals), "npxg_for"),
            non_penalty_xg_against=_shrink_to_league(
                _mean(npxg_against_vals), "npxg_against"
            ),
            average_shot_xg_for=_mean(avg_shot_for_vals),
            average_shot_xg_against=_mean(avg_shot_against_vals),
            home_attack_strength=_strength(home_attack, league_attack),
            home_defence_strength=_strength(home_defence, league_defence),
            away_attack_strength=_strength(away_attack, league_attack),
            away_defence_strength=_strength(away_defence, league_defence),
            recency_weighted_attack_rating=_weighted(attack_weights),
            recency_weighted_defence_rating=_weighted(defence_weights),
            set_piece_xg_for=_mean(set_for_vals),
            set_piece_xg_against=_mean(set_against_vals),
            goalkeeper_goals_prevented=goals_prevented,
        )

    def _load_team_match_stats(
        self,
        *,
        team: TeamModel,
        before_date,
        venue: Literal["home", "away"] | None,
        lookback_matches: int,
    ) -> list[tuple[HistoricalMatchModel, MatchAdvancedStatsModel, bool]]:
        name = team.name
        conditions = []
        if venue == "home":
            conditions.append(HistoricalMatchModel.home_team == name)
        elif venue == "away":
            conditions.append(HistoricalMatchModel.away_team == name)
        else:
            conditions.append(
                or_(
                    HistoricalMatchModel.home_team == name,
                    HistoricalMatchModel.away_team == name,
                )
            )

        matches = list(
            self.session.scalars(
                select(HistoricalMatchModel)
                .where(
                    HistoricalMatchModel.match_date < before_date,
                    *conditions,
                )
                .order_by(HistoricalMatchModel.match_date.desc())
                .limit(lookback_matches)
            ).all()
        )
        if not matches:
            return []

        match_ids = [match.id for match in matches]
        stats_rows = self.stats_repo.list_for_matches(match_ids, provider=self.provider)
        stats_by_match = {row.match_id: row for row in stats_rows}

        result: list[tuple[HistoricalMatchModel, MatchAdvancedStatsModel, bool]] = []
        for match in matches:
            stats = stats_by_match.get(match.id)
            if stats is None:
                continue
            is_home = match.home_team == name
            result.append((match, stats, is_home))
        return result

    def _league_averages(self, team: TeamModel, before_date) -> dict[str, float]:
        if team.league_id is None:
            return {}
        # Approximate league averages from recent stored stats for teams in league.
        team_names = list(
            self.session.scalars(
                select(TeamModel.name).where(TeamModel.league_id == team.league_id)
            ).all()
        )
        if not team_names:
            return {}
        matches = list(
            self.session.scalars(
                select(HistoricalMatchModel)
                .where(
                    HistoricalMatchModel.match_date < before_date,
                    or_(
                        HistoricalMatchModel.home_team.in_(team_names),
                        HistoricalMatchModel.away_team.in_(team_names),
                    ),
                )
                .order_by(HistoricalMatchModel.match_date.desc())
                .limit(500)
            ).all()
        )
        stats = self.stats_repo.list_for_matches(
            [match.id for match in matches], provider=self.provider
        )
        home_xg = [row.home_xg for row in stats if row.home_xg is not None]
        away_xg = [row.away_xg for row in stats if row.away_xg is not None]
        npxg = [
            value
            for row in stats
            for value in (row.home_non_penalty_xg, row.away_non_penalty_xg)
            if value is not None
        ]
        averages: dict[str, float] = {}
        if home_xg:
            averages["attack"] = sum(home_xg) / len(home_xg)
        if away_xg:
            averages["defence"] = sum(away_xg) / len(away_xg)
        if npxg:
            averages["npxg_for"] = sum(npxg) / len(npxg)
            averages["npxg_against"] = averages["npxg_for"]
        return averages

    @staticmethod
    def _empty(
        team_id: int,
        before: datetime,
        venue: Literal["home", "away"] | None,
        lookback_matches: int,
    ) -> TeamStrengthFeatures:
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
            set_piece_xg_for=None,
            set_piece_xg_against=None,
            goalkeeper_goals_prevented=None,
        )
