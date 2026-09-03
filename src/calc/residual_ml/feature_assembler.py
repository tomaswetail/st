"""Assemble residual ML features from existing calc modules."""

from __future__ import annotations

from datetime import date, datetime
from math import exp

from sqlalchemy.orm import Session

from calc.balance_and_environment import BalanceAndEnvironment
from calc.dixon_coles.model import DixonColesModel, DixonColesPrediction
from calc.dixon_coles.service import DixonColesService
from calc.league_behavior_calculator import LeagueBehaviorCalculator
from calc.market_probabilities import MarketProbabilities
from calc.player_availability_calculator import PlayerAvailabilityCalculator
from calc.rest_congestion_calculator import RestCongestionCalculator
from calc.strength_calculator import StrengthCalculator
from objects.models.st_match import STMatchModel
from objects.repositories.fixture_repository import FixtureRepository
from objects.repositories.league_repository import LeagueRepository
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.data_classes.residual_ml_features import ResidualMLFeatures
from objects.schema.data_classes.team_strength_features import MatchStrengthFeatures
from objects.schema.db.st_match_odds import STMatchOdds
from objects.schema.db.team import Team
from utils.common import ensure_unit_probabilities


class ResidualMLFeatureAssembler:
    """Build ResidualMLFeatures for one Stryktipset coupon match."""

    def __init__(
        self,
        session: Session,
        config: DataSourceConfig | None = None,
    ) -> None:
        self.session = session
        self.config = config or DataSourceConfig()
        self.fixture_repo = FixtureRepository(session)
        self.league_repo = LeagueRepository(session)
        self.strength_calculator = StrengthCalculator(session, config=self.config)
        self.balance_calculator = BalanceAndEnvironment(
            session,
            config=self.config,
            strength_calculator=self.strength_calculator,
        )
        self.league_behavior_calculator = LeagueBehaviorCalculator(
            session, config=self.config
        )
        self.rest_calculator = RestCongestionCalculator(session, config=self.config)
        self.availability_calculator = PlayerAvailabilityCalculator(
            session, config=self.config
        )
        # Reuse the strength calculator's HA instance so process() cache is shared.
        self.home_advantage_calculator = (
            self.strength_calculator.home_advantage_calculator()
        )
        self.dixon_coles_service = DixonColesService(session, config=self.config)
        self._league_external_id_cache: dict[str, int | None] = {}
        self._team_league_id_cache: dict[int, int | None] = {}
        self._team_fixtures_cache: dict[
            tuple[str, date, int], list
        ] = {}
        self._classic_dc_fit_cache: dict[
            tuple[int, date], DixonColesModel | None
        ] = {}
        self._classic_dc_fallback_logged: set[tuple[int, date]] = set()

    def clear_caches(self) -> None:
        """Drop lookback caches across nested calculators."""
        self._team_fixtures_cache.clear()
        self._classic_dc_fit_cache.clear()
        self._classic_dc_fallback_logged.clear()
        self.strength_calculator.clear_caches()
        self.league_behavior_calculator.clear_caches()
        self.rest_calculator.clear_caches()
        self.availability_calculator.clear_caches()
        self.home_advantage_calculator.clear_caches()

    def assemble(
        self,
        match: STMatchModel,
        *,
        draw_number: int | None = None,
        event_number: int | None = None,
    ) -> ResidualMLFeatures:
        if match.home_team is None or match.away_team is None:
            raise ValueError(f"Missing team on match id={match.id}")
        if match.start_time is None:
            raise ValueError(f"Missing start_time on match id={match.id}")

        cutoff = (
            match.start_time.date()
            if isinstance(match.start_time, datetime)
            else match.start_time
        )
        target_league_external_id = self._resolve_league_external_id(match)

        market_probs = self._market_probabilities(match)
        home_advantage_log, home_advantage_coefficient = self._home_advantage(
            match,
            cutoff,
            target_league_external_id=target_league_external_id,
        )
        strength = self.strength_calculator.get_fixture_features(
            match.home_team_id,
            match.away_team_id,
            match.start_time,
            match_id=match.id,
            target_league_external_id=target_league_external_id,
            home_advantage_coefficient=home_advantage_coefficient,
        )
        history_fixtures = self._load_balance_fixtures(match, cutoff)
        balance = self.balance_calculator.calculate(
            match,
            history_fixtures,
            market_probs,
            strength=strength,
        )
        league_behavior = self.league_behavior_calculator.calculate(match)
        home_previous, away_previous = self.rest_calculator.previous_fixtures(match)
        rest_seed = self.rest_calculator.calculate(match)
        availability = self.availability_calculator.calculate(
            match,
            favourite_strength=balance.favourite_strength,
            home_short_rest=rest_seed.home_short_rest,
            away_short_rest=rest_seed.away_short_rest,
            home_previous_fixture=home_previous,
            away_previous_fixture=away_previous,
        )
        rest = self.rest_calculator.calculate(match, availability=availability)

        league_avg_npxg = self._league_avg_npxg(match, cutoff)

        upset_rate = (
            1.0 - league_behavior.league_favourite_win_rate
            if league_behavior.league_favourite_win_rate is not None
            else None
        )
        congestion_difference = None
        if rest.home_congestion is not None and rest.away_congestion is not None:
            congestion_difference = rest.home_congestion - rest.away_congestion

        expected_home_goals, expected_away_goals, p_home_dc, p_draw_dc, p_away_dc = (
            self._engine_probabilities(match, cutoff, target_league_external_id, strength)
        )
        market_vs_dc_home = self._market_vs_dc(
            market_probs.get("1"), p_home_dc
        )
        market_vs_dc_draw = self._market_vs_dc(
            market_probs.get("X"), p_draw_dc
        )
        market_vs_dc_away = self._market_vs_dc(
            market_probs.get("2"), p_away_dc
        )

        return ResidualMLFeatures(
            match_id=match.id,
            draw_number=draw_number if draw_number is not None else match.stryktipset_round_id,
            feature_cutoff_date=cutoff,
            league_external_id=target_league_external_id,
            p_home_market=market_probs.get("1"),
            p_draw_market=market_probs.get("X"),
            p_away_market=market_probs.get("2"),
            home_npxg_for=strength.home_npxg_for,
            home_npxg_against=strength.home_npxg_against,
            away_npxg_for=strength.away_npxg_for,
            away_npxg_against=strength.away_npxg_against,
            home_opponent_adjusted_attack=strength.home_opponent_adjusted_attack,
            home_opponent_adjusted_defence=strength.home_opponent_adjusted_defence,
            away_opponent_adjusted_attack=strength.away_opponent_adjusted_attack,
            away_opponent_adjusted_defence=strength.away_opponent_adjusted_defence,
            home_shot_quality_for=strength.home_shot_quality_for,
            home_shot_quality_conceded=strength.home_shot_quality_conceded,
            away_shot_quality_for=strength.away_shot_quality_for,
            away_shot_quality_conceded=strength.away_shot_quality_conceded,
            home_set_piece_attack=strength.home_set_piece_attack,
            home_set_piece_defence=strength.home_set_piece_defence,
            away_set_piece_attack=strength.away_set_piece_attack,
            away_set_piece_defence=strength.away_set_piece_defence,
            home_goalkeeper_prevention=strength.home_goalkeeper_prevention,
            away_goalkeeper_prevention=strength.away_goalkeeper_prevention,
            expected_home_goals=expected_home_goals,
            expected_away_goals=expected_away_goals,
            p_home_dc=p_home_dc,
            p_draw_dc=p_draw_dc,
            p_away_dc=p_away_dc,
            market_vs_dc_home=market_vs_dc_home,
            market_vs_dc_draw=market_vs_dc_draw,
            market_vs_dc_away=market_vs_dc_away,
            attack_strength_difference=balance.attack_strength_difference,
            expected_goal_difference=balance.expected_goal_difference,
            expected_goal_total=balance.expected_goal_total,
            market_balance=balance.market_balance,
            defence_strength_difference=balance.defence_strength_difference,
            favourite_strength=balance.favourite_strength,
            combined_draw_rate=balance.combined_draw_rate,
            combined_one_goal_match_rate=balance.combined_one_goal_match_rate,
            combined_close_match_rate=balance.combined_close_match_rate,
            combined_low_scoring_rate=balance.combined_low_scoring_rate,
            league_draw_rate=league_behavior.league_draw_rate,
            league_home_win_rate=league_behavior.league_home_win_rate,
            league_away_win_rate=league_behavior.league_away_win_rate,
            league_avg_goals=league_behavior.league_avg_goals,
            league_goal_std=league_behavior.league_goal_std,
            league_favourite_win_rate=league_behavior.league_favourite_win_rate,
            league_competitive_balance=league_behavior.league_competitive_balance,
            league_avg_npxg=league_avg_npxg,
            league_upset_rate=upset_rate,
            league_prior_weight=league_behavior.league_prior_weight,
            home_rest_days=rest.home_rest_days,
            away_rest_days=rest.away_rest_days,
            rest_day_difference=rest.rest_day_difference,
            home_matches_last_14_days=rest.home_matches_last_14_days,
            away_matches_last_14_days=rest.away_matches_last_14_days,
            home_short_rest=rest.home_short_rest,
            away_short_rest=rest.away_short_rest,
            home_congestion=rest.home_congestion,
            away_congestion=rest.away_congestion,
            congestion_difference=congestion_difference,
            home_extra_time_in_previous_match=int(rest.home_extra_time_in_previous_match),
            away_extra_time_in_previous_match=int(rest.away_extra_time_in_previous_match),
            extra_time_x_short_rest=rest.extra_time_x_short_rest,
            fatigue_difference=rest.rest_day_difference,
            home_advantage_log=home_advantage_log,
            home_advantage_coefficient=home_advantage_coefficient,
            travel_distance_km=None,
            home_missing_player_value=availability.home_missing_player_value,
            away_missing_player_value=availability.away_missing_player_value,
            missing_value_difference=availability.missing_value_difference,
            home_unavailable_count=availability.home_unavailable_count,
            away_unavailable_count=availability.away_unavailable_count,
            home_lineup_changes=rest.home_lineup_changes,
            away_lineup_changes=rest.away_lineup_changes,
            missing_value_x_favourite=availability.missing_value_x_favourite,
            short_rest_x_missing_value=availability.short_rest_x_missing_value,
            congestion_x_squad_depth=rest.congestion_x_squad_depth,
            short_rest_x_rotation=rest.short_rest_x_rotation,
            has_availability=availability.has_availability,
        )

    def _engine_probabilities(
        self,
        match: STMatchModel,
        cutoff: date,
        league_external_id: int | None,
        strength: MatchStrengthFeatures,
    ) -> tuple[float | None, float | None, float | None, float | None, float | None]:
        """Return expected goals + 1X2 DC probs from classic or strength engine."""
        strength_tuple = (
            strength.expected_home_goals,
            strength.expected_away_goals,
            strength.dixon_coles_home_probability,
            strength.dixon_coles_draw_probability,
            strength.dixon_coles_away_probability,
        )
        if self.config.residual_ml_dc_engine != "classic":
            return strength_tuple

        classic = self._classic_dc_prediction(match, cutoff, league_external_id)
        if classic is None:
            return strength_tuple
        return (
            classic.lambda_home,
            classic.lambda_away,
            classic.p_home,
            classic.p_draw,
            classic.p_away,
        )

    def _classic_dc_prediction(
        self,
        match: STMatchModel,
        cutoff: date,
        league_external_id: int | None,
    ) -> DixonColesPrediction | None:
        if league_external_id is None:
            return None
        home_external_id = getattr(match.home_team, "external_id", None)
        away_external_id = getattr(match.away_team, "external_id", None)
        if home_external_id is None or away_external_id is None:
            return None

        cache_key = (league_external_id, cutoff)
        if cache_key not in self._classic_dc_fit_cache:
            try:
                model = self.dixon_coles_service.fit_league(
                    league_external_id, cutoff
                )
            except (ValueError, RuntimeError) as exc:
                if cache_key not in self._classic_dc_fallback_logged:
                    print(
                        f"Classic DC fit failed league={league_external_id} "
                        f"as_of={cutoff} ({exc}); falling back to strength DC",
                        flush=True,
                    )
                    self._classic_dc_fallback_logged.add(cache_key)
                model = None
            self._classic_dc_fit_cache[cache_key] = model

        model = self._classic_dc_fit_cache[cache_key]
        if model is None:
            return None
        try:
            return model.predict(int(home_external_id), int(away_external_id))
        except Exception as exc:
            if cache_key not in self._classic_dc_fallback_logged:
                print(
                    f"Classic DC predict failed league={league_external_id} "
                    f"as_of={cutoff} ({exc}); falling back to strength DC",
                    flush=True,
                )
                self._classic_dc_fallback_logged.add(cache_key)
            return None

    @staticmethod
    def _market_vs_dc(
        market_probability: float | None,
        dc_probability: float | None,
    ) -> float | None:
        if market_probability is None or dc_probability is None:
            return None
        return market_probability - dc_probability

    def _market_probabilities(self, match: STMatchModel) -> dict[str, float | None]:
        odds = match.match_odds
        if odds is None:
            return {"1": None, "X": None, "2": None}
        schema = STMatchOdds.model_validate(odds)
        raw = MarketProbabilities(schema).get_probs()
        return ensure_unit_probabilities(
            {"1": raw.get("1"), "X": raw.get("X"), "2": raw.get("2")}
        )

    def _resolve_league_external_id(self, match: STMatchModel) -> int | None:
        if match.league_name:
            if match.league_name in self._league_external_id_cache:
                return self._league_external_id_cache[match.league_name]
            league = self.league_repo.get_by_name(match.league_name)
            external_id = league.external_id if league is not None else None
            self._league_external_id_cache[match.league_name] = external_id
            if external_id is not None:
                return external_id
        if match.home_team is not None:
            team_id = match.home_team.id
            if team_id in self._team_league_id_cache:
                internal_id = self._team_league_id_cache[team_id]
            else:
                internal_id = self.fixture_repo.resolve_internal_league_id_for_team(
                    match.home_team
                )
                self._team_league_id_cache[team_id] = internal_id
            if internal_id is not None:
                league = self.league_repo.get(internal_id)
                if league is not None:
                    return league.external_id
        return None

    def _league_avg_npxg(self, match: STMatchModel, cutoff: date) -> float | None:
        if match.home_team is None:
            return None
        team_id = match.home_team.id
        if team_id in self._team_league_id_cache:
            league_id = self._team_league_id_cache[team_id]
        else:
            league_id = self.fixture_repo.resolve_internal_league_id_for_team(
                match.home_team
            )
            self._team_league_id_cache[team_id] = league_id
        if league_id is None:
            return None
        baselines = self.strength_calculator.league_averages_by_league_id(
            league_id, cutoff
        )
        value = baselines.get("npxg")
        return float(value) if value is not None else None

    def _home_advantage(
        self,
        match: STMatchModel,
        cutoff: date,
        *,
        target_league_external_id: int | None,
    ) -> tuple[float | None, float | None]:
        team = Team.model_validate(match.home_team)
        result = self.home_advantage_calculator.process(
            team,
            cutoff,
            target_league_external_id=target_league_external_id,
        )
        return result.home_advantage, exp(result.home_advantage)

    def _load_balance_fixtures(
        self, match: STMatchModel, cutoff: date
    ) -> list:
        lookback = max(self.config.balance_recent_matches * 4, 40)
        home_name = match.home_team.name
        away_name = match.away_team.name
        home_rows = self._team_fixtures_before(home_name, cutoff, lookback)
        away_rows = self._team_fixtures_before(away_name, cutoff, lookback)
        by_id = {row.id: row for row in home_rows}
        for row in away_rows:
            by_id.setdefault(row.id, row)
        return list(by_id.values())

    def _team_fixtures_before(
        self, team_name: str, cutoff: date, lookback: int
    ) -> list:
        cache_key = (team_name, cutoff, lookback)
        cached = self._team_fixtures_cache.get(cache_key)
        if cached is not None:
            return cached
        rows = self.fixture_repo.find_before_date_by_team(
            team_name=team_name,
            before_date=cutoff,
            venue=None,
            limit=lookback,
        )
        self._team_fixtures_cache[cache_key] = rows
        return rows
