"""Assemble residual ML features from existing calc modules."""

from __future__ import annotations

from datetime import date, datetime
from math import exp
from typing import Any

from sqlalchemy.orm import Session

from src.calc.balance_and_environment import BalanceAndEnvironment
from src.calc.league_behavior_calculator import LeagueBehaviorCalculator
from src.calc.market_probabilities import (
    MarketProbabilities,
    load_fixture_market_probabilities,
)
from src.calc.match_feature_context import (
    MatchFeatureContext,
    is_fixture_shaped,
    kickoff_datetime,
    resolve_cutoff_date,
)
from src.calc.player_availability_calculator import PlayerAvailabilityCalculator
from src.calc.rest_congestion_calculator import RestCongestionCalculator
from src.calc.strength_calculator import StrengthCalculator
from src.objects.repositories.fixture_repository import FixtureRepository
from src.objects.repositories.league_repository import LeagueRepository
from src.objects.repositories.team_repository import TeamRepository
from src.objects.schema.data_classes.data_sources import DataSourceConfig
from src.objects.schema.data_classes.market_probability import MarketProbabilityBreakdown
from src.objects.schema.data_classes.residual_ml_features import ResidualMLFeatures
from src.objects.schema.db.st_match_odds import STMatchOdds
from src.objects.schema.db.team import Team
from src.utils.common import ensure_unit_probabilities
from src.utils.fixture_fields import fixture_home_name, fixture_away_name


class ResidualMLFeatureAssembler:
    """Build ResidualMLFeatures for an ST coupon match or a historical fixture."""

    def __init__(
        self,
        session: Session,
        config: DataSourceConfig | None = None,
    ) -> None:
        self.session = session
        self.config = config or DataSourceConfig()
        self.fixture_repo = FixtureRepository(session)
        self.league_repo = LeagueRepository(session)
        self.team_repo = TeamRepository(session)
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
        self._league_external_id_cache: dict[tuple[str, str], int] = {}
        self._team_league_id_cache: dict[int, int | None] = {}
        self._team_fixtures_cache: dict[tuple[str, date, int], list] = {}

    def clear_caches(self) -> None:
        """Drop lookback caches across nested calculators."""
        self._league_external_id_cache.clear()
        self._team_league_id_cache.clear()
        self._team_fixtures_cache.clear()
        self.strength_calculator.clear_caches()
        self.league_behavior_calculator.clear_caches()
        self.rest_calculator.clear_caches()
        self.availability_calculator.clear_caches()
        self.home_advantage_calculator.clear_caches()

    def assemble(
        self,
        match: Any,
        *,
        draw_number: int | None = None,
        event_number: int | None = None,
        before_date: date | None = None,
        before: datetime | None = None,
        require_market: bool = True,
    ) -> ResidualMLFeatures:
        del event_number
        context = self._build_context(
            match, before_date=before_date, before=before
        )
        cutoff = context.cutoff
        target_league_external_id = context.league_external_id
        market = self._market_breakdown(match, required=require_market)
        market_probs = self._market_probs_dict(market)
        home_advantage_log, home_advantage_coefficient = self._home_advantage(
            context,
            cutoff,
            target_league_external_id=target_league_external_id,
        )
        strength_before: date | datetime = (
            cutoff if before_date is not None or before is not None else context.kickoff
        )
        strength = self.strength_calculator.get_fixture_features(
            context.home_team_internal_id,
            context.away_team_internal_id,
            strength_before,
            match_id=context.identity,
            target_league_external_id=target_league_external_id,
            home_advantage_coefficient=home_advantage_coefficient,
        )
        history_fixtures = self._load_balance_fixtures(context, cutoff)
        balance = self.balance_calculator.calculate(
            match,
            history_fixtures,
            market_probs,
            strength=strength,
            before_date=cutoff,
            context=context,
        )
        league_behavior = self.league_behavior_calculator.calculate(
            match,
            before_date=cutoff,
            context=context,
        )
        home_previous, away_previous = self.rest_calculator.previous_fixtures(
            match,
            before_date=cutoff,
            context=context,
        )
        rest_seed = self.rest_calculator.calculate(
            match,
            before_date=cutoff,
            context=context,
        )
        availability = self.availability_calculator.calculate(
            match,
            favourite_strength=balance.favourite_strength,
            home_short_rest=rest_seed.home_short_rest,
            away_short_rest=rest_seed.away_short_rest,
            home_previous_fixture=home_previous,
            away_previous_fixture=away_previous,
            before_date=cutoff,
            context=context,
        )
        rest = self.rest_calculator.calculate(
            match,
            availability=availability,
            before_date=cutoff,
            context=context,
        )

        league_avg_npxg = self._league_avg_npxg(context, cutoff)

        upset_rate = (
            1.0 - league_behavior.league_favourite_win_rate
            if league_behavior.league_favourite_win_rate is not None
            else None
        )
        congestion_difference = None
        if rest.home_congestion is not None and rest.away_congestion is not None:
            congestion_difference = rest.home_congestion - rest.away_congestion

        st_draw_number = getattr(match, "stryktipset_round_id", None)
        return ResidualMLFeatures(
            match_id=context.identity,
            draw_number=draw_number if draw_number is not None else st_draw_number,
            feature_cutoff_date=cutoff,
            league_external_id=target_league_external_id,
            p_home_market=market_probs.get("1"),
            p_draw_market=market_probs.get("X"),
            p_away_market=market_probs.get("2"),
            market_overround=market.overround if market is not None else None,
            market_entropy=market.market_entropy if market is not None else None,
            market_top_probability=(
                market.market_top_probability if market is not None else None
            ),
            market_second_probability=(
                market.market_second_probability if market is not None else None
            ),
            market_probability_gap=(
                market.market_probability_gap if market is not None else None
            ),
            market_price_type=market.price_type if market is not None else None,
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

    def _build_context(
        self,
        match: Any,
        *,
        before_date: date | None,
        before: datetime | None,
    ) -> MatchFeatureContext:
        cutoff = resolve_cutoff_date(match, before_date=before_date, before=before)
        kickoff = kickoff_datetime(match)
        if is_fixture_shaped(match):
            home_team, away_team = self._resolve_fixture_teams(match)
            league_external_id = getattr(match, "league_id", None)
            if league_external_id is not None:
                league_external_id = int(league_external_id)
            return MatchFeatureContext(
                identity=int(match.id),
                kickoff=kickoff,
                cutoff=cutoff,
                home_team=home_team,
                away_team=away_team,
                league_name=getattr(match, "league_name", None),
                league_country=getattr(match, "league_country", None),
                league_external_id=league_external_id,
                fixture_pk=int(match.id),
                source="fixture",
            )
        if match.home_team is None or match.away_team is None:
            raise ValueError(f"Missing team on match id={match.id}")
        if match.start_time is None:
            raise ValueError(f"Missing start_time on match id={match.id}")
        return MatchFeatureContext(
            identity=int(match.id),
            kickoff=kickoff,
            cutoff=cutoff,
            home_team=match.home_team,
            away_team=match.away_team,
            league_name=getattr(match, "league_name", None),
            league_country=getattr(match, "league_country_name", None),
            league_external_id=self._resolve_league_external_id(match),
            fixture_pk=None,
            source="stryktipset",
        )

    def _resolve_fixture_teams(self, fixture: Any) -> tuple[Any, Any]:
        home_external_id = getattr(fixture, "home_team_id", None)
        away_external_id = getattr(fixture, "away_team_id", None)
        if home_external_id is None or away_external_id is None:
            raise ValueError(
                f"Missing API-Football team id on fixture id={getattr(fixture, 'id', None)}"
            )
        home_team = self.team_repo.get_by_external_id(int(home_external_id))
        away_team = self.team_repo.get_by_external_id(int(away_external_id))
        if home_team is None:
            raise ValueError(
                f"Unresolved home team external_id={home_external_id} "
                f"on fixture id={getattr(fixture, 'id', None)} "
                f"({fixture_home_name(fixture)})"
            )
        if away_team is None:
            raise ValueError(
                f"Unresolved away team external_id={away_external_id} "
                f"on fixture id={getattr(fixture, 'id', None)} "
                f"({fixture_away_name(fixture)})"
            )
        return home_team, away_team

    def _market_breakdown(
        self, match: Any, *, required: bool = True
    ) -> MarketProbabilityBreakdown | None:
        if is_fixture_shaped(match):
            if not required:
                return None
            breakdown = load_fixture_market_probabilities(
                self.session,
                int(match.id),
                config=self.config,
            )
            if breakdown is None:
                raise ValueError(
                    f"No usable fixture odds for fixture id={match.id}"
                )
            return breakdown
        odds = getattr(match, "match_odds", None)
        if odds is None:
            return None
        schema = STMatchOdds.model_validate(odds)
        return MarketProbabilities(schema).describe()

    @staticmethod
    def _market_probs_dict(
        market: MarketProbabilityBreakdown | None,
    ) -> dict[str, float | None]:
        if market is None:
            return {"1": None, "X": None, "2": None}
        return ensure_unit_probabilities(market.as_probs())

    def _resolve_league_external_id(self, match: Any) -> int | None:
        league_name = (getattr(match, "league_name", None) or "").strip()
        country = (getattr(match, "league_country_name", None) or "").strip() or ""
        if league_name:
            cache_key = (league_name, country)
            if cache_key in self._league_external_id_cache:
                return self._league_external_id_cache[cache_key]
            league = None
            if country:
                league = self.league_repo.get_by_name_and_country(
                    league_name, country
                )
            if league is None:
                league = self.league_repo.get_by_name(league_name)
            if league is not None and league.external_id is not None:
                external_id = int(league.external_id)
                self._league_external_id_cache[cache_key] = external_id
                return external_id
            # Name miss: do not cache None — fall through to fixture/team path.

        return self.fixture_repo.resolve_league_external_id_for_match(
            match,
            kickoff_tolerance_minutes=self.config.kickoff_match_tolerance_minutes,
            skip_name=True,
        )

    def _league_avg_npxg(
        self, context: MatchFeatureContext, cutoff: date
    ) -> float | None:
        team_id = context.home_team_internal_id
        if team_id in self._team_league_id_cache:
            league_id = self._team_league_id_cache[team_id]
        elif context.source == "fixture" and context.league_external_id is not None:
            league = self.league_repo.get_by_external_id(context.league_external_id)
            league_id = league.id if league is not None else None
            self._team_league_id_cache[team_id] = league_id
        else:
            league_id = self.fixture_repo.resolve_internal_league_id_for_team(
                context.home_team
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
        context: MatchFeatureContext,
        cutoff: date,
        *,
        target_league_external_id: int | None,
    ) -> tuple[float | None, float | None]:
        team = Team.model_validate(context.home_team)
        result = self.home_advantage_calculator.process(
            team,
            cutoff,
            target_league_external_id=target_league_external_id,
        )
        return result.home_advantage, exp(result.home_advantage)

    def _load_balance_fixtures(
        self, context: MatchFeatureContext, cutoff: date
    ) -> list:
        lookback = max(self.config.balance_recent_matches * 4, 40)
        home_rows = self._team_fixtures_before(
            context.home_team_name, cutoff, lookback
        )
        away_rows = self._team_fixtures_before(
            context.away_team_name, cutoff, lookback
        )
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
