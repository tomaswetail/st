"""DB-backed wrapper around classic Dixon–Coles goals MLE."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from sqlalchemy.orm import Session

from src.calc.dixon_coles.model import DixonColesModel, DixonColesPrediction
from src.calc.dixon_coles.types import DixonColesMatch
from src.calc.dixon_coles.walk_forward import (
    EvalMatch,
    WalkForwardResult,
    run_walk_forward,
)
from src.calc.market_probabilities import MarketProbabilities
from src.data_sources.classic_dc_config import ClassicDcLeagueParams, load_league_params
from src.objects.repositories.fixture_repository import FixtureRepository
from src.objects.repositories.league_repository import LeagueRepository
from src.objects.repositories.st_match_repository import STMatchRepository
from src.objects.schema.data_classes.data_sources import DataSourceConfig
from src.objects.schema.db.st_match_odds import STMatchOdds
from src.utils.common import ensure_unit_probabilities


class DixonColesService:
    """Load finished fixtures from the DB and fit a per-league Dixon–Coles model."""

    def __init__(
        self,
        session: Session,
        config: DataSourceConfig | None = None,
    ) -> None:
        self.session = session
        self.config = config or DataSourceConfig()
        self.fixture_repo = FixtureRepository(session)
        self.st_match_repo = STMatchRepository(session)
        self.league_repo = LeagueRepository(session)
        self._league_external_id_by_name: dict[str, int | None] = {}
        self._league_params_by_id: dict[int, ClassicDcLeagueParams] | None = None

    def params_for_league(self, league_id: int) -> tuple[float, int, float]:
        """Return (xi, lookback_days, rho) for a league, with global fallback."""
        params = self._load_league_params().get(league_id)
        if params is None:
            return (
                self.config.classic_dc_xi,
                self.config.classic_dc_lookback_days,
                self.config.dixon_coles_rho,
            )
        return params.xi, params.lookback, params.rho

    def fit_league(
        self,
        league_id: int,
        as_of: date,
        *,
        xi: float | None = None,
        lookback_days: int | None = None,
        rho: float | None = None,
        min_team_matches: int | None = None,
    ) -> DixonColesModel:
        """Fit classic DC for one API league id as of ``as_of`` (exclusive)."""
        default_xi, default_lookback, default_rho = self.params_for_league(league_id)
        lookback = lookback_days if lookback_days is not None else default_lookback
        after_date = as_of - timedelta(days=lookback)
        fixtures = self.fixture_repo.find_finished_goals_before_date(
            before_date=as_of,
            league_id=league_id,
            after_date=after_date,
        )
        matches = [DixonColesMatch.from_fixture(fixture) for fixture in fixtures]
        return self.fit_league_from_matches(
            matches,
            league_id,
            as_of,
            xi=xi if xi is not None else default_xi,
            lookback_days=lookback,
            rho=rho if rho is not None else default_rho,
            min_team_matches=min_team_matches,
        )

    def fit_league_from_matches(
        self,
        matches: list[DixonColesMatch],
        league_id: int,
        as_of: date,
        *,
        xi: float | None = None,
        lookback_days: int | None = None,
        rho: float | None = None,
        min_team_matches: int | None = None,
    ) -> DixonColesModel:
        """Fit from an in-memory match list (optimizer / preloaded fixtures)."""
        default_xi, default_lookback, default_rho = self.params_for_league(league_id)
        lookback = lookback_days if lookback_days is not None else default_lookback
        model = DixonColesModel(
            xi=xi if xi is not None else default_xi,
            rho=rho if rho is not None else default_rho,
            max_goals=self.config.dixon_coles_max_goals,
            lookback_days=lookback,
            min_team_matches=(
                min_team_matches
                if min_team_matches is not None
                else self.config.classic_dc_min_team_matches
            ),
            as_of=as_of,
            fit_rho=self.config.classic_dc_fit_rho,
            rho_min=self.config.classic_dc_rho_min,
            rho_max=self.config.classic_dc_rho_max,
        )
        return model.fit(matches, as_of=as_of)

    def preload_league_fixtures(
        self,
        league_ids: list[int],
        *,
        before_date: date,
        max_lookback_days: int,
    ) -> dict[int, list[DixonColesMatch]]:
        """Load finished goal fixtures per league for optimizer preloading."""
        after_date = before_date - timedelta(days=max_lookback_days)
        fixtures_by_league: dict[int, list[DixonColesMatch]] = {}
        for league_id in league_ids:
            fixtures = self.fixture_repo.find_finished_goals_before_date(
                before_date=before_date,
                league_id=league_id,
                after_date=after_date,
            )
            fixtures_by_league[league_id] = [
                DixonColesMatch.from_fixture(fixture) for fixture in fixtures
            ]
        return fixtures_by_league

    def predict(
        self,
        model: DixonColesModel,
        home_team_id: int,
        away_team_id: int,
    ) -> DixonColesPrediction:
        return model.predict(home_team_id, away_team_id)

    def load_all_stryktipset_eval_matches(
        self,
        date_from: date,
        date_to: date,
        *,
        min_draw_number: int | None = None,
        max_draw_number: int | None = None,
    ) -> list[EvalMatch]:
        """All ST coupon matches in range with market probs and league ids."""
        rows = self.st_match_repo.find_finished_with_odds(
            date_from=date_from,
            date_to=date_to,
            min_draw_number=min_draw_number,
            max_draw_number=max_draw_number,
        )
        eval_matches: list[EvalMatch] = []
        for match in rows:
            league_external_id = self._resolve_league_external_id(match.league_name)
            if league_external_id is None:
                continue
            if match.home_team is None or match.away_team is None:
                continue
            if match.home_team.external_id is None or match.away_team.external_id is None:
                continue
            if match.match_odds is None or match.stryktipset_result not in {"1", "X", "2"}:
                continue
            if match.start_time is None:
                continue
            odds = STMatchOdds.model_validate(match.match_odds)
            raw = MarketProbabilities(odds).get_probs()
            unit = ensure_unit_probabilities(
                {"1": raw.get("1"), "X": raw.get("X"), "2": raw.get("2")}
            )
            if None in (unit.get("1"), unit.get("X"), unit.get("2")):
                continue
            match_date = (
                match.start_time.date()
                if isinstance(match.start_time, datetime)
                else match.start_time
            )
            eval_matches.append(
                EvalMatch(
                    match_date=match_date,
                    home_team_external_id=int(match.home_team.external_id),
                    away_team_external_id=int(match.away_team.external_id),
                    label=match.stryktipset_result,
                    p_home_market=float(unit["1"]),
                    p_draw_market=float(unit["X"]),
                    p_away_market=float(unit["2"]),
                    league_external_id=league_external_id,
                )
            )
        return eval_matches

    def load_stryktipset_eval_matches(
        self,
        league_id: int,
        date_from: date,
        date_to: date,
    ) -> list[EvalMatch]:
        """ST coupon matches in range for ``league_id`` with market probs."""
        return [
            match
            for match in self.load_all_stryktipset_eval_matches(date_from, date_to)
            if match.league_external_id == league_id
        ]

    def walk_forward_league(
        self,
        league_id: int,
        date_from: date,
        date_to: date,
    ) -> WalkForwardResult:
        """Day-grouped walk-forward classic DC vs market for one league."""
        eval_matches = self.load_stryktipset_eval_matches(
            league_id, date_from, date_to
        )
        fit_cache: dict[date, DixonColesModel | None] = {}

        def fit_for_date(day: date) -> DixonColesModel | None:
            if day in fit_cache:
                return fit_cache[day]
            try:
                model = self.fit_league(league_id, day)
            except (ValueError, RuntimeError):
                model = None
            fit_cache[day] = model
            return model

        return run_walk_forward(eval_matches, fit_for_date)

    def _load_league_params(self) -> dict[int, ClassicDcLeagueParams]:
        if self._league_params_by_id is None:
            self._league_params_by_id = load_league_params(
                self.config.classic_dc_league_params_path
            )
        return self._league_params_by_id

    def _resolve_league_external_id(self, league_name: str | None) -> int | None:
        if not league_name:
            return None
        if league_name in self._league_external_id_by_name:
            return self._league_external_id_by_name[league_name]
        league = self.league_repo.get_by_name(league_name)
        external_id = league.external_id if league is not None else None
        self._league_external_id_by_name[league_name] = external_id
        return external_id
