"""Classic Dixon–Coles goals MLE: fit attack/defence/HA, predict 1/X/2."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from calc.dixon_coles.types import DixonColesMatch
from calc.strength_calculator import (
    _scoreline_probability,
    dixon_coles_matrix,
)


@dataclass
class DixonColesPrediction:
    lambda_home: float
    lambda_away: float
    p_home: float
    p_draw: float
    p_away: float


@dataclass(frozen=True)
class FixtureDateIndex:
    """Sorted league fixtures for O(log n) lookback window slicing."""

    matches: tuple[DixonColesMatch, ...]
    dates: tuple[date, ...]

    @classmethod
    def from_matches(cls, matches: list[DixonColesMatch]) -> FixtureDateIndex:
        sorted_matches = tuple(
            sorted(
                matches,
                key=lambda match: (
                    match.match_date,
                    match.home_team_id,
                    match.away_team_id,
                ),
            )
        )
        return cls(
            matches=sorted_matches,
            dates=tuple(match.match_date for match in sorted_matches),
        )

    def window(self, *, as_of: date, lookback_days: int) -> list[DixonColesMatch]:
        """Keep matches with ``as_of - lookback_days <= match_date < as_of``."""
        after_date = as_of - timedelta(days=lookback_days)
        start = bisect_left(self.dates, after_date)
        end = bisect_left(self.dates, as_of)
        return list(self.matches[start:end])


def match_weight(days_before: int, xi: float) -> float:
    """Time-decay weight ``exp(-xi * days_before)``."""
    if days_before < 0:
        return 0.0
    return math.exp(-xi * days_before)


def filter_matches_by_lookback(
    matches: list[DixonColesMatch],
    *,
    as_of: date,
    lookback_days: int,
) -> list[DixonColesMatch]:
    """Keep matches with ``as_of - lookback_days <= match_date < as_of``."""
    return FixtureDateIndex.from_matches(matches).window(
        as_of=as_of,
        lookback_days=lookback_days,
    )


def _vectorized_scoreline_log_probability(
    goals_home: np.ndarray,
    goals_away: np.ndarray,
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Batched log scoreline probability with Dixon–Coles tau correction."""
    log_p_home = (
        goals_home * np.log(lambda_home)
        - lambda_home
        - gammaln(goals_home + 1)
    )
    log_p_away = (
        goals_away * np.log(lambda_away)
        - lambda_away
        - gammaln(goals_away + 1)
    )
    log_independent = log_p_home + log_p_away

    tau = np.ones_like(lambda_home)
    mask_00 = (goals_home == 0) & (goals_away == 0)
    mask_01 = (goals_home == 0) & (goals_away == 1)
    mask_10 = (goals_home == 1) & (goals_away == 0)
    mask_11 = (goals_home == 1) & (goals_away == 1)
    tau = np.where(mask_00, 1.0 - lambda_home * lambda_away * rho, tau)
    tau = np.where(mask_01, 1.0 + lambda_home * rho, tau)
    tau = np.where(mask_10, 1.0 + lambda_away * rho, tau)
    tau = np.where(mask_11, 1.0 - rho, tau)

    probability = np.exp(log_independent) * tau
    return np.log(np.maximum(probability, 1e-15))


def _scalar_neg_log_likelihood(
    theta: np.ndarray,
    *,
    unpack,
    usable: list[DixonColesMatch],
    team_index: dict[int, int],
    cutoff: date,
    xi: float,
    rho: float,
) -> float:
    attack, defence, home_advantage = unpack(theta)
    total = 0.0
    for match in usable:
        days_before = (cutoff - match.match_date).days
        weight = match_weight(days_before, xi)
        if weight <= 0:
            continue
        home_i = team_index[match.home_team_id]
        away_i = team_index[match.away_team_id]
        lambda_home = attack[home_i] * defence[away_i] * home_advantage
        lambda_away = attack[away_i] * defence[home_i]
        probability = _scoreline_probability(
            match.goals_home,
            match.goals_away,
            lambda_home,
            lambda_away,
            rho,
        )
        probability = max(probability, 1e-15)
        total -= weight * math.log(probability)
    return total


def _vectorized_neg_log_likelihood(
    theta: np.ndarray,
    *,
    unpack,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    goals_home: np.ndarray,
    goals_away: np.ndarray,
    weights: np.ndarray,
    rho: float,
) -> float:
    attack, defence, home_advantage = unpack(theta)
    lambda_home = attack[home_idx] * defence[away_idx] * home_advantage
    lambda_away = attack[away_idx] * defence[home_idx]
    log_prob = _vectorized_scoreline_log_probability(
        goals_home,
        goals_away,
        lambda_home,
        lambda_away,
        rho,
    )
    return float(-np.sum(weights * log_prob))


class DixonColesModel:
    """Goals-based Dixon–Coles with fixed rho and exponential time weights."""

    def __init__(
        self,
        *,
        xi: float = 0.0018,
        rho: float = -0.13,
        max_goals: int = 10,
        lookback_days: int = 730,
        min_team_matches: int = 5,
        as_of: date | None = None,
    ) -> None:
        self.xi = xi
        self.rho = rho
        self.max_goals = max_goals
        self.lookback_days = lookback_days
        self.min_team_matches = min_team_matches
        self.as_of = as_of
        self.team_ids: list[int] = []
        self.attack: dict[int, float] = {}
        self.defence: dict[int, float] = {}
        self.home_advantage: float = 1.0
        self.n_training_matches: int = 0
        self._fitted = False
        self._fitted_theta: np.ndarray | None = None
        self._last_fit_iterations: int | None = None

    @property
    def fitted_theta(self) -> np.ndarray | None:
        return None if self._fitted_theta is None else self._fitted_theta.copy()

    @property
    def last_fit_iterations(self) -> int | None:
        return self._last_fit_iterations

    def fit(
        self,
        matches: list[DixonColesMatch],
        *,
        as_of: date | None = None,
        initial_theta: np.ndarray | None = None,
    ) -> DixonColesModel:
        """Fit attack, defence, and home advantage on lookback-filtered matches."""
        cutoff = as_of or self.as_of
        if cutoff is None:
            if not matches:
                raise ValueError("Cannot fit without matches or as_of date")
            cutoff = max(match.match_date for match in matches) + timedelta(days=1)
        self.as_of = cutoff

        windowed = filter_matches_by_lookback(
            matches,
            as_of=cutoff,
            lookback_days=self.lookback_days,
        )
        usable = self.matches_with_enough_team_history(windowed)
        if not usable:
            raise ValueError("No matches left after lookback / min_team_matches filters")

        team_ids = sorted(
            {
                team_id
                for match in usable
                for team_id in (match.home_team_id, match.away_team_id)
            }
        )
        self.team_ids = team_ids
        team_index = {team_id: index for index, team_id in enumerate(team_ids)}
        n_teams = len(team_ids)

        # Parameters: log(attack[0..n-2]), log(defence[0..n-1]), log(home_advantage)
        # Attack of last team is set so geometric mean of attack is 1.
        n_params = (n_teams - 1) + n_teams + 1
        if (
            initial_theta is not None
            and len(initial_theta) == n_params
        ):
            x0 = np.asarray(initial_theta, dtype=float)
        else:
            x0 = np.zeros(n_params, dtype=float)

        def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
            log_attack_free = theta[: n_teams - 1]
            log_defence = theta[n_teams - 1 : n_teams - 1 + n_teams]
            log_home_advantage = float(theta[-1])
            # Pin geometric mean of attack to 1 => sum(log attack) = 0.
            log_attack_last = -float(np.sum(log_attack_free))
            log_attack = np.concatenate([log_attack_free, [log_attack_last]])
            attack = np.exp(log_attack)
            defence = np.exp(log_defence)
            home_advantage = math.exp(log_home_advantage)
            return attack, defence, home_advantage

        home_idx = np.fromiter(
            (team_index[match.home_team_id] for match in usable),
            dtype=np.intp,
        )
        away_idx = np.fromiter(
            (team_index[match.away_team_id] for match in usable),
            dtype=np.intp,
        )
        goals_home = np.fromiter(
            (match.goals_home for match in usable),
            dtype=np.intp,
        )
        goals_away = np.fromiter(
            (match.goals_away for match in usable),
            dtype=np.intp,
        )
        weights = np.fromiter(
            (
                match_weight((cutoff - match.match_date).days, self.xi)
                for match in usable
            ),
            dtype=float,
        )
        positive_weights = weights > 0
        home_idx = home_idx[positive_weights]
        away_idx = away_idx[positive_weights]
        goals_home = goals_home[positive_weights]
        goals_away = goals_away[positive_weights]
        weights = weights[positive_weights]

        def neg_log_likelihood(theta: np.ndarray) -> float:
            return _vectorized_neg_log_likelihood(
                theta,
                unpack=unpack,
                home_idx=home_idx,
                away_idx=away_idx,
                goals_home=goals_home,
                goals_away=goals_away,
                weights=weights,
                rho=self.rho,
            )

        result = minimize(
            neg_log_likelihood,
            x0,
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-8},
        )
        if not result.success and result.nit == 0:
            raise RuntimeError(f"Dixon–Coles fit failed: {result.message}")

        attack_arr, defence_arr, home_advantage = unpack(result.x)
        self.attack = {
            team_id: float(attack_arr[index])
            for team_id, index in team_index.items()
        }
        self.defence = {
            team_id: float(defence_arr[index])
            for team_id, index in team_index.items()
        }
        self.home_advantage = float(home_advantage)
        self.n_training_matches = len(usable)
        self._fitted_theta = result.x.copy()
        self._last_fit_iterations = int(result.nit)
        self._fitted = True
        return self

    def expected_goals(
        self,
        home_team_id: int,
        away_team_id: int,
    ) -> tuple[float, float]:
        self._require_fitted()
        attack_home = self.attack.get(home_team_id, 1.0)
        defence_home = self.defence.get(home_team_id, 1.0)
        attack_away = self.attack.get(away_team_id, 1.0)
        defence_away = self.defence.get(away_team_id, 1.0)
        lambda_home = attack_home * defence_away * self.home_advantage
        lambda_away = attack_away * defence_home
        return lambda_home, lambda_away

    def predict(
        self,
        home_team_id: int,
        away_team_id: int,
    ) -> DixonColesPrediction:
        """Return λ and renormalized 1/X/2 probabilities."""
        lambda_home, lambda_away = self.expected_goals(home_team_id, away_team_id)
        _matrix, p_home, p_draw, p_away = dixon_coles_matrix(
            lambda_home,
            lambda_away,
            rho=self.rho,
            max_goals=self.max_goals,
        )
        return DixonColesPrediction(
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            p_home=p_home,
            p_draw=p_draw,
            p_away=p_away,
        )

    def predict_dict(
        self,
        home_team_id: int,
        away_team_id: int,
    ) -> dict[str, float]:
        prediction = self.predict(home_team_id, away_team_id)
        return {
            "lambda_home": prediction.lambda_home,
            "lambda_away": prediction.lambda_away,
            "p_home": prediction.p_home,
            "p_draw": prediction.p_draw,
            "p_away": prediction.p_away,
        }

    def matches_with_enough_team_history(
        self,
        matches: list[DixonColesMatch],
    ) -> list[DixonColesMatch]:
        counts: Counter[int] = Counter()
        for match in matches:
            counts[match.home_team_id] += 1
            counts[match.away_team_id] += 1
        eligible = {
            team_id
            for team_id, count in counts.items()
            if count >= self.min_team_matches
        }
        return [
            match
            for match in matches
            if match.home_team_id in eligible and match.away_team_id in eligible
        ]

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("DixonColesModel.fit() must be called before predict")
