"""Per-league hyperparameter grid search for classic Dixon–Coles."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from itertools import product
from statistics import median
from time import perf_counter

import numpy as np

from src.calc.dixon_coles.model import DixonColesModel, FixtureDateIndex
from src.calc.dixon_coles.types import DixonColesMatch
from src.calc.dixon_coles.walk_forward import (
    EvalMatch,
    WalkForwardResult,
    run_walk_forward_per_league,
)

WindowCache = dict[tuple[int, date], list[DixonColesMatch]]


def _team_ids_for_window(
    matches: list[DixonColesMatch],
    *,
    min_team_matches: int,
) -> tuple[int, ...]:
    model = DixonColesModel(min_team_matches=min_team_matches)
    usable = model.matches_with_enough_team_history(matches)
    return tuple(
        sorted(
            {
                team_id
                for match in usable
                for team_id in (match.home_team_id, match.away_team_id)
            }
        )
    )


@dataclass(frozen=True)
class RhoFitDiagnostics:
    """Spread of MLE-fitted rho across walk-forward fits for one parameter set."""

    n_fits: int = 0
    median: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    at_bound_count: int = 0

    @property
    def at_bound_fraction(self) -> float:
        return self.at_bound_count / self.n_fits if self.n_fits else 0.0

    def summary(self) -> str:
        if not self.n_fits or self.median is None:
            return "rho_fitted=n/a"
        return (
            f"rho_fitted median={self.median:+.4f} "
            f"[{self.minimum:+.4f}, {self.maximum:+.4f}] "
            f"at_bound={self.at_bound_count}/{self.n_fits}"
        )


@dataclass(frozen=True)
class DixonColesParameterResult:
    league_id: int
    xi: float
    lookback: int
    rho: float
    log_loss: float
    rps: float
    num_predictions: int
    skipped_matches: int
    avg_training_matches: float
    rho_diagnostics: RhoFitDiagnostics = field(default_factory=RhoFitDiagnostics)


@dataclass
class DixonColesLeagueOptimizationResult:
    league_id: int
    best_xi: float
    best_lookback: int
    best_rho: float
    log_loss: float
    rps: float
    evaluated_matches: int
    skipped_matches: int
    validation_start: date | None
    validation_end: date | None
    parameter_results: list[DixonColesParameterResult] = field(default_factory=list)
    rho_diagnostics: RhoFitDiagnostics = field(default_factory=RhoFitDiagnostics)

    def top_parameter_results(self, n: int = 3) -> list[DixonColesParameterResult]:
        return sorted(self.parameter_results, key=lambda row: row.log_loss)[:n]


@dataclass
class DixonColesOptimizationResult:
    league_results: dict[int, DixonColesLeagueOptimizationResult]
    skipped_leagues: dict[int, str] = field(default_factory=dict)
    validation_start: date | None = None
    validation_end: date | None = None

    @property
    def pooled_log_loss(self) -> float:
        total_weight = 0
        weighted = 0.0
        for result in self.league_results.values():
            if result.evaluated_matches <= 0:
                continue
            weighted += result.log_loss * result.evaluated_matches
            total_weight += result.evaluated_matches
        return weighted / total_weight if total_weight else 0.0

    @property
    def pooled_market_log_loss(self) -> float:
        return 0.0

    def sorted_league_table(self) -> list[DixonColesLeagueOptimizationResult]:
        return sorted(
            self.league_results.values(),
            key=lambda row: row.log_loss,
        )


@dataclass(frozen=True)
class _LeagueOptimizationJob:
    league_id: int
    eval_matches: list[EvalMatch]
    league_fixtures: list[DixonColesMatch]
    validation_start: date
    validation_end: date
    xi_values: list[float]
    lookback_values: list[int]
    rho_values: list[float]
    min_training_matches: int
    min_team_matches: int
    max_goals: int
    log_progress: bool
    fit_rho: bool = False
    rho_min: float = -0.2
    rho_max: float = 0.2


def _average_training_size(
    *,
    fixture_index: FixtureDateIndex,
    eval_matches: list[EvalMatch],
    lookback: int,
    min_team_matches: int,
    window_cache: WindowCache | None = None,
) -> float:
    days = sorted({match.match_date for match in eval_matches})
    if not days:
        return 0.0
    sizes: list[int] = []
    for day in days:
        cache_key = (lookback, day)
        if window_cache is not None and cache_key in window_cache:
            windowed = window_cache[cache_key]
        else:
            windowed = fixture_index.window(as_of=day, lookback_days=lookback)
            if window_cache is not None:
                window_cache[cache_key] = windowed
        model = DixonColesModel(
            lookback_days=lookback,
            min_team_matches=min_team_matches,
            as_of=day,
        )
        usable = model.matches_with_enough_team_history(windowed)
        sizes.append(len(usable))
    return sum(sizes) / len(sizes)


def _evaluate_parameter_set(
    *,
    league_id: int,
    eval_matches: list[EvalMatch],
    fixture_index: FixtureDateIndex,
    xi: float,
    lookback: int,
    rho: float,
    min_team_matches: int,
    min_training_matches: int,
    max_goals: int,
    window_cache: WindowCache,
    use_warm_start: bool = True,
    fit_rho: bool = False,
    rho_min: float = -0.2,
    rho_max: float = 0.2,
) -> tuple[WalkForwardResult, RhoFitDiagnostics]:
    fit_cache: dict[date, DixonColesModel | None] = {}
    last_theta: np.ndarray | None = None
    last_team_ids: tuple[int, ...] | None = None

    def windowed_matches(day: date) -> list[DixonColesMatch]:
        cache_key = (lookback, day)
        if cache_key not in window_cache:
            window_cache[cache_key] = fixture_index.window(
                as_of=day,
                lookback_days=lookback,
            )
        return window_cache[cache_key]

    def fit_for_league_and_date(
        match_league_id: int,
        day: date,
    ) -> DixonColesModel | None:
        nonlocal last_theta, last_team_ids
        if match_league_id != league_id:
            return None
        if day in fit_cache:
            return fit_cache[day]
        matches = windowed_matches(day)
        if len(matches) < min_training_matches:
            fit_cache[day] = None
            return None
        initial_theta = None
        current_team_ids = _team_ids_for_window(
            matches,
            min_team_matches=min_team_matches,
        )
        if (
            use_warm_start
            and last_theta is not None
            and last_team_ids is not None
            and current_team_ids == last_team_ids
        ):
            initial_theta = last_theta
        try:
            model = DixonColesModel(
                xi=xi,
                rho=rho,
                max_goals=max_goals,
                lookback_days=lookback,
                min_team_matches=min_team_matches,
                as_of=day,
                fit_rho=fit_rho,
                rho_min=rho_min,
                rho_max=rho_max,
            ).fit(matches, as_of=day, initial_theta=initial_theta)
            if use_warm_start and model.fitted_theta is not None:
                last_theta = model.fitted_theta
                last_team_ids = tuple(model.team_ids)
        except (ValueError, RuntimeError):
            model = None
        fit_cache[day] = model
        return model

    walk_forward = run_walk_forward_per_league(eval_matches, fit_for_league_and_date)
    return walk_forward, _rho_diagnostics(fit_cache.values())


def _rho_diagnostics(models: Iterable[DixonColesModel | None]) -> RhoFitDiagnostics:
    fitted_models = [
        model for model in models if model is not None and model.fitted_rho is not None
    ]
    if not fitted_models:
        return RhoFitDiagnostics()
    fitted = [float(model.fitted_rho) for model in fitted_models]
    at_bound = sum(1 for model in fitted_models if model.rho_at_bound)
    return RhoFitDiagnostics(
        n_fits=len(fitted),
        median=float(median(fitted)),
        minimum=min(fitted),
        maximum=max(fitted),
        at_bound_count=at_bound,
    )


def optimize_single_league(
    league_id: int,
    eval_matches: list[EvalMatch],
    league_fixtures: list[DixonColesMatch],
    *,
    validation_start: date,
    validation_end: date,
    xi_values: list[float],
    lookback_values: list[int],
    rho_values: list[float],
    min_training_matches: int,
    min_team_matches: int,
    max_goals: int,
    log_progress: bool = True,
    use_warm_start: bool = True,
    fit_rho: bool = False,
    rho_min: float = -0.2,
    rho_max: float = 0.2,
    default_rho: float = -0.13,
) -> DixonColesLeagueOptimizationResult:
    """Grid-search walk-forward hyperparameters for one league.

    With ``fit_rho`` the rho dimension is dropped from the grid and rho is
    estimated by MLE inside each walk-forward fit instead.
    """
    if log_progress:
        print(
            f"Optimizing league={league_id} "
            f"({len(eval_matches)} validation matches)...",
            flush=True,
        )

    league_window_cache: WindowCache = {}
    fixture_index = FixtureDateIndex.from_matches(league_fixtures)
    avg_training_by_lookback: dict[int, float] = {}
    parameter_results: list[DixonColesParameterResult] = []
    if fit_rho:
        # Rho comes from the likelihood, so it must not also be swept.
        grid_combos = [
            (xi, lookback, None)
            for xi, lookback in product(xi_values, lookback_values)
        ]
    else:
        grid_combos = list(product(xi_values, lookback_values, rho_values))
    total_combos = len(grid_combos)
    league_started = perf_counter()

    for combo_index, (xi, lookback, rho) in enumerate(grid_combos, start=1):
        if log_progress and (
            combo_index == 1
            or combo_index == total_combos
            or combo_index % 10 == 0
        ):
            print(
                f"  league={league_id} grid {combo_index}/{total_combos} "
                f"xi={xi} lookback={lookback} "
                f"rho={'mle' if rho is None else rho}",
                flush=True,
            )

        walk_forward, rho_diagnostics = _evaluate_parameter_set(
            league_id=league_id,
            eval_matches=eval_matches,
            fixture_index=fixture_index,
            xi=xi,
            lookback=lookback,
            rho=default_rho if rho is None else rho,
            min_team_matches=min_team_matches,
            min_training_matches=min_training_matches,
            max_goals=max_goals,
            window_cache=league_window_cache,
            use_warm_start=use_warm_start,
            fit_rho=fit_rho,
            rho_min=rho_min,
            rho_max=rho_max,
        )

        if lookback not in avg_training_by_lookback:
            avg_training_by_lookback[lookback] = _average_training_size(
                fixture_index=fixture_index,
                eval_matches=eval_matches,
                lookback=lookback,
                min_team_matches=min_team_matches,
                window_cache=league_window_cache,
            )
        avg_training = avg_training_by_lookback[lookback]

        parameter_results.append(
            DixonColesParameterResult(
                league_id=league_id,
                xi=xi,
                lookback=lookback,
                # With fit_rho the representative rho is the median MLE fit.
                rho=(
                    rho_diagnostics.median
                    if rho is None and rho_diagnostics.median is not None
                    else (default_rho if rho is None else rho)
                ),
                log_loss=walk_forward.dc_log_loss,
                rps=walk_forward.dc_rps,
                num_predictions=walk_forward.n_scored,
                skipped_matches=walk_forward.n_skipped,
                avg_training_matches=avg_training,
                rho_diagnostics=rho_diagnostics,
            )
        )

    best = min(parameter_results, key=lambda row: row.log_loss)
    if log_progress:
        elapsed = perf_counter() - league_started
        print(
            f"Finished league={league_id} in {elapsed:.1f}s "
            f"(best log_loss={best.log_loss:.4f}) "
            f"{best.rho_diagnostics.summary()}",
            flush=True,
        )

    return DixonColesLeagueOptimizationResult(
        league_id=league_id,
        best_xi=best.xi,
        best_lookback=best.lookback,
        best_rho=best.rho,
        log_loss=best.log_loss,
        rps=best.rps,
        evaluated_matches=best.num_predictions,
        skipped_matches=best.skipped_matches,
        validation_start=validation_start,
        validation_end=validation_end,
        parameter_results=parameter_results,
        rho_diagnostics=best.rho_diagnostics,
    )


def _run_league_optimization_job(
    job: _LeagueOptimizationJob,
) -> DixonColesLeagueOptimizationResult:
    return optimize_single_league(
        job.league_id,
        job.eval_matches,
        job.league_fixtures,
        validation_start=job.validation_start,
        validation_end=job.validation_end,
        xi_values=job.xi_values,
        lookback_values=job.lookback_values,
        rho_values=job.rho_values,
        min_training_matches=job.min_training_matches,
        min_team_matches=job.min_team_matches,
        max_goals=job.max_goals,
        log_progress=job.log_progress,
        fit_rho=job.fit_rho,
        rho_min=job.rho_min,
        rho_max=job.rho_max,
    )


def _format_league_result_summary(
    league_result: DixonColesLeagueOptimizationResult,
    *,
    elapsed_seconds: float | None = None,
) -> str:
    timing = f" in {elapsed_seconds:.1f}s" if elapsed_seconds is not None else ""
    return (
        f"league={league_result.league_id}{timing} "
        f"log_loss={league_result.log_loss:.4f} "
        f"xi={league_result.best_xi} "
        f"lookback={league_result.best_lookback} "
        f"rho={league_result.best_rho} "
        f"scored={league_result.evaluated_matches}"
        + (
            f" {league_result.rho_diagnostics.summary()}"
            if league_result.rho_diagnostics.n_fits
            else ""
        )
    )


class DixonColesOptimizer:
    """Walk-forward grid search over xi, lookback, and rho per league."""

    def optimize(
        self,
        eval_matches_by_league: dict[int, list[EvalMatch]],
        fixtures_by_league: dict[int, list[DixonColesMatch]],
        *,
        validation_start: date,
        validation_end: date,
        xi_values: list[float],
        lookback_values: list[int],
        rho_values: list[float],
        min_training_matches: int,
        min_team_matches: int,
        min_eval_matches_per_league: int,
        max_goals: int = 10,
        jobs: int = 1,
        fit_rho: bool = False,
        rho_min: float = -0.2,
        rho_max: float = 0.2,
    ) -> DixonColesOptimizationResult:
        league_results: dict[int, DixonColesLeagueOptimizationResult] = {}
        skipped_leagues: dict[int, str] = {}
        jobs = max(1, jobs)

        eligible_leagues: list[tuple[int, list[EvalMatch]]] = []
        for league_id, eval_matches in sorted(eval_matches_by_league.items()):
            if len(eval_matches) < min_eval_matches_per_league:
                skipped_leagues[league_id] = (
                    f"too few validation matches ({len(eval_matches)} "
                    f"< {min_eval_matches_per_league})"
                )
                continue
            eligible_leagues.append((league_id, eval_matches))

        total_leagues = len(eligible_leagues)
        grid_combos_per_league = len(xi_values) * len(lookback_values)
        if not fit_rho:
            grid_combos_per_league *= len(rho_values)
        parallel_workers = min(jobs, total_leagues) if total_leagues else jobs
        print(
            f"League optimization: {total_leagues} leagues, "
            f"{grid_combos_per_league} grid combos/league, "
            f"{parallel_workers} worker(s), "
            f"rho={'MLE' if fit_rho else 'grid'}",
            flush=True,
        )
        if skipped_leagues:
            skipped_ids = ", ".join(str(league_id) for league_id in sorted(skipped_leagues))
            print(
                f"Skipping {len(skipped_leagues)} league(s): {skipped_ids}",
                flush=True,
            )
        for index, (league_id, eval_matches) in enumerate(eligible_leagues, start=1):
            fixture_count = len(fixtures_by_league.get(league_id, []))
            print(
                f"  [{index}/{total_leagues}] queued league={league_id} "
                f"({len(eval_matches)} validation matches, "
                f"{fixture_count} fixtures)",
                flush=True,
            )

        overall_started = perf_counter()
        worker_log_progress = jobs == 1

        if jobs == 1:
            for index, (league_id, eval_matches) in enumerate(eligible_leagues, start=1):
                print(
                    f"[{index}/{total_leagues}] starting league={league_id}",
                    flush=True,
                )
                league_started = perf_counter()
                league_results[league_id] = optimize_single_league(
                    league_id,
                    eval_matches,
                    fixtures_by_league.get(league_id, []),
                    validation_start=validation_start,
                    validation_end=validation_end,
                    xi_values=xi_values,
                    lookback_values=lookback_values,
                    rho_values=rho_values,
                    min_training_matches=min_training_matches,
                    min_team_matches=min_team_matches,
                    max_goals=max_goals,
                    log_progress=worker_log_progress,
                    fit_rho=fit_rho,
                    rho_min=rho_min,
                    rho_max=rho_max,
                )
                elapsed = perf_counter() - league_started
                print(
                    f"[{index}/{total_leagues}] finished "
                    f"{_format_league_result_summary(league_results[league_id], elapsed_seconds=elapsed)}",
                    flush=True,
                )
                if index < total_leagues:
                    overall_elapsed = perf_counter() - overall_started
                    remaining = total_leagues - index
                    eta_seconds = overall_elapsed / index * remaining
                    print(
                        f"Progress: {index}/{total_leagues} leagues done, "
                        f"elapsed {overall_elapsed:.0f}s, "
                        f"ETA ~{eta_seconds:.0f}s",
                        flush=True,
                    )
        else:
            optimization_jobs = [
                _LeagueOptimizationJob(
                    league_id=league_id,
                    eval_matches=eval_matches,
                    league_fixtures=fixtures_by_league.get(league_id, []),
                    validation_start=validation_start,
                    validation_end=validation_end,
                    xi_values=xi_values,
                    lookback_values=lookback_values,
                    rho_values=rho_values,
                    min_training_matches=min_training_matches,
                    min_team_matches=min_team_matches,
                    max_goals=max_goals,
                    log_progress=worker_log_progress,
                    fit_rho=fit_rho,
                    rho_min=rho_min,
                    rho_max=rho_max,
                )
                for league_id, eval_matches in eligible_leagues
            ]
            completed = 0
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                futures = {
                    executor.submit(_run_league_optimization_job, job): job.league_id
                    for job in optimization_jobs
                }
                for future in as_completed(futures):
                    league_id = futures[future]
                    league_result = future.result()
                    league_results[league_id] = league_result
                    completed += 1
                    overall_elapsed = perf_counter() - overall_started
                    remaining = total_leagues - completed
                    eta_seconds = (
                        overall_elapsed / completed * remaining / parallel_workers
                        if completed and remaining
                        else 0.0
                    )
                    print(
                        f"[{completed}/{total_leagues}] finished "
                        f"{_format_league_result_summary(league_result)} "
                        f"(elapsed {overall_elapsed:.0f}s"
                        f"{f', ETA ~{eta_seconds:.0f}s' if remaining else ''})",
                        flush=True,
                    )

        total_elapsed = perf_counter() - overall_started
        print(
            f"League optimization complete: {len(league_results)} leagues in "
            f"{total_elapsed:.1f}s",
            flush=True,
        )

        return DixonColesOptimizationResult(
            league_results=league_results,
            skipped_leagues=skipped_leagues,
            validation_start=validation_start,
            validation_end=validation_end,
        )


def group_eval_matches_by_league(
    eval_matches: list[EvalMatch],
) -> dict[int, list[EvalMatch]]:
    grouped: dict[int, list[EvalMatch]] = defaultdict(list)
    for match in eval_matches:
        grouped[match.league_external_id].append(match)
    return dict(grouped)
