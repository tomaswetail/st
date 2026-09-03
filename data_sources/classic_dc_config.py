"""Load classic Dixon–Coles optimization grid and per-league params from JSON."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from objects.schema.data_classes.data_sources import DataSourceConfig

logger = logging.getLogger(__name__)

CLASSIC_DC_LEAGUE_PARAMS_RELATIVE = Path("config/classic_dc_league_params.json")
CLASSIC_DC_OPTIMIZATION_GRID_RELATIVE = Path("config/classic_dc_optimization_grid.json")


@dataclass(frozen=True)
class ClassicDcOptimizationGrid:
    xi_values: list[float]
    lookback_values: list[int]
    rho_values: list[float]
    min_eval_matches_per_league: int
    min_training_matches: int
    min_team_matches: int


@dataclass(frozen=True)
class ClassicDcLeagueParams:
    xi: float
    lookback: int
    rho: float
    log_loss: float | None = None
    rps: float | None = None
    n_evaluated: int | None = None
    validation_start: str | None = None
    validation_end: str | None = None


def default_optimization_grid_path() -> Path:
    from utils.repo_paths import resolve_repo_path

    return resolve_repo_path(CLASSIC_DC_OPTIMIZATION_GRID_RELATIVE)


def default_league_params_path() -> Path:
    from utils.repo_paths import resolve_repo_path

    return resolve_repo_path(CLASSIC_DC_LEAGUE_PARAMS_RELATIVE)


def load_optimization_grid(path: Path | None = None) -> ClassicDcOptimizationGrid:
    config_path = path or default_optimization_grid_path()
    if not config_path.exists():
        raise FileNotFoundError(f"Classic DC optimization grid not found: {config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return ClassicDcOptimizationGrid(
        xi_values=[float(value) for value in raw["xi_values"]],
        lookback_values=[int(value) for value in raw["lookback_values"]],
        rho_values=[float(value) for value in raw["rho_values"]],
        min_eval_matches_per_league=int(raw.get("min_eval_matches_per_league", 15)),
        min_training_matches=int(raw.get("min_training_matches", 50)),
        min_team_matches=int(raw.get("min_team_matches", 5)),
    )


def load_league_params(path: Path | None = None) -> dict[int, ClassicDcLeagueParams]:
    config_path = path or default_league_params_path()
    if not config_path.exists():
        return {}
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        logger.exception("Failed reading classic DC league params %s", config_path)
        return {}
    if not isinstance(raw, dict):
        return {}

    params: dict[int, ClassicDcLeagueParams] = {}
    for league_key, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        try:
            league_id = int(league_key)
        except (TypeError, ValueError):
            continue
        params[league_id] = ClassicDcLeagueParams(
            xi=float(payload["xi"]),
            lookback=int(payload["lookback"]),
            rho=float(payload["rho"]),
            log_loss=(
                float(payload["log_loss"])
                if payload.get("log_loss") is not None
                else None
            ),
            rps=float(payload["rps"]) if payload.get("rps") is not None else None,
            n_evaluated=(
                int(payload["n_evaluated"])
                if payload.get("n_evaluated") is not None
                else None
            ),
            validation_start=payload.get("validation_start"),
            validation_end=payload.get("validation_end"),
        )
    return params


def write_league_params(
    league_results: dict[int, ClassicDcLeagueParams],
    path: Path,
) -> None:
    payload = {
        str(league_id): {
            "xi": entry.xi,
            "lookback": entry.lookback,
            "rho": entry.rho,
            **(
                {"log_loss": entry.log_loss}
                if entry.log_loss is not None
                else {}
            ),
            **({"rps": entry.rps} if entry.rps is not None else {}),
            **(
                {"n_evaluated": entry.n_evaluated}
                if entry.n_evaluated is not None
                else {}
            ),
            **(
                {"validation_start": entry.validation_start}
                if entry.validation_start
                else {}
            ),
            **(
                {"validation_end": entry.validation_end}
                if entry.validation_end
                else {}
            ),
        }
        for league_id, entry in sorted(league_results.items())
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
