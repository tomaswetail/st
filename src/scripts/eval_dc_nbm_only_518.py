#!/usr/bin/env python3
"""100% Dixon–Coles NBM on official 518 (eval-only). No production change.

Walk-forward fits ``DixonColesNBMModel`` per ``(league_external_id, as_of)``
from the live DB. Does not train HGB, does not call ``DixonColesService.fit_league``,
and does not fall back to CSV λ.

```bash
env -u PYTHONPATH /tmp/st-diag-venv/bin/python -m src.scripts.eval_dc_nbm_only_518
env -u PYTHONPATH /tmp/st-diag-venv/bin/python -m src.scripts.eval_dc_nbm_only_518 \
  --min-team-matches 3 --json artifacts/dc_nbm_only_518_min3.json
```
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import select, text
from sqlalchemy.orm import selectinload

from src.calc.dixon_coles.service import DixonColesService
from src.calc.dixon_coles.types import DixonColesMatch
from src.calc.dixon_coles_nbm import DixonColesNBMModel
from src.calc.probability_metrics import multiclass_log_loss
from src.calc.residual_ml import load_dataset_rows, select_backtest_rows
from src.calc.residual_ml.baseline import normalize_probabilities
from src.database import SessionLocal
from src.objects.models.st_match import STMatchModel
from src.objects.schema.data_classes.data_sources import DataSourceConfig
from src.utils.repo_paths import resolve_repo_path

OFFICIAL_N = 518
CLASSIC_VALID_DC_N_CITED = 465
CLASSIC_DC_LOG_LOSS_CITED = 1.0471
SHIP_SHRINK_LOG_LOSS_CITED = 1.0046
PRIOR_MIN5_NBM_LOG_LOSS = 1.0143
PRIOR_MIN5_N_SCORED = 269
PRIOR_MIN5_UNKNOWN_TEAM = 189
FIXED_RHO = -0.13
INITIAL_PHI = 0.05
LABEL_TO_INDEX = {"1": 0, "X": 1, "2": 2}
SKIP_REASONS = (
    "missing_label",
    "bad_as_of",
    "missing_league",
    "missing_st_match",
    "unknown_team_id",
    "fit_fail",
    "unknown_team",
    "nonfinite_1x2",
)
STOCKHOLM = ZoneInfo("Europe/Stockholm")


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def parse_as_of(row: dict[str, Any]) -> date | None:
    cutoff = row.get("feature_cutoff_date")
    if cutoff not in (None, ""):
        try:
            return date.fromisoformat(str(cutoff)[:10])
        except ValueError:
            pass
    match_date = row.get("match_date")
    if match_date not in (None, ""):
        try:
            return date.fromisoformat(str(match_date)[:10])
        except ValueError:
            pass
    return None


def parse_league_id(row: dict[str, Any]) -> int | None:
    raw = row.get("league_external_id")
    if raw in (None, ""):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def parse_match_id(row: dict[str, Any]) -> int | None:
    raw = row.get("match_id")
    if raw in (None, ""):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def parse_label(row: dict[str, Any]) -> str | None:
    label = str(row.get("label") or "").strip().upper()
    if label not in LABEL_TO_INDEX:
        return None
    return label


def team_api_id(team: Any) -> int | None:
    if team is None or getattr(team, "external_id", None) is None:
        return None
    try:
        return int(team.external_id)
    except (TypeError, ValueError):
        return None


def classic_dc_probs(row: dict[str, Any]) -> tuple[float, float, float] | None:
    norm = (
        _optional_float(row.get("p_home_dc_norm")),
        _optional_float(row.get("p_draw_dc_norm")),
        _optional_float(row.get("p_away_dc_norm")),
    )
    if all(value is not None for value in norm):
        return norm  # type: ignore[return-value]
    raw = (
        _optional_float(row.get("p_home_dc")),
        _optional_float(row.get("p_draw_dc")),
        _optional_float(row.get("p_away_dc")),
    )
    if any(value is None for value in raw):
        return None
    normalized = normalize_probabilities({"1": raw[0], "X": raw[1], "2": raw[2]})
    if normalized is None:
        return None
    return normalized["1"], normalized["X"], normalized["2"]


def market_probs(row: dict[str, Any]) -> tuple[float, float, float] | None:
    values = (
        _optional_float(row.get("p_home_market_norm")),
        _optional_float(row.get("p_draw_market_norm")),
        _optional_float(row.get("p_away_market_norm")),
    )
    if any(value is None for value in values):
        return None
    return values  # type: ignore[return-value]


def _finite_1x2(p_home: float, p_draw: float, p_away: float) -> bool:
    values = (p_home, p_draw, p_away)
    if any(not math.isfinite(value) for value in values):
        return False
    return not all(value == 0.0 for value in values)


def score_log_loss(
    labels: list[str],
    probability_rows: list[tuple[float, float, float]],
) -> float | None:
    if not labels:
        return None
    y_true = [LABEL_TO_INDEX[label] for label in labels]
    y_prob = [list(probs) for probs in probability_rows]
    return float(multiclass_log_loss(y_true, y_prob))


def try_open_session() -> tuple[Any | None, str | None]:
    try:
        session = SessionLocal()
        session.execute(text("SELECT 1"))
        return session, None
    except Exception as exc:  # noqa: BLE001 — diagnostic fallback
        return None, f"{type(exc).__name__}: {exc}"


def load_st_matches(session: Any, match_ids: list[int]) -> dict[int, STMatchModel]:
    if not match_ids:
        return {}
    matches = list(
        session.scalars(
            select(STMatchModel)
            .options(
                selectinload(STMatchModel.home_team),
                selectinload(STMatchModel.away_team),
            )
            .where(STMatchModel.id.in_(match_ids))
        ).all()
    )
    return {int(match.id): match for match in matches}


def _round4(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 4)


def _phi_stats(models: list[DixonColesNBMModel]) -> dict[str, Any]:
    phis = [float(model.phi) for model in models]
    if not phis:
        return {
            "n": 0,
            "min": None,
            "median": None,
            "max": None,
            "n_at_bound": 0,
        }
    return {
        "n": len(phis),
        "min": min(phis),
        "median": statistics.median(phis),
        "max": max(phis),
        "n_at_bound": sum(1 for model in models if model.phi_at_bound),
    }


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/dc_rho_mle_promotion/dataset.csv"),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("artifacts/dc_nbm_only_518.json"),
    )
    parser.add_argument(
        "--min-team-matches",
        type=int,
        default=None,
        help=(
            "Override DixonColesNBMModel min_team_matches. "
            "When unset, uses DataSourceConfig.classic_dc_min_team_matches."
        ),
    )
    args = parser.parse_args()

    dataset_path = resolve_repo_path(args.dataset)
    json_path = resolve_repo_path(args.json)
    if json_path.name.startswith("gate_eval"):
        raise SystemExit("Refusing to overwrite a gate_eval artifact")

    started_at = datetime.now(STOCKHOLM)
    interpreter = sys.executable
    config = DataSourceConfig()
    if args.min_team_matches is not None:
        if args.min_team_matches < 1:
            raise SystemExit("--min-team-matches must be >= 1")
        min_team_matches = int(args.min_team_matches)
        min_team_matches_source = "cli"
    else:
        min_team_matches = int(config.classic_dc_min_team_matches)
        min_team_matches_source = "config.classic_dc_min_team_matches"

    all_rows = load_dataset_rows(dataset_path)
    official_rows, official_filter = select_backtest_rows(all_rows)
    official_n = len(official_rows)
    print(
        f"interpreter={interpreter} official_n={official_n} filter={official_filter} "
        f"min_team_matches={min_team_matches} ({min_team_matches_source})",
        flush=True,
    )

    base_payload: dict[str, Any] = {
        "kind": "dc_nbm_only_518_diagnostic",
        "production_unchanged": True,
        "csv_lambda_fallback_used": False,
        "interpreter": interpreter,
        "dataset": str(dataset_path),
        "csv_n": len(all_rows),
        "official_n": official_n,
        "official_filter": official_filter,
        "started_at": started_at.isoformat(timespec="seconds"),
        "fixed_rho": FIXED_RHO,
        "fit_rho": False,
        "fit_phi": True,
        "min_team_matches": min_team_matches,
        "min_team_matches_source": min_team_matches_source,
        "config_classic_dc_min_team_matches_default": int(
            config.classic_dc_min_team_matches
        ),
        "prior_min5": {
            "min_team_matches": 5,
            "nbm_log_loss": PRIOR_MIN5_NBM_LOG_LOSS,
            "n_scored": PRIOR_MIN5_N_SCORED,
            "unknown_team": PRIOR_MIN5_UNKNOWN_TEAM,
        },
    }

    if official_n != OFFICIAL_N:
        payload = {
            **base_payload,
            "cannot_measure": True,
            "stop_reason": "official_n_mismatch",
            "expected_official_n": OFFICIAL_N,
        }
        _write_payload(json_path, payload)
        print(
            f"STOP: official n={official_n}, expected {OFFICIAL_N}. Cannot measure.",
            flush=True,
        )
        return 1

    session, connect_error = try_open_session()
    if session is None:
        payload = {
            **base_payload,
            "cannot_measure": True,
            "stop_reason": "database_unavailable",
            "connect_error": connect_error,
        }
        _write_payload(json_path, payload)
        print(
            f"STOP: cannot measure — database unavailable ({connect_error}). "
            "No CSV-λ fallback.",
            flush=True,
        )
        return 1

    skip_counts = Counter({reason: 0 for reason in SKIP_REASONS})
    scored: list[dict[str, Any]] = []
    cache: dict[tuple[int, date], DixonColesNBMModel | None] = {}
    fit_errors: list[str] = []
    try:
        service = DixonColesService(session, config=config)
        match_ids = [
            match_id
            for match_id in (parse_match_id(row) for row in official_rows)
            if match_id is not None
        ]
        matches_by_id = load_st_matches(session, match_ids)
        print(f"loaded_st_matches={len(matches_by_id)} of {len(match_ids)}", flush=True)

        needed_as_ofs: list[date] = []
        needed_leagues: set[int] = set()
        for row in official_rows:
            as_of = parse_as_of(row)
            league_id = parse_league_id(row)
            if as_of is None or league_id is None:
                continue
            needed_as_ofs.append(as_of)
            needed_leagues.add(league_id)

        fixtures_by_league: dict[int, list[DixonColesMatch]] = {}
        if needed_as_ofs and needed_leagues:
            max_as_of = max(needed_as_ofs)
            min_as_of = min(needed_as_ofs)
            max_lookback = max(
                service.params_for_league(league_id)[1] for league_id in needed_leagues
            )
            preload_lookback = (max_as_of - min_as_of).days + max_lookback
            print(
                f"preload leagues={sorted(needed_leagues)} "
                f"before={max_as_of} lookback_days={preload_lookback}",
                flush=True,
            )
            fixtures_by_league = service.preload_league_fixtures(
                sorted(needed_leagues),
                before_date=max_as_of,
                max_lookback_days=preload_lookback,
            )
            for league_id, history in fixtures_by_league.items():
                print(f"  league={league_id} history_n={len(history)}", flush=True)
    finally:
        session.close()

    for index, row in enumerate(official_rows, start=1):
        label = parse_label(row)
        if label is None:
            skip_counts["missing_label"] += 1
            continue
        as_of = parse_as_of(row)
        if as_of is None:
            skip_counts["bad_as_of"] += 1
            continue
        league_id = parse_league_id(row)
        if league_id is None:
            skip_counts["missing_league"] += 1
            continue
        match_id = parse_match_id(row)
        match = matches_by_id.get(match_id) if match_id is not None else None
        if match is None:
            skip_counts["missing_st_match"] += 1
            continue
        home_id = team_api_id(match.home_team)
        away_id = team_api_id(match.away_team)
        if home_id is None or away_id is None:
            skip_counts["unknown_team_id"] += 1
            continue

        cache_key = (league_id, as_of)
        if cache_key not in cache:
            xi, lookback_days, _ignored_rho = service.params_for_league(league_id)
            history = list(fixtures_by_league.get(league_id, []))
            model = DixonColesNBMModel(
                xi=xi,
                rho=FIXED_RHO,
                phi=INITIAL_PHI,
                max_goals=config.dixon_coles_max_goals,
                lookback_days=lookback_days,
                min_team_matches=min_team_matches,
                as_of=as_of,
                fit_rho=False,
                fit_phi=True,
            )
            print(
                f"fit {len(cache) + 1} league={league_id} as_of={as_of} "
                f"history_n={len(history)} xi={xi} lookback={lookback_days} "
                f"min_team_matches={min_team_matches}",
                flush=True,
            )
            try:
                model.fit(history, as_of=as_of)
            except (ValueError, RuntimeError) as exc:
                cache[cache_key] = None
                fit_errors.append(f"{cache_key}: {type(exc).__name__}: {exc}")
                print(f"  fit_fail {cache_key}: {exc}", flush=True)
            else:
                cache[cache_key] = model
                print(
                    f"  ok teams={len(model.team_ids)} "
                    f"phi={model.phi:.6f} rho={model.rho:.6f} "
                    f"n_train={model.n_training_matches}",
                    flush=True,
                )

        model = cache[cache_key]
        if model is None:
            skip_counts["fit_fail"] += 1
            continue
        if home_id not in model.team_ids or away_id not in model.team_ids:
            skip_counts["unknown_team"] += 1
            continue
        prediction = model.predict(home_id, away_id)
        if not _finite_1x2(prediction.p_home, prediction.p_draw, prediction.p_away):
            skip_counts["nonfinite_1x2"] += 1
            continue
        scored.append(
            {
                "match_id": match_id,
                "label": label,
                "league_id": league_id,
                "as_of": as_of.isoformat(),
                "rho": float(model.rho),
                "phi": float(model.phi),
                "p_home": float(prediction.p_home),
                "p_draw": float(prediction.p_draw),
                "p_away": float(prediction.p_away),
                "market": market_probs(row),
                "classic_dc": classic_dc_probs(row),
            }
        )
        if index % 50 == 0 or index == official_n:
            print(
                f"progress {index}/{official_n} scored={len(scored)} "
                f"skipped={official_n - len(scored)} cache={len(cache)}",
                flush=True,
            )

    n_scored = len(scored)
    n_skipped = official_n - n_scored
    successful_fits = [model for model in cache.values() if model is not None]
    cached_none = sum(1 for model in cache.values() if model is None)
    all_scored_rho_fixed = all(
        math.isclose(float(row["rho"]), FIXED_RHO, abs_tol=1e-12) for row in scored
    )
    nbm_log_loss = score_log_loss(
        [row["label"] for row in scored],
        [(row["p_home"], row["p_draw"], row["p_away"]) for row in scored],
    )
    market_same = [row for row in scored if row["market"] is not None]
    market_log_loss = score_log_loss(
        [row["label"] for row in market_same],
        [row["market"] for row in market_same],
    )

    intersection = [
        row for row in scored if row["classic_dc"] is not None
    ]
    intersection_nbm = score_log_loss(
        [row["label"] for row in intersection],
        [(row["p_home"], row["p_draw"], row["p_away"]) for row in intersection],
    )
    intersection_market_rows = [
        row for row in intersection if row["market"] is not None
    ]
    intersection_market = score_log_loss(
        [row["label"] for row in intersection_market_rows],
        [row["market"] for row in intersection_market_rows],
    )
    intersection_classic = score_log_loss(
        [row["label"] for row in intersection],
        [row["classic_dc"] for row in intersection],
    )

    official_classic_n = sum(
        1 for row in official_rows if classic_dc_probs(row) is not None
    )
    fail_closed_labels: list[str] = []
    fail_closed_probs: list[tuple[float, float, float]] = []
    fail_closed_nbm = 0
    fail_closed_market = 0
    scored_ids = {row["match_id"] for row in scored}
    scored_by_id = {row["match_id"]: row for row in scored}
    for row in official_rows:
        match_id = parse_match_id(row)
        label = parse_label(row)
        if label is None or match_id is None:
            continue
        if match_id in scored_ids:
            scored_row = scored_by_id[match_id]
            fail_closed_labels.append(label)
            fail_closed_probs.append(
                (scored_row["p_home"], scored_row["p_draw"], scored_row["p_away"])
            )
            fail_closed_nbm += 1
            continue
        market = market_probs(row)
        if market is None:
            continue
        fail_closed_labels.append(label)
        fail_closed_probs.append(market)
        fail_closed_market += 1
    fail_closed_log_loss = score_log_loss(fail_closed_labels, fail_closed_probs)

    finished_at = datetime.now(STOCKHOLM)
    payload = {
        **base_payload,
        "cannot_measure": False,
        "connect_error": connect_error,
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "n_scored": n_scored,
        "n_skipped": n_skipped,
        "skip_counts": dict(skip_counts),
        "unique_cached_keys": len(cache),
        "n_successful_fits": len(successful_fits),
        "n_cached_none": cached_none,
        "phi_over_successful_fits": _phi_stats(successful_fits),
        "every_scored_row_rho_neg_0_13": all_scored_rho_fixed,
        "csv_lambda_fallback_used": False,
        "nbm_log_loss": nbm_log_loss,
        "nbm_log_loss_4dp": _round4(nbm_log_loss),
        "market_same_rows": {
            "n": len(market_same),
            "log_loss": market_log_loss,
            "log_loss_4dp": _round4(market_log_loss),
        },
        "classic_valid_dc_official_n": official_classic_n,
        "intersection_classic_valid_dc": {
            "n": len(intersection),
            "nbm_log_loss": intersection_nbm,
            "nbm_log_loss_4dp": _round4(intersection_nbm),
            "market_n": len(intersection_market_rows),
            "market_log_loss": intersection_market,
            "market_log_loss_4dp": _round4(intersection_market),
            "csv_classic_dc_log_loss": intersection_classic,
            "csv_classic_dc_log_loss_4dp": _round4(intersection_classic),
        },
        "fail_closed_nbm_or_market": {
            "label": "not 100% NBM",
            "n": len(fail_closed_labels),
            "n_nbm": fail_closed_nbm,
            "n_market": fail_closed_market,
            "log_loss": fail_closed_log_loss,
            "log_loss_4dp": _round4(fail_closed_log_loss),
        },
        "cited": {
            "classic_100pct_dc_log_loss": CLASSIC_DC_LOG_LOSS_CITED,
            "classic_100pct_dc_n": CLASSIC_VALID_DC_N_CITED,
            "ship_shrink_0_7_log_loss": SHIP_SHRINK_LOG_LOSS_CITED,
            "ship_n": OFFICIAL_N,
        },
        "fit_errors": fit_errors,
        "timezone": "Europe/Stockholm",
    }
    _write_payload(json_path, payload)

    print(
        f"PRIMARY 100% NBM log-loss={_round4(nbm_log_loss)} n={n_scored} "
        f"skipped={n_skipped} min_team_matches={min_team_matches} "
        f"csv_lambda_fallback_used=false",
        flush=True,
    )
    print(
        f"prior_min5 NBM LL={PRIOR_MIN5_NBM_LOG_LOSS} n={PRIOR_MIN5_N_SCORED} "
        f"unknown_team={PRIOR_MIN5_UNKNOWN_TEAM}",
        flush=True,
    )
    print(
        f"market_same_rows log-loss={_round4(market_log_loss)} n={len(market_same)}",
        flush=True,
    )
    print(f"skip_counts={dict(skip_counts)}", flush=True)
    print(
        f"cache keys={len(cache)} successful_fits={len(successful_fits)} "
        f"cached_none={cached_none}",
        flush=True,
    )
    print(f"phi_stats={_phi_stats(successful_fits)}", flush=True)
    print(f"every_scored_row_rho_neg_0_13={all_scored_rho_fixed}", flush=True)
    if n_scored != CLASSIC_VALID_DC_N_CITED:
        print(
            f"intersection n={len(intersection)} "
            f"nbm={_round4(intersection_nbm)} "
            f"market={_round4(intersection_market)} "
            f"csv_classic={_round4(intersection_classic)}",
            flush=True,
        )
    print(
        f"fail_closed (not 100% NBM) log-loss={_round4(fail_closed_log_loss)} "
        f"n={len(fail_closed_labels)} nbm={fail_closed_nbm} market={fail_closed_market}",
        flush=True,
    )
    print(
        f"elapsed_s={(finished_at - started_at).total_seconds():.1f} "
        f"interpreter={interpreter}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
