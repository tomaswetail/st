#!/usr/bin/env python3
"""DC scoreline leftover diagnostic (read-only). No production change.

Joins CSV match_id → stryktipset_matches.id (ST FT scores preferred;
fixture fallback only when ST scores are null). Bins on classic DC λ_H+λ_A.
Does not train HGB, refit DC, or touch production files.

```bash
env -u PYTHONPATH /tmp/st-diag-venv/bin/python -m src.scripts.eval_dc_scoreline_leftover
```
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.orm import selectinload

from src.calc.player_availability_calculator import PlayerAvailabilityCalculator
from src.calc.residual_ml import load_dataset_rows, select_backtest_rows
from src.calc.residual_ml.lambda_shock import rho_for_league
from src.calc.strength_calculator import dixon_coles_matrix
from src.data_sources.classic_dc_config import load_league_params
from src.database import SessionLocal
from src.objects.models.st_match import STMatchModel
from src.objects.schema.data_classes.data_sources import DataSourceConfig
from src.utils.repo_paths import resolve_repo_path

FINISHED_FIXTURE_STATUSES = frozenset({"FT", "AET", "PEN", "AWD", "WO"})

LOCKED_BINS: tuple[tuple[str, float | None, float | None], ...] = (
    ("low", None, 2.2),
    ("mid", 2.2, 2.8),
    ("high", 2.8, None),
)
PREREGISTERED_BIN_N = {"low": 718, "mid": 905, "high": 672}
PREREGISTERED_VALID_LAMBDA_N = 2295
LOW_SCORING_SPLIT = 0.5
CLOSE_MATCH_SPLIT = 0.65
SCORELINE_EVENTS = ("00", "11", "22", "any")


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


def _lambda_pair(row: dict[str, Any]) -> tuple[float, float] | None:
    lambda_home = _optional_float(row.get("expected_home_goals"))
    lambda_away = _optional_float(row.get("expected_away_goals"))
    if lambda_home is None or lambda_away is None:
        return None
    return lambda_home, lambda_away


def _lambda_total(row: dict[str, Any]) -> float | None:
    pair = _lambda_pair(row)
    if pair is None:
        return None
    return pair[0] + pair[1]


def _bin_name(lambda_total: float) -> str:
    if lambda_total <= 2.2:
        return "low"
    if lambda_total <= 2.8:
        return "mid"
    return "high"


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _quantile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _lambda_stats(values: list[float]) -> dict[str, float | None]:
    ordered = sorted(values)
    return {
        "n": len(values),
        "min": ordered[0] if ordered else None,
        "p25": _quantile(ordered, 0.25),
        "median": _quantile(ordered, 0.50),
        "p75": _quantile(ordered, 0.75),
        "mean": _mean(values),
        "max": ordered[-1] if ordered else None,
    }


def _consistent_with_noise(leftover: float, standard_error: float, n: int) -> bool:
    if abs(leftover) <= 2.0 * standard_error:
        return True
    return n < 80 and abs(leftover) < 0.02


def _event_block(
    realized_rate: float,
    predicted_mean: float,
    n: int,
) -> dict[str, Any]:
    leftover = realized_rate - predicted_mean
    standard_error = math.sqrt(
        predicted_mean * (1.0 - predicted_mean) / n
    ) if n > 0 else float("nan")
    consistent = _consistent_with_noise(leftover, standard_error, n)
    return {
        "n": n,
        "realized": realized_rate,
        "mean_dc_p": predicted_mean,
        "leftover": leftover,
        "se": standard_error,
        "two_se": 2.0 * standard_error,
        "consistent_with_noise": consistent,
        "fails_noise": not consistent,
    }


def scoreline_leftovers(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    n = len(rows)
    realized = {
        "00": sum(1 for row in rows if row["home_goals"] == 0 and row["away_goals"] == 0) / n,
        "11": sum(1 for row in rows if row["home_goals"] == 1 and row["away_goals"] == 1) / n,
        "22": sum(1 for row in rows if row["home_goals"] == 2 and row["away_goals"] == 2) / n,
    }
    realized["any"] = realized["00"] + realized["11"] + realized["22"]
    predicted = {
        "00": _mean([row["p00"] for row in rows]) or 0.0,
        "11": _mean([row["p11"] for row in rows]) or 0.0,
        "22": _mean([row["p22"] for row in rows]) or 0.0,
        "any": _mean([row["p_any"] for row in rows]) or 0.0,
    }
    events = {key: _event_block(realized[key], predicted[key], n) for key in SCORELINE_EVENTS}
    return {"n": n, "events": events}


def draw_1x2_leftover(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    usable: list[dict[str, Any]] = []
    for row in rows:
        p_draw = _optional_float(row.get("p_draw_dc"))
        label = str(row.get("label") or "").strip().upper()
        if p_draw is None or label not in {"1", "X", "2"}:
            continue
        usable.append(row)
    if not usable:
        return None
    n = len(usable)
    realized = sum(1 for row in usable if str(row["label"]).strip().upper() == "X") / n
    predicted = _mean([float(row["p_draw_dc"]) for row in usable]) or 0.0
    block = _event_block(realized, predicted, n)
    block["source"] = "csv_p_draw_dc"
    return block


def attach_dc_scoreline_probs(
    rows: list[dict[str, Any]],
    *,
    league_params: dict[int, Any],
    default_rho: float,
) -> list[dict[str, Any]]:
    attached: list[dict[str, Any]] = []
    for row in rows:
        pair = _lambda_pair(row)
        if pair is None:
            continue
        lambda_home, lambda_away = pair
        rho = rho_for_league(
            row.get("league_external_id"),
            default_rho=default_rho,
            league_params=league_params,
        )
        matrix, _p_home, _p_draw, _p_away = dixon_coles_matrix(
            lambda_home, lambda_away, rho
        )
        copy = dict(row)
        copy["lambda_home"] = lambda_home
        copy["lambda_away"] = lambda_away
        copy["lambda_total"] = lambda_home + lambda_away
        copy["total_bin"] = _bin_name(copy["lambda_total"])
        copy["rho"] = rho
        copy["p00"] = matrix[0][0]
        copy["p11"] = matrix[1][1]
        copy["p22"] = matrix[2][2]
        copy["p_any"] = copy["p00"] + copy["p11"] + copy["p22"]
        attached.append(copy)
    return attached


def try_open_session() -> tuple[Any | None, str | None]:
    try:
        session = SessionLocal()
        session.execute(text("SELECT 1"))
        return session, None
    except Exception as exc:  # noqa: BLE001 — diagnostic fallback
        return None, f"{type(exc).__name__}: {exc}"


def join_scores(
    session: Any,
    rows: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    match_ids = [
        int(row["match_id"])
        for row in rows
        if row.get("match_id") is not None
    ]
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
    by_id = {int(match.id): match for match in matches}
    calculator = PlayerAvailabilityCalculator(session)
    joined: dict[int, dict[str, Any]] = {}
    counts = {
        "csv_rows": len(rows),
        "st_match_found": 0,
        "st_scores": 0,
        "fixture_fallback": 0,
        "missing_st_match": 0,
        "st_scores_null": 0,
        "no_fixture": 0,
        "unfinished_fixture": 0,
        "null_fixture_goals": 0,
        "skipped": 0,
    }
    for row in rows:
        raw_id = row.get("match_id")
        if raw_id is None:
            counts["skipped"] += 1
            continue
        match_id = int(raw_id)
        match = by_id.get(match_id)
        if match is None:
            counts["missing_st_match"] += 1
            counts["skipped"] += 1
            continue
        counts["st_match_found"] += 1
        if match.home_score is not None and match.away_score is not None:
            joined[match_id] = {
                "home_goals": int(match.home_score),
                "away_goals": int(match.away_score),
                "source": "st_match",
            }
            counts["st_scores"] += 1
            continue
        counts["st_scores_null"] += 1
        fixture = calculator._resolve_fixture(match)
        if fixture is None:
            counts["no_fixture"] += 1
            counts["skipped"] += 1
            continue
        status = str(fixture.status_short or "").upper()
        if status not in FINISHED_FIXTURE_STATUSES:
            counts["unfinished_fixture"] += 1
            counts["skipped"] += 1
            continue
        if fixture.goals_home is None or fixture.goals_away is None:
            counts["null_fixture_goals"] += 1
            counts["skipped"] += 1
            continue
        joined[match_id] = {
            "home_goals": int(fixture.goals_home),
            "away_goals": int(fixture.goals_away),
            "source": "fixture",
        }
        counts["fixture_fallback"] += 1
    return joined, counts


def apply_scores(
    rows: list[dict[str, Any]],
    joined: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in rows:
        raw_id = row.get("match_id")
        if raw_id is None:
            continue
        score = joined.get(int(raw_id))
        if score is None:
            continue
        copy = dict(row)
        copy["home_goals"] = score["home_goals"]
        copy["away_goals"] = score["away_goals"]
        copy["score_source"] = score["source"]
        scored.append(copy)
    return scored


def leftover_tables_by_bin(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    for name, _lo, _hi in LOCKED_BINS:
        bin_rows = [row for row in rows if row.get("total_bin") == name]
        tables[name] = scoreline_leftovers(bin_rows)
    return tables


def draw_1x2_by_bin(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    for name, _lo, _hi in LOCKED_BINS:
        bin_rows = [row for row in rows if row.get("total_bin") == name]
        tables[name] = draw_1x2_leftover(bin_rows)
    return tables


def environment_split(
    mid_rows: list[dict[str, Any]],
    column: str,
    threshold: float,
) -> dict[str, Any]:
    usable = [row for row in mid_rows if _optional_float(row.get(column)) is not None]
    low_side = [row for row in usable if float(row[column]) <= threshold]
    high_side = [row for row in usable if float(row[column]) > threshold]
    return {
        "column": column,
        "threshold": threshold,
        "n_mid": len(mid_rows),
        "n_missing_feature": len(mid_rows) - len(usable),
        "low": {
            "rule": f"{column} <= {threshold}",
            "n": len(low_side),
            "leftovers": scoreline_leftovers(low_side),
            "draw_1x2": draw_1x2_leftover(low_side),
        },
        "high": {
            "rule": f"{column} > {threshold}",
            "n": len(high_side),
            "leftovers": scoreline_leftovers(high_side),
            "draw_1x2": draw_1x2_leftover(high_side),
        },
    }


def leftover_sign(block: dict[str, Any] | None, event: str) -> float | None:
    if block is None:
        return None
    leftover = block["events"][event]["leftover"]
    if leftover == 0.0:
        return 0.0
    return 1.0 if leftover > 0.0 else -1.0


def apply_stop_rule(
    *,
    scores_joinable: bool,
    bin_tables: dict[str, Any] | None,
    env_split: dict[str, Any] | None,
) -> dict[str, Any]:
    if not scores_joinable or bin_tables is None:
        return {
            "branch": "cannot_measure",
            "one_sentence": (
                "Cannot measure: CSV has no scorelines and we cannot join fixtures "
                "without a rebuild / live DB."
            ),
            "persisting_events": [],
            "env_same_direction_events": [],
        }

    persisting: list[dict[str, Any]] = []
    for bin_name, table in bin_tables.items():
        if table is None:
            continue
        for event in SCORELINE_EVENTS:
            block = table["events"][event]
            if block["fails_noise"]:
                persisting.append(
                    {
                        "bin": bin_name,
                        "event": event,
                        "leftover": block["leftover"],
                        "n": block["n"],
                    }
                )

    if not persisting:
        return {
            "branch": "stop_regimes_noise",
            "one_sentence": (
                "Same totals, leftover 0-0/1-1 is noise — stop regimes."
            ),
            "persisting_events": [],
            "env_same_direction_events": [],
        }

    mid_table = bin_tables.get("mid")
    mid_persisting_events = {
        event
        for event in SCORELINE_EVENTS
        if mid_table is not None and mid_table["events"][event]["fails_noise"]
    }
    same_direction: list[str] = []
    if (
        env_split is not None
        and env_split["low"]["n"] >= 50
        and env_split["high"]["n"] >= 50
        and env_split["low"]["leftovers"] is not None
        and env_split["high"]["leftovers"] is not None
    ):
        for event in mid_persisting_events:
            low_block = env_split["low"]["leftovers"]["events"][event]
            high_block = env_split["high"]["leftovers"]["events"][event]
            if not low_block["fails_noise"] or not high_block["fails_noise"]:
                continue
            low_sign = leftover_sign(env_split["low"]["leftovers"], event)
            high_sign = leftover_sign(env_split["high"]["leftovers"], event)
            if (
                low_sign is not None
                and high_sign is not None
                and low_sign != 0.0
                and low_sign == high_sign
            ):
                same_direction.append(event)

    if same_direction:
        return {
            "branch": "rho_tail_slider_later",
            "one_sentence": (
                "Same totals, leftover lines up with low-scoring / close-match "
                "rates — a ρ/tail slider (not four named worlds) is worth a later sitting."
            ),
            "persisting_events": persisting,
            "env_same_direction_events": same_direction,
            "mid_persisting_events": sorted(mid_persisting_events),
        }

    return {
        "branch": "stop_regimes_leftover_unaligned",
        "one_sentence": (
            "Same totals, leftover 0-0/1-1 exists but does not line up with "
            "the locked environment split — stop named regimes."
        ),
        "persisting_events": persisting,
        "env_same_direction_events": [],
        "mid_persisting_events": sorted(mid_persisting_events),
    }


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
        default=Path("artifacts/dc_rho_mle_promotion/scoreline_leftover.json"),
    )
    args = parser.parse_args()

    dataset_path = resolve_repo_path(args.dataset)
    json_path = resolve_repo_path(args.json)
    if json_path.name.startswith("gate_eval"):
        raise SystemExit("Refusing to overwrite a gate_eval artifact")

    all_rows = load_dataset_rows(dataset_path)
    league_params = load_league_params()
    default_rho = DataSourceConfig().dixon_coles_rho
    valid_lambda_rows = attach_dc_scoreline_probs(
        all_rows,
        league_params=league_params,
        default_rho=default_rho,
    )

    lambda_totals = [row["lambda_total"] for row in valid_lambda_rows]
    bin_lambda_stats = {}
    for name, _lo, _hi in LOCKED_BINS:
        bin_lambda_stats[name] = _lambda_stats(
            [row["lambda_total"] for row in valid_lambda_rows if row["total_bin"] == name]
        )

    session, connect_error = try_open_session()
    join_counts: dict[str, Any] | None = None
    join_path = "none"
    scored_valid: list[dict[str, Any]] = []
    if session is None:
        scores_joinable = False
    else:
        try:
            joined, join_counts = join_scores(session, all_rows)
            scored_valid = apply_scores(valid_lambda_rows, joined)
            join_path = "st_match.home_score/away_score via match_id=stryktipset_matches.id"
            if join_counts["fixture_fallback"]:
                join_path += (
                    "; fixture fallback via PlayerAvailabilityCalculator._resolve_fixture"
                )
            scores_joinable = bool(scored_valid)
        finally:
            session.close()

    official_rows, official_filter = select_backtest_rows(
        all_rows, validation_fraction=0.20, max_draw=None
    )
    official_ids = {
        int(row["match_id"])
        for row in official_rows
        if row.get("match_id") is not None
    }
    official_valid = [
        row for row in valid_lambda_rows if int(row["match_id"]) in official_ids
    ]
    official_scored = [
        row for row in scored_valid if int(row["match_id"]) in official_ids
    ]

    bin_tables = leftover_tables_by_bin(scored_valid) if scores_joinable else None
    mid_scored = [row for row in scored_valid if row["total_bin"] == "mid"]
    env_split = (
        environment_split(mid_scored, "combined_low_scoring_rate", LOW_SCORING_SPLIT)
        if scores_joinable
        else None
    )
    close_split = (
        environment_split(mid_scored, "combined_close_match_rate", CLOSE_MATCH_SPLIT)
        if scores_joinable
        and env_split is not None
        and env_split["low"]["n"] >= 50
        and env_split["high"]["n"] >= 50
        else None
    )
    stop = apply_stop_rule(
        scores_joinable=scores_joinable,
        bin_tables=bin_tables,
        env_split=env_split,
    )

    payload = {
        "kind": "dc_scoreline_leftover_diagnostic",
        "production_unchanged": True,
        "interpreter_hint": "/tmp/st-diag-venv/bin/python",
        "dataset": str(dataset_path),
        "csv_n": len(all_rows),
        "valid_lambda_n": len(valid_lambda_rows),
        "classic_miss_n": len(all_rows) - len(valid_lambda_rows),
        "preregistered_valid_lambda_n": PREREGISTERED_VALID_LAMBDA_N,
        "valid_lambda_matches_preregistration": (
            len(valid_lambda_rows) == PREREGISTERED_VALID_LAMBDA_N
        ),
        "lambda_total_stats": _lambda_stats(lambda_totals),
        "bin_lambda_stats": bin_lambda_stats,
        "preregistered_bin_n": PREREGISTERED_BIN_N,
        "scores_joinable": scores_joinable,
        "join_path": join_path,
        "connect_error": connect_error,
        "join_counts": join_counts,
        "primary_scored_n": len(scored_valid),
        "score_source_counts": {
            "st_match": sum(1 for row in scored_valid if row.get("score_source") == "st_match"),
            "fixture": sum(1 for row in scored_valid if row.get("score_source") == "fixture"),
        },
        "draw_1x2_valid_lambda": draw_1x2_leftover(valid_lambda_rows),
        "draw_1x2_by_bin_valid_lambda": draw_1x2_by_bin(valid_lambda_rows),
        "draw_1x2_by_bin_scored": draw_1x2_by_bin(scored_valid) if scores_joinable else None,
        "scoreline_by_bin": bin_tables,
        "environment_split_mid": env_split,
        "close_match_split_mid": close_split,
        "official_518": {
            "filter": official_filter,
            "n": len(official_rows),
            "valid_lambda_n": len(official_valid),
            "scored_n": len(official_scored),
            "scoreline_by_bin": leftover_tables_by_bin(official_scored)
            if scores_joinable
            else None,
            "draw_1x2_by_bin": draw_1x2_by_bin(official_valid),
            "draw_1x2_scored_by_bin": draw_1x2_by_bin(official_scored)
            if scores_joinable
            else None,
        },
        "stop_rule": stop,
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(stop, indent=2), flush=True)
    print(
        f"csv_n={len(all_rows)} valid_lambda_n={len(valid_lambda_rows)} "
        f"scored_n={len(scored_valid)} official_518={len(official_rows)} "
        f"joinable={scores_joinable}",
        flush=True,
    )
    print(f"Wrote {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
