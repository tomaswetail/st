"""Derived xG metrics from provider shots and reported aggregates."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from objects.schema.data_classes.provider_dtos import (
    ProviderMatchDetails,
    ProviderShot,
)

logger = logging.getLogger(__name__)

SET_PIECE_SITUATIONS = {
    "setpiece",
    "set_piece",
    "set piece",
    "corner",
    "freekick",
    "free_kick",
    "free kick",
    "throwin",
    "throw_in",
    "directfreekick",
}


@dataclass
class DerivedMatchMetrics:
    """Shot- and aggregate-derived xG metrics for one match."""
    home_xg: float | None = None
    away_xg: float | None = None
    home_non_penalty_xg: float | None = None
    away_non_penalty_xg: float | None = None
    home_xgot: float | None = None
    away_xgot: float | None = None
    home_shots: int | None = None
    away_shots: int | None = None
    home_shots_on_target: int | None = None
    away_shots_on_target: int | None = None
    home_set_piece_xg: float | None = None
    away_set_piece_xg: float | None = None
    home_open_play_xg: float | None = None
    away_open_play_xg: float | None = None
    home_xg_from_shots: float | None = None
    away_xg_from_shots: float | None = None
    average_home_shot_xg: float | None = None
    average_away_shot_xg: float | None = None
    warnings: list[str] = field(default_factory=list)
    raw_payload_hash: str | None = None


def shot_fingerprint(
    *,
    match_id: int,
    provider: str,
    shot: ProviderShot,
    team_internal_id: int | None,
) -> str:
    """Stable SHA-256 fingerprint for idempotent shot upserts."""
    coords = shot.coordinates or {}
    payload = {
        "match_id": match_id,
        "provider": provider,
        "provider_shot_id": shot.provider_shot_id,
        "team_id": team_internal_id,
        "team_external_id": shot.team_id,
        "player_id": shot.player_id,
        "minute": shot.minute,
        "second": shot.second,
        "xg": shot.xg,
        "x": coords.get("x"),
        "y": coords.get("y"),
        "outcome": shot.outcome,
        "is_penalty": shot.is_penalty,
        "is_own_goal": shot.is_own_goal,
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sum_xg(shots: list[ProviderShot]) -> float | None:
    """Sum shot xG values, or None if none present."""
    values = [shot.xg for shot in shots if shot.xg is not None]
    if not values:
        return None
    return float(sum(values))


def _sum_xgot(shots: list[ProviderShot]) -> float | None:
    """Sum shot xGOT values, or None if none present."""
    values = [shot.xgot for shot in shots if shot.xgot is not None]
    if not values:
        return None
    return float(sum(values))


def _is_set_piece(shot: ProviderShot) -> bool:
    """True if the shot is a set piece or penalty."""
    if shot.is_penalty:
        return True
    situation = (shot.situation or "").strip().lower().replace("-", "_").replace(" ", "")
    normalized = (shot.situation or "").strip().lower()
    if normalized in SET_PIECE_SITUATIONS:
        return True
    compact = {
        s.replace(" ", "").replace("_", "")
        for s in SET_PIECE_SITUATIONS
    }
    return situation in compact


def _on_target(shot: ProviderShot) -> bool:
    """True if the shot outcome counts as on target."""
    outcome = (shot.outcome or "").strip().lower()
    return outcome in {
        "goal",
        "saved",
        "save",
        "blocked_on_target",
        "on_target",
        "shotontarget",
        "shot_on_target",
    }


def calculate_derived_metrics(
    details: ProviderMatchDetails,
    *,
    home_team_external_id: str,
    away_team_external_id: str,
    xg_aggregate_tolerance: float = 0.15,
) -> DerivedMatchMetrics:
    """Derive home/away xG metrics and disagreement warnings."""
    home_shots = [
        shot for shot in details.shots if shot.team_id == home_team_external_id
    ]
    away_shots = [
        shot for shot in details.shots if shot.team_id == away_team_external_id
    ]

    home_xg_from_shots = _sum_xg(home_shots)
    away_xg_from_shots = _sum_xg(away_shots)
    home_npxg = _sum_xg([s for s in home_shots if not s.is_penalty])
    away_npxg = _sum_xg([s for s in away_shots if not s.is_penalty])
    home_set = _sum_xg([s for s in home_shots if _is_set_piece(s)])
    away_set = _sum_xg([s for s in away_shots if _is_set_piece(s)])
    home_open = _sum_xg([s for s in home_shots if not _is_set_piece(s)])
    away_open = _sum_xg([s for s in away_shots if not _is_set_piece(s)])
    home_xgot = _sum_xgot(home_shots)
    away_xgot = _sum_xgot(away_shots)

    avg_home = (
        home_xg_from_shots / len(home_shots)
        if home_xg_from_shots is not None and home_shots
        else None
    )
    avg_away = (
        away_xg_from_shots / len(away_shots)
        if away_xg_from_shots is not None and away_shots
        else None
    )

    home_on_target = sum(1 for s in home_shots if _on_target(s)) if home_shots else None
    away_on_target = sum(1 for s in away_shots if _on_target(s)) if away_shots else None

    reported_home = details.home_xg
    reported_away = details.away_xg
    warnings: list[str] = []

    if (
        reported_home is not None
        and home_xg_from_shots is not None
        and abs(reported_home - home_xg_from_shots) > xg_aggregate_tolerance
    ):
        warnings.append(
            f"Home xG disagreement: provider={reported_home} shots={home_xg_from_shots}"
        )
    if (
        reported_away is not None
        and away_xg_from_shots is not None
        and abs(reported_away - away_xg_from_shots) > xg_aggregate_tolerance
    ):
        warnings.append(
            f"Away xG disagreement: provider={reported_away} shots={away_xg_from_shots}"
        )

    # Prefer provider-reported aggregates when present; keep shot-derived separate.
    home_xg = reported_home if reported_home is not None else home_xg_from_shots
    away_xg = reported_away if reported_away is not None else away_xg_from_shots
    home_xgot_final = (
        details.home_xgot if details.home_xgot is not None else home_xgot
    )
    away_xgot_final = (
        details.away_xgot if details.away_xgot is not None else away_xgot
    )

    payload_hash = hashlib.sha256(
        json.dumps(details.raw_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    return DerivedMatchMetrics(
        home_xg=home_xg,
        away_xg=away_xg,
        home_non_penalty_xg=home_npxg,
        away_non_penalty_xg=away_npxg,
        home_xgot=home_xgot_final,
        away_xgot=away_xgot_final,
        home_shots=len(home_shots) if details.shots else None,
        away_shots=len(away_shots) if details.shots else None,
        home_shots_on_target=home_on_target,
        away_shots_on_target=away_on_target,
        home_set_piece_xg=home_set,
        away_set_piece_xg=away_set,
        home_open_play_xg=home_open,
        away_open_play_xg=away_open,
        home_xg_from_shots=home_xg_from_shots,
        away_xg_from_shots=away_xg_from_shots,
        average_home_shot_xg=avg_home,
        average_away_shot_xg=avg_away,
        warnings=warnings,
        raw_payload_hash=payload_hash,
    )
