"""Conditional market/DC blend weight selection.

Pipeline order for Phase 3: conditional blend → draw adjust → HGB.
Missing league / injury signals fall back safely (skip rule or use defaults).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.utils.repo_paths import repo_root, resolve_repo_path

DEFAULT_CONFIG_PATH = repo_root() / "config" / "blend_weights.json"
DEFAULT_DC_PARAMS_PATH = repo_root() / "config" / "classic_dc_league_params.json"


@dataclass(frozen=True)
class BlendWeightRule:
    enabled: bool = True
    market_weight: float = 0.85
    dc_weight: float = 0.15
    threshold: float | None = None
    missing_league_market_weight: float | None = None
    missing_league_dc_weight: float | None = None
    max_acceptable_log_loss: float | None = None
    poor_fit_market_weight: float | None = None
    poor_fit_dc_weight: float | None = None
    missing_availability_market_weight: float | None = None
    missing_availability_dc_weight: float | None = None


@dataclass(frozen=True)
class BlendWeightsConfig:
    """Conditional blend weight policy loaded from JSON."""

    enabled: bool = False
    default_market_weight: float = 0.7
    default_dc_weight: float = 0.3
    league_overrides: dict[str, tuple[float, float]] = field(default_factory=dict)
    market_vs_dc: BlendWeightRule = field(default_factory=BlendWeightRule)
    dc_quality: BlendWeightRule = field(default_factory=BlendWeightRule)
    injury_uncertainty: BlendWeightRule = field(default_factory=BlendWeightRule)
    notes: str = ""


def _as_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_rule(raw: Any, *, defaults: BlendWeightRule) -> BlendWeightRule:
    if not isinstance(raw, Mapping):
        return defaults
    return BlendWeightRule(
        enabled=bool(raw.get("enabled", defaults.enabled)),
        market_weight=_as_float(raw.get("market_weight"), defaults.market_weight),
        dc_weight=_as_float(raw.get("dc_weight"), defaults.dc_weight),
        threshold=(
            None
            if raw.get("threshold") is None
            else _as_float(raw.get("threshold"), defaults.threshold or 0.0)
        ),
        missing_league_market_weight=(
            None
            if raw.get("missing_league_market_weight") is None
            else _as_float(
                raw.get("missing_league_market_weight"),
                defaults.missing_league_market_weight or 0.85,
            )
        ),
        missing_league_dc_weight=(
            None
            if raw.get("missing_league_dc_weight") is None
            else _as_float(
                raw.get("missing_league_dc_weight"),
                defaults.missing_league_dc_weight or 0.15,
            )
        ),
        max_acceptable_log_loss=(
            None
            if raw.get("max_acceptable_log_loss") is None
            else _as_float(
                raw.get("max_acceptable_log_loss"),
                defaults.max_acceptable_log_loss or 1.05,
            )
        ),
        poor_fit_market_weight=(
            None
            if raw.get("poor_fit_market_weight") is None
            else _as_float(
                raw.get("poor_fit_market_weight"),
                defaults.poor_fit_market_weight or 0.85,
            )
        ),
        poor_fit_dc_weight=(
            None
            if raw.get("poor_fit_dc_weight") is None
            else _as_float(
                raw.get("poor_fit_dc_weight"),
                defaults.poor_fit_dc_weight or 0.15,
            )
        ),
        missing_availability_market_weight=(
            None
            if raw.get("missing_availability_market_weight") is None
            else _as_float(
                raw.get("missing_availability_market_weight"),
                defaults.missing_availability_market_weight or 0.8,
            )
        ),
        missing_availability_dc_weight=(
            None
            if raw.get("missing_availability_dc_weight") is None
            else _as_float(
                raw.get("missing_availability_dc_weight"),
                defaults.missing_availability_dc_weight or 0.2,
            )
        ),
    )


def _parse_league_overrides(raw: Any) -> dict[str, tuple[float, float]]:
    if not isinstance(raw, Mapping):
        return {}
    parsed: dict[str, tuple[float, float]] = {}
    for league_id, weights in raw.items():
        if not isinstance(weights, Mapping):
            continue
        market = weights.get("market_weight")
        dc = weights.get("dc_weight")
        if market is None:
            continue
        market_weight = float(market)
        dc_weight = float(dc) if dc is not None else max(0.0, 1.0 - market_weight)
        parsed[str(league_id)] = (market_weight, dc_weight)
    return parsed


def load_blend_weights_config(
    path: str | Path | None = None,
) -> BlendWeightsConfig:
    """Load conditional blend policy (repo-relative or absolute)."""
    config_path = resolve_repo_path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return BlendWeightsConfig(enabled=False, notes="config file missing")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return BlendWeightsConfig(
        enabled=bool(payload.get("enabled", False)),
        default_market_weight=_as_float(payload.get("default_market_weight"), 0.7),
        default_dc_weight=_as_float(payload.get("default_dc_weight"), 0.3),
        league_overrides=_parse_league_overrides(payload.get("league_overrides")),
        market_vs_dc=_parse_rule(
            payload.get("market_vs_dc"),
            defaults=BlendWeightRule(
                enabled=True,
                market_weight=0.85,
                dc_weight=0.15,
                threshold=0.15,
            ),
        ),
        dc_quality=_parse_rule(
            payload.get("dc_quality"),
            defaults=BlendWeightRule(
                enabled=True,
                missing_league_market_weight=0.85,
                missing_league_dc_weight=0.15,
                max_acceptable_log_loss=1.05,
                poor_fit_market_weight=0.85,
                poor_fit_dc_weight=0.15,
            ),
        ),
        injury_uncertainty=_parse_rule(
            payload.get("injury_uncertainty"),
            defaults=BlendWeightRule(
                enabled=True,
                missing_availability_market_weight=0.8,
                missing_availability_dc_weight=0.2,
            ),
        ),
        notes=str(payload.get("notes", "") or ""),
    )


def load_dc_league_quality(
    path: str | Path | None = None,
) -> dict[str, float]:
    """Map league_external_id → validation log_loss from classic DC params."""
    params_path = (
        resolve_repo_path(path) if path is not None else DEFAULT_DC_PARAMS_PATH
    )
    if not params_path.exists():
        return {}
    payload = json.loads(params_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return {}
    quality: dict[str, float] = {}
    for league_id, entry in payload.items():
        if not isinstance(entry, Mapping):
            continue
        log_loss = entry.get("log_loss")
        if log_loss is None:
            continue
        try:
            quality[str(league_id)] = float(log_loss)
        except (TypeError, ValueError):
            continue
    return quality


def _feature_get(features: Any, name: str) -> Any:
    if isinstance(features, Mapping):
        return features.get(name)
    return getattr(features, name, None)


def _feature_key_present(features: Any, name: str) -> bool:
    if isinstance(features, Mapping):
        return name in features
    return hasattr(features, name)


def resolve_league_external_id(
    features: Any,
    *,
    league_external_id: str | int | None = None,
) -> str | None:
    """Prefer explicit arg; else features.league_external_id when present."""
    if league_external_id is not None and league_external_id != "":
        return str(league_external_id)
    value = _feature_get(features, "league_external_id")
    if value is None or value == "":
        return None
    return str(value)


def market_vs_dc_magnitude(features: Any) -> float | None:
    """Max |market_vs_dc_*| across home/draw/away; None if all missing."""
    magnitudes: list[float] = []
    for name in ("market_vs_dc_home", "market_vs_dc_draw", "market_vs_dc_away"):
        raw = _feature_get(features, name)
        if raw is None or raw == "":
            continue
        try:
            magnitudes.append(abs(float(raw)))
        except (TypeError, ValueError):
            continue
    if not magnitudes:
        return None
    return max(magnitudes)


def _normalize_pair(market_weight: float, dc_weight: float) -> tuple[float, float]:
    total = market_weight + dc_weight
    if total <= 0:
        return 0.7, 0.3
    return market_weight / total, dc_weight / total


def select_blend_weights(
    features: Any,
    config: BlendWeightsConfig | None = None,
    *,
    league_external_id: str | int | None = None,
    dc_league_log_loss: Mapping[str, float] | None = None,
    fallback_market_weight: float = 0.7,
    fallback_dc_weight: float = 0.3,
) -> tuple[float, float]:
    """Return ``(market_weight, dc_weight)`` for a match.

    When config is disabled/missing, returns fallback weights (DataSourceConfig).
    When enabled, a league override **replaces** the JSON default as the base
    pair (so 0.7/0.3 can apply even when default is 1.0/0.0). Other firing
    rules still compete via highest market weight (conservative toward market).
    Missing league/injury signals never raise.
    """
    resolved = config if config is not None else load_blend_weights_config()
    if not resolved.enabled:
        return _normalize_pair(fallback_market_weight, fallback_dc_weight)

    league_id = resolve_league_external_id(
        features, league_external_id=league_external_id
    )
    if league_id is not None and league_id in resolved.league_overrides:
        candidates: list[tuple[float, float]] = [resolved.league_overrides[league_id]]
    else:
        candidates = [
            (resolved.default_market_weight, resolved.default_dc_weight)
        ]

    mvdc = resolved.market_vs_dc
    if mvdc.enabled and mvdc.threshold is not None:
        magnitude = market_vs_dc_magnitude(features)
        if magnitude is not None and magnitude >= mvdc.threshold:
            candidates.append((mvdc.market_weight, mvdc.dc_weight))

    dc_rule = resolved.dc_quality
    if dc_rule.enabled:
        quality = (
            dict(dc_league_log_loss)
            if dc_league_log_loss is not None
            else load_dc_league_quality()
        )
        if league_id is None:
            # No league signal → skip DC-quality rule (safe default).
            pass
        elif league_id not in quality:
            market = dc_rule.missing_league_market_weight
            dc = dc_rule.missing_league_dc_weight
            if market is not None and dc is not None:
                candidates.append((market, dc))
        else:
            max_ll = dc_rule.max_acceptable_log_loss
            poor_market = dc_rule.poor_fit_market_weight
            poor_dc = dc_rule.poor_fit_dc_weight
            if (
                max_ll is not None
                and poor_market is not None
                and poor_dc is not None
                and quality[league_id] > max_ll
            ):
                candidates.append((poor_market, poor_dc))

    injury = resolved.injury_uncertainty
    if injury.enabled and _feature_key_present(features, "has_availability"):
        # Explicit 0 → uncertainty (more market). Absent key → skip (safe default).
        has_availability = _feature_get(features, "has_availability")
        try:
            availability_flag = int(float(has_availability))
        except (TypeError, ValueError):
            availability_flag = 0
        if availability_flag == 0:
            market = injury.missing_availability_market_weight
            dc = injury.missing_availability_dc_weight
            if market is not None and dc is not None:
                candidates.append((market, dc))

    best = max(candidates, key=lambda pair: pair[0])
    return _normalize_pair(best[0], best[1])
