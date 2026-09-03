"""Pure helpers shared by strength and home-advantage calculators."""

from __future__ import annotations

from objects.models.match_advanced_stats import MatchAdvancedStatsModel

# (metric_value, recency_weight) pairs, newest match first.
WeightedObservation = tuple[float, float]


def weighted_mean(
    values: list[float],
    weights: list[float],
) -> float | None:
    """Return the weighted arithmetic mean, or None if inputs are unusable."""
    if not values or not weights or len(values) != len(weights):
        return None
    total_weight = sum(weights)
    if total_weight <= 0:
        return None
    return sum(value * weight for value, weight in zip(values, weights)) / total_weight


def weighted_mean_from_pairs(
    observations: list[WeightedObservation],
) -> float | None:
    """Weighted mean from (value, weight) pairs."""
    if not observations:
        return None
    return weighted_mean(
        [value for value, _weight in observations],
        [weight for _value, weight in observations],
    )


def shrink(
    observed: float | None,
    sample_size: int,
    *,
    prior_rating: float = 1.0,
    prior_strength: int = 8,
) -> float | None:
    """Shrink ``observed`` toward ``prior_rating`` using sample vs prior strength."""
    if observed is None:
        return None
    if prior_strength <= 0:
        return observed
    return (
        sample_size * observed + prior_strength * prior_rating
    ) / (sample_size + prior_strength)


def recency_weights(match_count: int, decay: float = 0.9) -> list[float]:
    """Newest-first exponential weights: ``decay ** matches_ago``."""
    if match_count <= 0:
        return []
    return [decay**matches_ago for matches_ago in range(match_count)]


def normalize_strength(
    team_rate: float | None,
    league_rate: float | None,
) -> float | None:
    """Return team rate relative to league rate (1.0 = average)."""
    if team_rate is None or league_rate is None or league_rate <= 0:
        return None
    return team_rate / league_rate


def append_observation(
    bucket: list[WeightedObservation],
    value: float | None,
    weight: float,
) -> None:
    """Append a weighted observation when the metric is present."""
    if value is not None:
        bucket.append((value, weight))


def npxg_or_xg(
    non_penalty_xg: float | None, total_xg: float | None
) -> float | None:
    """Prefer non-penalty xG; fall back to total xG when npxG is missing."""
    if non_penalty_xg is not None:
        return non_penalty_xg
    return total_xg


def baselines_from_stats(
    advanced_stats_rows: list[MatchAdvancedStatsModel],
    decay: float,
) -> dict[str, float]:
    """Compute recency-weighted attack/defence/npxG league baselines."""
    match_weights = recency_weights(len(advanced_stats_rows), decay)

    home_xg: list[WeightedObservation] = []
    away_xg: list[WeightedObservation] = []
    home_npxg: list[WeightedObservation] = []
    away_npxg: list[WeightedObservation] = []
    home_set_piece: list[WeightedObservation] = []
    away_set_piece: list[WeightedObservation] = []

    for weight, row in zip(match_weights, advanced_stats_rows):
        append_observation(home_xg, row.home_xg, weight)
        append_observation(away_xg, row.away_xg, weight)
        append_observation(
            home_npxg, npxg_or_xg(row.home_non_penalty_xg, row.home_xg), weight
        )
        append_observation(
            away_npxg, npxg_or_xg(row.away_non_penalty_xg, row.away_xg), weight
        )
        append_observation(home_set_piece, row.home_set_piece_xg, weight)
        append_observation(away_set_piece, row.away_set_piece_xg, weight)

    baselines: dict[str, float] = {}
    attack = weighted_mean_from_pairs(home_xg)
    defence = weighted_mean_from_pairs(away_xg)
    home_npxg_mean = weighted_mean_from_pairs(home_npxg)
    away_npxg_mean = weighted_mean_from_pairs(away_npxg)
    overall_npxg = weighted_mean_from_pairs(home_npxg + away_npxg)
    set_piece_xg = weighted_mean_from_pairs(home_set_piece + away_set_piece)

    if attack is not None:
        baselines["attack"] = attack
    if defence is not None:
        baselines["defence"] = defence
    if home_npxg_mean is not None:
        baselines["home_npxg"] = home_npxg_mean
    if away_npxg_mean is not None:
        baselines["away_npxg"] = away_npxg_mean
    if overall_npxg is not None:
        baselines["npxg"] = overall_npxg
    if set_piece_xg is not None:
        baselines["set_piece_xg"] = set_piece_xg
    return baselines
