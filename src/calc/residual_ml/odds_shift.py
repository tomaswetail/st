"""Pure helpers for ST vs archive 1X2 probability shift."""

from __future__ import annotations

from src.calc.probability_metrics import clip_and_normalize_probs


def absolute_probability_deltas(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return ``|p_left − p_right|`` after clip+normalize over ``{1, X, 2}``."""
    left_home, left_draw, left_away = clip_and_normalize_probs(*left)
    right_home, right_draw, right_away = clip_and_normalize_probs(*right)
    return (
        abs(left_home - right_home),
        abs(left_draw - right_draw),
        abs(left_away - right_away),
    )
