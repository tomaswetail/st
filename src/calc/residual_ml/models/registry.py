"""Factory for seeded residual-ML registry backends."""

from __future__ import annotations

import importlib
from typing import Any

from src.calc.residual_ml.models.base import RegistryModel
from src.calc.residual_ml.models.backends import (
    DirectCatBoost,
    DirectHistGradient,
    DirectLightGBM,
    DirectLogistic,
    MarketBaselineModel,
    ResidualCatBoost,
    ResidualHistGradient,
    ResidualLightGBM,
    ResidualLogistic,
)

BACKENDS = (
    "market_baseline",
    "logistic",
    "hist_gradient",
    "lightgbm",
    "catboost",
)
OPTIONAL_PACKAGES = {
    "lightgbm": "lightgbm",
    "catboost": "catboost",
}
FAMILIES = ("market", "direct", "residual")

_BUILDERS: dict[tuple[str, str], type] = {
    ("market_baseline", "market"): MarketBaselineModel,
    ("logistic", "direct"): DirectLogistic,
    ("logistic", "residual"): ResidualLogistic,
    ("hist_gradient", "direct"): DirectHistGradient,
    ("hist_gradient", "residual"): ResidualHistGradient,
    ("lightgbm", "direct"): DirectLightGBM,
    ("lightgbm", "residual"): ResidualLightGBM,
    ("catboost", "direct"): DirectCatBoost,
    ("catboost", "residual"): ResidualCatBoost,
}


def import_optional_backend(backend: str) -> Any:
    """Import an optional package. Missing → clear error (not at package import)."""
    module_name = OPTIONAL_PACKAGES.get(backend)
    if module_name is None:
        return None
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Backend {backend!r} is not available: package {module_name!r} "
            "is not installed"
        ) from exc


def is_backend_available(backend: str) -> bool:
    if backend not in OPTIONAL_PACKAGES:
        return True
    try:
        import_optional_backend(backend)
    except ImportError:
        return False
    return True


def available_backends() -> list[str]:
    return [backend for backend in BACKENDS if is_backend_available(backend)]


def get_model(
    backend: str,
    family: str,
    *,
    random_state: int = 42,
    **kwargs: Any,
) -> RegistryModel:
    """Return a seeded registry model. Optional backends fail here, not on import."""
    if backend in OPTIONAL_PACKAGES:
        import_optional_backend(backend)
    key = (backend, family)
    builder = _BUILDERS.get(key)
    if builder is None:
        raise ValueError(
            f"Unknown backend/family pair: backend={backend!r}, family={family!r}. "
            f"Known pairs: {sorted(_BUILDERS)}"
        )
    return builder(random_state=random_state, **kwargs)
