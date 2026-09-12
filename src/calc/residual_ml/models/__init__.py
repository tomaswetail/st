"""Additive model registry. Import the factory from this package, not residual_ml."""

from src.calc.residual_ml.models.base import RegistryModel
from src.calc.residual_ml.models.registry import (
    available_backends,
    get_model,
    is_backend_available,
)

__all__ = [
    "RegistryModel",
    "available_backends",
    "get_model",
    "is_backend_available",
]
