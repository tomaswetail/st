"""Sarmanov–NB bivariate scoreline model (NB2 + Michels/Karlis mixer)."""

from src.calc.sarmanov_nb.matrix import (
    a_factor,
    admissible_rho_bounds,
    sarmanov_nb_matrix,
    sarmanov_nb_tau,
)
from src.calc.sarmanov_nb.model import (
    SarmanovNBModel,
    SarmanovNBPrediction,
)

__all__ = [
    "SarmanovNBModel",
    "SarmanovNBPrediction",
    "a_factor",
    "admissible_rho_bounds",
    "sarmanov_nb_matrix",
    "sarmanov_nb_tau",
]
