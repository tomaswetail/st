"""Dixon–Coles with Negative-Binomial (NB2) marginals."""

from src.calc.dixon_coles_nbm.matrix import (
    dixon_coles_nbm_matrix,
    dixon_coles_nbm_tau,
)
from src.calc.dixon_coles_nbm.model import (
    DixonColesNBMModel,
    DixonColesNBMPrediction,
)
from src.calc.dixon_coles_nbm.nb import (
    negative_binomial_log_pmf,
    negative_binomial_pmf,
)

__all__ = [
    "DixonColesNBMModel",
    "DixonColesNBMPrediction",
    "dixon_coles_nbm_matrix",
    "dixon_coles_nbm_tau",
    "negative_binomial_log_pmf",
    "negative_binomial_pmf",
]
