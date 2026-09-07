"""Stryktipset market + streckprocent coupon optimizer.

Optimizes coupon rows for EV vs public dilution using market odds as true
probabilities — not prediction accuracy. Not wired to ProbabilityManager,
residual ML, or Dixon–Coles.

Leakage UNKNOWN: odds and public % lack timestamps; regCloseTime is not
persisted on STRoundModel.
"""

from src.calc.stryktipset_optimizer.analyzer import (
    BankerSuggestion,
    MatchAnalysis,
    analyze_matches,
    banker_score,
    jensen_shannon_divergence,
)
from src.calc.stryktipset_optimizer.backtester import (
    BacktestResult,
    ParamConfig,
    run_backtest,
)
from src.calc.stryktipset_optimizer.candidates import (
    CandidateRow,
    build_score_table,
    market_favorite,
    top_candidates,
)
from src.calc.stryktipset_optimizer.data import (
    LEAKAGE_LIMITATIONS,
    CouponMatchInput,
    PreparedCoupon,
    load_coupon_from_session,
    prepare_matches,
)
from src.calc.stryktipset_optimizer.fair_probs import (
    InvalidOddsError,
    fair_probabilities_from_odds,
)
from src.calc.stryktipset_optimizer.optimize import CouponOptimizer
from src.calc.stryktipset_optimizer.params import OptimizerParams
from src.calc.stryktipset_optimizer.portfolio import (
    hamming_distance,
    hamming_similarity,
    select_diversified_portfolio,
)
from src.calc.stryktipset_optimizer.public_probs import (
    InvalidPublicShareError,
    normalize_public_shares,
)
from src.calc.stryktipset_optimizer.simulator import simulate_pool
from src.calc.stryktipset_optimizer.value import log_leverage, value_ratio

__all__ = [
    "BankerSuggestion",
    "BacktestResult",
    "CandidateRow",
    "CouponMatchInput",
    "CouponOptimizer",
    "InvalidOddsError",
    "InvalidPublicShareError",
    "LEAKAGE_LIMITATIONS",
    "MatchAnalysis",
    "OptimizerParams",
    "ParamConfig",
    "PreparedCoupon",
    "analyze_matches",
    "banker_score",
    "build_score_table",
    "fair_probabilities_from_odds",
    "hamming_distance",
    "hamming_similarity",
    "jensen_shannon_divergence",
    "load_coupon_from_session",
    "log_leverage",
    "market_favorite",
    "normalize_public_shares",
    "prepare_matches",
    "run_backtest",
    "select_diversified_portfolio",
    "simulate_pool",
    "top_candidates",
    "value_ratio",
]
