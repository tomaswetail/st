"""Stryktipset coupon optimizer (PREDICTION default / VALUE optional).

PREDICTION (default): maximize predictive coverage using vig-free market
probabilities only (MAX_P13 exact top-R by joint P). Public % is diagnostic.

VALUE: EV vs public dilution ``Σ log Pm − β Σ log Pp`` + Hamming diversity.

Not wired to ProbabilityManager, residual ML, or Dixon–Coles.

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
from src.calc.stryktipset_optimizer.coverage import (
    CoverageMetrics,
    compute_coverage,
    row_joint_probability,
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
from src.calc.stryktipset_optimizer.objectives import (
    greedy_approximate_portfolio,
)
from src.calc.stryktipset_optimizer.optimize import CouponOptimizer
from src.calc.stryktipset_optimizer.params import (
    OptimizerParams,
    OptimizerMode,
    PredictionObjective,
)
from src.calc.stryktipset_optimizer.portfolio import (
    hamming_distance,
    hamming_similarity,
    select_diversified_portfolio,
)
from src.calc.stryktipset_optimizer.prediction import (
    MarketRow,
    single_favorite_row,
    top_market_rows,
)
from src.calc.stryktipset_optimizer.public_probs import (
    InvalidPublicShareError,
    normalize_public_shares,
)
from src.calc.stryktipset_optimizer.reduced_system import (
    build_reduced_system,
    naive_closest_match_upgrades,
)
from src.calc.stryktipset_optimizer.simulator import simulate_pool
from src.calc.stryktipset_optimizer.value import log_leverage, value_ratio

__all__ = [
    "BankerSuggestion",
    "BacktestResult",
    "CandidateRow",
    "CoverageMetrics",
    "CouponMatchInput",
    "CouponOptimizer",
    "InvalidOddsError",
    "InvalidPublicShareError",
    "LEAKAGE_LIMITATIONS",
    "MarketRow",
    "MatchAnalysis",
    "OptimizerMode",
    "OptimizerParams",
    "ParamConfig",
    "PredictionObjective",
    "PreparedCoupon",
    "analyze_matches",
    "banker_score",
    "build_reduced_system",
    "build_score_table",
    "compute_coverage",
    "fair_probabilities_from_odds",
    "greedy_approximate_portfolio",
    "hamming_distance",
    "hamming_similarity",
    "jensen_shannon_divergence",
    "load_coupon_from_session",
    "log_leverage",
    "market_favorite",
    "naive_closest_match_upgrades",
    "normalize_public_shares",
    "prepare_matches",
    "row_joint_probability",
    "run_backtest",
    "select_diversified_portfolio",
    "simulate_pool",
    "single_favorite_row",
    "top_candidates",
    "top_market_rows",
    "value_ratio",
]
