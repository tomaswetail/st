"""Pydantic DTOs for Stryktipset coupon optimizer results."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.utils.common import Outcome


class OptimizerParametersDTO(BaseModel):
    mode: str = "PREDICTION"
    objective: str = "MAX_P13"
    beta: float
    lambda_diversity: float
    banker_value_weight: float
    candidate_count: int
    prediction_candidate_count: int = 2000
    public_epsilon: float
    coupon_size: int
    seed: int | None = None
    row_count: int = 1
    reduced_system: bool = False


class PortfolioRowDTO(BaseModel):
    rank: int
    outcomes: list[Outcome]
    row_score: float
    adjusted_score: float
    avg_hamming_similarity: float
    favorite_count: int
    home_count: int
    draw_count: int
    away_count: int
    joint_probability: float | None = None
    log_pm_sum: float | None = None


class CandidateRowDTO(BaseModel):
    outcomes: list[Outcome]
    row_score: float
    log_pm_sum: float
    log_pp_sum: float | None = None
    favorite_count: int
    home_count: int
    draw_count: int
    away_count: int
    joint_probability: float | None = None


class MatchAnalysisDTO(BaseModel):
    match_index: int
    match_label: str | None = None
    jensen_shannon: float
    max_abs_log_leverage: float
    highest_value_outcome: Outcome
    highest_value_ratio: float
    market_probs: dict[str, float]
    public_probs: dict[str, float]
    value_ratios: dict[str, float]
    log_leverages: dict[str, float]
    banker_scores: dict[str, float]


class BankerSuggestionDTO(BaseModel):
    match_index: int
    outcome: Outcome
    score: float
    log_pm: float
    log_leverage: float


class CoverageMetricsDTO(BaseModel):
    """P(best_correct=k) and cumulatives under market measure."""

    n_matches: int
    p_best_correct: list[float]
    p_full: float
    p_13: float
    p_12_or_better: float
    p_11_or_better: float
    p_10_or_better: float
    expected_best_correct: float


class PortfolioAnalysisDTO(BaseModel):
    selected_count: int
    candidate_pool_size: int
    mean_pairwise_hamming_similarity: float | None = None
    top_bankers: list[BankerSuggestionDTO] = Field(default_factory=list)
    sum_joint_probability: float | None = None


class SimulationTierCounts(BaseModel):
    """Relative tier hit counts (no SEK)."""

    correct_13: int = 0
    correct_12: int = 0
    correct_11: int = 0
    correct_10: int = 0


class SimulationMetricsDTO(BaseModel):
    n_simulations: int
    seed: int
    our_tiers: SimulationTierCounts
    public_tiers: SimulationTierCounts
    our_tier_rates: dict[str, float]
    public_tier_rates: dict[str, float]
    relative_tier_lifts: dict[str, float]
    mean_correct_ours: float
    mean_correct_public: float
    limitations: str


class StryktipsetOptimizationResult(BaseModel):
    """Full optimizer output for one coupon."""

    draw_number: int | None = None
    match_count: int
    mode: str = "PREDICTION"
    objective: str = "MAX_P13"
    exact_selection: bool = True
    parameters: OptimizerParametersDTO
    rows: list[PortfolioRowDTO]
    candidates: list[CandidateRowDTO] = Field(default_factory=list)
    match_analysis: list[MatchAnalysisDTO]
    portfolio_analysis: PortfolioAnalysisDTO
    coverage: CoverageMetricsDTO | None = None
    system_sign_pattern: list[list[Outcome]] | None = None
    system_sign_counts: list[int] | None = None
    limitations: str
    simulation: SimulationMetricsDTO | None = None
