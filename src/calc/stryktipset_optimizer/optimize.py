"""Orchestration: CouponOptimizer.optimize(...)."""

from __future__ import annotations

from src.calc.stryktipset_optimizer.analyzer import analyze_matches
from src.calc.stryktipset_optimizer.candidates import top_candidates
from src.calc.stryktipset_optimizer.data import (
    LEAKAGE_LIMITATIONS,
    CouponMatchInput,
    PreparedCoupon,
    prepare_matches,
)
from src.calc.stryktipset_optimizer.params import OptimizerParams
from src.calc.stryktipset_optimizer.portfolio import (
    hamming_similarity,
    select_diversified_portfolio,
)
from src.objects.schema.data_classes.stryktipset_optimizer import (
    BankerSuggestionDTO,
    CandidateRowDTO,
    MatchAnalysisDTO,
    OptimizerParametersDTO,
    PortfolioAnalysisDTO,
    PortfolioRowDTO,
    StryktipsetOptimizationResult,
)
from src.utils.common import Outcome


class CouponOptimizer:
    """Market + streckprocent coupon row optimizer (EV vs public dilution)."""

    def __init__(self, params: OptimizerParams | None = None) -> None:
        self.params = params or OptimizerParams()

    def optimize_prepared(
        self,
        coupon: PreparedCoupon,
        *,
        row_count: int = 1,
    ) -> StryktipsetOptimizationResult:
        if row_count < 1:
            raise ValueError(f"row_count must be >= 1, got {row_count}")

        params = self.params
        market_probs = [match.market_probs for match in coupon.matches]
        public_probs = [match.public_probs for match in coupon.matches]
        labels = [match.label for match in coupon.matches]

        candidates = top_candidates(
            market_probs,
            public_probs,
            beta=params.beta,
            candidate_count=params.candidate_count,
        )
        portfolio = select_diversified_portfolio(
            candidates,
            row_count=row_count,
            lambda_diversity=params.lambda_diversity,
        )
        match_analyses, banker_suggestions = analyze_matches(
            market_probs,
            public_probs,
            banker_value_weight=params.banker_value_weight,
            match_labels=labels,
        )

        return StryktipsetOptimizationResult(
            draw_number=coupon.draw_number,
            match_count=len(coupon.matches),
            parameters=OptimizerParametersDTO(
                beta=params.beta,
                lambda_diversity=params.lambda_diversity,
                banker_value_weight=params.banker_value_weight,
                candidate_count=params.candidate_count,
                public_epsilon=params.public_epsilon,
                coupon_size=params.coupon_size,
                seed=params.seed,
                row_count=row_count,
            ),
            rows=[
                PortfolioRowDTO(
                    rank=row.rank,
                    outcomes=list(row.outcomes),
                    row_score=row.row_score,
                    adjusted_score=row.adjusted_score,
                    avg_hamming_similarity=row.avg_hamming_similarity,
                    favorite_count=row.favorite_count,
                    home_count=row.home_count,
                    draw_count=row.draw_count,
                    away_count=row.away_count,
                )
                for row in portfolio
            ],
            candidates=[
                CandidateRowDTO(
                    outcomes=list(cand.outcomes),
                    row_score=cand.row_score,
                    log_pm_sum=cand.log_pm_sum,
                    log_pp_sum=cand.log_pp_sum,
                    favorite_count=cand.favorite_count,
                    home_count=cand.home_count,
                    draw_count=cand.draw_count,
                    away_count=cand.away_count,
                )
                for cand in candidates[: min(50, len(candidates))]
            ],
            match_analysis=[
                MatchAnalysisDTO(
                    match_index=item.match_index,
                    match_label=item.match_label,
                    jensen_shannon=item.jensen_shannon,
                    max_abs_log_leverage=item.max_abs_log_leverage,
                    highest_value_outcome=item.highest_value_outcome,
                    highest_value_ratio=item.highest_value_ratio,
                    market_probs=dict(item.market_probs),
                    public_probs=dict(item.public_probs),
                    value_ratios=dict(item.value_ratios),
                    log_leverages=dict(item.log_leverages),
                    banker_scores=dict(item.banker_scores),
                )
                for item in match_analyses
            ],
            portfolio_analysis=PortfolioAnalysisDTO(
                selected_count=len(portfolio),
                candidate_pool_size=len(candidates),
                mean_pairwise_hamming_similarity=_mean_pairwise_similarity(
                    [tuple(row.outcomes) for row in portfolio]
                ),
                top_bankers=[
                    BankerSuggestionDTO(
                        match_index=b.match_index,
                        outcome=b.outcome,
                        score=b.score,
                        log_pm=b.log_pm,
                        log_leverage=b.log_leverage,
                    )
                    for b in banker_suggestions[:10]
                ],
            ),
            limitations=coupon.limitations or LEAKAGE_LIMITATIONS,
        )

    def optimize(
        self,
        match_inputs: list[CouponMatchInput],
        *,
        row_count: int = 1,
        draw_number: int | None = None,
    ) -> StryktipsetOptimizationResult:
        coupon = prepare_matches(
            match_inputs,
            public_epsilon=self.params.public_epsilon,
            expected_count=self.params.coupon_size,
            draw_number=draw_number,
        )
        return self.optimize_prepared(coupon, row_count=row_count)


def _mean_pairwise_similarity(
    rows: list[tuple[Outcome, ...]],
) -> float | None:
    if len(rows) < 2:
        return None
    total = 0.0
    count = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            total += hamming_similarity(rows[i], rows[j])
            count += 1
    return total / count if count else None
