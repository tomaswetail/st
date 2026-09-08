"""Orchestration: CouponOptimizer.optimize(...).

Default mode=PREDICTION / objective=MAX_P13 (market Pm only).
VALUE mode preserves EV Σ log Pm − β Σ log Pp + Hamming diversity.
"""

from __future__ import annotations

from src.calc.stryktipset_optimizer.analyzer import analyze_matches
from src.calc.stryktipset_optimizer.candidates import top_candidates
from src.calc.stryktipset_optimizer.coverage import (
    CoverageMetrics,
    compute_coverage,
)
from src.calc.stryktipset_optimizer.data import (
    LEAKAGE_LIMITATIONS,
    CouponMatchInput,
    PreparedCoupon,
    prepare_matches,
)
from src.calc.stryktipset_optimizer.objectives import (
    APPROX_OBJECTIVES,
    greedy_approximate_portfolio,
)
from src.calc.stryktipset_optimizer.params import OptimizerParams
from src.calc.stryktipset_optimizer.portfolio import (
    hamming_similarity,
    select_diversified_portfolio,
)
from src.calc.stryktipset_optimizer.prediction import top_market_rows
from src.calc.stryktipset_optimizer.reduced_system import build_reduced_system
from src.objects.schema.data_classes.stryktipset_optimizer import (
    BankerSuggestionDTO,
    CandidateRowDTO,
    CoverageMetricsDTO,
    MatchAnalysisDTO,
    OptimizerParametersDTO,
    PortfolioAnalysisDTO,
    PortfolioRowDTO,
    StryktipsetOptimizationResult,
)
from src.utils.common import Outcome


class CouponOptimizer:
    """Stryktipset coupon optimizer (PREDICTION default; VALUE optional)."""

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
        if params.mode == "PREDICTION":
            return self._optimize_prediction(coupon, row_count=row_count)
        return self._optimize_value(coupon, row_count=row_count)

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

    def build_reduced_system(
        self,
        match_inputs: list[CouponMatchInput],
        *,
        row_budget: int,
        draw_number: int | None = None,
    ) -> StryktipsetOptimizationResult:
        """PREDICTION doubles/triples builder (marginal-gain upgrades)."""
        if self.params.mode != "PREDICTION":
            raise ValueError("build_reduced_system requires PREDICTION mode")
        coupon = prepare_matches(
            match_inputs,
            public_epsilon=self.params.public_epsilon,
            expected_count=self.params.coupon_size,
            draw_number=draw_number,
        )
        params = OptimizerParams(
            mode="PREDICTION",
            objective=self.params.objective,
            beta=self.params.beta,
            lambda_diversity=self.params.lambda_diversity,
            banker_value_weight=self.params.banker_value_weight,
            candidate_count=self.params.candidate_count,
            prediction_candidate_count=self.params.prediction_candidate_count,
            public_epsilon=self.params.public_epsilon,
            coupon_size=self.params.coupon_size,
            seed=self.params.seed,
            reduced_system=True,
        )
        return CouponOptimizer(params).optimize_prepared(
            coupon, row_count=row_budget
        )

    def _optimize_prediction(
        self,
        coupon: PreparedCoupon,
        *,
        row_count: int,
    ) -> StryktipsetOptimizationResult:
        params = self.params
        market_probs = [match.market_probs for match in coupon.matches]
        public_probs = [match.public_probs for match in coupon.matches]
        labels = [match.label for match in coupon.matches]
        limitations = coupon.limitations or LEAKAGE_LIMITATIONS

        system_sign_pattern: list[list[Outcome]] | None = None
        system_sign_counts: list[int] | None = None
        candidates_dto: list[CandidateRowDTO] = []
        pool_size = 0
        exact_selection = True

        if params.reduced_system:
            reduced = build_reduced_system(
                market_probs,
                row_budget=row_count,
                objective=params.objective,
            )
            portfolio_rows = [
                PortfolioRowDTO(
                    rank=rank,
                    outcomes=list(row.outcomes),
                    row_score=row.log_pm_sum,
                    adjusted_score=row.log_pm_sum,
                    avg_hamming_similarity=0.0,
                    favorite_count=row.favorite_count,
                    home_count=row.home_count,
                    draw_count=row.draw_count,
                    away_count=row.away_count,
                    joint_probability=row.joint_probability,
                    log_pm_sum=row.log_pm_sum,
                )
                for rank, row in enumerate(reduced.rows, start=1)
            ]
            coverage = reduced.coverage
            exact_selection = reduced.exact_selection
            pool_size = len(reduced.rows)
            system_sign_pattern = [list(signs) for signs in reduced.sign_pattern]
            system_sign_counts = list(reduced.sign_counts)
            limitations = f"{limitations} | {reduced.limitations}"
        elif params.objective == "MAX_P13":
            market_rows = top_market_rows(market_probs, row_count=row_count)
            portfolio_rows = [
                PortfolioRowDTO(
                    rank=rank,
                    outcomes=list(row.outcomes),
                    row_score=row.log_pm_sum,
                    adjusted_score=row.log_pm_sum,
                    avg_hamming_similarity=0.0,
                    favorite_count=row.favorite_count,
                    home_count=row.home_count,
                    draw_count=row.draw_count,
                    away_count=row.away_count,
                    joint_probability=row.joint_probability,
                    log_pm_sum=row.log_pm_sum,
                )
                for rank, row in enumerate(market_rows, start=1)
            ]
            coverage = compute_coverage(
                market_probs, [row.outcomes for row in market_rows]
            )
            exact_selection = True
            pool_size = len(market_rows)
            candidates_dto = [
                CandidateRowDTO(
                    outcomes=list(row.outcomes),
                    row_score=row.log_pm_sum,
                    log_pm_sum=row.log_pm_sum,
                    log_pp_sum=None,
                    favorite_count=row.favorite_count,
                    home_count=row.home_count,
                    draw_count=row.draw_count,
                    away_count=row.away_count,
                    joint_probability=row.joint_probability,
                )
                for row in market_rows[: min(50, len(market_rows))]
            ]
        else:
            if params.objective not in APPROX_OBJECTIVES:
                raise ValueError(f"unsupported objective: {params.objective}")
            approx = greedy_approximate_portfolio(
                market_probs,
                objective=params.objective,
                row_count=row_count,
                candidate_count=params.prediction_candidate_count,
            )
            portfolio_rows = [
                PortfolioRowDTO(
                    rank=rank,
                    outcomes=list(row.outcomes),
                    row_score=row.log_pm_sum,
                    adjusted_score=row.log_pm_sum,
                    avg_hamming_similarity=0.0,
                    favorite_count=row.favorite_count,
                    home_count=row.home_count,
                    draw_count=row.draw_count,
                    away_count=row.away_count,
                    joint_probability=row.joint_probability,
                    log_pm_sum=row.log_pm_sum,
                )
                for rank, row in enumerate(approx.rows, start=1)
            ]
            coverage = approx.coverage
            exact_selection = False
            pool_size = approx.candidate_pool_size
            limitations = f"{limitations} | {approx.limitations}"
            candidates_dto = [
                CandidateRowDTO(
                    outcomes=list(row.outcomes),
                    row_score=row.log_pm_sum,
                    log_pm_sum=row.log_pm_sum,
                    log_pp_sum=None,
                    favorite_count=row.favorite_count,
                    home_count=row.home_count,
                    draw_count=row.draw_count,
                    away_count=row.away_count,
                    joint_probability=row.joint_probability,
                )
                for row in approx.rows[: min(50, len(approx.rows))]
            ]

        match_analyses, banker_suggestions = analyze_matches(
            market_probs,
            public_probs,
            banker_value_weight=params.banker_value_weight,
            match_labels=labels,
        )
        sum_joint = sum(
            (row.joint_probability or 0.0) for row in portfolio_rows
        )

        return StryktipsetOptimizationResult(
            draw_number=coupon.draw_number,
            match_count=len(coupon.matches),
            mode="PREDICTION",
            objective=params.objective,
            exact_selection=exact_selection,
            parameters=_params_dto(params, row_count),
            rows=portfolio_rows,
            candidates=candidates_dto,
            match_analysis=_match_analysis_dtos(match_analyses),
            portfolio_analysis=PortfolioAnalysisDTO(
                selected_count=len(portfolio_rows),
                candidate_pool_size=pool_size,
                mean_pairwise_hamming_similarity=_mean_pairwise_similarity(
                    [tuple(row.outcomes) for row in portfolio_rows]
                ),
                top_bankers=_banker_dtos(banker_suggestions[:10]),
                sum_joint_probability=sum_joint,
            ),
            coverage=_coverage_dto(coverage),
            system_sign_pattern=system_sign_pattern,
            system_sign_counts=system_sign_counts,
            limitations=limitations,
        )

    def _optimize_value(
        self,
        coupon: PreparedCoupon,
        *,
        row_count: int,
    ) -> StryktipsetOptimizationResult:
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
        coverage = compute_coverage(
            market_probs, [row.outcomes for row in portfolio]
        )
        limitations = (
            f"{coupon.limitations or LEAKAGE_LIMITATIONS} | "
            "VALUE mode: public % affects selection; prior β-leverage "
            "OOS backtests apply to VALUE only, not PREDICTION default."
        )

        return StryktipsetOptimizationResult(
            draw_number=coupon.draw_number,
            match_count=len(coupon.matches),
            mode="VALUE",
            objective="VALUE_EV",
            exact_selection=False,
            parameters=_params_dto(params, row_count),
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
                    joint_probability=None,
                    log_pm_sum=None,
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
                    joint_probability=None,
                )
                for cand in candidates[: min(50, len(candidates))]
            ],
            match_analysis=_match_analysis_dtos(match_analyses),
            portfolio_analysis=PortfolioAnalysisDTO(
                selected_count=len(portfolio),
                candidate_pool_size=len(candidates),
                mean_pairwise_hamming_similarity=_mean_pairwise_similarity(
                    [tuple(row.outcomes) for row in portfolio]
                ),
                top_bankers=_banker_dtos(banker_suggestions[:10]),
                sum_joint_probability=None,
            ),
            coverage=_coverage_dto(coverage),
            system_sign_pattern=None,
            system_sign_counts=None,
            limitations=limitations,
        )


def _params_dto(params: OptimizerParams, row_count: int) -> OptimizerParametersDTO:
    return OptimizerParametersDTO(
        mode=params.mode,
        objective=params.objective if params.mode == "PREDICTION" else "VALUE_EV",
        beta=params.beta,
        lambda_diversity=params.lambda_diversity,
        banker_value_weight=params.banker_value_weight,
        candidate_count=params.candidate_count,
        prediction_candidate_count=params.prediction_candidate_count,
        public_epsilon=params.public_epsilon,
        coupon_size=params.coupon_size,
        seed=params.seed,
        row_count=row_count,
        reduced_system=params.reduced_system,
    )


def _coverage_dto(coverage: CoverageMetrics) -> CoverageMetricsDTO:
    return CoverageMetricsDTO(
        n_matches=coverage.n_matches,
        p_best_correct=list(coverage.p_best_correct),
        p_full=coverage.p_full,
        p_13=coverage.p_13,
        p_12_or_better=coverage.p_12_or_better,
        p_11_or_better=coverage.p_11_or_better,
        p_10_or_better=coverage.p_10_or_better,
        expected_best_correct=coverage.expected_best_correct,
    )


def _match_analysis_dtos(match_analyses) -> list[MatchAnalysisDTO]:
    return [
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
    ]


def _banker_dtos(bankers) -> list[BankerSuggestionDTO]:
    return [
        BankerSuggestionDTO(
            match_index=b.match_index,
            outcome=b.outcome,
            score=b.score,
            log_pm=b.log_pm,
            log_leverage=b.log_leverage,
        )
        for b in bankers
    ]


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
