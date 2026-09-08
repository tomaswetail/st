"""Unit tests for Stryktipset coupon optimizer (PREDICTION default / VALUE)."""

from __future__ import annotations

import itertools
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.calc.stryktipset_optimizer import (
    CouponMatchInput,
    CouponOptimizer,
    InvalidOddsError,
    InvalidPublicShareError,
    OptimizerParams,
    ParamConfig,
    build_reduced_system,
    compute_coverage,
    fair_probabilities_from_odds,
    hamming_distance,
    hamming_similarity,
    log_leverage,
    naive_closest_match_upgrades,
    normalize_public_shares,
    prepare_matches,
    row_joint_probability,
    run_backtest,
    select_diversified_portfolio,
    simulate_pool,
    top_candidates,
    top_market_rows,
    value_ratio,
)
from src.calc.stryktipset_optimizer.analyzer import (
    analyze_matches,
    jensen_shannon_divergence,
)
from src.calc.stryktipset_optimizer.backtester import order_coupons_chronologically
from src.calc.stryktipset_optimizer.candidates import build_score_table
from src.calc.stryktipset_optimizer.data import CouponDataError
from src.utils.common import OUTCOMES

# ---------------------------------------------------------------------------
# Fair probs
# ---------------------------------------------------------------------------


def test_fair_probs_worked_example():
    pm = fair_probabilities_from_odds(2.0, 3.5, 4.0)
    assert pm["1"] == pytest.approx(0.4827586207, rel=1e-6)
    assert pm["X"] == pytest.approx(0.2758620690, rel=1e-6)
    assert pm["2"] == pytest.approx(0.2413793103, rel=1e-6)
    assert sum(pm.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [1.0, 0.5, -1.0, 0.0])
def test_fair_probs_rejects_odds_le_1(bad: float):
    with pytest.raises(InvalidOddsError):
        fair_probabilities_from_odds(bad, 3.5, 4.0)


def test_fair_probs_rejects_nan_and_none():
    with pytest.raises(InvalidOddsError):
        fair_probabilities_from_odds(float("nan"), 3.5, 4.0)
    with pytest.raises(InvalidOddsError):
        fair_probabilities_from_odds(float("inf"), 3.5, 4.0)
    with pytest.raises(InvalidOddsError):
        fair_probabilities_from_odds(None, 3.5, 4.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Public shares
# ---------------------------------------------------------------------------


def test_public_percent_normalize_and_epsilon():
    pp = normalize_public_shares(50, 30, 20, public_epsilon=1e-6)
    assert sum(pp.values()) == pytest.approx(1.0)
    assert pp["1"] == pytest.approx(0.5, abs=1e-5)


def test_public_unit_scale():
    pp = normalize_public_shares(0.5, 0.3, 0.2)
    assert pp["1"] == pytest.approx(0.5, abs=1e-5)


def test_public_renorm_when_sum_not_100():
    pp = normalize_public_shares(40, 40, 40)  # sum 120 → renorm
    assert sum(pp.values()) == pytest.approx(1.0)
    assert pp["1"] == pytest.approx(1 / 3, abs=1e-4)


def test_public_rejects_sum_zero():
    with pytest.raises(InvalidPublicShareError):
        normalize_public_shares(0, 0, 0)


def test_public_one_percent_integers_are_percentage_scale():
    # STMatchBetModel ints: 1 means 1%, not unit-scale 1.0.
    pp = normalize_public_shares(50, 1, 49, public_epsilon=1e-6)
    assert sum(pp.values()) == pytest.approx(1.0)
    assert pp["1"] == pytest.approx(0.50, abs=1e-4)
    assert pp["X"] == pytest.approx(0.01, abs=1e-4)
    assert pp["2"] == pytest.approx(0.49, abs=1e-4)

    pp2 = normalize_public_shares(98, 1, 1, public_epsilon=1e-6)
    assert sum(pp2.values()) == pytest.approx(1.0)
    assert pp2["1"] == pytest.approx(0.98, abs=1e-4)
    assert pp2["X"] == pytest.approx(0.01, abs=1e-4)
    assert pp2["2"] == pytest.approx(0.01, abs=1e-4)


def test_public_fractional_percent_when_max_gt_1():
    # max>1 ⇒ percent scale; 0.3 means 0.3%.
    pp = normalize_public_shares(99.5, 0.3, 0.2, public_epsilon=1e-6)
    assert sum(pp.values()) == pytest.approx(1.0)
    assert pp["1"] == pytest.approx(0.995, abs=1e-4)
    assert pp["X"] == pytest.approx(0.003, abs=1e-4)
    assert pp["2"] == pytest.approx(0.002, abs=1e-4)


def test_public_rejects_negative_and_over_100():
    with pytest.raises(InvalidPublicShareError):
        normalize_public_shares(-1, 50, 50)
    with pytest.raises(InvalidPublicShareError):
        normalize_public_shares(101, 0, 0)


def test_public_tiny_unit_scale_gets_epsilon_floor():
    pp = normalize_public_shares(0.999, 0.0005, 0.0005, public_epsilon=1e-4)
    assert pp["X"] >= 1e-4 - 1e-12
    assert sum(pp.values()) == pytest.approx(1.0)


def test_prepare_matches_accepts_one_percent_public_on_13():
    inputs = [
        _make_match(public=(50, 1, 49) if i == 0 else (40, 30, 30))
        for i in range(13)
    ]
    coupon = prepare_matches(inputs, expected_count=13)
    assert coupon.matches[0].public_probs["X"] == pytest.approx(0.01, abs=1e-4)


# ---------------------------------------------------------------------------
# Value
# ---------------------------------------------------------------------------


def test_value_ratio_and_log_leverage():
    assert value_ratio(0.2, 0.1) == pytest.approx(2.0)
    assert log_leverage(0.2, 0.1) == pytest.approx(math.log(2.0))


# ---------------------------------------------------------------------------
# Row score / beta ranking flip (VALUE)
# ---------------------------------------------------------------------------


def _two_match_probs():
    # Match 0: market favorite is 1, but public piles on 1 → high beta prefers rare 2
    pm0 = {"1": 0.50, "X": 0.35, "2": 0.15}
    pp0 = {"1": 0.80, "X": 0.15, "2": 0.05}
    # Match 1: aligned (neutral)
    pm1 = {"1": 0.5, "X": 0.25, "2": 0.25}
    pp1 = {"1": 0.5, "X": 0.25, "2": 0.25}
    return [pm0, pm1], [pp0, pp1]


def test_row_score_sum_log_product_identity():
    market, public = _two_match_probs()
    table = build_score_table(market, public, beta=1.0)
    score = float(table[0, 0] + table[1, 0])
    expected = (
        math.log(0.50) - 1.0 * math.log(0.80)
        + math.log(0.5) - 1.0 * math.log(0.5)
    )
    assert score == pytest.approx(expected)
    log_pm = math.log(0.50) + math.log(0.5)
    log_pp = math.log(0.80) + math.log(0.5)
    assert score == pytest.approx(log_pm - 1.0 * log_pp)


def test_beta_ranking_flip():
    market, public = _two_match_probs()
    low_beta = top_candidates(market, public, beta=0.0, candidate_count=9)
    high_beta = top_candidates(market, public, beta=10.0, candidate_count=9)
    # beta=0 maximizes product of Pm → market favorite "1"
    assert low_beta[0].outcomes[0] == "1"
    # high beta rewards rare public outcomes → "2"
    assert high_beta[0].outcomes[0] == "2"
    assert low_beta[0].outcomes != high_beta[0].outcomes


# ---------------------------------------------------------------------------
# Hamming / portfolio diversity (VALUE)
# ---------------------------------------------------------------------------


def test_hamming_similarity_and_distance():
    a = ("1",) * 13
    b = ("1",) * 13
    c = ("1",) * 12 + ("X",)
    assert hamming_similarity(a, b) == pytest.approx(1.0)
    assert hamming_distance(a, b) == 0
    assert hamming_similarity(a, c) == pytest.approx(12 / 13)
    assert hamming_distance(a, c) == 1


def test_favorite_count_is_market_argmax_not_homes():
    # Match 0: market favorite is "1"; selecting "2" must not count as favorite.
    # Match 1: market favorite is "2"; selecting "2" must count as favorite.
    market = [
        {"1": 0.55, "X": 0.25, "2": 0.20},
        {"1": 0.20, "X": 0.25, "2": 0.55},
    ]
    public = [
        {"1": 0.4, "X": 0.3, "2": 0.3},
        {"1": 0.4, "X": 0.3, "2": 0.3},
    ]
    candidates = top_candidates(market, public, beta=1.0, candidate_count=9)
    by_row = {"".join(c.outcomes): c for c in candidates}

    away_then_away_fav = by_row["22"]
    assert away_then_away_fav.favorite_count == 1  # only match1
    assert away_then_away_fav.home_count == 0
    assert away_then_away_fav.away_count == 2

    both_favs = by_row["12"]
    assert both_favs.favorite_count == 2
    assert both_favs.home_count == 1
    assert both_favs.away_count == 1

    home_not_fav_on_match1 = by_row["11"]
    assert home_not_fav_on_match1.favorite_count == 1  # only match0
    assert home_not_fav_on_match1.home_count == 2


def test_lambda_diversity_diversifies_vs_zero():
    # Build candidates that share prefixes so diversity matters
    market = [{"1": 0.7, "X": 0.2, "2": 0.1} for _ in range(3)]
    # Skew public so top scores cluster on similar rows
    public = [{"1": 0.4, "X": 0.4, "2": 0.2} for _ in range(3)]
    candidates = top_candidates(market, public, beta=1.0, candidate_count=27)

    pure = select_diversified_portfolio(
        candidates, row_count=3, lambda_diversity=0.0
    )
    diversified = select_diversified_portfolio(
        candidates, row_count=3, lambda_diversity=10.0
    )

    def mean_pair(rows):
        vals = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                vals.append(
                    hamming_similarity(rows[i].outcomes, rows[j].outcomes)
                )
        return sum(vals) / len(vals)

    assert mean_pair(diversified) <= mean_pair(pure) + 1e-12
    assert pure[0].outcomes == candidates[0].outcomes


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------


def _make_match(
    odds=(2.0, 3.5, 4.0),
    public=(50, 30, 20),
    **kwargs,
) -> CouponMatchInput:
    return CouponMatchInput(
        odds_1=odds[0],
        odds_x=odds[1],
        odds_2=odds[2],
        public_1=public[0],
        public_x=public[1],
        public_2=public[2],
        **kwargs,
    )


def test_prepare_rejects_wrong_match_count():
    with pytest.raises(CouponDataError):
        prepare_matches([_make_match()] * 12, expected_count=13)


def test_prepare_rejects_invalid_odds():
    with pytest.raises(CouponDataError):
        prepare_matches(
            [_make_match(odds=(1.0, 3.5, 4.0))] * 13,
            expected_count=13,
        )


def test_end_to_end_optimize_small_coupon_prediction_default():
    inputs = [
        _make_match(
            odds=(2.1 + 0.1 * i, 3.4, 3.8 + 0.05 * i),
            public=(45, 30, 25),
        )
        for i in range(4)
    ]
    opt = CouponOptimizer(
        OptimizerParams(coupon_size=4, prediction_candidate_count=81)
    )
    result = opt.optimize(inputs, row_count=2, draw_number=9999)
    assert result.mode == "PREDICTION"
    assert result.objective == "MAX_P13"
    assert result.exact_selection is True
    assert result.draw_number == 9999
    assert result.match_count == 4
    assert len(result.rows) == 2
    assert result.coverage is not None
    assert "UNKNOWN" in result.limitations


def test_value_mode_end_to_end():
    inputs = [_make_match() for _ in range(3)]
    result = CouponOptimizer(
        OptimizerParams(
            mode="VALUE",
            candidate_count=27,
            coupon_size=3,
            lambda_diversity=0.0,
        )
    ).optimize(inputs, row_count=1)
    assert result.mode == "VALUE"
    assert len(result.rows) == 1
    assert result.parameters.beta == 1.0


def test_dto_has_required_fields():
    inputs = [_make_match() for _ in range(3)]
    result = CouponOptimizer(
        OptimizerParams(coupon_size=3)
    ).optimize(inputs, row_count=1)
    dumped = result.model_dump()
    for key in (
        "rows",
        "match_analysis",
        "portfolio_analysis",
        "parameters",
        "limitations",
        "mode",
        "objective",
        "coverage",
        "exact_selection",
    ):
        assert key in dumped


# ---------------------------------------------------------------------------
# PREDICTION / MAX_P13
# ---------------------------------------------------------------------------


def _small_market(n_matches: int = 3):
    markets = []
    for i in range(n_matches):
        # Distinct skewed probs so ranking is unambiguous.
        p1 = 0.45 + 0.02 * i
        px = 0.30 - 0.01 * i
        p2 = 1.0 - p1 - px
        markets.append({"1": p1, "X": px, "2": p2})
    return markets


def test_max_p13_equals_top_n_by_joint_p_brute_force():
    market = _small_market(3)
    row_count = 5
    heap_rows = top_market_rows(market, row_count=row_count)

    all_rows = []
    for combo in itertools.product(OUTCOMES, repeat=3):
        all_rows.append(
            (combo, row_joint_probability(market, combo))
        )
    all_rows.sort(key=lambda item: (-item[1], item[0]))
    expected = [row for row, _ in all_rows[:row_count]]
    got = [row.outcomes for row in heap_rows]
    assert got == expected


def test_max_p13_p_full_equals_sum_row_probs():
    market = _small_market(3)
    rows = top_market_rows(market, row_count=4)
    coverage = compute_coverage(market, [r.outcomes for r in rows])
    sum_p = sum(r.joint_probability for r in rows)
    assert coverage.p_full == pytest.approx(sum_p, rel=1e-9)
    assert coverage.p_13 == pytest.approx(sum_p, rel=1e-9)


def test_prediction_invariant_to_public_flip():
    inputs_a = [
        _make_match(odds=(2.0 + 0.2 * i, 3.5, 4.0), public=(60, 20, 20))
        for i in range(4)
    ]
    inputs_b = [
        _make_match(odds=(2.0 + 0.2 * i, 3.5, 4.0), public=(10, 10, 80))
        for i in range(4)
    ]
    params = OptimizerParams(mode="PREDICTION", objective="MAX_P13", coupon_size=4)
    a = CouponOptimizer(params).optimize(inputs_a, row_count=3)
    b = CouponOptimizer(params).optimize(inputs_b, row_count=3)
    assert [tuple(r.outcomes) for r in a.rows] == [
        tuple(r.outcomes) for r in b.rows
    ]


def test_coverage_metrics_sanity():
    market = _small_market(3)
    rows = [r.outcomes for r in top_market_rows(market, row_count=2)]
    coverage = compute_coverage(market, rows)
    assert len(coverage.p_best_correct) == 4
    assert all(p >= -1e-12 for p in coverage.p_best_correct)
    assert sum(coverage.p_best_correct) == pytest.approx(1.0, abs=1e-9)
    # Cumulatives monotone in threshold: P(best>=k) >= P(best>=k+1)
    cumul = [
        sum(coverage.p_best_correct[k:])
        for k in range(len(coverage.p_best_correct))
    ]
    for k in range(len(cumul) - 1):
        assert cumul[k] >= cumul[k + 1] - 1e-12
    assert coverage.p_full == pytest.approx(cumul[3], abs=1e-12)
    # Named 10+/11+/12+ are absolute thresholds (0 when N < threshold)
    assert coverage.p_12_or_better == 0.0
    assert coverage.p_11_or_better == 0.0
    assert coverage.p_10_or_better == 0.0
    assert 0.0 <= coverage.expected_best_correct <= 3.0 + 1e-9


def test_max_p13_n13_small_r_matches_sorted_joint():
    """N=13 with R=3: heap top-3 equals independent numpy full-enum top-3."""
    import numpy as np

    market = []
    for i in range(13):
        p1 = 0.50 + 0.01 * ((i * 3) % 5)
        px = 0.28 - 0.005 * ((i * 2) % 4)
        p2 = 1.0 - p1 - px
        market.append({"1": p1, "X": px, "2": p2})

    top3 = top_market_rows(market, row_count=3)

    # Independent brute force: all 3^13 joint probs via base-3 digits.
    pm = np.array([[m[o] for o in OUTCOMES] for m in market], dtype=np.float64)
    n_matches = 13
    n_worlds = 3**n_matches
    world_ids = np.arange(n_worlds, dtype=np.int32)
    worlds = np.empty((n_worlds, n_matches), dtype=np.int8)
    tmp = world_ids.copy()
    for match_index in range(n_matches - 1, -1, -1):
        worlds[:, match_index] = (tmp % 3).astype(np.int8)
        tmp //= 3
    log_p = np.log(pm)
    log_joint = np.zeros(n_worlds, dtype=np.float64)
    for match_index in range(n_matches):
        log_joint += log_p[match_index, worlds[:, match_index]]
    # Top-3 by joint P; ties: smaller world_id (DFS/"1","X","2" digit order).
    # heapq.nlargest unstable on ties — use lex key via structured sort.
    order = np.lexsort((world_ids, -log_joint))  # primary -log_joint, then id
    best_ids = order[:3]
    expected = []
    for world_id in best_ids:
        digits = worlds[world_id]
        expected.append(tuple(OUTCOMES[int(d)] for d in digits))

    assert [r.outcomes for r in top3] == expected


def test_default_params_are_prediction_max_p13():
    params = OptimizerParams()
    assert params.mode == "PREDICTION"
    assert params.objective == "MAX_P13"


# ---------------------------------------------------------------------------
# Reduced system
# ---------------------------------------------------------------------------


def test_reduced_system_marginal_beats_naive_closest():
    """Crafted coupon where closest-match upgrade ≠ max ΔP13 upgrade."""
    # Gaps: match2=0.30 (closest), match0=0.34, match1=0.35.
    # p2nd/fav ratios: match1 > match2 > match0 → MAX_P13 prefers match1.
    market = [
        {"1": 0.20, "X": 0.23, "2": 0.57},
        {"1": 0.05, "X": 0.30, "2": 0.65},
        {"1": 0.55, "X": 0.25, "2": 0.20},
    ]
    budget = 2  # one double only
    marginal = build_reduced_system(
        market, row_budget=budget, objective="MAX_P13"
    )
    naive_counts = naive_closest_match_upgrades(market, row_budget=budget)
    assert marginal.sign_counts != naive_counts
    assert marginal.sign_counts == (1, 2, 1)  # double match 1 (max ΔP13)
    assert naive_counts == (1, 1, 2)  # double match 2 (closest gap)



# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


def test_jensen_shannon_symmetric_and_zero_when_equal():
    p = {"1": 0.5, "X": 0.3, "2": 0.2}
    assert jensen_shannon_divergence(p, p) == pytest.approx(0.0)
    q = {"1": 0.2, "X": 0.3, "2": 0.5}
    assert jensen_shannon_divergence(p, q) == pytest.approx(
        jensen_shannon_divergence(q, p)
    )


def test_analyze_matches_ranks_by_js():
    market = [
        {"1": 0.5, "X": 0.3, "2": 0.2},
        {"1": 0.8, "X": 0.1, "2": 0.1},
    ]
    public = [
        {"1": 0.5, "X": 0.3, "2": 0.2},  # identical → JS≈0
        {"1": 0.1, "X": 0.1, "2": 0.8},  # large divergence
    ]
    analyses, bankers = analyze_matches(market, public, banker_value_weight=1.0)
    assert analyses[0].match_index == 1
    assert analyses[0].jensen_shannon > analyses[1].jensen_shannon
    assert bankers[0].score >= bankers[-1].score


# ---------------------------------------------------------------------------
# Phase 2: MC simulator
# ---------------------------------------------------------------------------


def test_simulate_pool_deterministic():
    market = [{"1": 0.5, "X": 0.3, "2": 0.2} for _ in range(5)]
    public = [{"1": 0.4, "X": 0.4, "2": 0.2} for _ in range(5)]
    rows = [("1", "1", "1", "X", "2")]
    a = simulate_pool(
        market, public, rows, n_simulations=100, seed=42, public_row_count=5
    )
    b = simulate_pool(
        market, public, rows, n_simulations=100, seed=42, public_row_count=5
    )
    c = simulate_pool(
        market, public, rows, n_simulations=100, seed=99, public_row_count=5
    )
    assert a.model_dump() == b.model_dump()
    assert a.mean_correct_ours != c.mean_correct_ours or a.our_tiers != c.our_tiers
    assert "13" in a.our_tier_rates
    assert a.limitations  # relative only; leakage documented


def test_simulate_no_sek_fields():
    market = [{"1": 0.6, "X": 0.2, "2": 0.2} for _ in range(3)]
    public = market
    sim = simulate_pool(
        market,
        public,
        [("1", "X", "2")],
        n_simulations=50,
        seed=0,
        public_row_count=5,
    )
    dumped = sim.model_dump()
    assert "sek" not in json.dumps(dumped).lower()
    assert "payout" not in json.dumps(dumped).lower()


def test_simulate_default_public_row_count_is_practical():
    from src.calc.stryktipset_optimizer.simulator import DEFAULT_PUBLIC_ROW_COUNT

    assert DEFAULT_PUBLIC_ROW_COUNT == 1000


# ---------------------------------------------------------------------------
# Phase 2: backtester chronological order (VALUE)
# ---------------------------------------------------------------------------


def test_backtest_chronological_no_shuffle():
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def coupon_at(day: int, draw: int):
        inputs = []
        for i in range(3):
            inputs.append(
                _make_match(
                    start_time=base + timedelta(days=day, hours=i),
                    result="1",
                )
            )
        return prepare_matches(inputs, expected_count=3, draw_number=draw)

    # Intentionally out of order in the input list
    coupons = [coupon_at(10, 100), coupon_at(1, 90), coupon_at(5, 95)]
    ordered = order_coupons_chronologically(coupons)
    assert [c.draw_number for c in ordered] == [90, 95, 100]

    result = run_backtest(
        coupons,
        [
            ParamConfig(
                mode="VALUE",
                beta=1.0,
                lambda_diversity=0.0,
                candidate_count=27,
                row_count=1,
            ),
            ParamConfig(
                mode="VALUE",
                beta=0.5,
                lambda_diversity=0.0,
                candidate_count=27,
                row_count=1,
            ),
        ],
        n_simulations=10,
        seed=0,
        mode="VALUE",
    )
    assert result.mode == "VALUE"
    assert result.primary_metric == "portfolio_leverage"
    assert result.best_by_mean_portfolio_leverage is not None
    assert result.best_by_primary is not None
    assert not hasattr(result, "best_by_mean_top_row_score")
    assert not hasattr(result, "best_by_mean_correct")
    assert "UNKNOWN" in result.limitations
    scored = [r for r in result.results if r.rounds_evaluated > 0]
    assert scored
    expected_best = max(scored, key=lambda r: r.mean_portfolio_leverage).params
    assert result.best_by_mean_portfolio_leverage == expected_best
    # Round evals follow chronological sort keys
    for cfg in result.results:
        keys = [ev.sort_key for ev in cfg.round_evals]
        assert keys == sorted(keys)
        for ev in cfg.round_evals:
            assert ev.realized_row_score is not None  # settled fixtures
            assert ev.portfolio_leverage == ev.portfolio_leverage  # finite


def test_backtest_best_not_trivial_max_beta_via_score_inflation():
    """High construction β inflates row_score; leverage ranking must not follow it.

    Fixture: market favorite has better Pm/Pp than a low-Pp longshot. High β
    chases the longshot (worse leverage); low/mid β keep the favorite.
    """
    coupons = []
    for draw, day in ((10, 1), (11, 2)):
        match_inputs = [
            CouponMatchInput(
                odds_1=1.67,  # ~0.60 fair
                odds_x=3.33,  # ~0.30
                odds_2=10.0,  # ~0.10
                public_1=50,
                public_x=40,
                public_2=10,
                result="1",
                start_time=datetime(2024, 1, day, tzinfo=timezone.utc),
            )
            for _ in range(3)
        ]
        coupons.append(
            prepare_matches(match_inputs, expected_count=3, draw_number=draw)
        )

    grid = [
        ParamConfig(
            mode="VALUE",
            beta=0.0,
            lambda_diversity=0.0,
            candidate_count=27,
            row_count=1,
        ),
        ParamConfig(
            mode="VALUE",
            beta=1.0,
            lambda_diversity=0.0,
            candidate_count=27,
            row_count=1,
        ),
        ParamConfig(
            mode="VALUE",
            beta=5.0,
            lambda_diversity=0.0,
            candidate_count=27,
            row_count=1,
        ),
    ]
    result = run_backtest(coupons, grid, n_simulations=5, seed=0, mode="VALUE")
    assert result.best_by_mean_portfolio_leverage is not None

    by_beta = {r.params.beta: r for r in result.results if r.rounds_evaluated > 0}
    assert set(by_beta) == {0.0, 1.0, 5.0}

    # Construction row_score diagnostic inflates with β (trap).
    assert by_beta[5.0].mean_top_row_score > by_beta[1.0].mean_top_row_score
    assert by_beta[1.0].mean_top_row_score > by_beta[0.0].mean_top_row_score

    # Leverage of high-β longshot portfolio is worse than low/mid β.
    assert by_beta[5.0].mean_portfolio_leverage < by_beta[0.0].mean_portfolio_leverage
    assert by_beta[5.0].mean_portfolio_leverage < by_beta[1.0].mean_portfolio_leverage

    best_beta = result.best_by_mean_portfolio_leverage.beta
    assert best_beta != 5.0
    assert best_beta in (0.0, 1.0)
    trap_best = max(result.results, key=lambda r: r.mean_top_row_score).params.beta
    assert trap_best == 5.0
    assert result.best_by_mean_portfolio_leverage.beta != trap_best


def test_backtest_falls_back_to_draw_number_when_no_start_time():
    early = prepare_matches(
        [_make_match(result="1") for _ in range(3)],
        expected_count=3,
        draw_number=10,
    )
    late = prepare_matches(
        [_make_match(result="1") for _ in range(3)],
        expected_count=3,
        draw_number=20,
    )
    ordered = order_coupons_chronologically([late, early])
    assert [c.draw_number for c in ordered] == [10, 20]


def test_backtest_prediction_uses_coverage_primary_not_leverage():
    coupons = [
        prepare_matches(
            [_make_match(result="1") for _ in range(3)],
            expected_count=3,
            draw_number=1,
        )
    ]
    result = run_backtest(
        coupons,
        [
            ParamConfig(
                mode="PREDICTION",
                objective="MAX_P13",
                row_count=1,
                prediction_candidate_count=27,
            )
        ],
        n_simulations=5,
        seed=0,
        mode="PREDICTION",
    )
    assert result.mode == "PREDICTION"
    assert result.primary_metric == "p_full"
    assert result.best_by_mean_portfolio_leverage is None
    assert result.best_by_primary is not None
    assert "VALUE-only" in result.limitations or "PREDICTION" in result.limitations


# ---------------------------------------------------------------------------
# N=13 smoke + microbench
# ---------------------------------------------------------------------------


def test_full_13_match_optimize_and_benchmark(record_property):
    inputs = [
        _make_match(
            odds=(2.0 + 0.05 * (i % 5), 3.2 + 0.1 * (i % 3), 3.5 + 0.15 * (i % 4)),
            public=(40 + (i % 10), 35 - (i % 5), 25),
        )
        for i in range(13)
    ]
    # PREDICTION MAX_P13 default
    params_pred = OptimizerParams(
        mode="PREDICTION",
        objective="MAX_P13",
        coupon_size=13,
        seed=0,
    )
    opt = CouponOptimizer(params_pred)
    t0 = time.perf_counter()
    result = opt.optimize(inputs, row_count=5, draw_number=1)
    elapsed_pred = time.perf_counter() - t0
    assert len(result.rows) == 5
    assert result.mode == "PREDICTION"
    assert result.coverage is not None
    assert result.coverage.p_full == pytest.approx(
        sum(r.joint_probability or 0.0 for r in result.rows),
        rel=1e-6,
    )

    # VALUE path still works for candidate_count microbench
    params_500 = OptimizerParams(
        mode="VALUE",
        beta=1.0,
        lambda_diversity=0.5,
        candidate_count=500,
        coupon_size=13,
        seed=0,
    )
    t1 = time.perf_counter()
    result_500 = CouponOptimizer(params_500).optimize(inputs, row_count=5)
    elapsed_500 = time.perf_counter() - t1
    assert result_500.portfolio_analysis.candidate_pool_size == 500

    params_50k = OptimizerParams(
        mode="VALUE",
        beta=1.0,
        lambda_diversity=0.5,
        candidate_count=50_000,
        coupon_size=13,
        seed=0,
    )
    t2 = time.perf_counter()
    result_50k = CouponOptimizer(params_50k).optimize(inputs, row_count=5)
    elapsed_50k = time.perf_counter() - t2
    assert result_50k.portfolio_analysis.candidate_pool_size == 50_000

    record_property("bench_prediction_max_p13_sec", elapsed_pred)
    record_property("bench_candidate_500_sec", elapsed_500)
    record_property("bench_candidate_50000_sec", elapsed_50k)
    assert elapsed_pred < 30.0
    assert elapsed_500 < 30.0
    assert elapsed_50k < 60.0
    print(
        f"\nBENCHMARK N=13 PREDICTION R=5: {elapsed_pred:.3f}s; "
        f"VALUE C=500: {elapsed_500:.3f}s; C=50000: {elapsed_50k:.3f}s"
    )


def test_no_residual_ml_or_probability_manager_imports():
    """Optimize path must not import production scoring stack."""
    import src.calc.stryktipset_optimizer as pkg
    import src.calc.stryktipset_optimizer.optimize as opt_mod
    import src.calc.stryktipset_optimizer.candidates as cand_mod
    import src.calc.stryktipset_optimizer.prediction as pred_mod
    import src.calc.stryktipset_optimizer.coverage as cov_mod
    import src.calc.stryktipset_optimizer.objectives as obj_mod
    import src.calc.stryktipset_optimizer.reduced_system as red_mod

    forbidden = (
        "src.calc.residual_ml",
        "src.calc.probability_manager",
        "src.calc.dixon_coles",
        "src.calc.dixon_coles_nbm",
    )
    for module in (pkg, opt_mod, cand_mod, pred_mod, cov_mod, obj_mod, red_mod):
        module_text = Path(module.__file__).read_text(encoding="utf-8")
        for name in forbidden:
            assert name not in module_text


def test_cli_fixture_roundtrip(tmp_path: Path):
    from src.scripts import optimize_stryktipset_coupon as cli

    fixture = {
        "draw_number": 42,
        "matches": [
            {
                "odds_1": 2.0,
                "odds_x": 3.5,
                "odds_2": 4.0,
                "public_1": 50,
                "public_x": 30,
                "public_2": 20,
            }
            for _ in range(13)
        ],
    }
    path = tmp_path / "coupon.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    out = tmp_path / "out.json"

    import sys

    argv = [
        "optimize_stryktipset_coupon",
        "--fixture",
        str(path),
        "--mode",
        "PREDICTION",
        "--objective",
        "MAX_P13",
        "--rows",
        "2",
        "--seed",
        "0",
        "--json",
        str(out),
    ]
    old = sys.argv
    try:
        sys.argv = argv
        cli.main()
    finally:
        sys.argv = old

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["draw_number"] == 42
    assert payload["mode"] == "PREDICTION"
    assert payload["objective"] == "MAX_P13"
    assert payload["exact_selection"] is True
    assert len(payload["rows"]) == 2
    assert payload["coverage"] is not None
    assert "UNKNOWN" in payload["limitations"]
