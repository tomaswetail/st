"""Tests for ResidualMLTrainer and ResidualMLModel."""

from __future__ import annotations

from pathlib import Path

import pytest

from calc.residual_ml.baseline import apply_residual_deltas, blend_baselines
from calc.residual_ml.model import ResidualMLModel
from calc.residual_ml.trainer import ResidualMLTrainer
from objects.schema.data_classes.residual_ml_features import ResidualMLFeatures


def _synthetic_rows(count: int = 120) -> list[dict]:
    rows = []
    for index in range(count):
        label = ("1", "X", "2")[index % 3]
        p_home = 0.45 + (index % 5) * 0.01
        p_draw = 0.28
        p_away = 1.0 - p_home - p_draw
        row = {
            "match_id": index + 1,
            "draw_number": 4700 + index // 13,
            "feature_cutoff_date": f"2024-{(index % 12) + 1:02d}-15",
            "match_date": f"2024-{(index % 12) + 1:02d}-15",
            "p_home_market": p_home,
            "p_draw_market": p_draw,
            "p_away_market": p_away,
            "expected_home_goals": 1.4,
            "expected_away_goals": 1.1,
            "p_home_dc": p_home - 0.02,
            "p_draw_dc": p_draw + 0.01,
            "p_away_dc": p_away + 0.01,
            "market_vs_dc_home": 0.02,
            "market_vs_dc_draw": -0.01,
            "market_vs_dc_away": -0.01,
            "attack_strength_difference": 0.1,
            "expected_goal_difference": 0.3,
            "expected_goal_total": 2.5,
            "market_balance": 0.05,
            "defence_strength_difference": 0.04,
            "favourite_strength": 0.05,
            "combined_draw_rate": 0.24,
            "league_draw_rate": 0.25,
            "league_home_win_rate": 0.44,
            "home_rest_days": 5,
            "away_rest_days": 4,
            "rest_day_difference": 1,
            "home_advantage_log": 0.1,
            "home_advantage_coefficient": 1.10,
            "label": label,
        }
        market = {"1": p_home, "X": p_draw, "2": p_away}
        engine = {
            "1": row["p_home_dc"],
            "X": row["p_draw_dc"],
            "2": row["p_away_dc"],
        }
        blend = blend_baselines(market, engine)
        assert blend is not None
        row["p_home_blend"] = blend["1"]
        row["p_draw_blend"] = blend["X"]
        row["p_away_blend"] = blend["2"]
        row["p_home_market_norm"] = p_home
        row["p_draw_market_norm"] = p_draw
        row["p_away_market_norm"] = p_away
        row["p_home_dc_norm"] = row["p_home_dc"]
        row["p_draw_dc_norm"] = row["p_draw_dc"]
        row["p_away_dc_norm"] = row["p_away_dc"]
        rows.append(row)
    return rows


def test_trainer_exclude_injury_features_omits_columns():
    rows = _synthetic_rows(10)
    for row in rows:
        row["has_availability"] = 1
        row["home_unavailable_count"] = 2
        row["missing_value_difference"] = 0.01
    all_names = ResidualMLTrainer.feature_names_from_rows(rows)
    without_injury = ResidualMLTrainer.feature_names_from_rows(
        rows,
        exclude_injury_features=True,
    )
    assert "has_availability" in all_names
    assert "has_availability" not in without_injury
    assert "home_unavailable_count" not in without_injury
    assert "attack_strength_difference" in without_injury


def test_trainer_fit_and_save_round_trip(tmp_path: Path):
    rows = _synthetic_rows()
    train_rows = rows[:100]
    valid_rows = rows[100:]
    trainer = ResidualMLTrainer()
    result = trainer.fit(train_rows, valid_rows)
    assert result.train_rows == 100
    assert result.validation_rows == 20
    assert result.validation_log_loss >= 0.0

    model_dir = tmp_path / "model"
    trainer.save(model_dir, version="test")
    loaded = ResidualMLTrainer.load(model_dir / "model.pkl")
    assert loaded.feature_names
    assert loaded.version == "test"


def test_residual_round_trip():
    baseline = {"1": 0.45, "X": 0.28, "2": 0.27}
    deltas = {"1": 0.15, "X": -0.05, "2": -0.10}
    adjusted = apply_residual_deltas(baseline, deltas)
    assert sum(adjusted.values()) == pytest.approx(1.0)
    assert adjusted["1"] > baseline["1"]
    assert adjusted["X"] < baseline["X"]


def test_model_predict_proba(tmp_path: Path):
    rows = _synthetic_rows(90)
    trainer = ResidualMLTrainer()
    trainer.fit(rows[:75], rows[75:])
    trainer.save(tmp_path, version="test")
    model = ResidualMLModel.load(tmp_path / "model.pkl")
    assert model is not None

    features = ResidualMLFeatures(
        match_id=1,
        draw_number=4750,
        feature_cutoff_date=__import__("datetime").date(2024, 6, 1),
        league_external_id=39,
        p_home_market=0.48,
        p_draw_market=0.27,
        p_away_market=0.25,
        home_npxg_for=1.4,
        home_npxg_against=1.0,
        away_npxg_for=1.2,
        away_npxg_against=1.1,
        home_opponent_adjusted_attack=1.1,
        home_opponent_adjusted_defence=0.9,
        away_opponent_adjusted_attack=1.0,
        away_opponent_adjusted_defence=1.0,
        home_shot_quality_for=0.12,
        home_shot_quality_conceded=0.10,
        away_shot_quality_for=0.11,
        away_shot_quality_conceded=0.09,
        home_set_piece_attack=1.0,
        home_set_piece_defence=1.0,
        away_set_piece_attack=1.0,
        away_set_piece_defence=1.0,
        home_goalkeeper_prevention=0.05,
        away_goalkeeper_prevention=0.02,
        expected_home_goals=1.5,
        expected_away_goals=1.1,
        p_home_dc=0.46,
        p_draw_dc=0.28,
        p_away_dc=0.26,
        market_vs_dc_home=0.02,
        market_vs_dc_draw=-0.01,
        market_vs_dc_away=-0.01,
        attack_strength_difference=0.1,
        expected_goal_difference=0.4,
        expected_goal_total=2.6,
        market_balance=0.08,
        defence_strength_difference=0.05,
        favourite_strength=0.08,
        combined_draw_rate=0.24,
        combined_one_goal_match_rate=0.3,
        combined_close_match_rate=0.35,
        combined_low_scoring_rate=0.2,
        league_draw_rate=0.25,
        league_home_win_rate=0.44,
        league_away_win_rate=0.31,
        league_avg_goals=2.7,
        league_goal_std=1.4,
        league_favourite_win_rate=0.58,
        league_competitive_balance=0.5,
        league_avg_npxg=1.35,
        league_upset_rate=0.42,
        league_prior_weight=0.8,
        home_rest_days=6,
        away_rest_days=4,
        rest_day_difference=2,
        home_matches_last_14_days=2,
        away_matches_last_14_days=3,
        home_short_rest=0,
        away_short_rest=1,
        home_congestion=0,
        away_congestion=1,
        congestion_difference=-1,
        home_extra_time_in_previous_match=0,
        away_extra_time_in_previous_match=1,
        extra_time_x_short_rest=1,
        fatigue_difference=2,
        home_advantage_log=0.12,
        home_advantage_coefficient=1.127,
    )
    probs = model.predict_proba(features)
    assert sum(probs.values()) == pytest.approx(1.0)
