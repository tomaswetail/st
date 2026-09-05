"""Tests for hyperparameter sweep behavior."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.calc.residual_ml.sweep import run_hyperparameter_sweep
from src.calc.residual_ml.trainer import ResidualMLTrainer, ResidualMLTrainingResult


def _sample_rows() -> list[dict]:
    return [
        {
            "label": "1",
            "match_date": f"2024-01-{index:02d}",
            "match_id": index,
            "draw_number": 4800 + index,
            "p_home_blend": 0.45,
            "p_draw_blend": 0.28,
            "p_away_blend": 0.27,
            "p_home_market_norm": 0.50,
            "p_draw_market_norm": 0.28,
            "p_away_market_norm": 0.22,
            "feature_a": 0.1 * index,
        }
        for index in range(1, 21)
    ]


def _fake_training_result(train_rows: list[dict], validation_rows: list[dict]) -> ResidualMLTrainingResult:
    return ResidualMLTrainingResult(
        version="",
        train_rows=len(train_rows),
        validation_rows=len(validation_rows),
        train_log_loss=1.0,
        validation_log_loss=0.9,
        market_validation_log_loss=1.01,
        blend_validation_log_loss=1.02,
        model_path=Path(),
        feature_schema_path=Path(),
        baseline_weights_path=Path(),
    )


def test_sweep_saves_best_trainer_without_extra_fit(tmp_path: Path):
    fit_calls = 0

    def fake_fit(self, train_rows, validation_rows):
        nonlocal fit_calls
        fit_calls += 1
        return _fake_training_result(train_rows, validation_rows)

    def fake_save(self, directory: Path, *, version: str = "v1") -> ResidualMLTrainingResult:
        model_path = directory / "model.pkl"
        model_path.write_bytes(b"model")
        return ResidualMLTrainingResult(
            version=version,
            train_rows=0,
            validation_rows=0,
            train_log_loss=0.0,
            validation_log_loss=0.0,
            market_validation_log_loss=None,
            blend_validation_log_loss=None,
            model_path=model_path,
            feature_schema_path=directory / "feature_schema.json",
            baseline_weights_path=directory / "baseline_weights.json",
        )

    with patch.object(ResidualMLTrainer, "fit", fake_fit):
        with patch.object(ResidualMLTrainer, "save", fake_save):
            with patch("src.calc.residual_ml.sweep.MAX_DEPTHS", [3]):
                with patch("src.calc.residual_ml.sweep.LEARNING_RATES", [0.05]):
                    with patch("src.calc.residual_ml.sweep.MAX_ITERS", [100]):
                        with patch("src.calc.residual_ml.sweep.LABEL_SMOOTHINGS", [0.0]):
                            best = run_hyperparameter_sweep(
                                _sample_rows(),
                                market_weight=0.7,
                                dc_weight=0.3,
                                output_dir=tmp_path,
                                validation_fraction=0.2,
                            )

    assert fit_calls == 1
    assert "model_path" in best
    assert Path(best["model_path"]).exists()
