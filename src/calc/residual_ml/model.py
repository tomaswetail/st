"""Load and run the residual 1X2 ML model against the market baseline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.calc.residual_ml.baseline import (
    apply_residual_deltas,
    market_baseline,
    shrink_toward_market,
)
from src.calc.residual_ml.vectorize import vectorize_features
from src.objects.schema.data_classes.data_sources import DataSourceConfig
from src.objects.schema.data_classes.residual_ml_features import ResidualMLFeatures

if TYPE_CHECKING:
    from src.calc.residual_ml.trainer import ResidualMLTrainer


class ResidualMLModel:
    """Predict 1X2 probabilities via residual correction on the market baseline."""

    def __init__(
        self,
        trainer: ResidualMLTrainer,
        *,
        version: str = "v1",
        final_shrink_to_market: float = 0.0,
    ) -> None:
        self.trainer = trainer
        self.version = version
        self.final_shrink_to_market = final_shrink_to_market

    @classmethod
    def load(
        cls,
        model_path: Path,
        config: DataSourceConfig | None = None,
    ) -> ResidualMLModel | None:
        if not model_path.exists():
            return None
        from src.calc.residual_ml.trainer import ResidualMLTrainer

        cfg = config or DataSourceConfig()
        trainer = ResidualMLTrainer.load(model_path)
        return cls(
            trainer,
            version=trainer.version,
            final_shrink_to_market=cfg.residual_ml_final_shrink_to_market,
        )

    def predict_proba(
        self,
        features: ResidualMLFeatures,
        *,
        baseline: dict[str, float] | None = None,
    ) -> dict[str, float]:
        market = baseline or market_baseline(
            {
                "1": features.p_home_market,
                "X": features.p_draw_market,
                "2": features.p_away_market,
            }
        )
        if market is None:
            raise ValueError("Cannot compute market baseline probabilities for prediction")

        vector = vectorize_features(
            features,
            feature_names=self.trainer.feature_names,
            global_medians=self.trainer.global_medians,
        )
        deltas = self.trainer.predict_deltas(vector)
        ml_probs = apply_residual_deltas(market, deltas)

        if self.final_shrink_to_market > 0:
            return shrink_toward_market(
                ml_probs,
                market,
                alpha=self.final_shrink_to_market,
            )
        return ml_probs
