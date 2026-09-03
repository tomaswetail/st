"""Load and run the residual 1X2 ML model."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from calc.residual_ml.baseline import (
    apply_residual_deltas,
    blend_baselines,
    engine_baseline,
    market_baseline,
    shrink_toward_market,
)
from calc.residual_ml.vectorize import vectorize_features
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.data_classes.residual_ml_features import ResidualMLFeatures

if TYPE_CHECKING:
    from calc.residual_ml.trainer import ResidualMLTrainer


class ResidualMLModel:
    """Predict 1X2 probabilities via residual correction on a blended baseline."""

    def __init__(
        self,
        trainer: ResidualMLTrainer,
        *,
        version: str = "v1",
        market_weight: float = 0.7,
        dc_weight: float = 0.3,
        final_shrink_to_market: float = 0.0,
    ) -> None:
        self.trainer = trainer
        self.version = version
        self.market_weight = market_weight
        self.dc_weight = dc_weight
        self.final_shrink_to_market = final_shrink_to_market

    @classmethod
    def load(
        cls,
        model_path: Path,
        config: DataSourceConfig | None = None,
    ) -> ResidualMLModel | None:
        if not model_path.exists():
            return None
        from calc.residual_ml.trainer import ResidualMLTrainer

        cfg = config or DataSourceConfig()
        trainer = ResidualMLTrainer.load(model_path)
        return cls(
            trainer,
            version=trainer.version,
            market_weight=trainer.market_weight,
            dc_weight=trainer.dc_weight,
            final_shrink_to_market=cfg.residual_ml_final_shrink_to_market,
        )

    def predict_proba(
        self,
        features: ResidualMLFeatures,
        *,
        baseline: dict[str, float] | None = None,
    ) -> dict[str, float]:
        market = market_baseline(
            {
                "1": features.p_home_market,
                "X": features.p_draw_market,
                "2": features.p_away_market,
            }
        )
        engine = engine_baseline(
            p_home_dc=features.p_home_dc,
            p_draw_dc=features.p_draw_dc,
            p_away_dc=features.p_away_dc,
        )
        blend = baseline or blend_baselines(
            market,
            engine,
            market_weight=self.market_weight,
            dc_weight=self.dc_weight,
        )
        if blend is None:
            raise ValueError("Cannot compute baseline probabilities for prediction")

        vector = vectorize_features(
            features,
            feature_names=self.trainer.feature_names,
            global_medians=self.trainer.global_medians,
        )
        deltas = self.trainer.predict_deltas(vector)
        ml_probs = apply_residual_deltas(blend, deltas)

        if market is not None and self.final_shrink_to_market > 0:
            return shrink_toward_market(
                ml_probs,
                market,
                alpha=self.final_shrink_to_market,
            )
        return ml_probs

    def predict_layers(
        self,
        features: ResidualMLFeatures,
    ) -> dict[str, dict[str, float] | None]:
        market = market_baseline(
            {
                "1": features.p_home_market,
                "X": features.p_draw_market,
                "2": features.p_away_market,
            }
        )
        engine = engine_baseline(
            p_home_dc=features.p_home_dc,
            p_draw_dc=features.p_draw_dc,
            p_away_dc=features.p_away_dc,
        )
        blend = blend_baselines(
            market,
            engine,
            market_weight=self.market_weight,
            dc_weight=self.dc_weight,
        )
        ml_probs = self.predict_proba(features, baseline=blend)
        return {
            "market_probabilities": market,
            "engine_probabilities": engine,
            "blend_probabilities": blend,
            "ml_probabilities": ml_probs,
        }
