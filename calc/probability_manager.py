from __future__ import annotations

from sqlalchemy.orm import Session

from calc.market_probabilities import MarketProbabilities
from calc.residual_ml import ResidualMLFeatureAssembler, ResidualMLModel
from calc.residual_ml.baseline import (
    blend_baselines,
    engine_baseline,
    market_baseline,
)
from objects.repositories.st_match_repository import STMatchRepository
from objects.repositories.st_round_repository import STRoundRepository
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.db.st_match_probability import STMatchProbabilityResult


class ProbabilityManager:
    """Compute market, engine, ML, and final 1X2 probabilities for a coupon."""

    def __init__(
        self,
        session: Session,
        config: DataSourceConfig | None = None,
    ) -> None:
        self.session = session
        self.config = config or DataSourceConfig()
        self.rounds_repo = STRoundRepository(session)
        self.matches_repo = STMatchRepository(session)
        self.assembler = ResidualMLFeatureAssembler(session, config=self.config)
        self.ml_model = (
            ResidualMLModel.load(self.config.residual_ml_model_path, config=self.config)
            if self.config.residual_ml_enabled
            else None
        )

    def process(self, draw_number: int) -> list[STMatchProbabilityResult]:
        round_model = self.rounds_repo.get_by_draw_number(draw_number)
        if round_model is None:
            raise ValueError(f"No Stryktipset round found for draw_number={draw_number}")

        matches = self.matches_repo.get_by_stryktipset_round_id(draw_number)
        if not matches:
            raise ValueError(f"No matches found for draw_number={draw_number}")

        results: list[STMatchProbabilityResult] = []
        for event_number, match in enumerate(matches, start=1):
            results.append(self.process_match(match, event_number=event_number))
        return results

    def process_match(
        self,
        match,
        *,
        event_number: int | None = None,
    ) -> STMatchProbabilityResult:
        if match.home_team is None or match.away_team is None:
            raise ValueError(f"Missing team on match id={match.id}")
        if match.start_time is None:
            raise ValueError(f"Missing start_time on match id={match.id}")

        features = self.assembler.assemble(
            match,
            draw_number=match.stryktipset_round_id
        )
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
            market_weight=self.config.residual_ml_market_weight,
            dc_weight=self.config.residual_ml_dc_weight,
        )

        ml_enabled = self.ml_model is not None
        ml_probs = None
        final_probs = blend or engine or market
        ml_notes: list[str] = []
        draw_boost_score = 0.0
        draw_value_gap = None

        if ml_enabled and self.ml_model is not None and blend is not None:
            try:
                ml_probs = self.ml_model.predict_proba(features, baseline=blend)
                final_probs = ml_probs
                if market is not None and ml_probs.get("X") is not None:
                    draw_boost_score = ml_probs["X"] - market.get("X", 0.0)
                    draw_value_gap = ml_probs["X"] - market["X"]
            except ValueError as exc:
                ml_notes.append(str(exc))
                final_probs = blend
        elif engine is not None:
            final_probs = engine
        elif market is not None:
            final_probs = market

        if final_probs is None:
            raise ValueError(f"Could not compute probabilities for match id={match.id}")

        return STMatchProbabilityResult(
            draw_number=match.stryktipset_round_id,
            event_number=event_number or 0,
            match_id=match.id,
            home_team=match.home_team.name,
            away_team=match.away_team.name,
            match_date=features.feature_cutoff_date,
            probabilities=dict(final_probs),
            engine_probabilities=engine,
            market_probabilities=market,
            ml_probabilities=ml_probs,
            final_probabilities=dict(final_probs),
            ml_enabled=ml_enabled and ml_probs is not None,
            ml_model_version=self.ml_model.version if self.ml_model else None,
            draw_boost_score=draw_boost_score,
            draw_value_gap=draw_value_gap,
            ml_notes=ml_notes,
        )
