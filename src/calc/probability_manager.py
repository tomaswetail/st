"""Compute market + optional residual ML + optional shrink 1X2 probabilities."""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.calc.residual_ml import ResidualMLFeatureAssembler, ResidualMLModel
from src.calc.residual_ml.baseline import market_baseline, shrink_toward_market
from src.objects.repositories.st_match_repository import STMatchRepository
from src.objects.repositories.st_round_repository import STRoundRepository
from src.objects.schema.data_classes.data_sources import DataSourceConfig
from src.objects.schema.db.st_match_probability import STMatchProbabilityResult


class ProbabilityManager:
    """Compute market → optional ML residual → optional shrink 1X2 probabilities."""

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
        if self.ml_model is not None:
            # Apply shrink here so the live pipeline is explicit and single-source.
            self.ml_model.final_shrink_to_market = 0.0

    def process(self, draw_number: int) -> list[STMatchProbabilityResult]:
        round_model = self.rounds_repo.get_by_draw_number(draw_number)
        if round_model is None:
            raise ValueError(f"No Stryktipset round found for draw_number={draw_number}")

        matches = self.matches_repo.get_by_stryktipset_round_id(round_model.id)
        if not matches:
            raise ValueError(f"No matches found for draw_number={draw_number}")

        return [
            self.process_match(match, event_number=event_number)
            for event_number, match in enumerate(matches, start=1)
        ]

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
            draw_number=match.stryktipset_round_id,
        )
        market = market_baseline(
            {
                "1": features.p_home_market,
                "X": features.p_draw_market,
                "2": features.p_away_market,
            }
        )
        if market is None:
            raise ValueError(
                f"Missing market probabilities for match id={match.id}"
            )

        ml_probabilities: dict[str, float] | None = None
        ml_notes: list[str] = []
        final_probabilities: dict[str, float] = market

        if self.ml_model is not None:
            try:
                ml_probabilities = self.ml_model.predict_proba(
                    features, baseline=market
                )
                final_probabilities = ml_probabilities
                shrink_alpha = float(self.config.residual_ml_final_shrink_to_market)
                if shrink_alpha > 0:
                    final_probabilities = shrink_toward_market(
                        ml_probabilities, market, alpha=shrink_alpha
                    )
            except ValueError as exc:
                ml_notes.append(str(exc))
                final_probabilities = market

        return STMatchProbabilityResult(
            draw_number=match.stryktipset_round_id,
            event_number=event_number or 0,
            match_id=match.id,
            home_team=match.home_team.name,
            away_team=match.away_team.name,
            match_date=features.feature_cutoff_date,
            probabilities=dict(final_probabilities),
            market_probabilities=market,
            ml_probabilities=ml_probabilities,
            final_probabilities=dict(final_probabilities),
            ml_enabled=self.ml_model is not None and ml_probabilities is not None,
            ml_model_version=self.ml_model.version if self.ml_model else None,
            ml_notes=ml_notes,
        )
