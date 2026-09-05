"""Load historical Stryktipset rows into a residual ML training dataset."""

from __future__ import annotations

import csv
import json
import time
from datetime import date
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from src.calc.draw_adjustment import DrawAdjustmentConfig, load_draw_adjustment_config
from src.calc.residual_ml.baseline import (
    apply_draw_adjustment,
    blend_baselines,
    engine_baseline,
    market_baseline,
)
from src.calc.residual_ml.blend_weights import (
    BlendWeightsConfig,
    load_blend_weights_config,
    load_dc_league_quality,
    select_blend_weights,
)
from src.calc.residual_ml.feature_assembler import ResidualMLFeatureAssembler
from src.objects.models.st_match import STMatchModel
from src.objects.models.st_round import STRoundModel
from src.objects.schema.data_classes.data_sources import DataSourceConfig
from src.objects.schema.data_classes.residual_ml_features import ResidualMLFeatures


class ResidualMLDatasetBuilder:
    """Build labeled feature rows from finished Stryktipset matches."""

    CACHE_FLUSH_EVERY = 500

    def __init__(
        self,
        session: Session,
        config: DataSourceConfig | None = None,
        *,
        draw_adjustment_config: DrawAdjustmentConfig | None = None,
        blend_weights_config: BlendWeightsConfig | None = None,
    ) -> None:
        self.session = session
        self.config = config or DataSourceConfig()
        self.assembler = ResidualMLFeatureAssembler(session, config=self.config)
        self.draw_adjustment_config = (
            draw_adjustment_config
            if draw_adjustment_config is not None
            else load_draw_adjustment_config()
        )
        self.blend_weights_config = (
            blend_weights_config
            if blend_weights_config is not None
            else load_blend_weights_config(self.config.residual_ml_blend_weights_path)
        )
        self._dc_league_log_loss = load_dc_league_quality()


    def iter_rows(
        self,
        *,
        min_draw_number: int | None = None,
        max_draw_number: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        matches = self._finished_matches(
            min_draw_number=min_draw_number,
            max_draw_number=max_draw_number,
        )
        print(
            f"Building residual ML dataset from {len(matches)} finished matches "
            f"(draws {min_draw_number}–{max_draw_number})",
            flush=True,
        )
        if self.config.residual_ml_home_advantage_mode == "fast":
            print(
                "Building with fast home advantage (league + competition only)",
                flush=True,
            )
        print(
            f"DC engine: {self.config.residual_ml_dc_engine}",
            flush=True,
        )
        emitted = 0
        batch_started = time.perf_counter()
        for index, match in enumerate(matches, start=1):
            if index % self.CACHE_FLUSH_EVERY == 0:
                self.assembler.clear_caches()
                self.session.expire_all()

            draw_number = self._draw_number(match)
            label = match.stryktipset_result
            if label not in {"1", "X", "2"}:
                continue
            if match.match_odds is None:
                print(
                    f"Skipping match_id={match.id} draw={draw_number} (no odds)",
                    flush=True,
                )
                continue
            if index == 1 or index % self.CACHE_FLUSH_EVERY == 0:
                home_name = match.home_team.name if match.home_team else "?"
                away_name = match.away_team.name if match.away_team else "?"
                batch_seconds = time.perf_counter() - batch_started
                print(
                    f"[{index}/{len(matches)}] Assembling draw={draw_number} "
                    f"match_id={match.id} {home_name} vs {away_name} "
                    f"[{batch_seconds:.1f}s]",
                    flush=True,
                )
                batch_started = time.perf_counter()
            try:
                features = self.assembler.assemble(
                    match,
                    draw_number=draw_number,
                    event_number=index,
                )
            except (ValueError, TypeError) as exc:
                print(
                    f"Skipping match_id={match.id} draw={draw_number} ({exc})",
                    flush=True,
                )
                continue
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
            # Pipeline: conditional blend → draw adjust → (HGB later).
            market_weight, dc_weight = select_blend_weights(
                features,
                self.blend_weights_config,
                league_external_id=features.league_external_id,
                dc_league_log_loss=self._dc_league_log_loss,
                fallback_market_weight=self.config.residual_ml_market_weight,
                fallback_dc_weight=self.config.residual_ml_dc_weight,
            )
            blend = blend_baselines(
                market,
                engine,
                market_weight=market_weight,
                dc_weight=dc_weight,
            )
            if blend is None:
                print(
                    f"Skipping match_id={match.id} draw={draw_number} "
                    "(no blend baseline)",
                    flush=True,
                )
                continue
            row = self._features_to_row(features)
            row["label"] = label
            row["match_date"] = features.feature_cutoff_date.isoformat()
            row["blend_market_weight"] = market_weight
            row["blend_dc_weight"] = dc_weight
            row["p_home_blend_pre_draw"] = blend["1"]
            row["p_draw_blend_pre_draw"] = blend["X"]
            row["p_away_blend_pre_draw"] = blend["2"]
            adjusted = apply_draw_adjustment(
                blend,
                features,
                self.draw_adjustment_config,
            )
            assert adjusted is not None
            row["p_home_blend"] = adjusted["1"]
            row["p_draw_blend"] = adjusted["X"]
            row["p_away_blend"] = adjusted["2"]
            if market is not None:
                row["p_home_market_norm"] = market["1"]
                row["p_draw_market_norm"] = market["X"]
                row["p_away_market_norm"] = market["2"]
            if engine is not None:
                row["p_home_dc_norm"] = engine["1"]
                row["p_draw_dc_norm"] = engine["X"]
                row["p_away_dc_norm"] = engine["2"]
            emitted += 1
            yield row
        final_batch_seconds = time.perf_counter() - batch_started
        print(
            f"Dataset build finished: {emitted} rows emitted "
            f"[{final_batch_seconds:.1f}s since last progress]",
            flush=True,
        )

    def build(
        self,
        *,
        min_draw_number: int | None = None,
        max_draw_number: int | None = None,
    ) -> list[dict[str, Any]]:
        return list(
            self.iter_rows(
                min_draw_number=min_draw_number,
                max_draw_number=max_draw_number,
            )
        )

    def export_csv(self, path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def export_csv_from_iter(
        self,
        path: Path,
        rows: Iterator[dict[str, Any]],
    ) -> int:
        """Stream rows to CSV; return number of rows written."""
        path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer: csv.DictWriter | None = None
            for row in rows:
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                    writer.writeheader()
                writer.writerow(row)
                written += 1
        if written == 0:
            path.write_text("", encoding="utf-8")
        return written

    def export_json(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def _finished_matches(
        self,
        *,
        min_draw_number: int | None = None,
        max_draw_number: int | None = None,
    ) -> list[STMatchModel]:
        query = (
            select(STMatchModel)
            .join(
                STRoundModel,
                STMatchModel.stryktipset_round_id == STRoundModel.id,
            )
            .options(
                selectinload(STMatchModel.home_team),
                selectinload(STMatchModel.away_team),
                selectinload(STMatchModel.match_odds),
                selectinload(STMatchModel.event),
            )
            .where(STMatchModel.stryktipset_result.in_(("1", "X", "2")))
        )
        if min_draw_number is not None:
            query = query.where(STRoundModel.draw_number >= min_draw_number)
        if max_draw_number is not None:
            query = query.where(STRoundModel.draw_number <= max_draw_number)
        query = query.order_by(STRoundModel.draw_number, STMatchModel.id)
        return list(self.session.scalars(query).all())

    @staticmethod
    def _draw_number(match: STMatchModel) -> int | None:
        event = getattr(match, "event", None)
        if event is not None:
            return event.draw_number
        return None

    @staticmethod
    def _features_to_row(features: ResidualMLFeatures) -> dict[str, Any]:
        row = features.to_dict()
        for key, value in list(row.items()):
            if isinstance(value, date):
                row[key] = value.isoformat()
        return row
