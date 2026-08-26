"""Load historical Stryktipset rows into a residual ML training dataset."""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from calc.residual_ml_baseline import (
    blend_baselines,
    engine_baseline,
    market_baseline,
)
from calc.residual_ml_feature_assembler import ResidualMLFeatureAssembler
from objects.models.st_match import STMatchModel
from objects.models.st_round import STRoundModel
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.data_classes.residual_ml_features import ResidualMLFeatures


class ResidualMLDatasetBuilder:
    """Build labeled feature rows from finished Stryktipset matches."""

    def __init__(
        self,
        session: Session,
        config: DataSourceConfig | None = None,
    ) -> None:
        self.session = session
        self.config = config or DataSourceConfig()
        self.assembler = ResidualMLFeatureAssembler(session, config=self.config)

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
        emitted = 0
        for index, match in enumerate(matches):
            label = match.stryktipset_result
            if label not in {"1", "X", "2"}:
                continue
            if match.match_odds is None:
                print(
                    f"Skipping match_id={match.id} draw={match.stryktipset_round_id} (no odds)",
                    flush=True,
                )
                continue
            home_name = match.home_team.name if match.home_team else "?"
            away_name = match.away_team.name if match.away_team else "?"
            print(
                f"[{index + 1}/{len(matches)}] Assembling draw={match.stryktipset_round_id} "
                f"match_id={match.id} {home_name} vs {away_name}",
                flush=True,
            )
            try:
                features = self.assembler.assemble(
                    match,
                    draw_number=match.stryktipset_round_id,
                    event_number=index + 1,
                )
            except (ValueError, TypeError) as exc:
                print(
                    f"Skipping match_id={match.id} draw={match.stryktipset_round_id} ({exc})",
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
            blend = blend_baselines(
                market,
                engine,
                market_weight=self.config.residual_ml_market_weight,
                dc_weight=self.config.residual_ml_dc_weight,
            )
            if blend is None:
                print(
                    f"Skipping match_id={match.id} draw={match.stryktipset_round_id} "
                    "(no blend baseline)",
                    flush=True,
                )
                continue
            row = self._features_to_row(features)
            row["label"] = label
            row["match_date"] = features.feature_cutoff_date.isoformat()
            row["p_home_blend"] = blend["1"]
            row["p_draw_blend"] = blend["X"]
            row["p_away_blend"] = blend["2"]
            if market is not None:
                row["p_home_market_norm"] = market["1"]
                row["p_draw_market_norm"] = market["X"]
                row["p_away_market_norm"] = market["2"]
            if engine is not None:
                row["p_home_dc_norm"] = engine["1"]
                row["p_draw_dc_norm"] = engine["X"]
                row["p_away_dc_norm"] = engine["2"]
            emitted += 1
            print(
                f"Assembled row {emitted} draw={match.stryktipset_round_id} "
                f"match_id={match.id} label={label}",
                flush=True,
            )
            yield row
        print(f"Dataset build finished: {emitted} rows emitted", flush=True)

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

    def export_json(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    @staticmethod
    def time_split(
        rows: list[dict[str, Any]],
        *,
        validation_fraction: float = 0.2,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        ordered = sorted(rows, key=lambda row: (row.get("match_date", ""), row.get("match_id", 0)))
        if not ordered:
            return [], []
        split_index = max(1, int(len(ordered) * (1.0 - validation_fraction)))
        if split_index >= len(ordered):
            split_index = len(ordered) - 1
        return ordered[:split_index], ordered[split_index:]

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
    def _features_to_row(features: ResidualMLFeatures) -> dict[str, Any]:
        row = features.to_dict()
        for key, value in list(row.items()):
            if isinstance(value, date):
                row[key] = value.isoformat()
        return row
