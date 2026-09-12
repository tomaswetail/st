"""Load historical Stryktipset rows into a residual ML training dataset."""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from src.calc.market_probabilities import load_fixture_market_probabilities
from src.calc.residual_ml.baseline import market_baseline
from src.calc.residual_ml.dual_price import (
    empty_dual_price_columns,
    price_block_from_breakdown,
    price_block_is_usable,
)
from src.calc.residual_ml.feature_assembler import ResidualMLFeatureAssembler
from src.objects.models.st_match import STMatchModel
from src.objects.models.st_round import STRoundModel
from src.objects.schema.data_classes.data_sources import DataSourceConfig
from src.objects.schema.data_classes.market_probability import MarketProbabilityBreakdown
from src.objects.schema.data_classes.residual_ml_features import ResidualMLFeatures
from src.utils.fixture_fields import fixture_match_date, fixture_outcome

SKIP_REASONS = (
    "timezone",
    "no_odds",
    "unresolved_team",
    "bad_label",
    "no_baseline",
    "missing_kickoff",
    "other",
)


def classify_skip(exc: BaseException) -> str:
    """Map an assemble/export exception to a skip-histogram bucket."""
    message = str(exc).lower()
    if isinstance(exc, TypeError) and (
        "can't compare" in message
        or "offset-naive" in message
        or "offset-aware" in message
        or ("naive" in message and "aware" in message)
    ):
        return "timezone"
    if "unresolved" in message and "team" in message:
        return "unresolved_team"
    if "no usable fixture odds" in message or "no odds" in message:
        return "no_odds"
    if "no market baseline" in message:
        return "no_baseline"
    if "kickoff" in message or "fixture_date" in message or "start_time" in message:
        return "missing_kickoff"
    return "other"


def format_skip_histogram(counts: Counter[str]) -> str:
    parts = [f"{reason}={counts.get(reason, 0)}" for reason in SKIP_REASONS]
    extras = [
        f"{reason}={count}"
        for reason, count in sorted(counts.items())
        if reason not in SKIP_REASONS
    ]
    return "Skip histogram: " + " ".join(parts + extras)


class ResidualMLDatasetBuilder:
    """Build labeled feature rows from finished Stryktipset matches."""

    CACHE_FLUSH_EVERY = 500

    def __init__(
        self,
        session: Session,
        config: DataSourceConfig | None = None,
    ) -> None:
        self.session = session
        self.config = config or DataSourceConfig()
        self.assembler = ResidualMLFeatureAssembler(session, config=self.config)
        self.skip_counts: Counter[str] = Counter()

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
        batch_started = time.perf_counter()
        for index, match in enumerate(matches, start=1):
            if index % self.CACHE_FLUSH_EVERY == 0:
                self.assembler.clear_caches()
                self.session.expire_all()

            draw_number = self._draw_number(match)
            label = match.stryktipset_result
            if label not in {"1", "X", "2"}:
                self.skip_counts["bad_label"] += 1
                continue
            if match.match_odds is None:
                self.skip_counts["no_odds"] += 1
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
                reason = classify_skip(exc)
                self.skip_counts[reason] += 1
                print(
                    f"Skipping match_id={match.id} draw={draw_number} "
                    f"[{reason}] ({exc})",
                    flush=True,
                )
                continue
            row = self._labeled_feature_row(features, label)
            if row is None:
                self.skip_counts["no_baseline"] += 1
                print(
                    f"Skipping match_id={match.id} draw={draw_number} "
                    "(no market baseline)",
                    flush=True,
                )
                continue
            row.update(empty_dual_price_columns())
            emitted += 1
            yield row
        final_batch_seconds = time.perf_counter() - batch_started
        print(
            f"Dataset build finished: {emitted} rows emitted "
            f"[{final_batch_seconds:.1f}s since last progress]",
            flush=True,
        )
        print(format_skip_histogram(self.skip_counts), flush=True)

    def iter_fixture_rows(
        self,
        *,
        league_external_id: int | None = None,
        league_season: int | None = None,
        dual_price: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """Yield labeled residual-ML rows from finished fixtures + fixture_odds.

        Default ``dual_price=True`` assembles features once and attaches
        opening and closing market columns side by side. ``dual_price=False``
        is the legacy single-triple export (skip if that price is missing).
        """
        if league_external_id is None and league_season is None:
            fixtures = self.assembler.fixture_repo.find_finished_with_odds(
                provider=self.config.fixture_odds_provider,
            )
            scope = "all leagues/seasons with odds"
        else:
            fixtures = self.assembler.fixture_repo.find_finished_for_league_season(
                league_id=league_external_id,
                league_season=league_season,
            )
            scope = f"league_id={league_external_id} season={league_season}"
        mode = "dual-price" if dual_price else "single-price"
        print(
            f"Building residual ML dataset from {len(fixtures)} finished fixtures "
            f"({scope}, {mode})",
            flush=True,
        )
        emitted = 0
        batch_started = time.perf_counter()
        for index, fixture in enumerate(fixtures, start=1):
            if index % self.CACHE_FLUSH_EVERY == 0:
                self.assembler.clear_caches()
                self.session.expire_all()

            label = fixture_outcome(fixture)
            if label not in {"1", "X", "2"}:
                self.skip_counts["bad_label"] += 1
                print(
                    f"Skipping fixture_id={fixture.id} (invalid label)",
                    flush=True,
                )
                continue
            cutoff = fixture_match_date(fixture)
            if cutoff is None:
                self.skip_counts["missing_kickoff"] += 1
                print(
                    f"Skipping fixture_id={fixture.id} (missing kickoff date)",
                    flush=True,
                )
                continue
            if index == 1 or index % self.CACHE_FLUSH_EVERY == 0:
                home_name = getattr(fixture, "home_team_name", None) or "?"
                away_name = getattr(fixture, "away_team_name", None) or "?"
                batch_seconds = time.perf_counter() - batch_started
                print(
                    f"[{index}/{len(fixtures)}] Assembling fixture_id={fixture.id} "
                    f"{home_name} vs {away_name} [{batch_seconds:.1f}s]",
                    flush=True,
                )
                batch_started = time.perf_counter()
            try:
                row = (
                    self._dual_price_fixture_row(fixture, label, index, cutoff)
                    if dual_price
                    else self._legacy_single_price_fixture_row(
                        fixture, label, index, cutoff
                    )
                )
            except (ValueError, TypeError) as exc:
                reason = classify_skip(exc)
                self.skip_counts[reason] += 1
                print(
                    f"Skipping fixture_id={fixture.id} [{reason}] ({exc})",
                    flush=True,
                )
                continue
            if row is None:
                continue
            emitted += 1
            yield row
        final_batch_seconds = time.perf_counter() - batch_started
        print(
            f"Dataset build finished: {emitted} rows emitted "
            f"[{final_batch_seconds:.1f}s since last progress]",
            flush=True,
        )
        print(format_skip_histogram(self.skip_counts), flush=True)

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

    def _load_fixture_price(
        self, fixture_id: int, price_type: str
    ) -> MarketProbabilityBreakdown | None:
        config = self.config.model_copy(update={"fixture_odds_price_type": price_type})
        return load_fixture_market_probabilities(
            self.session, fixture_id, config=config
        )

    def _legacy_single_price_fixture_row(
        self,
        fixture: Any,
        label: str,
        event_number: int,
        cutoff: date,
    ) -> dict[str, Any] | None:
        features = self.assembler.assemble(
            fixture,
            draw_number=None,
            event_number=event_number,
            before_date=cutoff,
            require_market=True,
        )
        row = self._labeled_feature_row(features, label)
        if row is None:
            self.skip_counts["no_baseline"] += 1
            print(
                f"Skipping fixture_id={fixture.id} (no market baseline)",
                flush=True,
            )
            return None
        return row

    def _dual_price_fixture_row(
        self,
        fixture: Any,
        label: str,
        event_number: int,
        cutoff: date,
    ) -> dict[str, Any] | None:
        features = self.assembler.assemble(
            fixture,
            draw_number=None,
            event_number=event_number,
            before_date=cutoff,
            require_market=False,
        )
        opening = self._load_fixture_price(int(fixture.id), "opening")
        closing = self._load_fixture_price(int(fixture.id), "closing")
        missing_diff = features.missing_value_difference
        opening_block = price_block_from_breakdown(
            opening, price_type="opening", missing_value_difference=missing_diff
        )
        closing_block = price_block_from_breakdown(
            closing, price_type="closing", missing_value_difference=missing_diff
        )
        if not price_block_is_usable(opening_block, "opening") and not (
            price_block_is_usable(closing_block, "closing")
        ):
            self.skip_counts["no_odds"] += 1
            print(
                f"Skipping fixture_id={fixture.id} (no usable opening or closing odds)",
                flush=True,
            )
            return None
        row = self._features_to_row(features)
        row["label"] = label
        row["match_date"] = features.feature_cutoff_date.isoformat()
        row["p_home_market_norm"] = None
        row["p_draw_market_norm"] = None
        row["p_away_market_norm"] = None
        row.update(opening_block)
        row.update(closing_block)
        return row

    @staticmethod
    def _labeled_feature_row(
        features: ResidualMLFeatures, label: str
    ) -> dict[str, Any] | None:
        market = market_baseline(
            {
                "1": features.p_home_market,
                "X": features.p_draw_market,
                "2": features.p_away_market,
            }
        )
        if market is None:
            return None
        row = ResidualMLDatasetBuilder._features_to_row(features)
        row["label"] = label
        row["match_date"] = features.feature_cutoff_date.isoformat()
        row["p_home_market_norm"] = market["1"]
        row["p_draw_market_norm"] = market["X"]
        row["p_away_market_norm"] = market["2"]
        return row

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
