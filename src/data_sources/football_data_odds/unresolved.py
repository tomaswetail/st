"""Read and write unresolved football-data.co.uk odds CSV rows."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

UNRESOLVED_ODDS_CSV_FIELDS = (
    "home_team",
    "away_team",
    "date",
    "league_code",
    "season",
    "reason",
)


@dataclass(frozen=True)
class UnresolvedOddsRow:
    """One Phase 1 miss-log row (or a remaining retry row)."""

    home_team: str
    away_team: str
    date: str
    league_code: str
    season: str
    reason: str

    @property
    def match_key(self) -> tuple[str, str, str, str]:
        return (self.home_team, self.away_team, self.date, self.league_code)


def append_unresolved_odds_row(
    csv_path: Path,
    *,
    home_team: str,
    away_team: str,
    match_date: str,
    league_code: str,
    season: str,
    reason: str,
) -> None:
    """Append one unresolved match; create the file with a header if needed."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    row = {
        "home_team": home_team,
        "away_team": away_team,
        "date": match_date,
        "league_code": league_code,
        "season": season,
        "reason": reason,
    }
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=UNRESOLVED_ODDS_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_unresolved_odds_rows(csv_path: Path) -> list[UnresolvedOddsRow]:
    """Load miss-log rows. Raises FileNotFoundError if the file is missing."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Unresolved odds CSV not found: {csv_path}")
    rows: list[UnresolvedOddsRow] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            home_team = (raw.get("home_team") or "").strip()
            away_team = (raw.get("away_team") or "").strip()
            match_date = (raw.get("date") or "").strip()
            league_code = (raw.get("league_code") or "").strip()
            season = (raw.get("season") or "").strip()
            reason = (raw.get("reason") or "").strip()
            if not home_team or not away_team or not match_date or not league_code:
                continue
            rows.append(
                UnresolvedOddsRow(
                    home_team=home_team,
                    away_team=away_team,
                    date=match_date,
                    league_code=league_code,
                    season=season,
                    reason=reason,
                )
            )
    return rows


def write_unresolved_odds_rows(
    csv_path: Path, rows: list[UnresolvedOddsRow]
) -> None:
    """Overwrite ``csv_path`` with the given rows (header always written)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=UNRESOLVED_ODDS_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "date": row.date,
                    "league_code": row.league_code,
                    "season": row.season,
                    "reason": row.reason,
                }
            )
