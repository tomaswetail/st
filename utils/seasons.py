from __future__ import annotations

from datetime import date

from data_sources.football_data_uk_xlsx_provider import start_year_to_season_code


def current_season_start_year(*, today: date | None = None) -> int:
    """Return the football season start year (Aug boundary)."""
    ref = today or date.today()
    return ref.year if ref.month >= 8 else ref.year - 1


def last_n_season_codes(n: int = 5, *, today: date | None = None) -> list[str]:
    """Return football-data season codes like '2425' for the last n seasons."""
    if n < 1:
        raise ValueError("n must be at least 1")

    start_year = current_season_start_year(today=today)
    return [
        start_year_to_season_code(start_year - offset)
        for offset in range(n - 1, -1, -1)
    ]
