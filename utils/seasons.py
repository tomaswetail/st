from __future__ import annotations

from datetime import date


def season_code_to_start_year(season: str) -> int:
    """Convert football season code (e.g. 2324) to start calendar year (2023)."""
    return 2000 + int(season[:2])


def start_year_to_season_code(start_year: int) -> str:
    """Convert start calendar year (2023) to football season code (2324)."""
    end = start_year + 1
    return f"{start_year % 100:02d}{end % 100:02d}"


def current_season_start_year(*, today: date | None = None) -> int:
    """Return the football season start year (Aug boundary)."""
    ref = today or date.today()
    return ref.year if ref.month >= 8 else ref.year - 1


def last_n_season_codes(n: int = 5, *, today: date | None = None) -> list[str]:
    """Return football season codes like '2425' for the last n seasons."""
    if n < 1:
        raise ValueError("n must be at least 1")

    start_year = current_season_start_year(today=today)
    return [
        start_year_to_season_code(start_year - offset)
        for offset in range(n - 1, -1, -1)
    ]
