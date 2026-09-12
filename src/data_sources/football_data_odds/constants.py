"""Constants for football-data.co.uk historical 1X2 odds ingestion."""

from __future__ import annotations

from datetime import date

PROVIDER = "football-data.co.uk"

MAIN_BASE_PATH = "/mmz4281"
EXTRA_BASE_PATH = "/new"
ODDS_HTTP_BASE_URL = "https://www.football-data.co.uk"

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

DEFAULT_DATE_FROM = date(2022, 2, 18)
DEFAULT_DATE_TO = date(2026, 6, 20)
DEFAULT_SEASONS = ("2122", "2223", "2324", "2425", "2526")

MAIN_CODES = (
    "E0",
    "E1",
    "E2",
    "E3",
    "SC0",
    "SP1",
    "I1",
    "D1",
    "F1",
    "N1",
    "P1",
    "B1",
)
EXTRA_CODES = ("SWE", "NOR", "FIN", "DNK")
ENGLISH_TIER_CODES = ("E0", "E1", "E2", "E3")

# (bookmaker, home_col, draw_col, away_col)
OPENING_BOOK_COLUMNS: tuple[tuple[str, str, str, str], ...] = (
    ("B365", "B365H", "B365D", "B365A"),
    ("PS", "PSH", "PSD", "PSA"),
    ("Avg", "AvgH", "AvgD", "AvgA"),
    ("Max", "MaxH", "MaxD", "MaxA"),
    ("BW", "BWH", "BWD", "BWA"),
    ("IW", "IWH", "IWD", "IWA"),
    ("WH", "WHH", "WHD", "WHA"),
    ("VC", "VCH", "VCD", "VCA"),
)

CLOSING_BOOK_COLUMNS: tuple[tuple[str, str, str, str], ...] = (
    ("B365", "B365CH", "B365CD", "B365CA"),
    ("PS", "PSCH", "PSCD", "PSCA"),
    ("Avg", "AvgCH", "AvgCD", "AvgCA"),
    ("Max", "MaxCH", "MaxCD", "MaxCA"),
    ("BFE", "BFECH", "BFECD", "BFECA"),
)

PRICE_OPENING = "opening"
PRICE_CLOSING = "closing"
