"""Configuration for external data sources and storage."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


class DataSourceConfig(BaseModel):
    """Paths and parameters for data ingestion and features."""

    team_aliases_path: Path = Field(
        default_factory=lambda: _project_root() / "config" / "team_aliases.json"
    )
    football_data_base_url: str = Field(
        default="https://www.football-data.co.uk/mmz4281"
    )
    football_data_extra_base_url: str = Field(
        default="https://www.football-data.co.uk/new"
    )
    football_data_world_cup_xlsx_url: str = Field(
        default="https://www.football-data.co.uk/WorldCup2026.xlsx"
    )
    default_leagues: list[str] = Field(default_factory=lambda: ["E0", "E1", "SP1"])
    default_seasons: list[str] = Field(default_factory=lambda: ["2324", "2425"])
    elo_start: float = Field(default=1500.0)
    elo_k: float = Field(default=20.0)
    elo_home_advantage: float = Field(default=60.0)
    form_matches: int = Field(default=5, ge=1)
    fuzzy_match_threshold: int = Field(default=85, ge=0, le=100)
    current_coupon_path: Path = Field(
        default_factory=lambda: _project_root() / "data" / "current_coupon.json"
    )
    current_odds_path: Path = Field(
        default_factory=lambda: _project_root() / "data" / "current_odds.json"
    )
    xg_provider: Literal["understat"] = Field(default="understat")
    understat_leagues: list[str] = Field(
        default_factory=lambda: [
            x.strip()
            for x in os.environ.get(
                "UNDERSTAT_LEAGUES", "EPL,La_liga,Bundesliga,Serie_A,Ligue_1"
            ).split(",")
            if x.strip()
        ]
    )
    understat_request_delay_sec: float = Field(
        default_factory=lambda: float(os.environ.get("UNDERSTAT_REQUEST_DELAY_SEC", "1.0")),
        ge=0.0,
    )
    footystats_api_key: str | None = Field(
        default_factory=lambda: os.environ.get("FOOTYSTATS_API_KEY")
    )
    thestatsapi_key: str | None = Field(
        default_factory=lambda: os.environ.get("THESTATSAPI_KEY")
    )
    raw_football_data_dir: Path = Field(
        default_factory=lambda: _project_root() / "data" / "raw" / "football-data"
    )
    svenskaspel_base_url: str = Field(default="https://api.spela.svenskaspel.se")
    svenskaspel_access_key: str | None = Field(
        default_factory=lambda: os.environ.get("SVENSKASPEL_ACCESS_KEY")
    )
    svenskaspel_draw_seed: int = Field(default=4950, ge=1)
    coupon_source: Literal["manual", "svenskaspel"] = Field(default="manual")
    odds_provider: Literal["svenskaspel", "the-odds-api", "manual"] = Field(default="svenskaspel")
    odds_aggregation_method: str = Field(default="average_probability")

    # Historical FotMob / SofaScore ingestion
    football_data_provider: Literal["fotmob", "sofascore"] = Field(
        default_factory=lambda: (
            "sofascore"
            if os.environ.get("FOOTBALL_DATA_PROVIDER", "fotmob").lower()
            == "sofascore"
            else "fotmob"
        )
    )
    fotmob_base_url: str = Field(
        default_factory=lambda: os.environ.get(
            "FOTMOB_BASE_URL", "https://www.fotmob.com/api/data"
        )
    )
    sofascore_base_url: str = Field(
        default_factory=lambda: os.environ.get(
            "SOFASCORE_BASE_URL", "https://api.sofascore.com/api/v1"
        )
    )
    football_data_request_delay_ms: int = Field(
        default_factory=lambda: int(
            os.environ.get("FOOTBALL_DATA_REQUEST_DELAY_MS", "500")
        ),
        ge=0,
    )
    football_data_max_retries: int = Field(
        default_factory=lambda: int(os.environ.get("FOOTBALL_DATA_MAX_RETRIES", "3")),
        ge=0,
    )
    football_data_cache_ttl_seconds: int = Field(
        default_factory=lambda: int(
            os.environ.get("FOOTBALL_DATA_CACHE_TTL_SECONDS", "3600")
        ),
        ge=0,
    )
    football_data_http_timeout_sec: float = Field(default=20.0, ge=1.0)
    football_data_user_agent: str = Field(
        default="st-football-data/1.0 (+historical-ingestion)"
    )
    football_data_cache_dir: Path = Field(
        default_factory=lambda: _project_root() / "data" / "cache" / "football-data"
    )
    kickoff_match_tolerance_minutes: int = Field(default=24 * 60, ge=0)
    xg_aggregate_tolerance: float = Field(default=0.15, ge=0.0)
    create_missing_historical_matches: bool = Field(default=False)
    football_data_feature_shrinkage_prior_matches: int = Field(default=10, ge=0)
    football_data_opponent_adjustment: Literal["none", "simple"] = Field(default="none")
    team_strength_lookback_matches: int = Field(
        default_factory=lambda: int(
            os.environ.get("TEAM_STRENGTH_LOOKBACK_MATCHES", "20")
        ),
        ge=1,
    )
    team_strength_recency_decay: float = Field(
        default_factory=lambda: float(
            os.environ.get("TEAM_STRENGTH_RECENCY_DECAY", "0.90")
        ),
        gt=0.0,
        le=1.0,
    )
    team_strength_prior_matches: int = Field(
        default_factory=lambda: int(
            os.environ.get("TEAM_STRENGTH_PRIOR_MATCHES", "8")
        ),
        ge=0,
    )
    team_strength_min_venue_matches: int = Field(default=5, ge=0)
    goalkeeper_prior_shots: int = Field(
        default_factory=lambda: int(os.environ.get("GOALKEEPER_PRIOR_SHOTS", "100")),
        ge=0,
    )
    dixon_coles_max_goals: int = Field(
        default_factory=lambda: int(os.environ.get("DIXON_COLES_MAX_GOALS", "10")),
        ge=1,
    )
    dixon_coles_rho: float = Field(
        default_factory=lambda: float(os.environ.get("DIXON_COLES_RHO", "-0.13")),
    )
    home_advantage_shrinkage_matches: int = Field(
        default_factory=lambda: int(
            os.environ.get("HOME_ADVANTAGE_SHRINKAGE_MATCHES", "30")
        ),
        ge=0,
    )
    max_team_home_advantage: float = Field(
        default_factory=lambda: float(
            os.environ.get("MAX_TEAM_HOME_ADVANTAGE", "0.30")
        ),
        gt=0.0,
    )
    home_advantage_epsilon: float = Field(
        default_factory=lambda: float(
            os.environ.get("HOME_ADVANTAGE_EPSILON", "0.05")
        ),
        gt=0.0,
    )
    home_advantage_recency_decay_rate: float = Field(
        default_factory=lambda: float(
            os.environ.get("HOME_ADVANTAGE_RECENCY_DECAY_RATE", "0.01")
        ),
        ge=0.0,
    )
    league_home_advantage_shrinkage_matches: int = Field(
        default_factory=lambda: int(
            os.environ.get("LEAGUE_HOME_ADVANTAGE_SHRINKAGE_MATCHES", "30")
        ),
        ge=0,
    )
    league_home_advantage_global_prior: float = Field(
        default_factory=lambda: float(
            os.environ.get("LEAGUE_HOME_ADVANTAGE_GLOBAL_PRIOR", "0.20")
        ),
    )
