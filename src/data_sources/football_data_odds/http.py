"""HTTP helper for football-data.co.uk CSV downloads."""

from __future__ import annotations

from pathlib import Path

from src.data_sources.football_data.http_client import ThrottledHttpClient
from src.data_sources.football_data_odds.constants import (
    BROWSER_USER_AGENT,
    ODDS_HTTP_BASE_URL,
)
from src.objects.schema.data_classes.data_sources import DataSourceConfig


def make_odds_http_client(
    config: DataSourceConfig | None = None,
    *,
    cache_dir: Path | None = None,
    enable_cache: bool | None = None,
) -> ThrottledHttpClient:
    """Throttled CSV client: browser User-Agent and redirect following."""
    resolved = config or DataSourceConfig()
    directory = cache_dir
    if directory is None:
        directory = resolved.football_data_cache_dir / "football_data_odds"
    cache_enabled = (
        enable_cache
        if enable_cache is not None
        else True
    )
    return ThrottledHttpClient(
        base_url=ODDS_HTTP_BASE_URL,
        timeout_sec=resolved.football_data_http_timeout_sec,
        max_retries=resolved.football_data_max_retries,
        request_delay_ms=resolved.football_data_request_delay_ms,
        cache_ttl_seconds=resolved.football_data_cache_ttl_seconds,
        cache_dir=directory,
        user_agent=BROWSER_USER_AGENT,
        enable_cache=cache_enabled,
    )
