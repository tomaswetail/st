"""Svenska Spel API client configuration schema."""

import os
from dataclasses import dataclass
from pathlib import Path

from objects.schema.data_classes.data_sources import DISK_CACHE_TTL_ONE_YEAR
from utils.repo_paths import repo_root


@dataclass
class SvenskaSpelConfig:
    svenskaspel_base_url: str = "https://api.spela.svenskaspel.se"
    svenskaspel_access_key: str = ""
    enable_cache: bool = True
    cache_dir: Path = repo_root() / "data" / "cache" / "svenskaspel"
    cache_ttl_open_seconds: int = 900
    cache_ttl_finalized_seconds: int = DISK_CACHE_TTL_ONE_YEAR
    cache_ttl_not_found_seconds: int = 3600

    @classmethod
    def from_env(cls) -> "SvenskaSpelConfig":
        cache_dir = os.environ.get("SVENSKASPEL_CACHE_DIR")
        return cls(
            svenskaspel_base_url=os.environ.get(
                "SVENSKASPEL_BASE_URL",
                "https://api.spela.svenskaspel.se",
            ),
            svenskaspel_access_key=os.environ.get("SVENSKASPEL_ACCESS_KEY", ""),
            enable_cache=os.environ.get("SVENSKASPEL_CACHE_ENABLED", "true").lower()
            in {"1", "true", "yes"},
            cache_dir=Path(cache_dir) if cache_dir else repo_root() / "data" / "cache" / "svenskaspel",
            cache_ttl_open_seconds=int(
                os.environ.get("SVENSKASPEL_CACHE_TTL_OPEN_SECONDS", "900")
            ),
            cache_ttl_finalized_seconds=int(
                os.environ.get(
                    "SVENSKASPEL_CACHE_TTL_FINALIZED_SECONDS",
                    str(DISK_CACHE_TTL_ONE_YEAR),
                )
            ),
            cache_ttl_not_found_seconds=int(
                os.environ.get("SVENSKASPEL_CACHE_TTL_NOT_FOUND_SECONDS", "3600")
            ),
        )
