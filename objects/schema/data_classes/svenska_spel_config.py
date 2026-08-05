"""Svenska Spel API client configuration schema."""

import os
from dataclasses import dataclass


@dataclass
class SvenskaSpelConfig:
    svenskaspel_base_url: str = "https://api.spela.svenskaspel.se"
    svenskaspel_access_key: str = ""

    @classmethod
    def from_env(cls) -> "SvenskaSpelConfig":
        return cls(
            svenskaspel_base_url=os.environ.get(
                "SVENSKASPEL_BASE_URL",
                "https://api.spela.svenskaspel.se",
            ),
            svenskaspel_access_key=os.environ.get("SVENSKASPEL_ACCESS_KEY", ""),
        )
