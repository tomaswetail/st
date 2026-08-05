"""Shared helpers for football_data tests."""

from __future__ import annotations

import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(*parts: str) -> dict | list:
    path = FIXTURES.joinpath(*parts)
    return json.loads(path.read_text(encoding="utf-8"))
