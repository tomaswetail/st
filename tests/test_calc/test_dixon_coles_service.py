"""Tests for per-league params in DixonColesService."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from src.calc.dixon_coles.service import DixonColesService
from src.objects.schema.data_classes.data_sources import DataSourceConfig


def test_params_for_league_uses_json_override(tmp_path: Path):
    params_path = tmp_path / "classic_dc_league_params.json"
    params_path.write_text(
        json.dumps({"39": {"xi": 0.003, "lookback": 365, "rho": -0.05}}),
        encoding="utf-8",
    )
    config = DataSourceConfig(classic_dc_league_params_path=params_path)
    service = DixonColesService(MagicMock(), config=config)
    assert service.params_for_league(39) == (0.003, 365, -0.05)
    assert service.params_for_league(140) == (
        config.classic_dc_xi,
        config.classic_dc_lookback_days,
        config.dixon_coles_rho,
    )
