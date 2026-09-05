"""Tests for classic DC config JSON helpers."""

from __future__ import annotations

import json
from pathlib import Path

from src.data_sources.classic_dc_config import (
    ClassicDcLeagueParams,
    load_league_params,
    load_optimization_grid,
    write_league_params,
)


def test_load_optimization_grid_smoke_file():
    grid = load_optimization_grid(
        Path("config/classic_dc_optimization_grid_smoke.json")
    )
    assert grid.xi_values == [0.0018]
    assert grid.lookback_values == [730]
    assert len(grid.rho_values) == 2
    # Absent from the JSON: rho stays grid-searched unless explicitly enabled.
    assert grid.fit_rho is False


def test_load_production_optimization_grid_fit_rho_true():
    grid = load_optimization_grid(
        Path("config/classic_dc_optimization_grid.json")
    )
    assert grid.fit_rho is True


def test_load_optimization_grid_reads_fit_rho(tmp_path: Path):
    grid_path = tmp_path / "grid.json"
    grid_path.write_text(
        json.dumps(
            {
                "xi_values": [0.001],
                "lookback_values": [730],
                "rho_values": [-0.13],
                "fit_rho": True,
            }
        ),
        encoding="utf-8",
    )
    assert load_optimization_grid(grid_path).fit_rho is True


def test_load_optimization_grid_fast_file():
    grid = load_optimization_grid(
        Path("config/classic_dc_optimization_grid_fast.json")
    )
    assert grid.xi_values == [0.00025, 0.0005, 0.001, 0.002, 0.003]
    assert grid.lookback_values == [365, 730, 1095]
    assert len(grid.rho_values) == 6
    assert len(grid.xi_values) * len(grid.lookback_values) * len(grid.rho_values) == 90


def test_write_and_load_league_params(tmp_path: Path):
    output = tmp_path / "league_params.json"
    write_league_params(
        {
            39: ClassicDcLeagueParams(
                xi=0.002,
                lookback=365,
                rho=-0.1,
                log_loss=1.01,
            )
        },
        output,
    )
    loaded = load_league_params(output)
    assert loaded[39].xi == 0.002
    assert loaded[39].lookback == 365
    assert loaded[39].rho == -0.1
    raw = json.loads(output.read_text(encoding="utf-8"))
    assert raw["39"]["log_loss"] == 1.01
