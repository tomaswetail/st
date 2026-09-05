"""Tests for residual ML validation fraction resolution."""

from __future__ import annotations

import json
from pathlib import Path

from src.calc.residual_ml.validation import resolve_validation_fraction


def test_resolve_validation_fraction_prefers_cli_value(tmp_path: Path):
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"")
    (tmp_path / "sweep_results.json").write_text(
        json.dumps({"best": {"validation_fraction": 0.15}}),
        encoding="utf-8",
    )
    assert resolve_validation_fraction(cli_fraction=0.2, model_path=model_path) == 0.2


def test_resolve_validation_fraction_reads_sweep_results(tmp_path: Path):
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"")
    (tmp_path / "sweep_results.json").write_text(
        json.dumps({"best": {"validation_fraction": 0.2}}),
        encoding="utf-8",
    )
    assert resolve_validation_fraction(cli_fraction=None, model_path=model_path) == 0.2


def test_resolve_validation_fraction_falls_back_to_default(tmp_path: Path):
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"")
    assert resolve_validation_fraction(cli_fraction=None, model_path=model_path) == 0.2
