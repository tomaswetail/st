"""Smoke tests for probability manager import paths."""

from __future__ import annotations


def test_probability_manager_import():
    from src.calc.probability_manager import ProbabilityManager

    assert ProbabilityManager is not None
