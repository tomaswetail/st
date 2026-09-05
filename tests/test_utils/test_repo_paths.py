"""Tests for repository path helpers."""

from __future__ import annotations

from pathlib import Path

from src.data_sources.classic_dc_config import (
    default_league_params_path,
    default_optimization_grid_path,
)
from src.utils.repo_paths import repo_root, resolve_repo_path


def test_repo_root_is_true_repository_root():
    root = repo_root()
    assert (root / "config").is_dir()
    assert (root / "docs").is_dir()
    assert (root / "src").is_dir()
    # Must not resolve to the application package parent alone
    utils_package_parent = Path(__file__).resolve().parents[2] / "src"
    assert root != utils_package_parent
    assert (root / "src" / "utils" / "repo_paths.py").is_file()


def test_resolve_repo_path_relative_is_under_repo_root():
    resolved = resolve_repo_path("config/foo.json")
    assert resolved.is_absolute()
    assert resolved == repo_root() / "config" / "foo.json"
    assert resolved.is_relative_to(repo_root())


def test_resolve_repo_path_absolute_unchanged(tmp_path):
    absolute = tmp_path / "dataset.csv"
    assert resolve_repo_path(absolute) == absolute


def test_dc_default_paths_resolve_under_repo_root():
    root = repo_root()
    assert default_league_params_path().is_relative_to(root)
    assert default_optimization_grid_path().is_relative_to(root)
