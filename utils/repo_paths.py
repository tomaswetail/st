"""Repository root and path resolution helpers."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return repository root (parent of ``utils/``)."""
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve ``path`` relative to repo root unless already absolute."""
    path = Path(path)
    if path.is_absolute():
        return path
    return repo_root() / path
