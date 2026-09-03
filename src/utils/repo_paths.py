"""Repository root and path resolution helpers."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the git/repository root (parent of ``src/``).

    Walks upward from this file looking for markers that identify the true
    repo root (``config/`` + ``docs/``, and/or ``.git``), so nesting changes
    under ``src/`` do not silently break path resolution.
    """
    start = Path(__file__).resolve().parent
    for candidate in (start, *start.parents):
        has_config_and_docs = (candidate / "config").is_dir() and (candidate / "docs").is_dir()
        has_git = (candidate / ".git").exists()
        if has_config_and_docs or has_git:
            return candidate
    # Fallback: parent of ``src/`` when this file lives at ``src/utils/repo_paths.py``
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve ``path`` relative to repo root unless already absolute."""
    path = Path(path)
    if path.is_absolute():
        return path
    return repo_root() / path
