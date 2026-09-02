"""Unit tests for TeamRepository.merge_duplicate."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from objects.repositories.team_repository import TeamRepository


def _repo_with_teams(
    *, keep_id: int = 17, remove_id: int = 42
) -> tuple[TeamRepository, MagicMock]:
    session = MagicMock()
    repo = TeamRepository(session)
    keep = SimpleNamespace(id=keep_id, name="Keep FC", external_id=1001)
    remove = SimpleNamespace(id=remove_id, name="Remove FC", external_id=1002)
    repo.get = MagicMock(
        side_effect=lambda team_id: {
            keep_id: keep,
            remove_id: remove,
        }.get(team_id)
    )
    repo.delete = MagicMock()
    return repo, session


def test_merge_duplicate_rejects_same_ids():
    repo, _session = _repo_with_teams()
    with pytest.raises(ValueError, match="must differ"):
        repo.merge_duplicate(keep_team_id=17, remove_team_id=17)


def test_merge_duplicate_rejects_missing_team():
    repo, _session = _repo_with_teams()
    repo.get = MagicMock(
        side_effect=lambda team_id: None
        if team_id == 99
        else SimpleNamespace(id=17, external_id=1)
    )
    with pytest.raises(ValueError, match="remove team"):
        repo.merge_duplicate(keep_team_id=17, remove_team_id=99)


def test_merge_duplicate_reassigns_and_removes_team():
    repo, session = _repo_with_teams(keep_id=17, remove_id=42)
    remove_team = repo.get(42)

    # fixtures home, fixtures away, st home, st away, shots, mappings
    execute_results = [
        SimpleNamespace(rowcount=3),  # fixtures home
        SimpleNamespace(rowcount=2),  # fixtures away
        SimpleNamespace(rowcount=1),  # st home
        SimpleNamespace(rowcount=0),  # st away
        SimpleNamespace(rowcount=4),  # shots
        SimpleNamespace(rowcount=2),  # mappings
    ]
    session.execute = MagicMock(side_effect=execute_results)

    counts = repo.merge_duplicate(keep_team_id=17, remove_team_id=42)

    assert counts == {
        "fixtures_updated": 5,
        "st_updated": 1,
        "shots_updated": 4,
        "mappings_deleted": 2,
        "removed_team_id": 42,
    }
    assert session.execute.call_count == 6
    repo.delete.assert_called_once_with(remove_team)
    session.commit.assert_called_once()
