"""Tests for team alias loading."""

from utils.team_name_matcher import _load_aliases


def test_load_aliases_includes_moved_and_duplicate_pairs():
    aliases = _load_aliases()
    assert aliases["Manchester City"] == "Man City"
    assert aliases["Luton Town"] == "Luton"
