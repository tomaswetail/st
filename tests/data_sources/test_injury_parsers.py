"""Tests for API-Football injury/lineup availability parsing."""

from __future__ import annotations

from datetime import datetime, timezone

from src.data_sources.injuries.parsers.api_football import parse_api_football_availability


def _lineups_payload() -> dict:
    return {
        "response": [
            {
                "team": {"id": 50, "name": "Home FC"},
                "formation": "4-3-3",
                "startXI": [
                    {"player": {"id": 1, "name": "GK", "number": 1, "pos": "G"}},
                    {"player": {"id": 2, "name": "DF", "number": 2, "pos": "D"}},
                ],
                "substitutes": [
                    {"player": {"id": 3, "name": "SUB", "number": 12, "pos": "M"}},
                ],
            },
            {
                "team": {"id": 42, "name": "Away FC"},
                "formation": "4-4-2",
                "startXI": [
                    {"player": {"id": 10, "name": "Away GK", "number": 1, "pos": "G"}},
                ],
                "substitutes": [],
            },
        ]
    }


def _injuries_payload() -> dict:
    return {
        "response": [
            {
                "player": {
                    "id": 99,
                    "name": "Injured Star",
                    "type": "Missing Fixture",
                    "reason": "Knee Injury",
                },
                "team": {"id": 50, "name": "Home FC"},
                "fixture": {
                    "id": 1000,
                    "date": "2025-08-15T18:00:00+00:00",
                    "timestamp": 1755280800,
                },
            },
            {
                "player": {
                    "id": 88,
                    "name": "Suspended",
                    "type": "Missing Fixture",
                    "reason": "Suspended",
                },
                "team": {"id": 42, "name": "Away FC"},
            },
        ]
    }


def test_parse_combines_lineups_and_injuries() -> None:
    kickoff = datetime(2025, 8, 15, 18, 0, tzinfo=timezone.utc)
    snapshot = parse_api_football_availability(
        fixture_id=7,
        home_team_external_id=50,
        away_team_external_id=42,
        kickoff_at=kickoff,
        lineups_payload=_lineups_payload(),
        injuries_payload=_injuries_payload(),
    )
    assert snapshot is not None
    assert snapshot.provider == "api-football"
    assert snapshot.source == "api_football_injuries_lineups"
    assert snapshot.home_starter_count == 2
    assert snapshot.away_starter_count == 1
    assert snapshot.home_unavailable_count == 1
    assert snapshot.away_unavailable_count == 1
    assert snapshot.home_missing_value == 1.0
    assert snapshot.away_missing_value == 1.0
    statuses = {player.external_id: player.status for player in snapshot.players}
    assert statuses["99"] == "injured"
    assert statuses["88"] == "suspended"
    assert statuses["1"] == "starter"


def test_parse_returns_none_without_data() -> None:
    assert (
        parse_api_football_availability(
            fixture_id=1,
            home_team_external_id=1,
            away_team_external_id=2,
            kickoff_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            lineups_payload={"response": []},
            injuries_payload={"response": []},
        )
        is None
    )


def test_injury_overrides_lineup_player() -> None:
    lineups = {
        "response": [
            {
                "team": {"id": 50, "name": "Home"},
                "startXI": [
                    {"player": {"id": 99, "name": "Maybe Out", "number": 9}},
                ],
                "substitutes": [],
            },
            {
                "team": {"id": 42, "name": "Away"},
                "startXI": [{"player": {"id": 10, "name": "OK", "number": 1}}],
                "substitutes": [],
            },
        ]
    }
    injuries = {
        "response": [
            {
                "player": {
                    "id": 99,
                    "name": "Maybe Out",
                    "type": "Missing Fixture",
                    "reason": "Injury",
                },
                "team": {"id": 50},
            }
        ]
    }
    snapshot = parse_api_football_availability(
        fixture_id=1,
        home_team_external_id=50,
        away_team_external_id=42,
        kickoff_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        lineups_payload=lineups,
        injuries_payload=injuries,
    )
    assert snapshot is not None
    home_statuses = [
        player.status
        for player in snapshot.players
        if player.external_id == "99"
    ]
    assert home_statuses == ["injured"]
    assert snapshot.home_starter_count == 0
