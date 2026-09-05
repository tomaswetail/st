"""Unit tests for FotMob league team parsing."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.data_sources.football_data.providers.fotmob import (
    FotMobProvider,
    parse_fotmob_league_teams,
)


def test_parse_fotmob_league_teams_from_table_all():
    payload = {
        "table": [
            {
                "data": {
                    "table": {
                        "all": [
                            {"id": 1, "name": "Alpha", "shortName": "Alp"},
                            {"id": 2, "name": "Beta", "shortName": "Bet"},
                            {"id": 1, "name": "Alpha FC", "shortName": "Alp"},
                        ]
                    }
                }
            }
        ]
    }
    teams = parse_fotmob_league_teams(payload)
    assert [(t.provider_team_id, t.name, t.short_name) for t in teams] == [
        ("1", "Alpha", "Alp"),
        ("2", "Beta", "Bet"),
    ]


def test_parse_fotmob_league_teams_from_nested_tables():
    payload = {
        "table": [
            {
                "data": {
                    "tables": [
                        {
                            "leagueName": "Group A",
                            "table": {
                                "all": [{"id": 10, "name": "Gamma"}],
                            },
                        },
                        {
                            "leagueName": "Group B",
                            "table": {
                                "all": [{"id": 11, "name": "Delta"}],
                            },
                        },
                    ]
                }
            }
        ]
    }
    teams = parse_fotmob_league_teams(payload)
    assert {t.provider_team_id for t in teams} == {"10", "11"}


def test_fetch_teams_for_leagues_dedupes_across_leagues():
    provider = FotMobProvider(client=MagicMock())
    provider.fetch_league_teams = MagicMock(
        side_effect=[
            [
                MagicMock(provider_team_id="1", name="Alpha"),
                MagicMock(provider_team_id="2", name="Beta"),
            ],
            [
                MagicMock(provider_team_id="2", name="Beta"),
                MagicMock(provider_team_id="3", name="Gamma"),
            ],
        ]
    )

    teams = provider.fetch_teams_for_leagues([47, 48], country_codes={47: "ENG"})

    assert [t.provider_team_id for t in teams] == ["1", "2", "3"]
    assert provider.fetch_league_teams.call_args_list[0].kwargs["country_code"] == "ENG"
    assert provider.fetch_league_teams.call_args_list[1].kwargs["country_code"] is None
