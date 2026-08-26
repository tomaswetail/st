import csv
from pathlib import Path
from typing import Any

from data_sources.api_football_client import get_all_leagues, APIFootballClient, get_team_names_by_league
from data_sources.football_data.providers.fotmob import FotMobProvider
from data_sources.svenskaspel_api_client import SvenskaSpelClient
from database import init_db, SessionLocal
from objects.repositories.team_repository import TeamRepository
from utils.common import LEAGUES_EXTERNAL_IDS, API_FOOTBALL_TO_FOTMOB_LEAGUE_MAPPING, FOTMOBLEAGUE_EXTERNAL_ID_TO_CCODE

EXTRA = {
    8814: "BRA",
    132: "ENG",
    117: "ENG",
    8944: "ENG",
    8947: "ENG",
    10176: "ENG",
    9084: "ENG",
}

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def _participant_by_type(
    participants: list[dict[str, Any]], role: str
) -> dict[str, Any]:
    for participant in participants:
        if participant.get("type") == role:
            return participant
    raise ValueError(f"Missing participant with type={role!r}")

def get_fotmob_team_names(provider: FotMobProvider | None = None) -> list[str]:
    country_codes = {}

    for k, v in API_FOOTBALL_TO_FOTMOB_LEAGUE_MAPPING.items():
        if v:
            country_codes[v] = FOTMOBLEAGUE_EXTERNAL_ID_TO_CCODE[k]

    for k, v in EXTRA.items():
        country_codes[k] = v
    fotmob_team_ids = list(API_FOOTBALL_TO_FOTMOB_LEAGUE_MAPPING.values()) + list(
        EXTRA.keys()
    )
    fotmob = provider or FotMobProvider()
    teams = fotmob.fetch_teams_for_leagues(
        fotmob_team_ids,
        country_codes=country_codes,
    )
    _teams = [{'external_id': t.provider_team_id, 'name': t.name} for t in teams]

    write_csv('fotmob_teams', _teams)

def get_ss_teams():
    client = SvenskaSpelClient()
    _teams = []

    for draw_number in range(4760, 4959):
        payload = client.fetch_draw(draw_number)
        draw = payload["draw"]

        draw_num = draw["drawNumber"]

        for draw_event in draw.get("drawEvents") or []:
            match_data = draw_event.get("match") or {}
            participants = match_data.get("participants") or []
            home_participant = _participant_by_type(participants, "home")
            away_participant = _participant_by_type(participants, "away")
            _teams.append({'external_id': away_participant['id'], 'name': away_participant['name']})
            _teams.append({'external_id': home_participant['id'], 'name': home_participant['name']})

    write_csv('svenska_spel_teams', _unique_teams(_teams))

def get_api_football_teams():
    init_db()
    session = SessionLocal()
    teams = []
    client = APIFootballClient()
    for season in ["2223", "2324", "2425", "2526"]:
        for league_id in LEAGUES_EXTERNAL_IDS:
            _teams = get_team_names_by_league(client, "42", "2627")
            create_teams(session, _teams, is_national_team=False)
            teams += _teams
    teams += get_wc_teams()
    teams += get_ec_teams()

    #write_csv('api_football_teams', _unique_teams(teams))

def get_wc_teams():
    init_db()
    session = SessionLocal()
    teams = []
    client = APIFootballClient()
    for season in [2010, 2014, 2018, 2022, 2026]:
        print(f"League {1} year {season}")
        _teams = get_team_names_by_league(client, 1, season)
        #create_teams(session, _teams, is_national_team=True)
        teams += _teams
    return _unique_teams(teams)

def get_ec_teams():
    init_db()
    session = SessionLocal()
    teams = []
    client = APIFootballClient()
    for season in [2008, 2012, 2016, 2020, 2024]:
        print(f"League {4} year {season}")
        _teams = get_team_names_by_league(client, 4, season)
        #create_teams(session, _teams, is_national_team=True)
        teams += _teams

    return _unique_teams(teams)

def _unique_teams(teams: list[dict]):
    ret = []
    for team in teams:
        team_ids = [t['external_id'] for t in ret if 'external_id' in t]
        if team['external_id'] not in team_ids:
            ret.append(
                team
            )
    return ret

def write_csv(filename: str, data: list[dict]):
    csv_path = Path(_project_root() / "data" / f"{filename}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

def create_teams(session, teams: list, is_national_team=False):
    team_repo = TeamRepository(session)
    for t in teams:
        team_repo.create_from_provider_team(external_id=int(t['external_id']), name=t['team_name'],national=is_national_team)
    team_repo.commit()


def _get_leagues():
    client = APIFootballClient()
    all =  get_all_leagues(client)

    h = [f for f in all if f.country_name=='Portugal']

    for u in h:
        print(u.league_name)
    return get_all_leagues(client)

def _get_fotmob_teams(league_id, country_code):
    teams = FotMobProvider().fetch_teams_for_leagues(
        [league_id],
        country_codes={league_id:country_code},
    )
    teams = [p.name for p in teams]
    return teams


def st_teams():
    from services.draw_manager import STDrawManager
    init_db()
    session = SessionLocal()
    d = STDrawManager(session)
    h = d.import_all_draws_and_name_check()
    print(h)

get_api_football_teams()


g = {

}
