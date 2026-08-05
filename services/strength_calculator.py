from sqlalchemy.orm import Session

from objects.repositories.historical_match_repository import HistoricalMatchRepository
from objects.repositories.team_repository import TeamRepository

HOME_ADV = 0.45
DEFAULT_LEAGUE_AVG_GOALS = 1.35


class StrengthCalculator:

    def __init__(
        self,
        home_team_name: str,
        away_team_name: str,
        session: Session | None = None,
    ) -> None:
        team_repo = TeamRepository(session)
        home_team = team_repo.get_by_name(home_team_name)
        away_team = team_repo.get_by_name(away_team_name)
        self.home_team = home_team
        self.away_team = away_team
        self.historical_match_repo = HistoricalMatchRepository(session)
        if home_team is None or home_team.league_id is None:
            self.league_avg_home_goals = DEFAULT_LEAGUE_AVG_GOALS
            self.league_avg_away_goals = DEFAULT_LEAGUE_AVG_GOALS
        else:
            try:
                self.league_avg_home_goals = self.historical_match_repo.get_home_goal_average_by_league(
                    home_team.league_id
                )
                self.league_avg_away_goals = self.historical_match_repo.get_away_goal_average_by_league(
                    home_team.league_id
                )
            except (KeyError, TypeError, ZeroDivisionError):
                self.league_avg_home_goals = DEFAULT_LEAGUE_AVG_GOALS
                self.league_avg_away_goals = DEFAULT_LEAGUE_AVG_GOALS
        self.league_avg_goals = (self.league_avg_home_goals + self.league_avg_away_goals) / 2

    def _safe_team_stat(self, team, getter) -> float:
        if team is None or team.league_id is None:
            return self.league_avg_goals
        try:
            value = getter(team)
        except (KeyError, TypeError, AttributeError):
            return self.league_avg_goals
        if value is None:
            return self.league_avg_goals
        return float(value)

    def calc_goal_stats(self, home_adv: float | None = None):
        """Beräknar lambda med korrekt genomsnitt per match"""
        if home_adv is None:
            home_adv = self.historical_match_repo.get_home_advantage()

        # === Hemmalaget ===
        home_attack_home = self.historical_match_repo.get_scored_home_goals_by_team(self.home_team)
        home_attack_away = self.historical_match_repo.get_scored_away_goals_by_team(self.home_team)
        home_conceded_home = self.historical_match_repo.get_lost_home_goals_by_team(self.home_team)
        home_conceded_away = self.historical_match_repo.get_lost_away_goals_by_team(self.home_team)

        # === Bortalaget ===
        away_attack_home = self.historical_match_repo.get_scored_home_goals_by_team(self.away_team)
        away_attack_away = self.historical_match_repo.get_scored_away_goals_by_team(self.away_team)
        away_conceded_home = self.historical_match_repo.get_lost_home_goals_by_team(self.away_team)
        away_conceded_away = self.historical_match_repo.get_lost_away_goals_by_team(self.away_team)

        # === Hämta antal matcher (VIKTIGT!) ===
        home_matches = self.historical_match_repo.get_number_of_home_matches_by_team(self.home_team)
        away_matches = self.historical_match_repo.get_number_of_away_matches_by_team(self.away_team)

        # Genomsnitt per match
        home_attack_avg = (home_attack_home + home_attack_away) / (2 * home_matches) if home_matches > 0 else 1.4
        home_conceded_avg = (home_conceded_home + home_conceded_away) / (2 * home_matches) if home_matches > 0 else 1.4

        away_attack_avg = (away_attack_home + away_attack_away) / (2 * away_matches) if away_matches > 0 else 1.3
        away_conceded_avg = (away_conceded_home + away_conceded_away) / (2 * away_matches) if away_matches > 0 else 1.4

        # Poisson lambda
        lambda_home = home_attack_avg * away_conceded_avg * (1 + home_adv)
        lambda_away = away_attack_avg * home_conceded_avg

        return lambda_home, lambda_away

    def calc_relative_lambdas_without_home_adv(self) -> tuple[float, float]:
        """Return (lambda_home_base, lambda_away) before applying home advantage."""
        league_avg = self.league_avg_goals

        home_attack_home = self._safe_team_stat(
            self.home_team, self.historical_match_repo.get_scored_home_goals_by_team
        )
        home_attack_away = self._safe_team_stat(
            self.home_team, self.historical_match_repo.get_scored_away_goals_by_team
        )
        home_conceded_home = self._safe_team_stat(
            self.home_team, self.historical_match_repo.get_lost_home_goals_by_team
        )
        home_conceded_away = self._safe_team_stat(
            self.home_team, self.historical_match_repo.get_lost_away_goals_by_team
        )

        away_attack_home = self._safe_team_stat(
            self.away_team, self.historical_match_repo.get_scored_home_goals_by_team
        )
        away_attack_away = self._safe_team_stat(
            self.away_team, self.historical_match_repo.get_scored_away_goals_by_team
        )
        away_conceded_home = self._safe_team_stat(
            self.away_team, self.historical_match_repo.get_lost_home_goals_by_team
        )
        away_conceded_away = self._safe_team_stat(
            self.away_team, self.historical_match_repo.get_lost_away_goals_by_team
        )

        home_attack_avg = (home_attack_home + home_attack_away) / 2
        home_conceded_avg = (home_conceded_home + home_conceded_away) / 2

        away_attack_avg = (away_attack_home + away_attack_away) / 2
        away_conceded_avg = (away_conceded_home + away_conceded_away) / 2

        home_attack_rel = home_attack_avg / league_avg
        home_conceded_rel = home_conceded_avg / league_avg

        away_attack_rel = away_attack_avg / league_avg
        away_conceded_rel = away_conceded_avg / league_avg

        lambda_home_base = home_attack_rel * away_conceded_rel
        lambda_away = away_attack_rel * home_conceded_rel
        return lambda_home_base, lambda_away

    def calc_goal_stats_relative(self, home_adv: float = 0.35):
        """Beräknar relativa styrkor och lambda för hemmalag och bortalag"""
        lambda_home_base, lambda_away = self.calc_relative_lambdas_without_home_adv()
        lambda_home = lambda_home_base * (1 + home_adv)
        return lambda_home, lambda_away
