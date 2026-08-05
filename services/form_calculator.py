from datetime import date

from sqlalchemy.orm import Session

from objects.models.team import TeamModel
from objects.repositories.historical_match_repository import HistoricalMatchRepository
from objects.repositories.team_repository import TeamRepository
from objects.schema.data_classes.data_sources import DataSourceConfig
from objects.schema.data_classes.recent_form_stats import RecentFormStats, TeamRestDays
from objects.schema.data_classes.team_rest_days import MatchRecentFormFeatures, MatchRestDaysFeatures
from objects.schema.db.historical_match import HistoricalMatch
from services.recent_form import build_match_recent_form_features, calculate_recent_form
from services.congestion_calculator import calculate_team_congestion
from services.restdays_calculator import build_match_rest_days_features
from services.team_form import TeamFormCalculator


class FormCalculator:
    """Calculate team and match form features from historical matches."""

    def __init__(
        self,
        home_team_name: str,
        away_team_name: str,
        config: DataSourceConfig | None = None,
        session: Session | None = None,
    ) -> None:
        """Load home/away teams and repositories for a fixture."""
        self.config = DataSourceConfig()
        self.historical_match_repo = HistoricalMatchRepository(session)
        team_repo = TeamRepository(session)
        home_team = team_repo.get_by_name(home_team_name)
        away_team = team_repo.get_by_name(away_team_name)
        self.home_team = home_team
        self.away_team = away_team

    def _fixture_matches(self) -> list[HistoricalMatch]:
        """Return deduplicated historical matches for both fixture teams."""
        home_matches = self.historical_match_repo.get_matches_by_team(self.home_team)
        away_matches = self.historical_match_repo.get_matches_by_team(self.away_team)
        by_id = {m.id: m for m in home_matches + away_matches}
        return sorted(by_id.values(), key=lambda m: m.match_date)

    def get_team_form(
        self,
        team: TeamModel,
        before_date: date,
    ) -> tuple[int | None, int | None, int | None]:
        """Return unweighted (points, goals_for, goals_against) for last N matches."""
        if team is None:
            raise ValueError("team is required")

        matches = self.historical_match_repo.get_matches_by_team(team)
        calculator = TeamFormCalculator(self.config.form_matches)
        calculator.ingest(matches)
        return calculator.form_before(team.name, before_date)

    def get_recent_team_form(
        self,
        team: TeamModel,
        before_date: date,
    ) -> RecentFormStats:
        """Return weighted recent form stats for one team before a date."""
        if team is None:
            raise ValueError("team is required")

        matches = self.historical_match_repo.get_matches_by_team(team)
        return calculate_recent_form(
            matches,
            team.name,
            before_date,
            lookback=self.config.form_matches,
        )

    def get_match_recent_form(self, before_date: date) -> MatchRecentFormFeatures:
        """Return home vs away recent form comparison for the fixture."""
        if self.home_team is None or self.away_team is None:
            raise ValueError("home_team and away_team are required")

        matches = self._fixture_matches()
        return build_match_recent_form_features(
            matches,
            self.home_team.name,
            self.away_team.name,
            before_date,
            lookback=self.config.form_matches,
        )

    def get_match_rest_days(self, match_date: date) -> MatchRestDaysFeatures:
        """Return home vs away rest-day comparison for the fixture."""
        if self.home_team is None or self.away_team is None:
            raise ValueError("home_team and away_team are required")

        matches = self._fixture_matches()
        return build_match_rest_days_features(
            matches,
            self.home_team.name,
            self.away_team.name,
            match_date,
        )

    def get_match_congestion_scores(
        self, match_date: date
    ) -> tuple[float, float, float]:
        """Return home, away, and away-minus-home congestion scores."""
        if self.home_team is None or self.away_team is None:
            raise ValueError("home_team and away_team are required")

        matches = self._fixture_matches()
        home_congestion = calculate_team_congestion(
            matches, self.home_team.name, match_date
        )
        away_congestion = calculate_team_congestion(
            matches, self.away_team.name, match_date
        )
        return (
            home_congestion,
            away_congestion,
            away_congestion - home_congestion,
        )

    def get_match_congestion_advantage(self, match_date: date) -> float:
        """Return away minus home congestion score for the fixture."""
        return self.get_match_congestion_scores(match_date)[2]
