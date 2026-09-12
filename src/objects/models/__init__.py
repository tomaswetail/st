"""SQLAlchemy model package — import side-effects register tables for create_all."""

from src.objects.models.fixture import FixtureModel
from src.objects.models.fixture_odds import FixtureOddsModel
from src.objects.models.injury_snapshot import InjurySnapshotModel
from src.objects.models.league import LeagueModel
from src.objects.models.match_availability import MatchAvailabilityModel
from src.objects.models.player import PlayerModel
from src.objects.models.match_advanced_stats import MatchAdvancedStatsModel
from src.objects.models.match_shot import MatchShotModel
from src.objects.models.meta_data import MetadataRow
from src.objects.models.st_match import STMatchModel
from src.objects.models.st_match_bet import STMatchBetModel
from src.objects.models.st_match_odds import STMatchOddsModel
from src.objects.models.st_round import STRoundModel
from src.objects.models.team import TeamModel

__all__ = [
    "FixtureModel",
    "FixtureOddsModel",
    "InjurySnapshotModel",
    "LeagueModel",
    "MatchAvailabilityModel",
    "PlayerModel",
    "MatchAdvancedStatsModel",
    "MatchShotModel",
    "MetadataRow",
    "STMatchModel",
    "STMatchBetModel",
    "STMatchOddsModel",
    "STRoundModel",
    "TeamModel",
]
