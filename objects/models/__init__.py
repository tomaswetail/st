"""SQLAlchemy model package — import side-effects register tables for create_all."""

from objects.models.external_entity_mapping import ExternalEntityMappingModel
from objects.models.fixture import FixtureModel
from objects.models.league import LeagueModel
from objects.models.match_advanced_stats import MatchAdvancedStatsModel
from objects.models.match_shot import MatchShotModel
from objects.models.meta_data import MetadataRow
from objects.models.st_match import STMatchModel
from objects.models.st_match_bet import STMatchBetModel
from objects.models.st_round import STRoundModel
from objects.models.team import TeamModel

__all__ = [
    "ExternalEntityMappingModel",
    "FixtureModel",
    "LeagueModel",
    "MatchAdvancedStatsModel",
    "MatchShotModel",
    "MetadataRow",
    "STMatchModel",
    "STMatchBetModel",
    "STRoundModel",
    "TeamModel",
]
