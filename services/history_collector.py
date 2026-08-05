import typer

from data_sources.data_collector import DataCollector
from objects.schema.data_classes.data_sources import DataSourceConfig


def import_football_data() -> None:

    seasons = "2223,2324,2425"
    """Download and import Football-Data.co.uk CSVs into PostgreSQL."""
    ds = DataSourceConfig()
    builder = DataCollector(ds)
    try:
        season_list = [x.strip() for x in seasons.split(",") if x.strip()]

        league_count, tournament_count = builder.import_all(season_list)
        typer.echo(f"Imported {league_count} league matches and {tournament_count} tournament matches")
    except Exception as exc:
        raise typer.Exit(code=1) from exc
    finally:
        builder.close()


def refresh_all_historical_matches() -> None:
    """Refresh all historical match data (all leagues, last 5 seasons)."""
    builder = DataCollector()
    try:
        count = builder.refresh_all_matches()
        typer.echo(f"Refreshed {count} historical matches")
    except Exception as exc:
        raise typer.Exit(code=1) from exc
    finally:
        builder.close()
