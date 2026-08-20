"""Typer CLI for historical SofaScore xG and shot data ingestion."""

from __future__ import annotations

from typing import Optional

import typer

from data_sources.football_data.league_catalogue import LeagueCatalogueService
from data_sources.football_data.service import ExtendedMatchDataService
from database import SessionLocal, init_db
from objects.schema.data_classes.data_sources import DataSourceConfig

app = typer.Typer(
    name="football-data",
    help="Import historical football xG and shot data from SofaScore.",
    no_args_is_help=True,
)


def _require_sofascore(provider: str) -> None:
    if provider != "sofascore":
        raise typer.BadParameter("provider must be sofascore")


def _build_service(
    *,
    provider: str,
    dry_run: bool,
    request_delay: Optional[int],
) -> ExtendedMatchDataService:
    _require_sofascore(provider)
    config = DataSourceConfig()
    if request_delay is not None:
        config.football_data_request_delay_ms = request_delay
    session = SessionLocal()
    return ExtendedMatchDataService(
        provider=provider,  # type: ignore[arg-type]
        session=session,
        config=config,
        dry_run=dry_run,
    )


def _build_catalogue(*, provider: str) -> LeagueCatalogueService:
    _require_sofascore(provider)
    return LeagueCatalogueService(
        provider=provider,  # type: ignore[arg-type]
        session=SessionLocal(),
    )


@app.command("import-league")
def import_league(
    league_id: int = typer.Option(..., "--league-id", help="Internal leagues.id"),
    provider: str = typer.Option("sofascore", "--provider", help="sofascore"),
    season: Optional[str] = typer.Option(None, "--season", help="Season label or provider id"),
    force_refresh: bool = typer.Option(False, "--force-refresh"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    request_delay: Optional[int] = typer.Option(
        None, "--request-delay", help="Delay between HTTP requests in ms"
    ),
    limit: Optional[int] = typer.Option(None, "--limit", help="Max matches to import"),
) -> None:
    """Fetch and store league history for an internal league."""
    init_db()
    service = _build_service(
        provider=provider,
        dry_run=dry_run,
        request_delay=request_delay,
    )
    try:
        result = service.fetch_and_store_league_history(
            league_id=league_id,
            season=season,
            force_refresh=force_refresh,
            limit=limit,
        )
        typer.echo(
            "requested={requested} imported={imported} updated={updated} "
            "skipped={skipped} unresolved={unresolved} failed={failed}".format(
                requested=result.requested,
                imported=result.imported,
                updated=result.updated,
                skipped=result.skipped,
                unresolved=result.unresolved,
                failed=result.failed,
            )
        )
        if result.failed:
            raise typer.Exit(code=1)
    finally:
        service.close()


@app.command("import-match")
def import_match(
    match_id: int = typer.Option(
        ..., "--match-id", help="Internal fixtures.id"
    ),
    provider: str = typer.Option("sofascore", "--provider", help="sofascore"),
    force_refresh: bool = typer.Option(False, "--force-refresh"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    request_delay: Optional[int] = typer.Option(
        None, "--request-delay", help="Delay between HTTP requests in ms"
    ),
) -> None:
    """Fetch and store advanced stats for one historical match."""
    init_db()
    service = _build_service(
        provider=provider,
        dry_run=dry_run,
        request_delay=request_delay,
    )
    try:
        result = service.fetch_and_store_match(
            match_id=match_id,
            force_refresh=force_refresh,
        )
        typer.echo(
            f"match_id={result.internal_match_id} status={result.status} "
            f"shots={result.shots_imported} error={result.error}"
        )
        if result.warnings:
            for warning in result.warnings:
                typer.echo(f"warning: {warning}")
        if result.status == "failed":
            raise typer.Exit(code=1)
    finally:
        service.close()


@app.command("list-leagues")
def list_leagues(
    provider: str = typer.Option("sofascore", "--provider"),
    query: Optional[str] = typer.Option(None, "--query"),
    country: Optional[str] = typer.Option(None, "--country"),
) -> None:
    """List provider league catalogue ids."""
    catalogue = _build_catalogue(provider=provider)
    try:
        leagues = catalogue.find_leagues(query=query, country=country)
        for league in leagues:
            typer.echo(
                f"{league.provider_league_id}\t{league.name}\t"
                f"{league.country or ''}\t{league.country_code or ''}"
            )
    finally:
        catalogue.close()


@app.command("suggest-league-mappings")
def suggest_league_mappings(
    provider: str = typer.Option("sofascore", "--provider"),
) -> None:
    """Suggest provider ids for internal leagues (does not write)."""
    init_db()
    catalogue = _build_catalogue(provider=provider)
    try:
        for suggestion in catalogue.suggest_mappings():
            candidate = suggestion.candidate
            external = (
                f"{candidate.provider_league_id}\t{candidate.name}"
                if candidate
                else "\t"
            )
            typer.echo(
                f"{suggestion.internal_league_id}\t{suggestion.internal_name}\t"
                f"{suggestion.method}\t{suggestion.confidence:.3f}\t{external}"
            )
    finally:
        catalogue.close()


@app.command("map-league")
def map_league(
    league_id: int = typer.Option(..., "--league-id"),
    provider: str = typer.Option("sofascore", "--provider"),
    external_id: Optional[str] = typer.Option(None, "--external-id"),
    query: Optional[str] = typer.Option(None, "--query"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Map an internal leagues.id to a SofaScore league id."""
    init_db()
    catalogue = _build_catalogue(provider=provider)
    try:
        result = catalogue.map_league(
            league_id,
            external_entity_id=external_id,
            query=query,
            dry_run=dry_run,
        )
        typer.echo(
            f"league_id={result.internal_league_id} status={result.status} "
            f"external_id={result.external_entity_id} error={result.error}"
        )
        for candidate in result.candidates:
            typer.echo(
                f"candidate\t{candidate.provider_league_id}\t{candidate.name}\t"
                f"{candidate.country or ''}"
            )
        if result.status in {"failed", "unresolved"}:
            raise typer.Exit(code=1)
    finally:
        catalogue.close()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
