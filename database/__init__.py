from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Generator

from sqlalchemy.orm import Session, sessionmaker

from database.models import Base

logger = logging.getLogger(__name__)

__all__ = [
    "Base",
    "SessionLocal",
    "init_db",
    "session_scope",
]

_engine = None
_session_factory: sessionmaker | None = None


def _database_url() -> str:
    if url := os.environ.get("DATABASE_URL"):
        return url

    user = os.environ.get("POSTGRES_USER", "st_poisson_distribution_user")
    password = os.environ.get("POSTGRES_PASSWORD", "password")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "st_poisson_distribution")
    if password:
        return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"
    return f"postgresql+psycopg://{user}@{host}:{port}/{db}"


def _get_engine():
    global _engine
    if _engine is None:
        from sqlalchemy import create_engine

        _engine = create_engine(_database_url(), future=True)
    return _engine


def _get_session_factory() -> sessionmaker:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=_get_engine(), expire_on_commit=False, future=True
        )
    return _session_factory


class _SessionLocalProxy:
    """Delay engine/psycopg setup until a session is actually created."""

    def __call__(self, *args: Any, **kwargs: Any) -> Session:
        return _get_session_factory()(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(_get_session_factory(), name)


SessionLocal = _SessionLocalProxy()


def __getattr__(name: str):
    if name == "DATABASE_URL":
        return _database_url()
    if name == "engine":
        return _get_engine()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def init_db() -> None:
    import objects.models  # noqa: F401
    from sqlalchemy import text
    from sqlalchemy.exc import ProgrammingError

    engine = _get_engine()
    Base.metadata.create_all(engine)
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        for extension_name in ("pg_trgm", "unaccent"):
            already_installed = connection.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = :name"),
                {"name": extension_name},
            ).scalar()
            if already_installed:
                continue
            try:
                connection.execute(
                    text(f"CREATE EXTENSION IF NOT EXISTS {extension_name}")
                )
            except ProgrammingError:
                logger.warning(
                    "Could not create extension %s (superuser required). "
                    "Run as a superuser: CREATE EXTENSION IF NOT EXISTS %s;",
                    extension_name,
                    extension_name,
                )


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
