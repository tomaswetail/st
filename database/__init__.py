from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from database.models import Base


def _database_url() -> str:
    if url := os.environ.get("DATABASE_URL"):
        return url

    user = os.environ.get("POSTGRES_USER", "st_poisson_distribution_user" )
    password = os.environ.get("POSTGRES_PASSWORD", "password")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "st_poisson_distribution")
    if password:
        return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"
    return f"postgresql+psycopg://{user}@{host}:{port}/{db}"


DATABASE_URL = _database_url()

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db() -> None:
    import objects.models  # noqa: F401

    Base.metadata.create_all(engine)


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
