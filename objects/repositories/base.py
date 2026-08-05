from __future__ import annotations

from typing import Generic, TypeVar

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database import SessionLocal

ModelT = TypeVar("ModelT")


class BaseRepository(Generic[ModelT]):
    """Base repository with session handling and CRUD operations."""

    model: type[ModelT]

    def __init__(self, session: Session | None = None) -> None:
        self._session = session or SessionLocal()
        self._owns_session = session is None

    @property
    def session(self) -> Session:
        return self._session

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    def __enter__(self) -> BaseRepository[ModelT]:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self._owns_session:
            return
        if exc_type is None:
            self._session.commit()
        else:
            self._session.rollback()
        self._session.close()

    def commit(self) -> None:
        self._session.commit()

    def rollback(self) -> None:
        self._session.rollback()

    def flush(self) -> None:
        self._session.flush()

    def get(self, id: int) -> ModelT | None:
        return self._session.get(self.model, id)

    def get_all(self) -> list[ModelT]:
        return list(self._session.scalars(select(self.model)).all())

    def count(self) -> int:
        return int(
            self._session.scalar(
                select(func.count()).select_from(self.model)
            )
        )

    def create(self, **kwargs) -> ModelT:
        entity = self.model(**kwargs)
        self._session.add(entity)
        return entity

    def save(self, entity: ModelT) -> ModelT:
        self._session.add(entity)
        return entity

    def delete(self, entity: ModelT) -> None:
        self._session.delete(entity)

    def delete_by_id(self, id: int) -> bool:
        entity = self.get(id)
        if entity is None:
            return False
        self.delete(entity)
        return True
