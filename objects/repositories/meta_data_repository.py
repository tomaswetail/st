from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert

from objects.models.meta_data import MetadataRow

from objects.repositories.base import BaseRepository


class MetaDataRepository(BaseRepository[MetadataRow]):
    model = MetadataRow

    def get_by_key(self, key: str) -> MetadataRow | None:
        return self.session.get(self.model, key)

    def get_value(self, key: str) -> str | None:
        row = self.get_by_key(key)
        return row.value if row else None

    def set_value(self, key: str, value: str) -> None:
        stmt = pg_insert(self.model).values(key=key, value=value)
        stmt = stmt.on_conflict_do_update(
            index_elements=["key"],
            set_={
                "value": stmt.excluded.value,
                "updated_at": func.now(),
            },
        )
        self.session.execute(stmt)
        self.session.commit()
