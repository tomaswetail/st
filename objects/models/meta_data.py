from datetime import datetime

from sqlalchemy import DateTime, func, String
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


class MetadataRow(Base):
    __tablename__ = "metadata"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str] = mapped_column(String(100), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
