from typing import Optional

from sqlalchemy import (
    Integer,
    String, Boolean, UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base

class STRoundModel(Base):
    __tablename__ = "stryktipset_rounds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Draw/event info
    product_id: Mapped[int] = mapped_column(Integer, nullable=False)
    product_name: Mapped[str] = mapped_column(String(100), nullable=False)

    draw_number: Mapped[int] = mapped_column(Integer, nullable=False)
    event_number: Mapped[int] = mapped_column(Integer, nullable=False)

    description: Mapped[Optional[str]] = mapped_column(String(255))
    comment: Mapped[Optional[str]] = mapped_column(String(255))
    cancelled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    matches: Mapped[list["STMatchModel"]] = relationship(
        back_populates="event",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint(
            "product_id",
            "draw_number",
            "event_number",
            name="uq_event_product_draw_event",
        ),
    )
