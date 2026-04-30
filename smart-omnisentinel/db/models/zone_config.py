"""db/models/zone_config.py — Zone sensitivity and suppression rules."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import Float, String, Text, JSON

from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from db.models.camera import Camera


class ZoneConfig(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "zone_configs"

    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # 0.0 = lowest sensitivity, 1.0 = highest sensitivity
    sensitivity: Mapped[float] = mapped_column(Float, default=0.7, nullable=False)

    # EventType strings to suppress in this zone, e.g. ["FALL_COLLAPSE"] for gym
    suppress_event_types: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    # Schedule of peak hours: [{"start": "08:00", "end": "09:30", "days": [1,2,3,4,5]}]
    peak_hours_schedule: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    cameras: Mapped[List["Camera"]] = relationship(
        "Camera", back_populates="zone", lazy="select"
    )

    def __repr__(self) -> str:
        return f"<ZoneConfig name={self.name!r} sensitivity={self.sensitivity}>"
