"""db/models/camera.py — Camera registration and status tracking."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import DateTime, Enum as SQLEnum, ForeignKey, SmallInteger, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.constants import CameraStatus
from db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from db.models.incident import Incident
    from db.models.zone_config import ZoneConfig
    from db.models.threshold_profile import ThresholdProfile


class Camera(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "cameras"

    name: Mapped[str] = mapped_column(String(100), nullable=False)
    stream_url: Mapped[str] = mapped_column(Text, nullable=False)
    location: Mapped[str] = mapped_column(String(200), nullable=False)

    zone_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("zone_configs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    threshold_profile_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("threshold_profiles.id", ondelete="SET NULL"),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        SQLEnum(CameraStatus, name="camera_status_enum"),
        default=CameraStatus.DISABLED,
        nullable=False,
        index=True,
    )
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    fps_target: Mapped[int] = mapped_column(SmallInteger, default=4, nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    zone: Mapped[Optional["ZoneConfig"]] = relationship(
        "ZoneConfig", back_populates="cameras", lazy="joined"
    )
    threshold_profile: Mapped[Optional["ThresholdProfile"]] = relationship(
        "ThresholdProfile", back_populates="cameras", lazy="joined"
    )
    incidents: Mapped[List["Incident"]] = relationship(
        "Incident", back_populates="camera", lazy="select"
    )

    def __repr__(self) -> str:
        return f"<Camera id={self.id} name={self.name!r} status={self.status}>"
