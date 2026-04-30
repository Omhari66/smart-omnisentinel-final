"""db/models/threshold_profile.py — Per-deployment confidence and risk thresholds."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from sqlalchemy import Boolean, Float, SmallInteger, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from db.models.camera import Camera


class ThresholdProfile(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "threshold_profiles"

    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)

    # Per-classifier minimum confidence to consider event a candidate
    violence_confidence_min: Mapped[float] = mapped_column(Float, default=0.72)
    fall_confidence_min: Mapped[float] = mapped_column(Float, default=0.68)
    crowd_confidence_min: Mapped[float] = mapped_column(Float, default=0.65)

    # Frames event must persist at above-threshold confidence before confirming
    persistence_frames: Mapped[int] = mapped_column(SmallInteger, default=12)

    # Risk score tier boundaries
    risk_low_max: Mapped[int] = mapped_column(SmallInteger, default=29)
    risk_medium_max: Mapped[int] = mapped_column(SmallInteger, default=64)

    # Smoothing
    ema_alpha: Mapped[float] = mapped_column(Float, default=0.35)

    # Cooldown: seconds before same camera+event_type can fire another HIGH alert
    cooldown_seconds: Mapped[int] = mapped_column(SmallInteger, default=300)

    # Escalation: seconds before unreviewed MEDIUM auto-escalates to HIGH
    escalation_timeout_seconds: Mapped[int] = mapped_column(SmallInteger, default=300)

    is_default: Mapped[bool] = mapped_column(Boolean, default=False)

    cameras: Mapped[List["Camera"]] = relationship(
        "Camera", back_populates="threshold_profile", lazy="select"
    )

    def __repr__(self) -> str:
        return f"<ThresholdProfile name={self.name!r} default={self.is_default}>"
