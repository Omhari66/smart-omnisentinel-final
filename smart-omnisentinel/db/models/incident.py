"""db/models/incident.py — Core incident record."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum as SQLEnum,
    Float,
    ForeignKey,
    SmallInteger,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.constants import EventType, IncidentStatus, ReviewAction, Severity
from db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from db.models.camera import Camera
    from db.models.alert import Alert
    from db.models.evidence_clip import EvidenceClip
    from db.models.review_action import ReviewActionRecord
    from db.models.model_version import ModelVersion
    from db.models.user import User


class Incident(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "incidents"

    camera_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("cameras.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_type: Mapped[str] = mapped_column(
        SQLEnum(EventType, name="event_type_enum"),
        nullable=False,
        index=True,
    )
    severity: Mapped[str] = mapped_column(
        SQLEnum(Severity, name="severity_enum"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        SQLEnum(IncidentStatus, name="incident_status_enum"),
        default=IncidentStatus.PENDING_REVIEW,
        nullable=False,
        index=True,
    )

    risk_score: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    confidence_at_alert: Mapped[float] = mapped_column(Float, nullable=False)
    people_count: Mapped[int] = mapped_column(SmallInteger, default=0, nullable=False)

    event_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    event_end: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    model_version_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("model_versions.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Review fields
    reviewer_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    review_action: Mapped[Optional[str]] = mapped_column(
        SQLEnum(ReviewAction, name="review_action_enum"),
        nullable=True,
    )

    # Evidence locking
    is_locked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    retention_days: Mapped[int] = mapped_column(SmallInteger, default=3, nullable=False)

    # AI-generated summary (populated by local LLM after detection)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    camera: Mapped["Camera"] = relationship("Camera", back_populates="incidents")
    alerts: Mapped[List["Alert"]] = relationship(
        "Alert", back_populates="incident", lazy="select"
    )
    evidence_clips: Mapped[List["EvidenceClip"]] = relationship(
        "EvidenceClip", back_populates="incident", lazy="select"
    )
    review_records: Mapped[List["ReviewActionRecord"]] = relationship(
        "ReviewActionRecord", back_populates="incident", lazy="select"
    )
    model_version: Mapped[Optional["ModelVersion"]] = relationship(
        "ModelVersion", lazy="joined"
    )
    reviewer: Mapped[Optional["User"]] = relationship(
        "User", foreign_keys=[reviewer_id], lazy="joined"
    )

    def __repr__(self) -> str:
        return (
            f"<Incident id={self.id} type={self.event_type} "
            f"severity={self.severity} status={self.status}>"
        )
