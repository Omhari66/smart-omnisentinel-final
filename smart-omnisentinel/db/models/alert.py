"""db/models/alert.py — Alert records and escalation chain."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import DateTime, Enum as SQLEnum, ForeignKey, String, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.constants import Severity
from db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from db.models.incident import Incident


class Alert(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "alerts"

    incident_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("incidents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    severity: Mapped[str] = mapped_column(
        SQLEnum(Severity, name="alert_severity_enum"),
        nullable=False,
    )
    triggered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    # JSON list of channels used: ["email", "webhook", "websocket"]
    notification_channels: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    cooldown_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Self-referential: if this was auto-escalated from a prior alert
    escalated_from_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("alerts.id", ondelete="SET NULL"),
        nullable=True,
    )
    escalation_reason: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # Relationships
    incident: Mapped["Incident"] = relationship("Incident", back_populates="alerts")
    escalated_from: Mapped[Optional["Alert"]] = relationship(
        "Alert", remote_side="Alert.id"
    )
