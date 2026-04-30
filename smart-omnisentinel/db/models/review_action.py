"""db/models/review_action.py — Reviewer decisions on incidents."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Enum as SQLEnum, Float, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.constants import ReviewAction
from db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from db.models.incident import Incident
    from db.models.user import User


class ReviewActionRecord(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "review_actions"

    incident_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("incidents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    reviewer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    action: Mapped[str] = mapped_column(
        SQLEnum(ReviewAction, name="review_action_record_enum"),
        nullable=False,
        index=True,
    )
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence_override: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    incident: Mapped["Incident"] = relationship(
        "Incident", back_populates="review_records"
    )
    reviewer: Mapped["User"] = relationship("User")
