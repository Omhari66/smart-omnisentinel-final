"""db/models/evidence_clip.py — Evidence clip file records."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import BigInteger, Boolean, DateTime, Enum as SQLEnum, Float, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.constants import RetentionTag
from db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from db.models.incident import Incident


class EvidenceClip(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "evidence_clips"

    incident_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("incidents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sha256_hash: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_locked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_anonymized: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    retention_tag: Mapped[str] = mapped_column(
        SQLEnum(RetentionTag, name="retention_tag_enum"),
        default=RetentionTag.TEMP_72H,
        nullable=False,
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    incident: Mapped["Incident"] = relationship("Incident", back_populates="evidence_clips")
