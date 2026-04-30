"""
db/models/audit_log.py
----------------------
Append-only audit trail for all sensitive actions.
Uses BigInteger PK for high-volume insert performance.
Application DB role must NOT have UPDATE or DELETE on this table.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Integer, DateTime, Enum as SQLEnum, String, func, JSON
from sqlalchemy.dialects.postgresql import INET, UUID
from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class AuditActionType(str):
    CLIP_ACCESSED = "CLIP_ACCESSED"
    CLIP_EXPORTED = "CLIP_EXPORTED"
    INCIDENT_REVIEWED = "INCIDENT_REVIEWED"
    ALERT_ESCALATED = "ALERT_ESCALATED"
    CONFIG_CHANGED = "CONFIG_CHANGED"
    USER_LOGIN = "USER_LOGIN"
    USER_LOGOUT = "USER_LOGOUT"
    SYSTEM_AUTO_ESCALATED = "SYSTEM_AUTO_ESCALATED"
    EVIDENCE_LOCKED = "EVIDENCE_LOCKED"
    FALSE_POSITIVE_FLAGGED = "FALSE_POSITIVE_FLAGGED"


class AuditLog(Base):
    __tablename__ = "audit_log"

    # Sequential IDs make append-only easier to reason about
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # actor_id is null for system-generated entries
    actor_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    actor_type: Mapped[str] = mapped_column(
        String(10), default="SYSTEM", nullable=False
    )  # "USER" | "SYSTEM"

    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    target_type: Mapped[str] = mapped_column(String(50), nullable=False)
    target_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    detail: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    def __repr__(self) -> str:
        return f"<AuditLog id={self.id} action={self.action!r} actor={self.actor_id}>"
