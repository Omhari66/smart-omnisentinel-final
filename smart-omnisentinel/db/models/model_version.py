"""db/models/model_version.py — Deployed ML model version registry."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import Boolean, DateTime, String, Text, JSON

from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base, TimestampMixin, UUIDMixin


class ModelVersion(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "model_versions"

    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Which EventType strings this model handles
    event_types: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    checkpoint_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    onnx_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Per-class evaluation metrics stored as JSON
    precision_scores: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    recall_scores: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    deployed_at: Mapped[Optional[str]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def __repr__(self) -> str:
        return f"<ModelVersion name={self.name!r} active={self.is_active}>"
