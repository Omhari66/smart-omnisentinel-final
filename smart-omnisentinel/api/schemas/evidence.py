"""api/schemas/evidence.py — Evidence clip access schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from api.schemas.common import OrmBase
from core.constants import RetentionTag


class EvidenceClipResponse(OrmBase):
    id: uuid.UUID
    incident_id: uuid.UUID
    file_size_bytes: Optional[int]
    duration_seconds: Optional[float]
    sha256_hash: Optional[str]
    is_locked: bool
    is_anonymized: bool
    retention_tag: RetentionTag
    expires_at: Optional[datetime]
    created_at: datetime


class SignedURLResponse(BaseModel):
    evidence_id: uuid.UUID
    signed_url: str
    expires_at: datetime
    sha256_hash: Optional[str]


class EvidenceManifest(BaseModel):
    """Full JSON manifest — returned from /evidence/{id}/manifest."""
    incident_id: uuid.UUID
    camera_id: uuid.UUID
    camera_name: str
    location: str
    event_type: str
    risk_score: int
    severity: str
    detected_at: datetime
    event_start: datetime
    event_end: Optional[datetime]
    clip_path: str
    clip_sha256: Optional[str]
    clip_duration_seconds: Optional[float]
    people_count: int
    model_version: Optional[str]
    confidence_at_alert: float
    reviewer_id: Optional[uuid.UUID]
    reviewer_action: Optional[str]
    reviewer_notes: Optional[str]
    reviewed_at: Optional[datetime]
    is_locked: bool
    retention_days: int
    created_at: datetime
