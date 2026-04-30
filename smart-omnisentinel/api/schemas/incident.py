"""api/schemas/incident.py — Incident request/response schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from api.schemas.auth import UserSummary
from api.schemas.camera import CameraResponse
from api.schemas.common import OrmBase
from core.constants import EventType, IncidentStatus, ReviewAction, Severity


class IncidentFilters(BaseModel):
    camera_id: Optional[uuid.UUID] = None
    severity: Optional[List[Severity]] = None
    status: Optional[List[IncidentStatus]] = None
    event_type: Optional[List[EventType]] = None
    from_dt: Optional[datetime] = Field(default=None, alias="from")
    to_dt: Optional[datetime] = Field(default=None, alias="to")
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=25, ge=1, le=100)

    model_config = {"populate_by_name": True}


class ModelVersionSummary(OrmBase):
    id: uuid.UUID
    name: str


class EvidenceClipSummary(OrmBase):
    id: uuid.UUID
    duration_seconds: Optional[float]
    sha256_hash: Optional[str]
    is_locked: bool


class IncidentResponse(OrmBase):
    id: uuid.UUID
    camera_id: uuid.UUID
    event_type: EventType
    severity: Severity
    status: IncidentStatus
    risk_score: int
    confidence_at_alert: float
    people_count: int
    event_start: datetime
    event_end: Optional[datetime]
    detected_at: datetime
    duration_seconds: Optional[float]
    is_locked: bool
    retention_days: int
    review_action: Optional[ReviewAction]
    reviewed_at: Optional[datetime]
    reviewer: Optional[UserSummary]
    model_version: Optional[ModelVersionSummary]
    evidence_clips: List[EvidenceClipSummary] = []
    created_at: datetime


class IncidentSummary(OrmBase):
    """Lightweight version for list endpoints."""
    id: uuid.UUID
    camera_id: uuid.UUID
    event_type: EventType
    severity: Severity
    status: IncidentStatus
    risk_score: int
    confidence_at_alert: float
    detected_at: datetime
    is_locked: bool


class IncidentAnalyticsSummary(BaseModel):
    total: int
    by_severity: dict[str, int]
    by_event_type: dict[str, int]
    by_status: dict[str, int]
    false_positive_rate: float
    period_from: datetime
    period_to: datetime
