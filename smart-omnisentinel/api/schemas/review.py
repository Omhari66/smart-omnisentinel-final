"""api/schemas/review.py — Review queue and action schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from api.schemas.common import OrmBase
from core.constants import IncidentStatus, ReviewAction, Severity


class ReviewActionRequest(BaseModel):
    action: ReviewAction
    notes: Optional[str] = None
    confidence_override: Optional[float] = None


class ReviewQueueItem(BaseModel):
    incident_id: uuid.UUID
    camera_name: str
    location: str
    event_type: str
    severity: Severity
    risk_score: int
    detected_at: datetime
    escalation_deadline: Optional[datetime]  # for MEDIUM incidents
    queue_position: int


class ReviewActionResponse(OrmBase):
    id: uuid.UUID
    incident_id: uuid.UUID
    reviewer_id: uuid.UUID
    action: ReviewAction
    notes: Optional[str]
    created_at: datetime
