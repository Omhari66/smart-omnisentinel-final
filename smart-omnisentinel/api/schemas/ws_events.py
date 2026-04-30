"""
api/schemas/ws_events.py
------------------------
Pydantic schemas for all WebSocket events pushed to the dashboard.
Every event shares the WSEventEnvelope wrapper.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from core.constants import CameraStatus, EventType, Severity, WSEventType


class WSEventEnvelope(BaseModel):
    """Outer envelope for all WebSocket messages."""
    event_id: uuid.UUID = Field(default_factory=uuid4)
    event_type: WSEventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    payload: Any


# ---------------------------------------------------------------------------
# Payload schemas (one per WSEventType)
# ---------------------------------------------------------------------------

class ConnectedPayload(BaseModel):
    active_incidents: int
    cameras_online: int
    cameras_offline: int
    pending_reviews: int
    server_time: datetime


class IncidentCreatedPayload(BaseModel):
    incident_id: uuid.UUID
    camera_id: uuid.UUID
    camera_name: str
    location: str
    event_type: EventType
    severity: Severity
    risk_score: int
    confidence: float
    detected_at: datetime


class AlertSeverityChangedPayload(BaseModel):
    incident_id: uuid.UUID
    previous_severity: Severity
    new_severity: Severity
    reason: str
    changed_at: datetime


class CameraStatusChangedPayload(BaseModel):
    camera_id: uuid.UUID
    camera_name: str
    previous_status: CameraStatus
    new_status: CameraStatus
    reason: str
    changed_at: datetime


class EvidenceReadyPayload(BaseModel):
    incident_id: uuid.UUID
    evidence_id: uuid.UUID
    duration_seconds: Optional[float]


class ReviewRequestedPayload(BaseModel):
    incident_id: uuid.UUID
    severity: Severity
    queue_position: int
    escalation_deadline: Optional[datetime]


class IncidentResolvedPayload(BaseModel):
    incident_id: uuid.UUID
    resolution: str
    resolved_by: Optional[str]
    resolved_at: datetime


class SystemHealthDegradedPayload(BaseModel):
    component: str
    metric: str
    value: Any
    threshold: Any
    severity: str  # "WARNING" | "CRITICAL"


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_event(event_type: WSEventType, payload: BaseModel) -> WSEventEnvelope:
    return WSEventEnvelope(
        event_type=event_type,
        payload=payload.model_dump(mode="json"),
    )
