"""
services/alert_manager/alert_dispatcher.py
-------------------------------------------
Receives scored incidents from the risk engine.
Routes to LOW / MEDIUM / HIGH handler based on severity tier.
Writes all routing decisions to the audit trail.
Triggers cooldown on the event analyzer after HIGH alert fires.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core.constants import EventType, IncidentStatus, Severity, WSEventType
from core.event_bus import get_event_bus
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ScoredIncident:
    """Parsed representation of a message from 'scored_incidents' channel."""
    incident_id: uuid.UUID
    camera_id: str
    event_type: EventType
    severity: Severity
    risk_score: int
    smoothed_confidence: float
    people_count: int
    motion_magnitude: float
    event_start_timestamp: float
    timestamp: float
    track_ids: list

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScoredIncident":
        return cls(
            incident_id=uuid.UUID(d["incident_id"]) if "incident_id" in d else uuid.uuid4(),
            camera_id=d["camera_id"],
            event_type=EventType(d["event_type"]),
            severity=Severity(d["severity"]),
            risk_score=int(d["risk_score"]),
            smoothed_confidence=float(d["smoothed_confidence"]),
            people_count=int(d.get("people_count", 1)),
            motion_magnitude=float(d.get("motion_magnitude", 0.0)),
            event_start_timestamp=float(d.get("event_start_timestamp", time.time())),
            timestamp=float(d.get("timestamp", time.time())),
            track_ids=d.get("track_ids", []),
        )


class AlertDispatcher:
    """
    Central dispatcher: receives scored incidents, routes to severity handlers.
    One instance shared across the application.
    """

    def __init__(self):
        self._bus = get_event_bus()
        self._active_incidents: Dict[str, uuid.UUID] = {}  # key → incident_id
        # Deduplication: track open incidents per (camera_id, event_type)

    def _dedup_key(self, camera_id: str, event_type: str) -> str:
        return f"{camera_id}::{event_type}"

    def _is_duplicate(self, incident: ScoredIncident) -> bool:
        """
        Returns True if this incident is a continuation of an already-active one.
        Duplicates should update the existing incident, not create a new one.
        """
        key = self._dedup_key(incident.camera_id, incident.event_type.value)
        return key in self._active_incidents

    def _register_incident(self, incident: ScoredIncident) -> None:
        key = self._dedup_key(incident.camera_id, incident.event_type.value)
        self._active_incidents[key] = incident.incident_id

    def close_incident(self, camera_id: str, event_type: str) -> None:
        key = self._dedup_key(camera_id, event_type)
        self._active_incidents.pop(key, None)

    async def dispatch(self, message: Dict[str, Any]) -> None:
        """
        Handler for messages on 'scored_incidents' channel.
        Called by event bus consumer.
        """
        try:
            incident = ScoredIncident.from_dict(message)
        except Exception as exc:
            logger.error("alert_dispatch_parse_error", error=str(exc), message=message)
            return

        is_dup = self._is_duplicate(incident)

        if is_dup:
            logger.debug(
                "incident_deduplicated",
                camera_id=incident.camera_id,
                event_type=incident.event_type,
                severity=incident.severity,
            )
            # For persistent MEDIUM/HIGH events, check if escalation needed
            # (handled by escalation_timer.py separately)
            return

        self._register_incident(incident)
        logger.info(
            "alert_dispatching",
            camera_id=incident.camera_id,
            event_type=incident.event_type,
            severity=incident.severity,
            risk_score=incident.risk_score,
        )

        if incident.severity == Severity.LOW:
            await self._handle_low(incident)
        elif incident.severity == Severity.MEDIUM:
            await self._handle_medium(incident)
        elif incident.severity == Severity.HIGH:
            await self._handle_high(incident)

    # -----------------------------------------------------------------------
    # Severity handlers
    # -----------------------------------------------------------------------

    async def _handle_low(self, incident: ScoredIncident) -> None:
        """
        LOW: Persist incident, queue for review. No external alert.
        """
        from services.alert_manager.audit_writer import write_audit
        from services.evidence_manager.clip_extractor import request_clip_extraction

        logger.info("alert_low_handling", incident_id=str(incident.incident_id))

        # Persist incident to DB
        db_incident = await self._persist_incident(incident, IncidentStatus.PENDING_REVIEW)
        if db_incident is None:
            return

        # Request evidence clip (30s retention)
        await request_clip_extraction(
            incident_id=db_incident.id,
            camera_id=incident.camera_id,
            retention_tag="TEMP_72H",
        )

        # Audit
        await write_audit(
            actor_type="SYSTEM",
            action="ALERT_LOW_QUEUED",
            target_type="incident",
            target_id=db_incident.id,
            detail={"risk_score": incident.risk_score, "severity": "LOW"},
        )

        # WebSocket push (soft notification — dashboard review queue only)
        await self._push_ws_event(
            WSEventType.INCIDENT_CREATED, incident, db_incident.id
        )

    async def _handle_medium(self, incident: ScoredIncident) -> None:
        """
        MEDIUM: Notify security personnel, request review, start escalation timer.
        """
        from services.alert_manager.audit_writer import write_audit
        from services.evidence_manager.clip_extractor import request_clip_extraction
        from services.notification.webhook_sender import send_webhook

        logger.info("alert_medium_handling", incident_id=str(incident.incident_id))

        db_incident = await self._persist_incident(incident, IncidentStatus.ACTIVE_REVIEW)
        if db_incident is None:
            return

        # Request evidence clip (30-day retention)
        await request_clip_extraction(
            incident_id=db_incident.id,
            camera_id=incident.camera_id,
            retention_tag="STANDARD_30D",
        )

        # Notify via configured channels
        await self._dispatch_notifications(incident, db_incident.id, severity="MEDIUM")

        # Audit
        await write_audit(
            actor_type="SYSTEM",
            action="ALERT_MEDIUM_DISPATCHED",
            target_type="incident",
            target_id=db_incident.id,
            detail={"risk_score": incident.risk_score, "channels": ["websocket", "webhook"]},
        )

        # Push WS event (amber alert on dashboard)
        await self._push_ws_event(
            WSEventType.INCIDENT_CREATED, incident, db_incident.id
        )

        # Register with escalation timer
        from services.alert_manager.escalation_timer import get_escalation_timer
        timer = get_escalation_timer()
        timer.register(str(db_incident.id), incident.camera_id, incident.event_type.value)

    async def _handle_high(self, incident: ScoredIncident) -> None:
        """
        HIGH: Full emergency workflow. Lock evidence. Notify all channels.
        Trigger cooldown to prevent alert storm.
        """
        from services.alert_manager.audit_writer import write_audit
        from services.event_analysis.event_analyzer import get_event_analyzer
        from services.evidence_manager.clip_extractor import request_clip_extraction

        logger.warning(
            "alert_high_emergency",
            incident_id=str(incident.incident_id),
            camera_id=incident.camera_id,
            event_type=incident.event_type,
            risk_score=incident.risk_score,
        )

        db_incident = await self._persist_incident(
            incident,
            IncidentStatus.EMERGENCY,
            is_locked=True,
        )
        if db_incident is None:
            return

        # Request locked evidence clip (365-day retention)
        await request_clip_extraction(
            incident_id=db_incident.id,
            camera_id=incident.camera_id,
            retention_tag="LONG_365D",
            lock=True,
        )

        # Full notification dispatch
        await self._dispatch_notifications(incident, db_incident.id, severity="HIGH")

        # Set cooldown on event analyzer to prevent alert storm
        analyzer = get_event_analyzer()
        analyzer.set_cooldown(incident.camera_id, incident.event_type.value)

        # Audit
        await write_audit(
            actor_type="SYSTEM",
            action="ALERT_HIGH_EMERGENCY",
            target_type="incident",
            target_id=db_incident.id,
            detail={
                "risk_score": incident.risk_score,
                "confidence": incident.smoothed_confidence,
                "people_count": incident.people_count,
            },
        )

        # Push full-screen red alert to dashboard
        await self._push_ws_event(
            WSEventType.INCIDENT_CREATED, incident, db_incident.id
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    async def _persist_incident(
        self,
        incident: ScoredIncident,
        status: IncidentStatus,
        is_locked: bool = False,
    ):
        """Write incident record to the database."""
        from datetime import timedelta

        from db.session import get_session_factory
        from db.models.incident import Incident as IncidentModel

        retention_map = {
            Severity.LOW: 3,
            Severity.MEDIUM: 30,
            Severity.HIGH: 365,
        }

        factory = get_session_factory()
        try:
            async with factory() as db:
                db_incident = IncidentModel(
                    camera_id=uuid.UUID(incident.camera_id)
                    if len(incident.camera_id) == 36 else None,
                    event_type=incident.event_type.value,
                    severity=incident.severity.value,
                    status=status.value,
                    risk_score=incident.risk_score,
                    confidence_at_alert=incident.smoothed_confidence,
                    people_count=incident.people_count,
                    event_start=datetime.fromtimestamp(
                        incident.event_start_timestamp, tz=timezone.utc
                    ),
                    detected_at=datetime.fromtimestamp(
                        incident.timestamp, tz=timezone.utc
                    ),
                    is_locked=is_locked,
                    retention_days=retention_map.get(incident.severity, 3),
                )
                db.add(db_incident)
                await db.commit()
                await db.refresh(db_incident)
                logger.info(
                    "incident_persisted",
                    incident_id=str(db_incident.id),
                    status=status,
                )
                return db_incident
        except Exception as exc:
            logger.error("incident_persist_failed", error=str(exc))
            return None

    async def _dispatch_notifications(
        self,
        incident: ScoredIncident,
        incident_id: uuid.UUID,
        severity: str,
    ) -> None:
        """Dispatch notifications for MEDIUM and HIGH alerts."""
        from services.notification.webhook_sender import send_webhook
        from services.notification.email_sender import send_alert_email

        payload = {
            "incident_id": str(incident_id),
            "camera_id": incident.camera_id,
            "event_type": incident.event_type.value,
            "severity": severity,
            "risk_score": incident.risk_score,
            "confidence": round(incident.smoothed_confidence, 3),
            "people_count": incident.people_count,
            "timestamp": datetime.fromtimestamp(
                incident.timestamp, tz=timezone.utc
            ).isoformat(),
        }

        # Fire-and-forget — don't block alert pipeline on notification latency
        asyncio.create_task(send_webhook(payload))
        asyncio.create_task(send_alert_email(payload))

    async def _push_ws_event(
        self,
        event_type: WSEventType,
        incident: ScoredIncident,
        incident_id: uuid.UUID,
    ) -> None:
        """Push WebSocket event to all connected dashboard clients."""
        from api.schemas.ws_events import IncidentCreatedPayload, make_event
        from api.websocket_manager import get_ws_manager

        try:
            ws_manager = get_ws_manager()
            payload = IncidentCreatedPayload(
                incident_id=incident_id,
                camera_id=uuid.UUID(incident.camera_id)
                if len(incident.camera_id) == 36 else uuid.uuid4(),
                camera_name=incident.camera_id,  # Enriched in V2 with DB lookup
                location="",
                event_type=incident.event_type,
                severity=incident.severity,
                risk_score=incident.risk_score,
                confidence=incident.smoothed_confidence,
                detected_at=datetime.fromtimestamp(incident.timestamp, tz=timezone.utc),
            )
            event = make_event(event_type, payload)
            await ws_manager.broadcast(event.model_dump(mode="json"))
        except Exception as exc:
            logger.warning("ws_push_failed", error=str(exc))


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_dispatcher: Optional[AlertDispatcher] = None


def get_alert_dispatcher() -> AlertDispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = AlertDispatcher()
    return _dispatcher
