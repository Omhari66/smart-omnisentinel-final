"""
services/alert_manager/escalation_timer.py
--------------------------------------------
Background task: monitors MEDIUM incidents for review timeout.
If a MEDIUM incident is not reviewed within escalation_timeout_seconds,
it is automatically escalated to HIGH.

Runs as an asyncio task every 30 seconds.
Escalation writes a new Alert record with reason="AUTO_ESCALATED_TIMEOUT".
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

from core.config import get_settings
from core.constants import IncidentStatus, Severity, WSEventType
from core.logger import get_logger

logger = get_logger(__name__)

CHECK_INTERVAL_SECONDS = 30   # How often the timer loop checks for overdue incidents


@dataclass
class EscalationEntry:
    """Tracks one MEDIUM incident awaiting review."""
    incident_id: str
    camera_id: str
    event_type: str
    registered_at: float = field(default_factory=time.time)
    timeout_seconds: float = 300.0

    @property
    def is_overdue(self) -> bool:
        return time.time() - self.registered_at > self.timeout_seconds


class EscalationTimer:
    """
    Maintains a registry of open MEDIUM incidents.
    Background loop checks for overdue incidents and escalates them.
    """

    def __init__(self):
        self._registry: Dict[str, EscalationEntry] = {}  # incident_id → entry
        settings = get_settings()
        # Default timeout from settings; individual incidents may override
        self._default_timeout = 300.0

    def register(
        self,
        incident_id: str,
        camera_id: str,
        event_type: str,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        """Register a MEDIUM incident for escalation monitoring."""
        self._registry[incident_id] = EscalationEntry(
            incident_id=incident_id,
            camera_id=camera_id,
            event_type=event_type,
            timeout_seconds=timeout_seconds or self._default_timeout,
        )
        logger.info(
            "escalation_registered",
            incident_id=incident_id,
            timeout_seconds=timeout_seconds or self._default_timeout,
        )

    def deregister(self, incident_id: str) -> None:
        """Remove from registry when incident is reviewed."""
        self._registry.pop(incident_id, None)

    async def run(self) -> None:
        """
        Background loop. Runs forever until cancelled.
        Called as asyncio.create_task(timer.run()).
        """
        logger.info("escalation_timer_started")
        while True:
            try:
                await asyncio.sleep(CHECK_INTERVAL_SECONDS)
                await self._check_overdue()
            except asyncio.CancelledError:
                logger.info("escalation_timer_stopped")
                raise
            except Exception as exc:
                logger.error("escalation_timer_error", error=str(exc))

    async def _check_overdue(self) -> None:
        overdue = [
            entry for entry in self._registry.values()
            if entry.is_overdue
        ]
        if not overdue:
            return

        logger.info("escalation_check_overdue", count=len(overdue))

        for entry in overdue:
            await self._escalate(entry)
            self.deregister(entry.incident_id)

    async def _escalate(self, entry: EscalationEntry) -> None:
        """Escalate a MEDIUM incident to HIGH."""
        from db.session import get_session_factory
        from db.models.incident import Incident
        from db.models.alert import Alert
        from db.models.audit_log import AuditLog
        from sqlalchemy import select

        logger.warning(
            "auto_escalating_incident",
            incident_id=entry.incident_id,
            camera_id=entry.camera_id,
            event_type=entry.event_type,
            overdue_by_seconds=round(
                time.time() - entry.registered_at - entry.timeout_seconds, 1
            ),
        )

        factory = get_session_factory()
        try:
            async with factory() as db:
                result = await db.execute(
                    select(Incident).where(
                        Incident.id == uuid.UUID(entry.incident_id)
                    )
                )
                incident = result.scalar_one_or_none()

                if incident is None:
                    return

                # Only escalate if still unreviewed
                if incident.status not in (
                    IncidentStatus.ACTIVE_REVIEW,
                    IncidentStatus.PENDING_REVIEW,
                ):
                    return

                old_severity = incident.severity
                incident.severity = Severity.HIGH.value
                incident.status = IncidentStatus.EMERGENCY.value
                incident.is_locked = True

                # Create escalation alert record
                escalation_alert = Alert(
                    incident_id=incident.id,
                    severity=Severity.HIGH.value,
                    triggered_at=datetime.now(tz=timezone.utc),
                    escalation_reason="AUTO_ESCALATED_TIMEOUT",
                )
                db.add(escalation_alert)

                # Audit
                audit = AuditLog(
                    actor_type="SYSTEM",
                    action="SYSTEM_AUTO_ESCALATED",
                    target_type="incident",
                    target_id=incident.id,
                    detail={
                        "previous_severity": old_severity,
                        "new_severity": Severity.HIGH.value,
                        "reason": "unreviewed_timeout",
                        "overdue_seconds": round(
                            time.time() - entry.registered_at - entry.timeout_seconds
                        ),
                    },
                )
                db.add(audit)
                await db.commit()

        except Exception as exc:
            logger.error("escalation_db_error", incident_id=entry.incident_id, error=str(exc))
            return

        # Push WebSocket severity-changed event
        await self._push_escalation_event(entry, old_severity=old_severity)

        # Trigger emergency notifications
        await self._notify_escalation(entry)

    async def _push_escalation_event(self, entry: EscalationEntry, old_severity: str) -> None:
        from api.schemas.ws_events import AlertSeverityChangedPayload, make_event
        from api.websocket_manager import get_ws_manager
        from datetime import datetime, timezone

        try:
            ws_manager = get_ws_manager()
            payload = AlertSeverityChangedPayload(
                incident_id=uuid.UUID(entry.incident_id),
                previous_severity=Severity(old_severity),
                new_severity=Severity.HIGH,
                reason="AUTO_ESCALATED_TIMEOUT",
                changed_at=datetime.now(tz=timezone.utc),
            )
            event = make_event(WSEventType.ALERT_SEVERITY_CHANGED, payload)
            await ws_manager.broadcast(event.model_dump(mode="json"))
        except Exception as exc:
            logger.warning("escalation_ws_push_failed", error=str(exc))

    async def _notify_escalation(self, entry: EscalationEntry) -> None:
        from services.notification.webhook_sender import send_webhook
        from services.notification.email_sender import send_alert_email

        payload = {
            "incident_id": entry.incident_id,
            "camera_id": entry.camera_id,
            "event_type": entry.event_type,
            "severity": "HIGH",
            "reason": "AUTO_ESCALATED — no reviewer response within timeout",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        asyncio.create_task(send_webhook(payload))
        asyncio.create_task(send_alert_email(payload))


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_timer: Optional[EscalationTimer] = None


def get_escalation_timer() -> EscalationTimer:
    global _timer
    if _timer is None:
        _timer = EscalationTimer()
    return _timer
