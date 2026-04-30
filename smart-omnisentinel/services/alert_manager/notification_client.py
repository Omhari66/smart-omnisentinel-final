"""
services/alert_manager/notification_client.py
----------------------------------------------
Unified notification dispatcher.
Routes alert payloads to configured channels (email, webhook, SMS).
Called by alert handlers and escalation timer.

Design principle: every notification is fire-and-forget.
Failures are logged but never allowed to block the alert pipeline.
All dispatches run as asyncio tasks.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from core.config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)


class NotificationClient:
    """
    Central hub for all outbound notifications.
    Selects and dispatches to configured channels based on severity.
    """

    def __init__(self):
        self._settings = get_settings()

    async def notify(
        self,
        payload: Dict[str, Any],
        severity: str,
        channels: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Send notification to appropriate channels.

        Args:
            payload:  Incident/alert data dict to send
            severity: "LOW" | "MEDIUM" | "HIGH"
            channels: Override channel list (default = auto-select by severity)

        Returns:
            List of channel names actually dispatched to.
        """
        if channels is None:
            channels = self._select_channels(severity)

        dispatched = []
        tasks = []

        for channel in channels:
            if channel == "email":
                tasks.append(self._dispatch_email(payload))
                dispatched.append("email")
            elif channel == "webhook":
                tasks.append(self._dispatch_webhook(payload))
                dispatched.append("webhook")
            elif channel == "sms":
                tasks.append(self._dispatch_sms(payload))
                dispatched.append("sms")

        if tasks:
            # Fire-and-forget: wrap in create_task so caller is not blocked
            for task_coro in tasks:
                asyncio.create_task(
                    self._safe_dispatch(task_coro, payload),
                    name=f"notify_{severity}_{payload.get('incident_id', 'unknown')[:8]}",
                )

        logger.info(
            "notifications_dispatched",
            severity=severity,
            channels=dispatched,
            incident_id=payload.get("incident_id"),
        )
        return dispatched

    def _select_channels(self, severity: str) -> List[str]:
        """
        Determine which channels to use based on severity.

        LOW    → no external notification (review queue only)
        MEDIUM → webhook (security room system)
        HIGH   → webhook + email + SMS (if configured)
        """
        if severity == "LOW":
            return []
        elif severity == "MEDIUM":
            channels = ["webhook"]
            if self._settings.notifications.smtp_host:
                channels.append("email")
            return channels
        else:  # HIGH
            channels = ["webhook", "email"]
            if self._settings.notifications.twilio_enabled:
                channels.append("sms")
            return channels

    async def _dispatch_email(self, payload: Dict[str, Any]) -> None:
        from services.notification.email_sender import send_alert_email
        await send_alert_email(payload)

    async def _dispatch_webhook(self, payload: Dict[str, Any]) -> None:
        from services.notification.webhook_sender import send_webhook
        await send_webhook(payload)

    async def _dispatch_sms(self, payload: Dict[str, Any]) -> None:
        from services.notification.sms_sender import send_alert_sms
        await send_alert_sms(payload)

    @staticmethod
    async def _safe_dispatch(coro, payload: Dict[str, Any]) -> None:
        """Wrap coroutine to catch and log failures without propagating."""
        try:
            await coro
        except Exception as exc:
            logger.error(
                "notification_dispatch_failed",
                error=str(exc),
                incident_id=payload.get("incident_id"),
            )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_client: Optional[NotificationClient] = None


def get_notification_client() -> NotificationClient:
    global _client
    if _client is None:
        _client = NotificationClient()
    return _client
