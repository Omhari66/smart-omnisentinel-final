"""
services/notification/sms_sender.py
--------------------------------------
Twilio SMS sender for HIGH-severity emergency alerts.
This is a V2 feature — disabled by default (NOTIFICATIONS__TWILIO_ENABLED=false).

Activation:
    Set in .env:
        NOTIFICATIONS__TWILIO_ENABLED=true
        NOTIFICATIONS__TWILIO_ACCOUNT_SID=ACxxxxxxxx
        NOTIFICATIONS__TWILIO_AUTH_TOKEN=xxxxxxxx
        NOTIFICATIONS__TWILIO_FROM_NUMBER=+15551234567
        NOTIFICATIONS__TWILIO_TO_NUMBERS=["+15559876543", "+15557654321"]

SMS is reserved for HIGH-severity only.
Character limit is enforced (160 chars) to avoid multi-part SMS charges.
"""

from __future__ import annotations

from typing import Any, Dict, List

from core.config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)

MAX_SMS_CHARS = 160


def _build_sms_body(payload: Dict[str, Any]) -> str:
    """
    Build a concise SMS alert message under 160 characters.
    Truncates gracefully if fields are long.
    """
    severity = payload.get("severity", "ALERT")
    event_type = payload.get("event_type", "INCIDENT")
    camera_id = payload.get("camera_id", "")[:20]
    risk = payload.get("risk_score", "?")
    ts = str(payload.get("timestamp", ""))[:16]  # "2024-03-15T02:34"

    msg = f"[{severity}] SmartOmniSentinel: {event_type} detected. Camera: {camera_id} Risk: {risk} At: {ts}"

    if len(msg) > MAX_SMS_CHARS:
        msg = msg[:MAX_SMS_CHARS - 3] + "..."

    return msg


async def send_alert_sms(payload: Dict[str, Any]) -> None:
    """
    Send SMS alert to all configured emergency contacts.
    Silently skips if Twilio is not enabled or not configured.
    """
    settings = get_settings()
    notif = settings.notifications

    if not notif.twilio_enabled:
        logger.debug("sms_skipped_not_enabled")
        return

    if not all([notif.twilio_account_sid, notif.twilio_auth_token,
                notif.twilio_from_number, notif.twilio_to_numbers]):
        logger.warning("sms_skipped_incomplete_config")
        return

    body = _build_sms_body(payload)

    # Twilio does not have an official async client.
    # We run the synchronous call in a thread pool to avoid blocking.
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        _send_sync,
        notif.twilio_account_sid,
        notif.twilio_auth_token,
        notif.twilio_from_number,
        notif.twilio_to_numbers,
        body,
        payload,
    )


def _send_sync(
    account_sid: str,
    auth_token: str,
    from_number: str,
    to_numbers: List[str],
    body: str,
    payload: Dict[str, Any],
) -> None:
    """Synchronous Twilio dispatch — runs in thread pool executor."""
    try:
        from twilio.rest import Client  # type: ignore[import]
    except ImportError:
        logger.error(
            "twilio_not_installed",
            hint="pip install twilio",
        )
        return

    client = Client(account_sid, auth_token)

    for to_number in to_numbers:
        try:
            message = client.messages.create(
                body=body,
                from_=from_number,
                to=to_number,
            )
            logger.info(
                "sms_sent",
                to=to_number,
                sid=message.sid,
                incident_id=payload.get("incident_id"),
            )
        except Exception as exc:
            logger.error(
                "sms_failed",
                to=to_number,
                error=str(exc),
                incident_id=payload.get("incident_id"),
            )
