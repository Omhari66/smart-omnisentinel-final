"""
services/notification/email_sender.py
---------------------------------------
Async SMTP email sender for alert notifications.
Uses aiosmtplib for non-blocking email delivery.
Gracefully no-ops if SMTP is not configured.
"""

from __future__ import annotations

import json
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List

from core.config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)


def _build_email_body(payload: Dict[str, Any]) -> str:
    """Build plain-text email body from incident payload."""
    lines = [
        "SmartOmniSentinel — Incident Alert",
        "=" * 40,
        f"Incident ID:  {payload.get('incident_id', 'N/A')}",
        f"Severity:     {payload.get('severity', 'N/A')}",
        f"Event Type:   {payload.get('event_type', 'N/A')}",
        f"Camera:       {payload.get('camera_id', 'N/A')}",
        f"Risk Score:   {payload.get('risk_score', 'N/A')}",
        f"Confidence:   {payload.get('confidence', 'N/A')}",
        f"People Count: {payload.get('people_count', 'N/A')}",
        f"Timestamp:    {payload.get('timestamp', 'N/A')}",
        "",
        "Log in to the dashboard to review this incident.",
        "",
        "This is an automated message from SmartOmniSentinel.",
    ]
    return "\n".join(lines)


async def send_alert_email(payload: Dict[str, Any]) -> None:
    """
    Send alert email to all configured recipients.
    Silently skips if SMTP is not configured.
    """
    settings = get_settings()
    notif = settings.notifications

    if not notif.smtp_host or not notif.alert_recipients:
        logger.debug("email_skipped_not_configured")
        return

    severity = payload.get("severity", "ALERT")
    subject = f"[{severity}] SmartOmniSentinel Incident — {payload.get('event_type', '')}"
    body = _build_email_body(payload)

    try:
        import aiosmtplib

        for recipient in notif.alert_recipients:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = notif.smtp_from
            msg["To"] = recipient
            msg.attach(MIMEText(body, "plain"))

            await aiosmtplib.send(
                msg,
                hostname=notif.smtp_host,
                port=notif.smtp_port,
                username=notif.smtp_user or None,
                password=notif.smtp_password or None,
                start_tls=True,
            )
            logger.info(
                "alert_email_sent",
                to=recipient,
                severity=severity,
                incident_id=payload.get("incident_id"),
            )

    except Exception as exc:
        logger.error(
            "alert_email_failed",
            error=str(exc),
            incident_id=payload.get("incident_id"),
        )
