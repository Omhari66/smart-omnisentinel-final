"""
services/notification/webhook_sender.py
-----------------------------------------
HTTP POST webhook sender for external alert integrations.
Sends to all configured webhook URLs with a standard JSON payload.
Retries once on failure; then logs and gives up (fire-and-forget).
"""

from __future__ import annotations

from typing import Any, Dict

import httpx

from core.config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)

WEBHOOK_TIMEOUT_SECONDS = 5
MAX_RETRIES = 1


async def send_webhook(payload: Dict[str, Any]) -> None:
    """
    POST payload to all configured webhook URLs.
    Silently skips if no URLs configured.
    """
    settings = get_settings()
    urls = settings.notifications.webhook_urls

    if not urls:
        logger.debug("webhook_skipped_no_urls")
        return

    async with httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT_SECONDS) as client:
        for url in urls:
            for attempt in range(MAX_RETRIES + 1):
                try:
                    response = await client.post(
                        url,
                        json=payload,
                        headers={
                            "Content-Type": "application/json",
                            "X-Sentinel-Source": "SmartOmniSentinel",
                        },
                    )
                    response.raise_for_status()
                    logger.info(
                        "webhook_sent",
                        url=url,
                        status=response.status_code,
                        incident_id=payload.get("incident_id"),
                    )
                    break
                except Exception as exc:
                    if attempt < MAX_RETRIES:
                        logger.warning(
                            "webhook_retry",
                            url=url,
                            attempt=attempt + 1,
                            error=str(exc),
                        )
                    else:
                        logger.error(
                            "webhook_failed",
                            url=url,
                            error=str(exc),
                            incident_id=payload.get("incident_id"),
                        )
