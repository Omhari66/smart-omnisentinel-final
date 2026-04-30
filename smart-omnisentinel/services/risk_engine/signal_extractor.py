"""
services/risk_engine/signal_extractor.py
------------------------------------------
Extracts all contextual signals needed for risk scoring
by querying the database for camera zone config, recent alert history,
and event metadata.

Called by the risk engine pipeline before compute_risk_score().
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from core.logger import get_logger

logger = get_logger(__name__)

# How far back to look for "recent alerts" on same camera
RECENT_ALERT_WINDOW_MINUTES = 10


async def extract_signals(
    camera_id: str,
    event: dict,
) -> dict:
    """
    Fetch all contextual signals for one confirmed event.

    Returns a dict of signal values ready to build RiskSignals:
        zone_sensitivity, recent_alert, hour_of_day
    Other signals come directly from the event dict itself.
    """
    zone_sensitivity = await _get_zone_sensitivity(camera_id)
    recent_alert = await _check_recent_alert(camera_id, event.get("event_type", ""))
    hour_of_day = datetime.now(tz=timezone.utc).hour

    return {
        "zone_sensitivity": zone_sensitivity,
        "recent_alert": recent_alert,
        "hour_of_day": hour_of_day,
    }


async def _get_zone_sensitivity(camera_id: str) -> float:
    """Load zone sensitivity for the camera. Defaults to 0.7 on error."""
    try:
        from db.session import get_session_factory
        from db.models.camera import Camera
        from sqlalchemy import select

        factory = get_session_factory()
        async with factory() as db:
            result = await db.execute(
                select(Camera).where(Camera.id == UUID(camera_id))
            )
            camera = result.scalar_one_or_none()
            if camera and camera.zone:
                return float(camera.zone.sensitivity)
    except Exception as exc:
        logger.warning("zone_sensitivity_fetch_failed", camera_id=camera_id, error=str(exc))
    return 0.7  # Default


async def _check_recent_alert(camera_id: str, event_type: str) -> bool:
    """
    Returns True if this camera fired a HIGH alert in the last N minutes.
    Used to boost risk score for repeated incidents.
    """
    try:
        from db.session import get_session_factory
        from db.models.alert import Alert
        from db.models.incident import Incident
        from sqlalchemy import select, and_

        cutoff = datetime.now(tz=timezone.utc) - timedelta(
            minutes=RECENT_ALERT_WINDOW_MINUTES
        )
        factory = get_session_factory()
        async with factory() as db:
            result = await db.execute(
                select(Alert)
                .join(Alert.incident)
                .where(and_(
                    Incident.camera_id == UUID(camera_id),
                    Alert.triggered_at >= cutoff,
                    Alert.severity == "HIGH",
                ))
                .limit(1)
            )
            return result.scalar_one_or_none() is not None
    except Exception as exc:
        logger.warning("recent_alert_check_failed", camera_id=camera_id, error=str(exc))
    return False
