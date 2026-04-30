"""
services/risk_engine/severity_classifier.py
---------------------------------------------
Maps a numeric risk score to a severity tier.
Also provides the full risk pipeline orchestrator that:
  1. Extracts signals
  2. Computes risk score
  3. Publishes to 'scored_incidents' channel

This is the entry point called by the event bus consumer
listening on 'confirmed_events'.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from core.constants import Severity
from core.event_bus import get_event_bus
from core.logger import get_logger
from services.risk_engine.risk_scorer import (
    RiskSignals,
    build_signals_from_event,
    compute_risk_score,
)
from services.risk_engine.signal_extractor import extract_signals

logger = get_logger(__name__)


def score_to_severity(
    score: int,
    risk_low_max: int = 29,
    risk_medium_max: int = 64,
) -> Severity:
    """Map numeric risk score to severity tier."""
    if score <= risk_low_max:
        return Severity.LOW
    elif score <= risk_medium_max:
        return Severity.MEDIUM
    return Severity.HIGH


async def process_confirmed_event(message: dict[str, Any]) -> None:
    """
    Handler for messages on 'confirmed_events' channel.
    Full pipeline: extract signals → compute risk → publish scored_incident.
    """
    camera_id = message.get("camera_id", "")
    event_type = message.get("event_type", "")

    # Fetch contextual signals from DB/cache
    ctx = await extract_signals(camera_id=camera_id, event=message)

    # Build signals object
    signals = build_signals_from_event(
        event=message,
        zone_sensitivity=ctx.get("zone_sensitivity", 0.7),
        recent_alert=ctx.get("recent_alert", False),
        event_start_timestamp=message.get("event_start_timestamp"),
    )

    # Compute risk score
    risk = compute_risk_score(signals)

    # Build scored incident payload
    scored = {
        "incident_id": str(uuid.uuid4()),
        "camera_id": camera_id,
        "event_type": event_type,
        "severity": risk.severity.value,
        "risk_score": risk.score,
        "smoothed_confidence": signals.confidence,
        "raw_confidence": message.get("raw_confidence", signals.confidence),
        "people_count": signals.people_count,
        "motion_magnitude": signals.motion_magnitude,
        "event_start_timestamp": message.get("event_start_timestamp", time.time()),
        "timestamp": time.time(),
        "track_ids": message.get("track_ids", []),
        "factor_breakdown": risk.factor_breakdown,
    }

    bus = get_event_bus()
    await bus.publish("scored_incidents", scored)

    logger.info(
        "incident_scored",
        camera_id=camera_id,
        event_type=event_type,
        risk_score=risk.score,
        severity=risk.severity,
        confidence=round(signals.confidence, 3),
    )
