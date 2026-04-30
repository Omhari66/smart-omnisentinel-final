"""
services/review_service/false_positive_logger.py
--------------------------------------------------
Logs false positive feedback to a dedicated table/file for model retraining.
When a reviewer marks an incident as FALSE_POSITIVE, the event metadata
is stored so the training pipeline can use it as a hard negative.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

from core.config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)

FP_LOG_FILENAME = "false_positives.jsonl"


async def log_false_positive(
    incident_id: str,
    camera_id: str,
    event_type: str,
    confidence_at_alert: float,
    risk_score: int,
    reviewer_notes: str | None,
) -> None:
    """
    Append a false positive record to the FP log file.
    This file is consumed by the training pipeline for hard negative mining.
    """
    settings = get_settings()
    log_path = os.path.join(settings.storage.evidence_root, FP_LOG_FILENAME)

    record = {
        "incident_id": incident_id,
        "camera_id": camera_id,
        "event_type": event_type,
        "confidence_at_alert": confidence_at_alert,
        "risk_score": risk_score,
        "reviewer_notes": reviewer_notes,
        "logged_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(
            "false_positive_logged",
            incident_id=incident_id,
            event_type=event_type,
            confidence=confidence_at_alert,
        )
    except Exception as exc:
        logger.error("false_positive_log_failed", error=str(exc))
