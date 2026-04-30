"""
services/review_service/review_queue.py
-----------------------------------------
Manages the ordered review queue for human-in-the-loop verification.
Incidents are sorted by: severity tier (HIGH first), then risk_score desc,
then detected_at asc (oldest first within same severity).
"""
from __future__ import annotations

from typing import List

from core.constants import IncidentStatus, Severity
from core.logger import get_logger

logger = get_logger(__name__)

# Severity sort order (lower index = higher priority)
_SEVERITY_ORDER = {
    Severity.HIGH.value: 0,
    Severity.MEDIUM.value: 1,
    Severity.LOW.value: 2,
}


async def get_prioritized_queue(db) -> list:
    """
    Fetch incidents pending review, sorted by priority.
    Returns ORM Incident objects in priority order.
    """
    from db.models.incident import Incident
    from sqlalchemy import select

    result = await db.execute(
        select(Incident)
        .where(Incident.status.in_([
            IncidentStatus.PENDING_REVIEW.value,
            IncidentStatus.ACTIVE_REVIEW.value,
        ]))
    )
    incidents = list(result.scalars().all())

    # Sort: severity tier first, then risk_score desc, then detected_at asc
    def sort_key(inc):
        severity_rank = _SEVERITY_ORDER.get(inc.severity, 99)
        return (severity_rank, -inc.risk_score, inc.detected_at)

    incidents.sort(key=sort_key)
    return incidents
