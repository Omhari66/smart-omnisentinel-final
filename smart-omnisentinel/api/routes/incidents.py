"""
api/routes/incidents.py
-----------------------
Incident listing, detail retrieval, and analytics endpoints.
Write operations on incidents happen through review routes only.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, func, select
from sqlalchemy.orm import selectinload

from api.dependencies import CurrentUser, DBSession, Pagination, require_role
from api.schemas.common import PaginatedResponse
from api.schemas.incident import (
    IncidentAnalyticsSummary,
    IncidentResponse,
    IncidentSummary,
)
from core.constants import EventType, IncidentStatus, ReviewAction, Severity, UserRole
from core.logger import get_logger
from db.models.incident import Incident

router = APIRouter(prefix="/incidents", tags=["incidents"])
logger = get_logger(__name__)


def _build_filter_clauses(
    camera_id: Optional[uuid.UUID],
    severity: Optional[List[Severity]],
    status: Optional[List[IncidentStatus]],
    event_type: Optional[List[EventType]],
    from_dt: Optional[datetime],
    to_dt: Optional[datetime],
):
    """Build SQLAlchemy WHERE clauses from query parameters."""
    clauses = []
    if camera_id:
        clauses.append(Incident.camera_id == camera_id)
    if severity:
        clauses.append(Incident.severity.in_([s.value for s in severity]))
    if status:
        clauses.append(Incident.status.in_([s.value for s in status]))
    if event_type:
        clauses.append(Incident.event_type.in_([e.value for e in event_type]))
    if from_dt:
        clauses.append(Incident.detected_at >= from_dt)
    if to_dt:
        clauses.append(Incident.detected_at <= to_dt)
    return clauses


@router.get("", response_model=PaginatedResponse[IncidentSummary])
async def list_incidents(
    db: DBSession,
    current_user: CurrentUser,
    pagination: Pagination,
    camera_id: Optional[uuid.UUID] = Query(default=None),
    severity: Optional[List[Severity]] = Query(default=None),
    status: Optional[List[IncidentStatus]] = Query(default=None),
    event_type: Optional[List[EventType]] = Query(default=None),
    from_dt: Optional[datetime] = Query(default=None, alias="from"),
    to_dt: Optional[datetime] = Query(default=None, alias="to"),
):
    """
    Paginated incident list with multi-field filtering.
    Returns lightweight IncidentSummary (no evidence/review sub-objects).
    """
    clauses = _build_filter_clauses(
        camera_id, severity, status, event_type, from_dt, to_dt
    )
    base_query = select(Incident)
    if clauses:
        base_query = base_query.where(and_(*clauses))

    # Count
    count_result = await db.execute(
        select(func.count()).select_from(base_query.subquery())
    )
    total = count_result.scalar_one()

    # Paginated results
    result = await db.execute(
        base_query
        .order_by(Incident.detected_at.desc())
        .offset(pagination.offset)
        .limit(pagination.page_size)
    )
    items = list(result.scalars().all())

    return PaginatedResponse[IncidentSummary].build(
        items=[IncidentSummary.model_validate(i) for i in items],
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
    )


@router.get("/analytics/summary", response_model=IncidentAnalyticsSummary)
async def get_analytics_summary(
    db: DBSession,
    current_user: User = Depends(require_role(UserRole.VIEWER)),
    camera_id: Optional[uuid.UUID] = Query(default=None),
    from_dt: Optional[datetime] = Query(
        default=None,
        alias="from",
        description="ISO datetime, defaults to 30 days ago",
    ),
    to_dt: Optional[datetime] = Query(default=None, alias="to"),
):
    """Aggregated analytics: counts by severity, event type, status, FP rate."""
    from datetime import timedelta

    now = datetime.now(tz=timezone.utc)
    period_from = from_dt or (now - timedelta(days=30))
    period_to = to_dt or now

    clauses = [
        Incident.detected_at >= period_from,
        Incident.detected_at <= period_to,
    ]
    if camera_id:
        clauses.append(Incident.camera_id == camera_id)

    base_query = select(Incident).where(and_(*clauses))
    result = await db.execute(base_query)
    incidents = list(result.scalars().all())

    total = len(incidents)
    by_severity: dict = {s.value: 0 for s in Severity}
    by_event_type: dict = {e.value: 0 for e in EventType}
    by_status: dict = {s.value: 0 for s in IncidentStatus}
    fp_count = 0

    for inc in incidents:
        by_severity[inc.severity] = by_severity.get(inc.severity, 0) + 1
        by_event_type[inc.event_type] = by_event_type.get(inc.event_type, 0) + 1
        by_status[inc.status] = by_status.get(inc.status, 0) + 1
        if inc.review_action == ReviewAction.FALSE_POSITIVE:
            fp_count += 1

    reviewed = sum(
        1 for i in incidents if i.review_action is not None
    )
    fp_rate = round(fp_count / reviewed, 4) if reviewed > 0 else 0.0

    return IncidentAnalyticsSummary(
        total=total,
        by_severity=by_severity,
        by_event_type=by_event_type,
        by_status=by_status,
        false_positive_rate=fp_rate,
        period_from=period_from,
        period_to=period_to,
    )


@router.get("/{incident_id}", response_model=IncidentResponse)
async def get_incident(
    incident_id: uuid.UUID,
    db: DBSession,
    current_user: CurrentUser,
):
    """Full incident detail including evidence clips and review history."""
    result = await db.execute(
        select(Incident)
        .where(Incident.id == incident_id)
        .options(
            selectinload(Incident.evidence_clips),
            selectinload(Incident.review_records),
            selectinload(Incident.reviewer),
            selectinload(Incident.model_version),
        )
    )
    incident = result.scalar_one_or_none()
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found.")

    return IncidentResponse.model_validate(incident)
