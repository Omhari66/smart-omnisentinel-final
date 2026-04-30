"""
api/routes/review.py
--------------------
Human-in-the-loop review workflow.
Guards and supervisors confirm, dismiss, or escalate incidents.
All actions are logged to the audit trail.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from api.dependencies import CurrentUser, DBSession, Pagination, require_role
from api.schemas.common import PaginatedResponse
from api.schemas.review import ReviewActionRequest, ReviewActionResponse, ReviewQueueItem
from core.constants import (
    IncidentStatus,
    ReviewAction,
    Severity,
    UserRole,
    WSEventType,
)
from core.logger import get_logger
from db.models.audit_log import AuditLog
from db.models.incident import Incident
from db.models.review_action import ReviewActionRecord
from db.models.user import User

router = APIRouter(prefix="/review", tags=["review"])
logger = get_logger(__name__)


@router.get("/queue", response_model=List[ReviewQueueItem])
async def get_review_queue(
    db: DBSession,
    current_user: User = Depends(require_role(UserRole.GUARD)),
):
    """
    Returns incidents awaiting human review, ordered by severity then detected_at.
    MEDIUM incidents show escalation deadline.
    """
    result = await db.execute(
        select(Incident)
        .where(
            Incident.status.in_([
                IncidentStatus.PENDING_REVIEW,
                IncidentStatus.ACTIVE_REVIEW,
            ])
        )
        .options(selectinload(Incident.camera))
        .order_by(
            # HIGH/EMERGENCY first, then MEDIUM, then LOW
            Incident.risk_score.desc(),
            Incident.detected_at.asc(),
        )
    )
    incidents = list(result.scalars().all())

    items: List[ReviewQueueItem] = []
    for idx, inc in enumerate(incidents, start=1):
        # Escalation deadline: detected_at + escalation_timeout (default 5 min)
        escalation_deadline = None
        if inc.severity == Severity.MEDIUM and inc.camera.threshold_profile:
            timeout = inc.camera.threshold_profile.escalation_timeout_seconds
            from datetime import timedelta
            escalation_deadline = inc.detected_at + timedelta(seconds=timeout)

        items.append(
            ReviewQueueItem(
                incident_id=inc.id,
                camera_name=inc.camera.name if inc.camera else "Unknown",
                location=inc.camera.location if inc.camera else "",
                event_type=inc.event_type,
                severity=inc.severity,
                risk_score=inc.risk_score,
                detected_at=inc.detected_at,
                escalation_deadline=escalation_deadline,
                queue_position=idx,
            )
        )
    return items


@router.post("/{incident_id}/action", response_model=ReviewActionResponse, status_code=201)
async def submit_review_action(
    incident_id: uuid.UUID,
    body: ReviewActionRequest,
    background_tasks: BackgroundTasks,
    db: DBSession,
    current_user: User = Depends(require_role(UserRole.GUARD)),
):
    """
    Submit a review decision for an incident.
    Updates incident status and writes to review_actions and audit_log.
    """
    result = await db.execute(
        select(Incident).options(selectinload(Incident.camera)).where(Incident.id == incident_id)
    )
    incident = result.scalar_one_or_none()
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found.")

    # Allow changing status during demo (removed 409 restriction)

    # Map action → new status
    status_map = {
        ReviewAction.CONFIRM: IncidentStatus.EMERGENCY,
        ReviewAction.DISMISS: IncidentStatus.RESOLVED,
        ReviewAction.ESCALATE: IncidentStatus.EMERGENCY,
        ReviewAction.FALSE_POSITIVE: IncidentStatus.FALSE_POSITIVE,
    }

    new_status = status_map[body.action]
    incident.status = new_status
    incident.reviewer_id = current_user.id
    incident.reviewed_at = datetime.now(tz=timezone.utc)
    incident.review_action = body.action

    # Lock evidence on CONFIRM or ESCALATE
    if body.action in (ReviewAction.CONFIRM, ReviewAction.ESCALATE):
        incident.is_locked = True
        
    # Dispatch real-world SMS/Email alerts on ESCALATE
    if body.action == ReviewAction.ESCALATE:
        from services.alert_manager.notifier import NotificationService
        notifier = NotificationService()
        background_tasks.add_task(notifier.dispatch_emergency_alert, incident)

    # Record the review action
    review_record = ReviewActionRecord(
        incident_id=incident.id,
        reviewer_id=current_user.id,
        action=body.action,
        notes=body.notes,
        confidence_override=body.confidence_override,
    )
    db.add(review_record)

    # Write audit log
    audit = AuditLog(
        actor_id=current_user.id,
        actor_type="USER",
        action="INCIDENT_REVIEWED",
        target_type="incident",
        target_id=incident.id,
        detail={
            "action": body.action,
            "new_status": new_status,
            "notes": body.notes,
        },
    )
    db.add(audit)
    await db.flush()

    # Push WebSocket event
    from api.schemas.ws_events import IncidentResolvedPayload, WSEventEnvelope, make_event
    from api.websocket_manager import get_ws_manager
    ws_manager = get_ws_manager()
    event = make_event(
        WSEventType.INCIDENT_RESOLVED,
        IncidentResolvedPayload(
            incident_id=incident.id,
            resolution=body.action,
            resolved_by=current_user.email,
            resolved_at=incident.reviewed_at,
        ),
    )
    await ws_manager.broadcast(event.model_dump(mode="json"))

    logger.info(
        "incident_reviewed",
        incident_id=str(incident_id),
        action=body.action,
        reviewer=current_user.email,
    )
    return ReviewActionResponse.model_validate(review_record)


@router.get("/history", response_model=PaginatedResponse[ReviewActionResponse])
async def get_review_history(
    db: DBSession,
    pagination: Pagination,
    current_user: User = Depends(require_role(UserRole.SUPERVISOR)),
):
    """All completed review actions, newest first."""
    from sqlalchemy import func

    total_result = await db.execute(
        select(func.count()).select_from(ReviewActionRecord)
    )
    total = total_result.scalar_one()

    result = await db.execute(
        select(ReviewActionRecord)
        .order_by(ReviewActionRecord.created_at.desc())
        .offset(pagination.offset)
        .limit(pagination.page_size)
    )
    records = list(result.scalars().all())

    return PaginatedResponse[ReviewActionResponse].build(
        items=[ReviewActionResponse.model_validate(r) for r in records],
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
    )
