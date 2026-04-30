"""
api/routes/alerts.py
---------------------
Alert history, escalation logs, and manual escalation endpoint.
Alerts are created automatically by the alert_dispatcher.
This router provides READ access and the manual escalation action.
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
from core.constants import Severity, UserRole
from core.logger import get_logger
from db.models.alert import Alert
from db.models.audit_log import AuditLog
from db.models.incident import Incident
from pydantic import BaseModel

router = APIRouter(prefix="/alerts", tags=["alerts"])
logger = get_logger(__name__)


class AlertResponse(BaseModel):
    id: uuid.UUID
    incident_id: uuid.UUID
    severity: str
    triggered_at: datetime
    notification_channels: Optional[list]
    cooldown_expires_at: Optional[datetime]
    escalated_from_id: Optional[uuid.UUID]
    escalation_reason: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class EscalationRequest(BaseModel):
    reason: str
    notify_authorities: bool = False


class EscalationResponse(BaseModel):
    alert_id: uuid.UUID
    incident_id: uuid.UUID
    escalated_at: datetime
    reason: str


@router.get("", response_model=PaginatedResponse[AlertResponse])
async def list_alerts(
    db: DBSession,
    current_user: CurrentUser,
    pagination: Pagination,
    severity: Optional[Severity] = Query(default=None),
    incident_id: Optional[uuid.UUID] = Query(default=None),
    from_dt: Optional[datetime] = Query(default=None, alias="from"),
    to_dt: Optional[datetime] = Query(default=None, alias="to"),
):
    """
    Paginated alert history with filters.
    Returns alerts newest-first.
    """
    clauses = []
    if severity:
        clauses.append(Alert.severity == severity.value)
    if incident_id:
        clauses.append(Alert.incident_id == incident_id)
    if from_dt:
        clauses.append(Alert.triggered_at >= from_dt)
    if to_dt:
        clauses.append(Alert.triggered_at <= to_dt)

    base_query = select(Alert)
    if clauses:
        base_query = base_query.where(and_(*clauses))

    total_result = await db.execute(
        select(func.count()).select_from(base_query.subquery())
    )
    total = total_result.scalar_one()

    result = await db.execute(
        base_query
        .order_by(Alert.triggered_at.desc())
        .offset(pagination.offset)
        .limit(pagination.page_size)
    )
    alerts = list(result.scalars().all())

    return PaginatedResponse[AlertResponse].build(
        items=[AlertResponse.model_validate(a) for a in alerts],
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
    )


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(
    alert_id: uuid.UUID,
    db: DBSession,
    current_user: CurrentUser,
):
    """Full alert detail including escalation chain."""
    result = await db.execute(
        select(Alert).where(Alert.id == alert_id)
    )
    alert = result.scalar_one_or_none()
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found.")
    return AlertResponse.model_validate(alert)


@router.get("/escalations/history")
async def get_escalation_history(
    db: DBSession,
    pagination: Pagination,
    current_user: User = Depends(require_role(UserRole.SUPERVISOR)),
):
    """
    All auto-escalation events (MEDIUM → HIGH timeout escalations).
    Ordered newest-first.
    """
    result = await db.execute(
        select(Alert)
        .where(Alert.escalation_reason.isnot(None))
        .order_by(Alert.triggered_at.desc())
        .offset(pagination.offset)
        .limit(pagination.page_size)
    )
    escalations = list(result.scalars().all())
    return [AlertResponse.model_validate(a) for a in escalations]


@router.post("/{alert_id}/escalate", response_model=EscalationResponse)
async def manual_escalate(
    alert_id: uuid.UUID,
    body: EscalationRequest,
    db: DBSession,
    current_user: User = Depends(require_role(UserRole.SUPERVISOR)),
):
    """
    Manual escalation of an alert to authorities.
    Requires SUPERVISOR role. Writes to audit log.
    Does NOT automatically contact authorities — that requires human confirmation
    of the notification action through external integrations.
    """
    result = await db.execute(
        select(Alert).where(Alert.id == alert_id)
    )
    alert = result.scalar_one_or_none()
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found.")

    # Lock the associated incident
    inc_result = await db.execute(
        select(Incident).where(Incident.id == alert.incident_id)
    )
    incident = inc_result.scalar_one_or_none()
    if incident:
        incident.is_locked = True
        incident.severity = Severity.HIGH.value

    # Create escalation record
    escalation_alert = Alert(
        incident_id=alert.incident_id,
        severity=Severity.HIGH.value,
        triggered_at=datetime.now(tz=timezone.utc),
        escalated_from_id=alert_id,
        escalation_reason=f"MANUAL_ESCALATION: {body.reason}",
    )
    db.add(escalation_alert)

    # Audit
    audit = AuditLog(
        actor_id=current_user.id,
        actor_type="USER",
        action="ALERT_ESCALATED",
        target_type="alert",
        target_id=alert_id,
        detail={
            "reason": body.reason,
            "notify_authorities": body.notify_authorities,
            "escalated_by": current_user.email,
        },
    )
    db.add(audit)
    await db.flush()

    logger.warning(
        "manual_escalation",
        alert_id=str(alert_id),
        incident_id=str(alert.incident_id),
        escalated_by=current_user.email,
        reason=body.reason,
    )

    # Fire notifications if requested
    if body.notify_authorities:
        from services.notification.webhook_sender import send_webhook
        import asyncio
        asyncio.create_task(send_webhook({
            "type": "MANUAL_ESCALATION",
            "alert_id": str(alert_id),
            "incident_id": str(alert.incident_id),
            "reason": body.reason,
            "escalated_by": current_user.email,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }))

    return EscalationResponse(
        alert_id=escalation_alert.id,
        incident_id=alert.incident_id,
        escalated_at=escalation_alert.triggered_at,
        reason=body.reason,
    )
