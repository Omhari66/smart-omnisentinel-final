"""
services/alert_manager/audit_writer.py
----------------------------------------
Centralized function for writing to the append-only audit_log table.
All system and user actions touching incidents, clips, or config
must call write_audit() — never write to audit_log directly elsewhere.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from core.logger import get_logger

logger = get_logger(__name__)


async def write_audit(
    action: str,
    target_type: str,
    actor_type: str = "SYSTEM",
    actor_id: Optional[uuid.UUID] = None,
    target_id: Optional[uuid.UUID] = None,
    detail: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
) -> None:
    """
    Write one record to the audit_log table.
    Fire-and-forget style: logs errors but does not raise.
    Audit failures must never block the alert pipeline.
    """
    from db.models.audit_log import AuditLog
    from db.session import get_session_factory

    factory = get_session_factory()
    try:
        async with factory() as db:
            entry = AuditLog(
                actor_id=actor_id,
                actor_type=actor_type,
                action=action,
                target_type=target_type,
                target_id=target_id,
                detail=detail or {},
                ip_address=ip_address,
            )
            db.add(entry)
            await db.commit()
    except Exception as exc:
        # Log but never raise — audit failures must not disrupt the alert pipeline
        logger.error(
            "audit_write_failed",
            action=action,
            target_type=target_type,
            target_id=str(target_id) if target_id else None,
            error=str(exc),
        )
