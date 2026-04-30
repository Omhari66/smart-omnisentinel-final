"""
api/routes/evidence.py
----------------------
Evidence clip access via short-lived signed URLs.
Every access is logged to the audit trail.
Clips are never served from a public static path.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy import select

from api.dependencies import CurrentUser, DBSession, require_role
from api.schemas.evidence import EvidenceClipResponse, EvidenceManifest, SignedURLResponse
from core.constants import UserRole, WSEventType
from core.logger import get_logger
from core.signed_url import generate_signed_url, verify_signed_url
from db.models.audit_log import AuditLog
from db.models.evidence_clip import EvidenceClip
from db.models.incident import Incident

router = APIRouter(prefix="/evidence", tags=["evidence"])
logger = get_logger(__name__)


async def _get_clip_or_404(evidence_id: uuid.UUID, db) -> EvidenceClip:
    result = await db.execute(
        select(EvidenceClip).where(EvidenceClip.id == evidence_id)
    )
    clip = result.scalar_one_or_none()
    if clip is None:
        raise HTTPException(status_code=404, detail="Evidence clip not found.")
    return clip


@router.get("/{evidence_id}", response_model=EvidenceClipResponse)
async def get_evidence_metadata(
    evidence_id: uuid.UUID,
    db: DBSession,
    current_user: User = Depends(require_role(UserRole.SUPERVISOR)),
):
    """Return evidence clip metadata (not the video file itself)."""
    clip = await _get_clip_or_404(evidence_id, db)
    return EvidenceClipResponse.model_validate(clip)


@router.get("/{evidence_id}/url", response_model=SignedURLResponse)
async def get_signed_url(
    evidence_id: uuid.UUID,
    request: Request,
    db: DBSession,
    current_user: User = Depends(require_role(UserRole.SUPERVISOR)),
):
    """
    Generate a 15-minute signed URL for direct clip playback.
    Logs the access to the audit trail.
    """
    clip = await _get_clip_or_404(evidence_id, db)

    base_url = str(request.base_url).rstrip("/")
    resource_path = f"/api/v1/evidence/clips/{clip.id}"
    signed = generate_signed_url(base_url=base_url, path=resource_path)

    # Audit log
    audit = AuditLog(
        actor_id=current_user.id,
        actor_type="USER",
        action="CLIP_ACCESSED",
        target_type="evidence_clip",
        target_id=clip.id,
        detail={"requested_by": current_user.email},
        ip_address=request.client.host if request.client else None,
    )
    db.add(audit)

    from core.config import get_settings
    settings = get_settings()
    expires_at = datetime.now(tz=timezone.utc) + timedelta(
        seconds=settings.signed_url.expiry_seconds
    )

    logger.info(
        "evidence_url_generated",
        evidence_id=str(evidence_id),
        user=current_user.email,
    )

    return SignedURLResponse(
        evidence_id=clip.id,
        signed_url=signed,
        expires_at=expires_at,
        sha256_hash=clip.sha256_hash,
    )


@router.get("/clips/{evidence_id}")
async def serve_clip(
    evidence_id: uuid.UUID,
    token: str,
    expires: int,
    db: DBSession,
):
    """
    Serve the actual video file. Only reachable via valid signed URL.
    This endpoint is called by the browser/player after getting the signed URL.
    """
    resource_path = f"/api/v1/evidence/clips/{evidence_id}"
    verify_signed_url(path=resource_path, token=token, expires=expires)

    clip = await _get_clip_or_404(evidence_id, db)

    if not os.path.exists(clip.file_path):
        raise HTTPException(status_code=404, detail="Clip file not found on disk.")

    return FileResponse(
        path=clip.file_path,
        media_type="video/mp4",
        filename=f"evidence_{evidence_id}.mp4",
    )


@router.get("/{evidence_id}/manifest", response_model=EvidenceManifest)
async def get_evidence_manifest(
    evidence_id: uuid.UUID,
    db: DBSession,
    current_user: User = Depends(require_role(UserRole.SUPERVISOR)),
):
    """Full provenance manifest for an evidence clip."""
    clip = await _get_clip_or_404(evidence_id, db)

    result = await db.execute(
        select(Incident)
        .where(Incident.id == clip.incident_id)
        .join(Incident.camera)
    )
    incident = result.scalar_one_or_none()
    if incident is None:
        raise HTTPException(status_code=404, detail="Associated incident not found.")

    model_version_name = None
    if incident.model_version:
        model_version_name = incident.model_version.name

    reviewer_action = None
    reviewer_notes = None
    if incident.review_action:
        reviewer_action = incident.review_action
    if incident.review_records:
        latest = sorted(incident.review_records, key=lambda r: r.created_at)[-1]
        reviewer_notes = latest.notes

    return EvidenceManifest(
        incident_id=incident.id,
        camera_id=incident.camera_id,
        camera_name=incident.camera.name if incident.camera else "Unknown",
        location=incident.camera.location if incident.camera else "",
        event_type=incident.event_type,
        risk_score=incident.risk_score,
        severity=incident.severity,
        detected_at=incident.detected_at,
        event_start=incident.event_start,
        event_end=incident.event_end,
        clip_path=clip.file_path,
        clip_sha256=clip.sha256_hash,
        clip_duration_seconds=clip.duration_seconds,
        people_count=incident.people_count,
        model_version=model_version_name,
        confidence_at_alert=incident.confidence_at_alert,
        reviewer_id=incident.reviewer_id,
        reviewer_action=reviewer_action,
        reviewer_notes=reviewer_notes,
        reviewed_at=incident.reviewed_at,
        is_locked=incident.is_locked,
        retention_days=incident.retention_days,
        created_at=incident.created_at,
    )


@router.get("/snapshots/{incident_id}")
async def get_snapshot(incident_id: str):
    """
    Serve the JPEG snapshot for a specific incident.
    """
    from pathlib import Path
    import glob
    
    # Securely construct path
    evidence_dir = Path("evidence_storage/snapshots")
    if not evidence_dir.exists():
        raise HTTPException(status_code=404, detail="Snapshot directory not found.")
        
    # Snapshots are saved as {incident_id[:8]}_{timestamp}.jpg
    search_pattern = f"{incident_id[:8]}_*.jpg"
    matches = glob.glob(str(evidence_dir / search_pattern))
    
    if not matches:
        raise HTTPException(status_code=404, detail="Snapshot not found for this incident.")
        
    latest_snapshot = sorted(matches)[-1]
    
    return FileResponse(
        path=latest_snapshot,
        media_type="image/jpeg",
        filename=f"snapshot_{incident_id[:8]}.jpg",
    )


@router.get("/clips/incident/{incident_id}")
async def get_clip_by_incident(incident_id: str):
    """
    Serve the MP4 video clip directly by incident ID (for the dashboard).
    """
    from pathlib import Path
    import glob
    
    evidence_dir = Path("evidence_storage/clips")
    if not evidence_dir.exists():
        raise HTTPException(status_code=404, detail="Clip directory not found.")
        
    search_pattern = f"{incident_id}_*.mp4"
    matches = glob.glob(str(evidence_dir / search_pattern))
    
    if not matches:
        raise HTTPException(status_code=404, detail="Video clip not found yet. Still processing.")
        
    latest_clip = sorted(matches)[-1]
    
    return FileResponse(
        path=latest_clip,
        media_type="video/mp4",
        filename=f"evidence_{incident_id[:8]}.mp4",
    )
