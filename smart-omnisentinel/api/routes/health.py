"""api/routes/health.py — System health endpoints."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi import APIRouter

from api.dependencies import CurrentUser
from core.config import get_settings
from core.logger import get_logger

router = APIRouter(prefix="/health", tags=["health"])
logger = get_logger(__name__)


@router.get("")
async def system_health(current_user: CurrentUser):
    """Aggregate system health summary."""
    settings = get_settings()
    evidence_root = settings.storage.evidence_root

    # Disk usage check
    try:
        stat = os.statvfs(evidence_root) if os.path.exists(evidence_root) else None
        if stat:
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
            total_gb = (stat.f_blocks * stat.f_frsize) / (1024 ** 3)
            disk_usage_pct = round((1 - free_gb / total_gb) * 100, 1)
        else:
            free_gb = total_gb = disk_usage_pct = None
    except Exception:
        free_gb = total_gb = disk_usage_pct = None

    return {
        "status": "ok",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "version": "1.0.0",
        "environment": settings.app_env,
        "storage": {
            "evidence_root": evidence_root,
            "free_gb": round(free_gb, 2) if free_gb else None,
            "total_gb": round(total_gb, 2) if total_gb else None,
            "usage_pct": disk_usage_pct,
        },
    }


@router.get("/ping")
async def ping():
    """Unauthenticated liveness probe for load balancers."""
    return {"status": "ok", "ts": datetime.now(tz=timezone.utc).isoformat()}
