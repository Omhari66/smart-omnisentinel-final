"""
api/routes/health.py
--------------------
System health endpoints.

  GET /api/v1/health/ping  — unauthenticated liveness probe (no DB, no Redis)
  GET /api/v1/health       — authenticated summary (disk, version, env)
  GET /api/v1/health/ready — unauthenticated readiness probe (checks DB + Redis)
                             Returns 200 if all critical dependencies are up,
                             503 if any dependency is down.
                             Used by Docker/K8s to gate traffic before service is ready.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import sqlalchemy
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from api.dependencies import CurrentUser
from core.config import get_settings
from core.logger import get_logger

router = APIRouter(prefix="/health", tags=["health"])
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Liveness probe (always fast — no I/O)
# ---------------------------------------------------------------------------

@router.get("/ping")
async def ping():
    """
    Unauthenticated liveness probe for load balancers and Docker healthcheck.
    Returns instantly without touching DB or Redis.
    A 200 here means the process is alive and the event loop is running.
    """
    return {"status": "ok", "ts": datetime.now(tz=timezone.utc).isoformat()}


# ---------------------------------------------------------------------------
# Readiness probe — deep dependency check
# ---------------------------------------------------------------------------

@router.get("/ready")
async def readiness():
    """
    Unauthenticated readiness probe.
    Returns 200 only when ALL critical dependencies respond:
      - PostgreSQL / SQLite (via SQLAlchemy)
      - Redis (if enabled)

    Returns 503 with a detailed breakdown if any dependency is down.
    Kubernetes / Docker Compose should use this as the readiness probe,
    NOT the liveness probe, to avoid routing traffic to an unready pod.
    """
    settings = get_settings()
    checks: dict = {}
    overall_ok = True
    t_start = time.monotonic()

    # ── Database check ───────────────────────────────────────────────────────
    db_ok = False
    db_detail = ""
    try:
        from db.session import get_session_factory
        factory = get_session_factory()
        t0 = time.monotonic()
        async with factory() as db:
            await db.execute(sqlalchemy.text("SELECT 1"))
        db_latency_ms = round((time.monotonic() - t0) * 1000, 1)
        db_ok = True
        db_detail = f"{db_latency_ms}ms"
    except Exception as exc:
        db_detail = f"error: {type(exc).__name__}: {exc}"
        overall_ok = False
        logger.warning("readiness_db_check_failed", error=str(exc))

    checks["database"] = {"ok": db_ok, "detail": db_detail}

    # ── Redis check (only if enabled) ────────────────────────────────────────
    redis_ok = True   # Assume ok if disabled
    redis_detail = "disabled"
    if settings.redis.enabled:
        redis_ok = False
        try:
            import redis.asyncio as aioredis
            t0 = time.monotonic()
            r = aioredis.from_url(settings.redis.url, socket_connect_timeout=2)
            await r.ping()
            await r.aclose()
            redis_latency_ms = round((time.monotonic() - t0) * 1000, 1)
            redis_ok = True
            redis_detail = f"{redis_latency_ms}ms"
        except Exception as exc:
            redis_detail = f"error: {type(exc).__name__}: {exc}"
            overall_ok = False
            logger.warning("readiness_redis_check_failed", error=str(exc))

    checks["redis"] = {"ok": redis_ok, "detail": redis_detail}

    # ── Build response ────────────────────────────────────────────────────────
    total_ms = round((time.monotonic() - t_start) * 1000, 1)
    http_status = status.HTTP_200_OK if overall_ok else status.HTTP_503_SERVICE_UNAVAILABLE

    payload = {
        "status": "ready" if overall_ok else "not_ready",
        "checks": checks,
        "total_ms": total_ms,
        "ts": datetime.now(tz=timezone.utc).isoformat(),
    }

    if not overall_ok:
        logger.error("readiness_probe_failed", checks=checks)

    return JSONResponse(content=payload, status_code=http_status)


# ---------------------------------------------------------------------------
# Authenticated system health summary
# ---------------------------------------------------------------------------

@router.get("")
async def system_health(current_user: CurrentUser):
    """Aggregate system health summary (authenticated)."""
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
