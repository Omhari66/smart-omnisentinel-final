"""
api/main.py
-----------
FastAPI application factory.
Registers all routers, middleware, exception handlers, and lifecycle events.
Entry point for uvicorn:  uvicorn api.main:app --reload
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

import sqlalchemy
from fastapi import FastAPI

from api.middleware import register_exception_handlers, register_middleware
from api.routes import alerts, auth, cameras, evidence, health, incidents, review, ws
from api.routes import config_api as config_api_router
from api.routes import status as status_router
from api.routes.zones import profiles_router, zones_router
from core.config import get_settings
from core.event_bus import get_event_bus
from core.logger import get_logger, setup_logging
from db.session import close_engine, get_session_factory


# ---------------------------------------------------------------------------
# Lifecycle: startup → yield → shutdown
# Modern lifespan pattern (replaces deprecated @app.on_event decorator).
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Single async context manager that handles both startup and shutdown.
    Everything before `yield` runs on startup; everything after runs on shutdown.
    FastAPI guarantees the shutdown block runs even if startup raises.
    """
    settings = get_settings()
    setup_logging()
    logger = get_logger(__name__)

    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("sentinel_starting", environment=settings.app_env)

    # Warm up DB connection pool — fail fast if DB is unreachable
    factory = get_session_factory()
    async with factory() as db:
        await db.execute(sqlalchemy.text("SELECT 1"))
    logger.info("db_pool_warmed")

    # Initialize event bus (Redis-backed in production, asyncio.Queue in dev)
    bus = get_event_bus()
    logger.info("event_bus_initialized", type=type(bus).__name__)

    # In development, background workers run in-process.
    # In production they run as separate Docker containers (inference_worker service).
    if settings.app_env == "development":
        await _start_background_workers(app)

    logger.info("sentinel_ready")

    yield  # ── Application is running ─────────────────────────────────────

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("sentinel_shutting_down")

    # Gracefully cancel any in-process background tasks (dev mode only)
    tasks: list[asyncio.Task] = getattr(app.state, "background_tasks", [])
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    await close_engine()
    logger.info("sentinel_stopped")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="SmartOmniSentinel",
        description=(
            "Edge AI-Based Real-Time Incident Detection and Response "
            "for CCTV Networks"
        ),
        version="1.0.0",
        lifespan=lifespan,                          # ← modern pattern, no deprecation warnings
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
    )

    # Middleware (order matters — last registered runs first)
    register_middleware(app)
    register_exception_handlers(app)

    # API versioned prefix
    API_PREFIX = "/api/v1"

    # Routers
    app.include_router(auth.router, prefix=API_PREFIX)
    app.include_router(cameras.router, prefix=API_PREFIX)
    app.include_router(incidents.router, prefix=API_PREFIX)
    app.include_router(review.router, prefix=API_PREFIX)
    app.include_router(evidence.router, prefix=API_PREFIX)
    app.include_router(alerts.router, prefix=API_PREFIX)
    app.include_router(zones_router, prefix=API_PREFIX)
    app.include_router(profiles_router, prefix=API_PREFIX)
    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(ws.router, prefix=API_PREFIX)
    app.include_router(config_api_router.router, prefix=API_PREFIX)
    app.include_router(status_router.router, prefix=API_PREFIX)

    return app


async def _start_background_workers(app: FastAPI) -> None:
    """
    In development mode, start background workers as asyncio tasks
    within the same process.  In production, each service runs in its
    own container and this function is never called.
    """
    logger = get_logger(__name__)
    logger.info("starting_background_workers_dev_mode")

    tasks: list[asyncio.Task] = []

    # Alert escalation timer — checks for MEDIUM incidents past deadline
    from services.alert_manager.escalation_timer import EscalationTimer
    timer = EscalationTimer()
    tasks.append(asyncio.create_task(timer.run(), name="escalation_timer"))

    # Evidence retention manager — cleans expired staging clips
    from services.evidence_manager.retention_manager import RetentionManager
    retention = RetentionManager()
    tasks.append(asyncio.create_task(retention.run(), name="retention_manager"))

    app.state.background_tasks = tasks
    logger.info("background_workers_started", count=len(tasks))


# ---------------------------------------------------------------------------
# Application instance (imported by uvicorn)
# ---------------------------------------------------------------------------

app = create_app()
