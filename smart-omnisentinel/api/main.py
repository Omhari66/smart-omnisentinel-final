"""
api/main.py
-----------
FastAPI application factory.
Registers all routers, middleware, exception handlers, and lifecycle events.
Entry point for uvicorn:  uvicorn api.main:app --reload
"""
from __future__ import annotations

import json
import os
import asyncio
import sqlalchemy
from typing import AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

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
# Observability helpers
# ---------------------------------------------------------------------------

def _init_sentry(settings) -> None:
    """
    Initialise Sentry SDK if a DSN is configured.
    Safe to call in dev — an empty DSN is a no-op.
    """
    if not settings.observability.sentry_dsn:
        return
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

        sentry_sdk.init(
            dsn=settings.observability.sentry_dsn,
            environment=settings.app_env,
            release="smartomnisentinel@1.0.0",
            traces_sample_rate=settings.observability.sentry_traces_sample_rate,
            profiles_sample_rate=settings.observability.sentry_profiles_sample_rate,
            integrations=[
                FastApiIntegration(transaction_style="endpoint"),
                SqlalchemyIntegration(),
            ],
            # Don't send PII (IP addresses, user emails) to Sentry
            send_default_pii=False,
        )
        get_logger(__name__).info("sentry_initialized", env=settings.app_env)
    except ImportError:
        get_logger(__name__).warning(
            "sentry_sdk_not_installed",
            hint="pip install sentry-sdk[fastapi]",
        )


def _init_prometheus(app: FastAPI, settings) -> None:
    """
    Mount the Prometheus /metrics endpoint using prometheus-fastapi-instrumentator.
    Records: request count, request latency (p50/p95/p99), request size,
             response size, and HTTP status code breakdowns per endpoint.
    """
    if not settings.observability.prometheus_enabled:
        return
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator(
            should_group_status_codes=True,     # Group 2xx, 4xx, 5xx
            should_ignore_untemplated=True,      # Skip unmapped paths (e.g. favicon)
            should_respect_env_var=False,
            should_instrument_requests_inprogress=True,
            excluded_handlers=[
                "/api/v1/health/ping",           # Skip liveness probe from metrics
                "/metrics",                      # Skip metrics scrapes themselves
            ],
            inprogress_name="http_requests_inprogress",
            inprogress_labels=True,
        ).instrument(app).expose(
            app,
            endpoint="/metrics",
            include_in_schema=False,             # Hide from Swagger UI
            tags=["observability"],
        )
        get_logger(__name__).info("prometheus_metrics_enabled", endpoint="/metrics")
    except ImportError:
        get_logger(__name__).warning(
            "prometheus_fastapi_instrumentator_not_installed",
            hint="pip install prometheus-fastapi-instrumentator",
        )


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

    # Initialise Sentry error tracking (no-op if DSN not configured)
    _init_sentry(settings)

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
        lifespan=lifespan,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
    )

    # Middleware (order matters — last registered runs first)
    register_middleware(app)
    register_exception_handlers(app)

    # ── Prometheus metrics (/metrics endpoint) ─────────────────────────────
    # Must be set up BEFORE routers so the instrumentator wraps all routes.
    _init_prometheus(app, settings)

    # ── API versioned prefix ────────────────────────────────────────────────
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
