"""
services/ingestion/__main__.py
--------------------------------
Entry point for the ingestion service when run as a standalone process.

In development: ingestion is wired into the FastAPI process via event bus.
In production:  this runs in its own Docker container (Dockerfile.inference).

Usage:
    python -m services.ingestion

The service:
  1. Connects to PostgreSQL to fetch all ACTIVE cameras
  2. Starts a StreamReader task per camera
  3. Subscribes to ingestion.commands bus channel for dynamic START/STOP
  4. Subscribes to confirmed_events to feed the risk engine
  5. Runs indefinitely until interrupted
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.config import get_settings
from core.event_bus import get_event_bus
from core.logger import get_logger, setup_logging
from db.session import close_engine, get_session_factory
from services.ingestion.camera_registry import get_camera_registry
from services.risk_engine.severity_classifier import process_confirmed_event

setup_logging()
logger = get_logger(__name__)


async def load_active_cameras() -> list:
    """Fetch all cameras with ACTIVE status from the database."""
    from db.models.camera import Camera
    from sqlalchemy import select
    from core.constants import CameraStatus

    factory = get_session_factory()
    async with factory() as db:
        result = await db.execute(
            select(Camera).where(Camera.status == CameraStatus.ACTIVE.value)
        )
        cameras = list(result.scalars().all())
        logger.info("active_cameras_loaded", count=len(cameras))
        return cameras


async def main() -> None:
    settings = get_settings()
    logger.info(
        "ingestion_service_starting",
        device=settings.inference.device,
        effective_fps=settings.inference.effective_fps,
    )

    # Load ML models
    logger.info("loading_ml_models")
    try:
        from services.inference.model_loader import load_models
        load_models()
        logger.info("ml_models_loaded")
    except Exception as exc:
        logger.error("ml_model_load_failed", error=str(exc))
        logger.warning("running_without_trained_classifier")

    bus = get_event_bus()
    registry = get_camera_registry()

    # Start all active cameras
    cameras = await load_active_cameras()
    for camera in cameras:
        await registry.start_camera(
            camera_id=str(camera.id),
            stream_url=camera.stream_url,
        )

    # Wire up the pipeline:
    #   event_candidates.* → event_analyzer (in event_analysis service)
    #   confirmed_events   → risk engine
    #   scored_incidents   → alert dispatcher
    tasks = [
        asyncio.create_task(
            registry.listen_for_control_commands(),
            name="camera_control_listener",
        ),
        asyncio.create_task(
            bus.consume("confirmed_events", process_confirmed_event),
            name="risk_engine_consumer",
        ),
    ]

    logger.info(
        "ingestion_service_ready",
        active_cameras=len(cameras),
        tasks=len(tasks),
    )

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        logger.info("ingestion_service_interrupted")
    finally:
        logger.info("ingestion_service_stopping")
        await registry.stop_all()
        await close_engine()
        logger.info("ingestion_service_stopped")


if __name__ == "__main__":
    asyncio.run(main())
