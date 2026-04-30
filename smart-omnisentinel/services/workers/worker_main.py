"""
services/workers/worker_main.py
---------------------------------
Entry point for background worker containers.
WORKER_TYPE environment variable selects which worker runs.

Valid WORKER_TYPE values:
    alert_manager     - alert escalation timer + scored_incidents consumer
    evidence_manager  - retention manager + clip extraction consumer
    health_monitor    - camera health + service health monitors
    event_pipeline    - confirmed_events → risk scoring → scored_incidents

In development (single process), all workers run in api/main.py as tasks.
In production, each runs in its own Docker container.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.config import get_settings
from core.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


async def run_alert_manager() -> None:
    """Runs: escalation timer + scored_incidents dispatcher."""
    from core.event_bus import get_event_bus
    from services.alert_manager.alert_dispatcher import get_alert_dispatcher
    from services.alert_manager.escalation_timer import get_escalation_timer

    logger.info("worker_starting", type="alert_manager")
    bus = get_event_bus()
    dispatcher = get_alert_dispatcher()
    timer = get_escalation_timer()

    await asyncio.gather(
        bus.consume("scored_incidents", dispatcher.dispatch),
        timer.run(),
    )


async def run_evidence_manager() -> None:
    """Runs: retention manager."""
    from services.evidence_manager.retention_manager import RetentionManager

    logger.info("worker_starting", type="evidence_manager")
    retention = RetentionManager()
    await retention.run()


async def run_health_monitor() -> None:
    """Runs: camera health monitor + service health monitor + reporter."""
    from services.health_monitor.camera_health import CameraHealthMonitor
    from services.health_monitor.service_health import ServiceHealthMonitor
    from services.health_monitor.health_reporter import HealthReporter

    logger.info("worker_starting", type="health_monitor")
    await asyncio.gather(
        CameraHealthMonitor().run(),
        ServiceHealthMonitor().run(),
        HealthReporter().run(),
    )


async def run_event_pipeline() -> None:
    """Runs: confirmed_events → risk scoring → publishes scored_incidents."""
    from core.event_bus import get_event_bus
    from services.risk_engine.severity_classifier import process_confirmed_event

    logger.info("worker_starting", type="event_pipeline")
    bus = get_event_bus()
    await bus.consume("confirmed_events", process_confirmed_event)


WORKER_MAP = {
    "alert_manager": run_alert_manager,
    "evidence_manager": run_evidence_manager,
    "health_monitor": run_health_monitor,
    "event_pipeline": run_event_pipeline,
}


async def main() -> None:
    worker_type = os.environ.get("WORKER_TYPE", "").strip().lower()
    if worker_type not in WORKER_MAP:
        logger.error(
            "invalid_worker_type",
            worker_type=worker_type,
            valid=list(WORKER_MAP.keys()),
        )
        sys.exit(1)

    logger.info("worker_main_starting", worker_type=worker_type)
    try:
        await WORKER_MAP[worker_type]()
    except KeyboardInterrupt:
        logger.info("worker_interrupted")
    except Exception as exc:
        logger.error("worker_fatal_error", error=str(exc), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
