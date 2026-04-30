"""
core/logger.py
--------------
Structured logging setup using structlog.
All services import `get_logger` from here.

In development: human-readable colored console output.
In production:  JSON lines to stdout (compatible with log aggregators).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from core.config import get_settings


def setup_logging() -> None:
    """
    Configure structlog. Call once at application startup (api/main.py).
    """
    settings = get_settings()

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.is_production:
        # JSON output for log aggregators (Loki, Datadog, CloudWatch, etc.)
        renderer = structlog.processors.JSONRenderer()
    else:
        # Human-readable colored output for development
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(
        logging.DEBUG if settings.debug else logging.INFO
    )

    # Silence noisy third-party loggers
    for noisy in ["uvicorn.access", "sqlalchemy.engine", "ultralytics"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Returns a bound logger for the given module name.

    Usage:
        from core.logger import get_logger
        logger = get_logger(__name__)
        logger.info("event_detected", camera_id=cam_id, event_type="VIOLENT_INTERACTION")
    """
    return structlog.get_logger(name)
