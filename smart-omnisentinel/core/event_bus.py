"""
core/event_bus.py
-----------------
Internal publish/subscribe bus connecting pipeline services.

MVP Mode (REDIS__ENABLED=false):
    Uses asyncio.Queue — zero infrastructure overhead,
    works perfectly for single-machine / single-process deployment.

V2 Mode (REDIS__ENABLED=true):
    Swaps to Redis Streams — enables multi-process, multi-machine scaling.
    Service code does NOT change; only this module changes.

Channel naming convention:
    "frames.{camera_id}"
    "event_candidates.{camera_id}"
    "confirmed_events"
    "scored_incidents"
    "alert_actions"
    "camera_health"
    "system_health"
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, AsyncIterator, Callable, Coroutine

from core.logger import get_logger

logger = get_logger(__name__)

# Type alias for async handler callbacks
Handler = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class InProcessEventBus:
    """
    Simple asyncio-Queue-based pub/sub bus.
    Suitable for single-process MVP deployment.
    Multiple subscribers on the same channel each get a separate queue.
    """

    def __init__(self) -> None:
        # channel → list of queues (one per subscriber)
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def publish(self, channel: str, message: dict[str, Any]) -> None:
        """Broadcast message to all subscribers of the channel."""
        queues = self._subscribers.get(channel, [])
        for q in queues:
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                logger.warning(
                    "event_bus_queue_full",
                    channel=channel,
                    queue_size=q.maxsize,
                )

    def subscribe(self, channel: str, maxsize: int = 256) -> asyncio.Queue:
        """
        Register a new subscriber queue for the given channel.
        Returns the queue; the caller reads from it.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subscribers[channel].append(q)
        logger.debug("event_bus_subscribed", channel=channel)
        return q

    def unsubscribe(self, channel: str, queue: asyncio.Queue) -> None:
        if channel in self._subscribers:
            try:
                self._subscribers[channel].remove(queue)
            except ValueError:
                pass

    async def consume(
        self,
        channel: str,
        handler: Handler,
        maxsize: int = 256,
    ) -> None:
        """
        Convenience method: subscribe and consume in a loop.
        Designed to be run as an asyncio task.

        Usage:
            asyncio.create_task(
                bus.consume("confirmed_events", my_handler)
            )
        """
        queue = self.subscribe(channel, maxsize=maxsize)
        logger.info("event_bus_consumer_started", channel=channel)
        try:
            while True:
                message = await queue.get()
                try:
                    await handler(message)
                except Exception as exc:
                    logger.error(
                        "event_bus_handler_error",
                        channel=channel,
                        error=str(exc),
                        exc_info=True,
                    )
                finally:
                    queue.task_done()
        except asyncio.CancelledError:
            self.unsubscribe(channel, queue)
            logger.info("event_bus_consumer_stopped", channel=channel)
            raise

    async def consume_prefix(
        self,
        prefix: str,
        handler: Handler,
        maxsize: int = 256,
    ) -> AsyncIterator[None]:
        """
        Subscribe to all channels matching a prefix (e.g. "frames.").
        NOTE: In MVP, channels must be pre-registered. This is a pattern helper.
        For true dynamic channel subscription, upgrade to Redis Streams.
        """
        raise NotImplementedError(
            "Dynamic prefix subscription requires Redis Streams backend. "
            "Use explicit channel names in MVP mode."
        )


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------

_bus_instance: InProcessEventBus | None = None


def get_event_bus() -> InProcessEventBus:
    """
    Returns the singleton event bus instance.
    Call once at startup in api/main.py; inject via FastAPI dependency.
    """
    global _bus_instance
    if _bus_instance is None:
        _bus_instance = InProcessEventBus()
    return _bus_instance
