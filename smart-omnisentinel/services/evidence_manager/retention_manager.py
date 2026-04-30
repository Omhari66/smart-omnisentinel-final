"""
services/evidence_manager/retention_manager.py
------------------------------------------------
Background task: enforces retention policy by deleting expired evidence clips.
Runs every hour. Only deletes clips where is_locked=False and expires_at < NOW.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

from core.logger import get_logger
from db.models.evidence_clip import EvidenceClip

logger = get_logger(__name__)

RETENTION_CHECK_INTERVAL = 3600  # Run every hour


class RetentionManager:
    """Deletes evidence clips whose retention period has expired."""

    async def run(self) -> None:
        logger.info("retention_manager_started")
        while True:
            try:
                await asyncio.sleep(RETENTION_CHECK_INTERVAL)
                await self._purge_expired()
            except asyncio.CancelledError:
                logger.info("retention_manager_stopped")
                raise
            except Exception as exc:
                logger.error("retention_manager_error", error=str(exc))

    async def _purge_expired(self) -> None:
        from db.session import get_session_factory
        from sqlalchemy import select

        now = datetime.now(tz=timezone.utc)
        factory = get_session_factory()

        async with factory() as db:
            result = await db.execute(
                select(EvidenceClip).where(
                    EvidenceClip.is_locked == False,
                    EvidenceClip.expires_at <= now,
                )
            )
            expired_clips = list(result.scalars().all())

        if not expired_clips:
            return

        logger.info("retention_purging", count=len(expired_clips))

        async with factory() as db:
            for clip in expired_clips:
                # Delete file from disk
                if clip.file_path and os.path.exists(clip.file_path):
                    try:
                        os.remove(clip.file_path)
                        logger.info(
                            "evidence_clip_deleted",
                            clip_id=str(clip.id),
                            path=clip.file_path,
                        )
                    except OSError as exc:
                        logger.warning(
                            "evidence_clip_delete_failed",
                            clip_id=str(clip.id),
                            error=str(exc),
                        )
                # Remove DB record
                await db.delete(clip)
            await db.commit()

        logger.info("retention_purge_complete", purged=len(expired_clips))
