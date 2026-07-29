"""
services/evidence_manager/clip_extractor.py
---------------------------------------------
Generates evidence clips by merging pre-event buffer frames with
live post-event capture.

Workflow:
  1. On event confirmed: flush pre-event circular buffer to staging/
  2. Continue capturing post-event frames for POST_EVENT_BUFFER_SECONDS
  3. Merge pre + post clips into final clip in evidence_storage/clips/
  4. Compute SHA-256 hash of final clip
  5. Write metadata manifest
  6. Persist EvidenceClip record to DB
  7. Publish EVIDENCE_READY WebSocket event
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import cv2
import numpy as np

from core.config import get_settings
from core.constants import POST_EVENT_BUFFER_SECONDS, RetentionTag, WSEventType
from core.logger import get_logger
from services.inference.frame_buffer import BufferedFrame, get_buffer

logger = get_logger(__name__)


@dataclass
class ClipRequest:
    incident_id: uuid.UUID
    camera_id: str
    retention_tag: str
    lock: bool = False


_active_clip_tasks = set()

async def request_clip_extraction(
    incident_id: uuid.UUID,
    camera_id: str,
    retention_tag: str = "TEMP_72H",
    lock: bool = False,
) -> None:
    """
    Entry point called by alert handlers.
    Schedules clip extraction as a background asyncio task.
    Non-blocking — alert pipeline continues immediately.
    """
    request = ClipRequest(
        incident_id=incident_id,
        camera_id=camera_id,
        retention_tag=retention_tag,
        lock=lock,
    )
    task = asyncio.create_task(
        _extract_clip(request),
        name=f"clip_extract_{incident_id}",
    )
    _active_clip_tasks.add(task)
    task.add_done_callback(_active_clip_tasks.discard)
    logger.info(
        "clip_extraction_scheduled",
        incident_id=str(incident_id),
        camera_id=camera_id,
    )


async def _extract_clip(request: ClipRequest) -> None:
    """
    Background task: flush buffer, wait for post-event frames, merge, hash, persist.
    """
    settings = get_settings()
    evidence_root = settings.storage.evidence_root
    staging_dir = os.path.join(evidence_root, "staging")
    clips_dir = os.path.join(evidence_root, "clips")
    manifests_dir = os.path.join(evidence_root, "manifests")

    os.makedirs(staging_dir, exist_ok=True)
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(manifests_dir, exist_ok=True)

    incident_str = str(request.incident_id)

    # --- Step 1: Flush pre-event buffer ---
    buffer = get_buffer(request.camera_id)
    pre_frames = buffer.flush()

    if not pre_frames:
        logger.warning("clip_no_pre_frames", incident_id=incident_str)

    # --- Step 2: Write pre-event clip to staging ---
    pre_clip_path = os.path.join(staging_dir, f"{incident_str}_pre.mp4")
    fps = settings.inference.effective_fps or 4

    loop = asyncio.get_event_loop()
    pre_written = await loop.run_in_executor(
        None, _write_frames_to_video, pre_frames, pre_clip_path, fps
    )

    # --- Step 3: Capture post-event frames ---
    # In MVP, we wait POST_EVENT_BUFFER_SECONDS and then flush whatever
    # has accumulated in the buffer since the event.
    # In V2, the ingestion service streams directly into a post-event recorder.
    await asyncio.sleep(POST_EVENT_BUFFER_SECONDS)
    post_frames = buffer.flush()

    post_clip_path = os.path.join(staging_dir, f"{incident_str}_post.mp4")
    post_written = await loop.run_in_executor(
        None, _write_frames_to_video, post_frames, post_clip_path, fps
    )

    # --- Step 4: Merge pre + post clips ---
    final_clip_path = os.path.join(clips_dir, f"{incident_str}_full.mp4")
    all_frames = pre_frames + post_frames

    merged = await loop.run_in_executor(
        None, _write_frames_to_video, all_frames, final_clip_path, fps
    )

    if not merged:
        logger.error("clip_merge_failed", incident_id=incident_str)
        return

    # --- Step 5: Compute SHA-256 hash ---
    sha256_hash = await loop.run_in_executor(
        None, _compute_sha256, final_clip_path
    )
    file_size = os.path.getsize(final_clip_path) if os.path.exists(final_clip_path) else 0
    duration_seconds = len(all_frames) / fps if fps > 0 else 0.0

    logger.info(
        "clip_merged",
        incident_id=incident_str,
        path=final_clip_path,
        frames=len(all_frames),
        duration_s=round(duration_seconds, 1),
        sha256=sha256_hash[:16] + "..." if sha256_hash else None,
    )

    # --- Step 6: Persist EvidenceClip record ---
    await _persist_evidence_clip(
        incident_id=request.incident_id,
        file_path=final_clip_path,
        file_size_bytes=file_size,
        duration_seconds=duration_seconds,
        sha256_hash=sha256_hash,
        retention_tag=request.retention_tag,
        lock=request.lock,
    )

    # --- Step 7: Clean up staging files ---
    for path in (pre_clip_path, post_clip_path):
        if os.path.exists(path):
            os.remove(path)

    # --- Step 8: Push EVIDENCE_READY WebSocket event ---
    await _push_evidence_ready(request.incident_id, duration_seconds)


def _write_frames_to_video(
    frames: List[BufferedFrame],
    output_path: str,
    fps: int,
) -> bool:
    """Write a list of BufferedFrames to an MP4 file. Returns True on success."""
    if not frames:
        return False

    h, w = frames[0].frame.shape[:2]
    # Changed from 'avc1' to 'mp4v' to prevent the missing Cisco OpenH264 DLL error on Windows
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    if not writer.isOpened():
        logger.error("video_writer_failed_to_open", path=output_path)
        return False

    for bf in frames:
        writer.write(bf.frame)

    writer.release()
    return True


def _compute_sha256(file_path: str) -> Optional[str]:
    """Compute SHA-256 hash of a file for integrity verification."""
    if not os.path.exists(file_path):
        return None
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as exc:
        logger.error("sha256_computation_failed", path=file_path, error=str(exc))
        return None


async def _persist_evidence_clip(
    incident_id: uuid.UUID,
    file_path: str,
    file_size_bytes: int,
    duration_seconds: float,
    sha256_hash: Optional[str],
    retention_tag: str,
    lock: bool,
) -> None:
    """Write EvidenceClip record to the database."""
    from db.models.evidence_clip import EvidenceClip
    from db.session import get_session_factory

    retention_days_map = {
        "TEMP_72H": 3,
        "STANDARD_30D": 30,
        "LONG_365D": 365,
        "PERMANENT": None,
    }
    retention_days = retention_days_map.get(retention_tag, 3)

    expires_at = None
    if retention_days:
        expires_at = datetime.now(tz=timezone.utc) + timedelta(days=retention_days)

    factory = get_session_factory()
    try:
        async with factory() as db:
            clip = EvidenceClip(
                incident_id=incident_id,
                file_path=file_path,
                file_size_bytes=file_size_bytes,
                duration_seconds=duration_seconds,
                sha256_hash=sha256_hash,
                is_locked=lock,
                retention_tag=RetentionTag(retention_tag),
                expires_at=expires_at,
            )
            db.add(clip)
            await db.commit()
            logger.info(
                "evidence_clip_persisted",
                incident_id=str(incident_id),
                clip_id=str(clip.id),
            )
    except Exception as exc:
        logger.error(
            "evidence_clip_persist_failed",
            incident_id=str(incident_id),
            error=str(exc),
        )


async def _push_evidence_ready(
    incident_id: uuid.UUID,
    duration_seconds: float,
) -> None:
    from api.schemas.ws_events import EvidenceReadyPayload, make_event
    from api.websocket_manager import get_ws_manager

    try:
        ws_manager = get_ws_manager()
        payload = EvidenceReadyPayload(
            incident_id=incident_id,
            evidence_id=uuid.uuid4(),  # Populated from DB in V2
            duration_seconds=duration_seconds,
        )
        event = make_event(WSEventType.EVIDENCE_READY, payload)
        await ws_manager.broadcast(event.model_dump(mode="json"))
    except Exception as exc:
        logger.warning("evidence_ready_push_failed", error=str(exc))
