"""
services/evidence_manager/metadata_writer.py
----------------------------------------------
Writes JSON metadata manifests for evidence clips.
Manifests are stored alongside clips in evidence_storage/manifests/.
Each manifest is a standalone JSON file containing full provenance data.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from core.config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)


async def write_manifest(
    incident_id: UUID,
    manifest_data: dict[str, Any],
) -> str:
    """
    Write a JSON manifest file for the given incident.
    Returns the path to the written manifest.
    """
    settings = get_settings()
    manifests_dir = os.path.join(settings.storage.evidence_root, "manifests")
    os.makedirs(manifests_dir, exist_ok=True)

    filename = f"{incident_id}.json"
    manifest_path = os.path.join(manifests_dir, filename)

    manifest_data["manifest_written_at"] = datetime.now(tz=timezone.utc).isoformat()

    try:
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=2, default=str)
        logger.info("manifest_written", incident_id=str(incident_id), path=manifest_path)
    except Exception as exc:
        logger.error("manifest_write_failed", incident_id=str(incident_id), error=str(exc))
        raise

    return manifest_path


async def read_manifest(incident_id: UUID) -> Optional[dict[str, Any]]:
    """Read an existing manifest file."""
    settings = get_settings()
    manifest_path = os.path.join(
        settings.storage.evidence_root, "manifests", f"{incident_id}.json"
    )
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path) as f:
            return json.load(f)
    except Exception as exc:
        logger.error("manifest_read_failed", incident_id=str(incident_id), error=str(exc))
        return None
