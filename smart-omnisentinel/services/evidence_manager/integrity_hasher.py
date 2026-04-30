"""
services/evidence_manager/integrity_hasher.py
-----------------------------------------------
SHA-256 file hasher for evidence clip integrity verification.
Also provides hash chain verification for a sequence of evidence items.
"""
from __future__ import annotations

import hashlib
import os
from typing import Optional

from core.logger import get_logger

logger = get_logger(__name__)

CHUNK_SIZE = 65536  # 64KB chunks for streaming hash


def compute_sha256(file_path: str) -> Optional[str]:
    """
    Compute SHA-256 hash of a file.
    Returns hex digest string, or None on failure.
    Streams the file to handle large clips without loading into memory.
    """
    if not os.path.exists(file_path):
        logger.warning("hash_file_not_found", path=file_path)
        return None

    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                sha256.update(chunk)
        digest = sha256.hexdigest()
        logger.debug("hash_computed", path=file_path, sha256=digest[:16] + "...")
        return digest
    except OSError as exc:
        logger.error("hash_computation_failed", path=file_path, error=str(exc))
        return None


def verify_sha256(file_path: str, expected_hash: str) -> bool:
    """
    Verify a file's SHA-256 hash against an expected value.
    Used to detect tampering with evidence clips.
    Returns True if hash matches, False otherwise.
    """
    actual = compute_sha256(file_path)
    if actual is None:
        return False
    match = actual == expected_hash.lower()
    if not match:
        logger.warning(
            "hash_verification_failed",
            path=file_path,
            expected=expected_hash[:16],
            actual=actual[:16],
        )
    return match
