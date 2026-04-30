"""
core/signed_url.py
------------------
HMAC-based signed URL generator for protected evidence clip access.
Clips are never served from publicly addressable paths.
Every access requires a short-lived signed URL.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from urllib.parse import urlencode

from core.config import get_settings
from core.exceptions import AuthenticationError
from core.logger import get_logger

logger = get_logger(__name__)


def _compute_signature(path: str, expires: int, secret: str) -> str:
    """Compute HMAC-SHA256 signature for (path, expires) pair."""
    message = f"{path}:{expires}".encode()
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def generate_signed_url(
    base_url: str,
    path: str,
    expiry_seconds: int | None = None,
) -> str:
    """
    Generate a time-limited signed URL for a protected resource.

    Args:
        base_url: e.g. "https://sentinel.local/api/v1/evidence/clips"
        path:     Relative resource path, e.g. "/INC-uuid_full.mp4"
        expiry_seconds: Override default expiry from settings.

    Returns:
        Full URL with ?token=...&expires=... query params appended.
    """
    settings = get_settings()
    secret = settings.signed_url.secret
    ttl = expiry_seconds or settings.signed_url.expiry_seconds

    expires_at = int(time.time()) + ttl
    signature = _compute_signature(path, expires_at, secret)

    params = urlencode({"token": signature, "expires": expires_at})
    return f"{base_url}{path}?{params}"


def verify_signed_url(path: str, token: str, expires: str | int) -> None:
    """
    Verify a signed URL's token and expiry.
    Raises AuthenticationError if invalid or expired.
    """
    settings = get_settings()
    secret = settings.signed_url.secret

    try:
        expires_int = int(expires)
    except (TypeError, ValueError) as exc:
        raise AuthenticationError("Invalid signed URL: bad expires param.") from exc

    if time.time() > expires_int:
        raise AuthenticationError("Signed URL has expired.")

    expected = _compute_signature(path, expires_int, secret)
    if not hmac.compare_digest(expected, token):
        logger.warning("signed_url_invalid_token", path=path)
        raise AuthenticationError("Signed URL token is invalid.")
