"""
core/security.py
----------------
JWT token creation/verification and bcrypt password handling.
All auth logic flows through here — never scattered in route handlers.

Token revocation (Phase 5):
  Every token now carries a `jti` (JWT ID) — a unique UUID per token.
  On logout/revoke, the jti is stored in a Redis blocklist until the token expires.
  decode_token() checks the blocklist and raises AuthenticationError for revoked tokens.
  Falls back to in-process set when Redis is disabled (development/testing).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from core.config import get_settings
from core.exceptions import AuthenticationError
from core.logger import get_logger

logger = get_logger(__name__)
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# In-process blocklist fallback (used when Redis is disabled — dev/test only).
# Not suitable for production multi-process deployments.
_in_process_blocklist: set[str] = set()


# ---------------------------------------------------------------------------
# Password utilities
# ---------------------------------------------------------------------------

def hash_password(plain: str) -> str:
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


# ---------------------------------------------------------------------------
# JWT utilities
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def create_access_token(
    subject: str | uuid.UUID,
    role: str,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    settings = get_settings()
    expire = _now_utc() + timedelta(
        minutes=settings.jwt.access_token_expire_minutes
    )
    jti = str(uuid.uuid4())   # Unique token ID for revocation
    payload: dict[str, Any] = {
        "sub": str(subject),
        "role": role,
        "type": "access",
        "jti": jti,
        "exp": expire,
        "iat": _now_utc(),
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(
        payload,
        settings.secret_key,
        algorithm=settings.jwt.algorithm,
    )


def create_refresh_token(subject: str | uuid.UUID) -> str:
    settings = get_settings()
    expire = _now_utc() + timedelta(
        days=settings.jwt.refresh_token_expire_days
    )
    jti = str(uuid.uuid4())   # Unique token ID for revocation
    payload: dict[str, Any] = {
        "sub": str(subject),
        "type": "refresh",
        "jti": jti,
        "exp": expire,
        "iat": _now_utc(),
    }
    return jwt.encode(
        payload,
        settings.secret_key,
        algorithm=settings.jwt.algorithm,
    )


def decode_token(token: str) -> dict[str, Any]:
    """
    Decodes and validates a JWT token.
    Also checks the Redis revocation blocklist — raises AuthenticationError
    if the token's jti has been revoked (logout/revoke endpoint called).
    Raises AuthenticationError on any failure.
    """
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt.algorithm],
        )
    except JWTError as exc:
        logger.warning("jwt_decode_failed", error=str(exc))
        raise AuthenticationError("Token is invalid or expired.") from exc

    # Check revocation blocklist
    jti: str | None = payload.get("jti")
    if jti and _is_revoked(jti):
        logger.warning("jwt_revoked_token_used", jti=jti)
        raise AuthenticationError("Token has been revoked. Please log in again.")

    return payload


def extract_user_id(token: str) -> str:
    payload = decode_token(token)
    user_id: str | None = payload.get("sub")
    if not user_id:
        raise AuthenticationError("Token missing subject claim.")
    return user_id


def extract_user_role(token: str) -> str:
    payload = decode_token(token)
    role: str | None = payload.get("role")
    if not role:
        raise AuthenticationError("Token missing role claim.")
    return role


# ---------------------------------------------------------------------------
# Token revocation (Redis blocklist)
# ---------------------------------------------------------------------------

def _redis_blocklist_key(jti: str) -> str:
    return f"token_blocklist:{jti}"


def _is_revoked(jti: str) -> bool:
    """
    Check if a jti is in the revocation blocklist.
    Tries Redis first; falls back to the in-process set.
    """
    settings = get_settings()
    if settings.redis.enabled:
        try:
            import redis as _redis
            r = _redis.from_url(settings.redis.url, decode_responses=True)
            return r.exists(_redis_blocklist_key(jti)) > 0
        except Exception as exc:
            logger.error("blocklist_redis_check_failed", error=str(exc))
            # Fail open — don't block every request if Redis is temporarily down.
            # Log the error so alerting catches it.
            return False
    # Dev/test: in-process set
    return jti in _in_process_blocklist


async def revoke_token(jti: str, expires_at: datetime) -> None:
    """
    Add a jti to the revocation blocklist until the token's expiry.
    Uses Redis TTL so the entry auto-expires — no manual cleanup needed.

    Falls back to in-process set when Redis is disabled.
    """
    settings = get_settings()
    now = _now_utc()
    ttl_seconds = max(0, int((expires_at - now).total_seconds()))

    if settings.redis.enabled:
        try:
            import redis.asyncio as aioredis
            r = aioredis.from_url(settings.redis.url, decode_responses=True)
            await r.setex(_redis_blocklist_key(jti), ttl_seconds, "1")
            await r.aclose()
            logger.info("token_revoked_redis", jti=jti, ttl=ttl_seconds)
            return
        except Exception as exc:
            logger.error("blocklist_redis_write_failed", error=str(exc))
            # Fall through to in-process as emergency fallback

    # Dev/test: in-process set (no TTL — cleared on process restart)
    _in_process_blocklist.add(jti)
    logger.info("token_revoked_in_process", jti=jti)
