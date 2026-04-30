"""
core/security.py
----------------
JWT token creation/verification and bcrypt password handling.
All auth logic flows through here — never scattered in route handlers.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from jose import JWTError, jwt
from passlib.context import CryptContext

from core.config import get_settings
from core.exceptions import AuthenticationError
from core.logger import get_logger

logger = get_logger(__name__)
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


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
    subject: str | UUID,
    role: str,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    settings = get_settings()
    expire = _now_utc() + timedelta(
        minutes=settings.jwt.access_token_expire_minutes
    )
    payload: dict[str, Any] = {
        "sub": str(subject),
        "role": role,
        "type": "access",
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


def create_refresh_token(subject: str | UUID) -> str:
    settings = get_settings()
    expire = _now_utc() + timedelta(
        days=settings.jwt.refresh_token_expire_days
    )
    payload: dict[str, Any] = {
        "sub": str(subject),
        "type": "refresh",
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
    Raises AuthenticationError on any failure.
    """
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt.algorithm],
        )
        return payload
    except JWTError as exc:
        logger.warning("jwt_decode_failed", error=str(exc))
        raise AuthenticationError("Token is invalid or expired.") from exc


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
