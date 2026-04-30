"""
api/dependencies.py
-------------------
FastAPI dependency functions injected into route handlers.
Keep route handlers thin — all heavy lifting is done in services.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Optional

from fastapi import Depends, Header, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import Settings, get_settings
from core.constants import UserRole
from core.exceptions import AuthenticationError, AuthorizationError
from core.security import decode_token
from db.models.user import User
from db.session import get_db_session

_bearer_scheme = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def get_config() -> Settings:
    return get_settings()


# ---------------------------------------------------------------------------
# Database session
# ---------------------------------------------------------------------------

DBSession = Annotated[AsyncSession, Depends(get_db_session)]


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

async def _get_current_user(
    credentials: Annotated[
        Optional[HTTPAuthorizationCredentials], Depends(_bearer_scheme)
    ],
    db: DBSession,
) -> User:
    """
    Extract and validate JWT. Load and return the corresponding User.
    Raises HTTP 401 if token is missing/invalid.
    Raises HTTP 403 if user is inactive.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = decode_token(credentials.credentials)
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    user_id_str: str | None = payload.get("sub")
    if not user_id_str:
        raise HTTPException(status_code=401, detail="Invalid token payload.")

    from sqlalchemy import select

    result = await db.execute(
        select(User).where(User.id == uuid.UUID(user_id_str))
    )
    user = result.scalar_one_or_none()
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive.",
        )
    return user


CurrentUser = Annotated[User, Depends(_get_current_user)]


# ---------------------------------------------------------------------------
# Role-based access control
# ---------------------------------------------------------------------------

_ROLE_HIERARCHY = {
    UserRole.VIEWER: 0,
    UserRole.GUARD: 1,
    UserRole.SUPERVISOR: 2,
    UserRole.ADMIN: 3,
}


def require_role(minimum_role: UserRole):
    """
    Dependency factory for role-gated routes.

    Usage:
        @router.get("/sensitive")
        async def route(user: User = Depends(require_role(UserRole.SUPERVISOR))):
            ...
    """

    async def _check_role(current_user: CurrentUser) -> User:
        user_level = _ROLE_HIERARCHY.get(UserRole(current_user.role), -1)
        required_level = _ROLE_HIERARCHY.get(minimum_role, 999)
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{minimum_role}' or higher required.",
            )
        return current_user

    return _check_role


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

class PaginationParams:
    def __init__(
        self,
        page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
        page_size: int = Query(default=25, ge=1, le=100, description="Items per page"),
    ):
        self.page = page
        self.page_size = page_size

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


Pagination = Annotated[PaginationParams, Depends(PaginationParams)]
