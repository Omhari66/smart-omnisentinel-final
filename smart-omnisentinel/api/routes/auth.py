"""api/routes/auth.py — Authentication endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import CurrentUser, DBSession, get_config
from api.schemas.auth import (
    LoginRequest,
    RefreshRequest,
    TokenResponse,
    UserCreate,
    UserResponse,
    UserSummary,
)
from core.config import Settings
from core.constants import UserRole
from core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from db.models.user import User

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    db: DBSession,
    settings: Settings = Depends(get_config),
):
    result = await db.execute(select(User).where(User.email == body.email))
    user: User | None = result.scalar_one_or_none()

    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled.",
        )

    # Update last_login_at
    await db.execute(
        update(User)
        .where(User.id == user.id)
        .values(last_login_at=datetime.now(tz=timezone.utc))
    )

    access_token = create_access_token(
        subject=user.id, role=user.role
    )
    refresh_token = create_refresh_token(subject=user.id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.jwt.access_token_expire_minutes * 60,
        user=UserSummary.model_validate(user),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    body: RefreshRequest,
    db: DBSession,
    settings: Settings = Depends(get_config),
):
    from core.exceptions import AuthenticationError

    try:
        payload = decode_token(body.refresh_token)
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Not a refresh token.")

    import uuid
    result = await db.execute(
        select(User).where(User.id == uuid.UUID(payload["sub"]))
    )
    user: User | None = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive.")

    access_token = create_access_token(subject=user.id, role=user.role)
    new_refresh = create_refresh_token(subject=user.id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh,
        expires_in=settings.jwt.access_token_expire_minutes * 60,
        user=UserSummary.model_validate(user),
    )


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: CurrentUser):
    return current_user


@router.post("/users", response_model=UserResponse, status_code=201)
async def create_user(
    body: UserCreate,
    db: DBSession,
    current_user: CurrentUser,
):
    """Admin-only: create a new system user."""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin role required.")

    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered.")

    user = User(
        email=body.email,
        hashed_password=hash_password(body.password),
        full_name=body.full_name,
        role=body.role,
    )
    db.add(user)
    await db.flush()
    return user
