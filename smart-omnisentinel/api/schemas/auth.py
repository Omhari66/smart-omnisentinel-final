"""api/schemas/auth.py — Auth request/response schemas."""

from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel, EmailStr, Field

from api.schemas.common import OrmBase
from core.constants import UserRole


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # access token TTL in seconds
    user: "UserSummary"


class RefreshRequest(BaseModel):
    refresh_token: str


class UserSummary(OrmBase):
    id: uuid.UUID
    email: str
    full_name: str
    role: UserRole


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: str = Field(min_length=1, max_length=200)
    role: UserRole = UserRole.VIEWER


class UserResponse(OrmBase):
    id: uuid.UUID
    email: str
    full_name: str
    role: UserRole
    is_active: bool
