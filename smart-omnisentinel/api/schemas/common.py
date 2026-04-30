"""api/schemas/common.py — Shared pagination, error, and response wrappers."""

from __future__ import annotations

import math
from typing import Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int
    page_size: int
    pages: int

    @classmethod
    def build(
        cls,
        items: List[T],
        total: int,
        page: int,
        page_size: int,
    ) -> "PaginatedResponse[T]":
        pages = math.ceil(total / page_size) if page_size > 0 else 0
        return cls(items=items, total=total, page=page, page_size=page_size, pages=pages)


class ErrorResponse(BaseModel):
    detail: str
    code: Optional[str] = None


class MessageResponse(BaseModel):
    message: str


class OrmBase(BaseModel):
    """Base for all ORM-backed response schemas."""
    model_config = ConfigDict(from_attributes=True)
