"""
db/session.py
-------------
Async SQLAlchemy engine and session factory.
All DB access goes through AsyncSession; never use sync session.
"""

from __future__ import annotations

from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from core.config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)

_engine = None
_session_factory = None


def get_engine():
    global _engine

    if _engine is None:
        settings = get_settings()
        db_url = settings.db.url

        if "sqlite" in db_url:
            # SQLite: no pooling arguments
            _engine = create_async_engine(
                db_url,
                echo=settings.db.echo,
            )
        else:
            # PostgreSQL / others: use pooling
            _engine = create_async_engine(
                db_url,
                pool_size=settings.db.pool_size,
                max_overflow=settings.db.max_overflow,
                echo=settings.db.echo,
                pool_pre_ping=True,
            )

        logger.info("db_engine_created", url=db_url.split("@")[-1])

    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,  # Avoid lazy-load errors post-commit
            autoflush=False,
            autocommit=False,
        )
    return _session_factory


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """
    FastAPI dependency that yields an AsyncSession per request.
    Session is committed on success, rolled back on error.

    Usage in route:
        async def my_route(db: AsyncSession = Depends(get_db_session)):
            ...
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def close_engine() -> None:
    """Called at application shutdown to cleanly close DB connections."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        logger.info("db_engine_closed")
        _engine = None
