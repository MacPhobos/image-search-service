"""Database session management with lazy initialization."""

from collections.abc import AsyncGenerator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from image_search_service.core.config import get_settings
from image_search_service.core.logging import get_logger

logger = get_logger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_sync_engine: Engine | None = None


def get_engine() -> AsyncEngine:
    """Get or create async database engine (lazy initialization)."""
    global _engine

    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            echo=False,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
        logger.info("Database engine initialized")

    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create async session factory."""
    global _session_factory

    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        logger.info("Session factory initialized")

    return _session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for FastAPI to get database session."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        finally:
            await session.close()


def get_sync_engine() -> Engine:
    """Get synchronous engine for RQ workers.

    Returns:
        Synchronous SQLAlchemy engine
    """
    global _sync_engine

    if _sync_engine is None:
        settings = get_settings()
        # Convert async URL to sync (replace asyncpg with psycopg2)
        sync_url = settings.database_url.replace("postgresql+asyncpg", "postgresql+psycopg2")
        _sync_engine = create_engine(sync_url, pool_pre_ping=True)
        logger.info("Sync database engine initialized for workers")

    return _sync_engine


async def close_db() -> None:
    """Close database engine and cleanup resources."""
    global _engine, _session_factory, _sync_engine

    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Async database engine closed")

    if _sync_engine is not None:
        _sync_engine.dispose()
        _sync_engine = None
        logger.info("Sync database engine closed")
