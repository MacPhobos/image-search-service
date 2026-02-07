"""PostgreSQL integration test fixtures.

🔴 CRITICAL SAFETY REQUIREMENT 🔴
This file uses testcontainers ONLY. It NEVER connects to production databases.
All PostgreSQL connections come from PostgresContainer which uses ephemeral
Docker containers with random ports. Production DB (localhost:5432) is never touched.

These fixtures spin up a real PostgreSQL container via testcontainers
and provide both async and sync sessions for testing database-specific behavior.

Usage: Mark tests with @pytest.mark.postgres to use these fixtures.
"""

import uuid
from collections.abc import AsyncGenerator, Generator
from typing import Any

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session as SyncSession
from testcontainers.postgres import PostgresContainer  # type: ignore

from image_search_service.db.models import Base


# Session-scoped container (one container per test session)
@pytest.fixture(scope="session")
def postgres_container() -> Generator[PostgresContainer, None, None]:
    """Start PostgreSQL container for integration tests.

    The container is started once per test session and stopped after
    all tests complete. Uses PostgreSQL 16-alpine to match production.

    SAFETY: This container is completely isolated from production databases.
    It uses random ports and ephemeral storage.
    """
    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


@pytest.fixture(scope="session")
def pg_connection_url(postgres_container: PostgresContainer) -> str:
    """Get async PostgreSQL connection URL.

    Returns URL in postgresql+asyncpg:// format for SQLAlchemy async.

    SAFETY: URL points to testcontainer only, never production.
    """
    url: str = postgres_container.get_connection_url()
    # testcontainers returns psycopg2 URL; convert to asyncpg
    async_url = url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")
    # Verify we're not connecting to production (port should NOT be 5432 on localhost)
    assert "localhost:5432" not in async_url or "testcontainers" in async_url.lower(), (
        f"SAFETY VIOLATION: Attempted to use production database! URL: {async_url}"
    )
    return async_url


@pytest.fixture(scope="session")
def pg_sync_connection_url(postgres_container: PostgresContainer) -> str:
    """Get sync PostgreSQL connection URL.

    Returns URL in postgresql:// format for SQLAlchemy sync.

    SAFETY: URL points to testcontainer only, never production.
    """
    url: str = postgres_container.get_connection_url()
    # Verify we're not connecting to production
    assert "localhost:5432" not in url or "testcontainers" in url.lower(), (
        f"SAFETY VIOLATION: Attempted to use production database! URL: {url}"
    )
    return url


@pytest.fixture
async def pg_engine(pg_connection_url: str) -> AsyncGenerator[AsyncEngine, None]:
    """Create async PostgreSQL engine (function-scoped due to async).

    Creates all tables via Base.metadata.create_all.
    For migration testing, use the fresh_pg_database fixture instead.

    Note: This is function-scoped instead of session-scoped to avoid
    event loop issues with pytest-asyncio. Tables are created/dropped
    per test, but the underlying PostgreSQL container is session-scoped.
    """
    engine = create_async_engine(pg_connection_url, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def pg_session(pg_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Create async PostgreSQL session with per-test rollback.

    Each test gets a fresh session that rolls back after the test,
    ensuring test isolation.
    """
    session_factory = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session
        # Rollback any uncommitted changes for test isolation
        await session.rollback()


@pytest.fixture
def pg_sync_engine(pg_sync_connection_url: str) -> Generator[Engine, None, None]:
    """Create sync PostgreSQL engine for background job tests."""
    engine = create_engine(pg_sync_connection_url, echo=False)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def pg_sync_session(pg_sync_engine: Engine) -> Generator[SyncSession, None, None]:
    """Create sync PostgreSQL session with per-test rollback."""
    session = SyncSession(pg_sync_engine)
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def fresh_pg_database(postgres_container: PostgresContainer) -> Generator[str, None, None]:
    """Create a fresh database for each migration test.

    This ensures complete isolation between Alembic upgrade/downgrade tests.
    Each test gets a new database, preventing contamination from previous migrations.

    SAFETY: Creates databases inside testcontainer only.
    """
    import psycopg2  # type: ignore

    db_name = f"test_{uuid.uuid4().hex[:8]}"
    base_url: str = postgres_container.get_connection_url()

    # psycopg2.connect needs postgresql:// not postgresql+psycopg2://
    psycopg2_url = base_url.replace("postgresql+psycopg2://", "postgresql://")

    # Connect to default DB and create new one
    conn: Any = psycopg2.connect(psycopg2_url)
    conn.autocommit = True
    cursor: Any = conn.cursor()
    cursor.execute(f"CREATE DATABASE {db_name}")
    cursor.close()
    conn.close()

    # Return URL pointing to new database (keeping postgresql+psycopg2 for Alembic)
    fresh_url = base_url.rsplit("/", 1)[0] + f"/{db_name}"
    yield fresh_url

    # Cleanup: drop database
    conn = psycopg2.connect(psycopg2_url)
    conn.autocommit = True
    cursor = conn.cursor()
    cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
    cursor.close()
    conn.close()
