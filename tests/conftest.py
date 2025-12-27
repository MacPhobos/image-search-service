"""Pytest configuration and fixtures."""

import hashlib
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from image_search_service.core.config import get_settings
from image_search_service.db.models import Base
from image_search_service.db.session import get_db
from image_search_service.main import create_app
from image_search_service.services.embedding import get_embedding_service
from image_search_service.vector.qdrant import get_qdrant_client

# Use SQLite for tests (no external dependencies)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear settings cache before and after each test to prevent production settings leaking.

    CRITICAL: This prevents tests from using cached production collection names,
    which could cause deletion of live Qdrant data.
    """
    from image_search_service.core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def use_test_settings(monkeypatch):
    """Override settings with test-safe collection names.

    CRITICAL: This ensures tests never use production collection names
    like "image_assets" or "faces", preventing accidental data deletion.
    """
    from image_search_service.core.config import get_settings

    get_settings.cache_clear()

    # Set environment variables for test collection names
    monkeypatch.setenv("QDRANT_COLLECTION", "test_image_assets")
    monkeypatch.setenv("QDRANT_FACE_COLLECTION", "test_faces")

    # Clear cache again so new env vars are picked up
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class MockEmbeddingService:
    """Mock embedding service that returns deterministic vectors without loading OpenCLIP."""

    @property
    def embedding_dim(self) -> int:
        """Return fixed embedding dimension."""
        return 512

    def embed_text(self, text: str) -> list[float]:
        """Generate deterministic vector from text using hash.

        Args:
            text: Text to embed

        Returns:
            Deterministic 512-dim vector normalized to [0, 1]
        """
        # Use MD5 hash to generate deterministic values
        h = hashlib.md5(text.encode()).hexdigest()

        # Generate 512 values from repeating the 32 hex characters
        vector = []
        for i in range(512):
            # Take 2 hex chars, convert to int, normalize to [0, 1]
            idx = (i * 2) % len(h)
            val = int(h[idx : idx + 2], 16) / 255.0
            vector.append(val)

        return vector

    def embed_image(self, image_path: str | Path) -> list[float]:
        """Generate deterministic vector from image path.

        Args:
            image_path: Path to image (unused, uses path string for hash)

        Returns:
            Deterministic 512-dim vector
        """
        # Use path as seed for deterministic embedding
        return self.embed_text(str(image_path))


@pytest.fixture
async def db_engine() -> AsyncGenerator[AsyncEngine, None]:
    """Create test database engine with SQLite in-memory."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session with transaction rollback.

    Each test gets a fresh session that rolls back after the test,
    ensuring test isolation.
    """
    async_session = async_sessionmaker(db_engine, expire_on_commit=False)

    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
def qdrant_client() -> QdrantClient:
    """Create in-memory Qdrant client for testing.

    No external Qdrant server needed - uses :memory: mode.
    """
    client = QdrantClient(":memory:")

    # Create test collection with 512-dim vectors
    settings = get_settings()
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )

    return client


@pytest.fixture
def mock_embedding_service() -> MockEmbeddingService:
    """Create mock embedding service that doesn't load OpenCLIP."""
    return MockEmbeddingService()


@pytest.fixture
def temp_image_factory(tmp_path: Path) -> Callable[..., Path]:
    """Factory fixture for creating temporary test images.

    Args:
        tmp_path: pytest's tmp_path fixture

    Returns:
        Factory function that creates test images

    Example:
        def test_something(temp_image_factory):
            image_path = temp_image_factory("test.jpg", width=100, height=100)
            # use image_path
    """

    def _create_image(filename: str = "test.jpg", width: int = 10, height: int = 10) -> Path:
        """Create a small test image.

        Args:
            filename: Output filename
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Path to created image
        """
        # Create RGB image with gradient pattern
        img = Image.new("RGB", (width, height))
        pixels = img.load()

        # Simple gradient pattern for visual variety
        if pixels is not None:
            for x in range(width):
                for y in range(height):
                    r = int((x / width) * 255)
                    g = int((y / height) * 255)
                    b = 128
                    pixels[x, y] = (r, g, b)

        # Save to temp directory
        image_path = tmp_path / filename
        img.save(image_path)

        return image_path

    return _create_image


@pytest.fixture
async def test_client(
    db_session: AsyncSession,
    qdrant_client: QdrantClient,
    mock_embedding_service: MockEmbeddingService,
) -> AsyncGenerator[AsyncClient, None]:
    """Create test client for FastAPI application with dependency overrides.

    Overrides:
        - get_db: Uses test database session (SQLite in-memory)
        - get_qdrant_client: Uses in-memory Qdrant client
        - get_embedding_service: Uses mock service (no OpenCLIP loading)

    Args:
        db_session: Test database session fixture
        qdrant_client: In-memory Qdrant client fixture
        mock_embedding_service: Mock embedding service fixture

    Yields:
        AsyncClient for making test requests
    """
    app = create_app()

    # Override dependencies to use test fixtures
    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    def override_get_qdrant() -> QdrantClient:
        return qdrant_client

    def override_get_embedding() -> MockEmbeddingService:
        return mock_embedding_service

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_qdrant_client] = override_get_qdrant
    app.dependency_overrides[get_embedding_service] = override_get_embedding

    # Create test client
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    # Clear overrides after test
    app.dependency_overrides.clear()


@pytest.fixture
def mock_queue() -> dict[str, Any]:
    """Mock RQ queue for testing background job enqueuing.

    Returns:
        Dictionary to track enqueued jobs

    Example:
        def test_enqueue(mock_queue, monkeypatch):
            monkeypatch.setattr("module.get_queue", lambda: mock_queue)
            # Test code that enqueues jobs
            assert len(mock_queue["jobs"]) == 1
    """
    jobs: list[dict[str, Any]] = []

    class MockQueue:
        def enqueue(self, func: str, *args: Any, **kwargs: Any) -> None:
            jobs.append({"func": func, "args": args, "kwargs": kwargs})

    queue = MockQueue()
    return {"queue": queue, "jobs": jobs}
