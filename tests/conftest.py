"""Pytest configuration and fixtures."""

import hashlib
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from image_search_service.core.config import get_settings
from image_search_service.db.models import Base
from image_search_service.db.session import get_db, get_sync_db
from image_search_service.main import create_app
from image_search_service.services.embedding import get_embedding_service
from image_search_service.vector.face_qdrant import get_face_qdrant_client
from image_search_service.vector.qdrant import get_qdrant_client

# Import PostgreSQL fixtures (only used when @pytest.mark.postgres tests run)
# These imports make the fixtures available to tests marked with @pytest.mark.postgres
from tests.conftest_postgres import (  # noqa: F401
    fresh_pg_database,
    pg_connection_url,
    pg_engine,
    pg_session,
    pg_sync_connection_url,
    pg_sync_engine,
    pg_sync_session,
    postgres_container,
)

# Use SQLite for tests (no external dependencies)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
TEST_SYNC_DATABASE_URL = "sqlite:///:memory:"


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


@pytest.fixture(autouse=True)
def clear_embedding_cache(monkeypatch):
    """Clear embedding service cache and redirect to mock implementation.

    CRITICAL: This prevents tests from loading the real OpenCLIP/SigLIP models,
    which are expensive and not needed for unit tests.

    Approach:
    1. Clear @lru_cache on get_embedding_service()
    2. Monkeypatch embed_text() and embed_image() methods to use SemanticMockEmbeddingService
    3. This avoids loading the real models while keeping EmbeddingService structure

    This ensures all tests use the lightweight mock embedding service with semantic similarity.
    """
    from image_search_service.services.embedding import EmbeddingService, get_embedding_service

    # Clear any cached embedding services from previous tests
    get_embedding_service.cache_clear()

    # Create a shared mock instance for consistent semantic embeddings
    semantic_mock = SemanticMockEmbeddingService()

    # Monkeypatch EmbeddingService methods to use SemanticMockEmbeddingService
    original_embed_text = EmbeddingService.embed_text
    original_embed_image = EmbeddingService.embed_image
    original_embed_images_batch = EmbeddingService.embed_images_batch
    original_embedding_dim = EmbeddingService.embedding_dim

    def mock_embed_text(self, text: str) -> list[float]:
        """Use semantic mock instead of real CLIP."""
        return semantic_mock.embed_text(text)

    def mock_embed_image(self, image_path: str | Path) -> list[float]:
        """Use semantic mock instead of real CLIP."""
        return semantic_mock.embed_image(image_path)

    def mock_embed_images_batch(self, images: list) -> list[list[float]]:
        """Use semantic mock instead of real CLIP."""
        return semantic_mock.embed_images_batch(images)

    def mock_embedding_dim(self) -> int:
        """Return 768-dim to match semantic mock."""
        return semantic_mock.embedding_dim

    monkeypatch.setattr(EmbeddingService, "embed_text", mock_embed_text)
    monkeypatch.setattr(EmbeddingService, "embed_image", mock_embed_image)
    monkeypatch.setattr(EmbeddingService, "embed_images_batch", mock_embed_images_batch)
    monkeypatch.setattr(EmbeddingService, "embedding_dim", property(mock_embedding_dim))

    yield

    # Cleanup after test
    get_embedding_service.cache_clear()
    # Restore original methods (monkeypatch handles this automatically)


@pytest.fixture(autouse=True)
def validate_embedding_dimensions():
    """Validate that test embedding dimensions match expected values.

    This guard catches the most common mock accuracy issue: dimension
    mismatch between test and production embedding services.

    Expected dimensions:
    - Image search (CLIP/SigLIP): 768
    - Face recognition (InsightFace/ArcFace): 512

    This runs at import time before any tests execute to fail fast.
    """
    # Pre-test validation - check class constant
    assert MockEmbeddingService.EMBEDDING_DIM == 768, (
        f"Mock embedding service dimension mismatch: "
        f"got {MockEmbeddingService.EMBEDDING_DIM}, expected 768 (CLIP/SigLIP). "
        f"Make sure MockEmbeddingService = SemanticMockEmbeddingService in conftest.py"
    )

    # Also verify instance dimension property
    test_instance = MockEmbeddingService()
    assert test_instance.embedding_dim == 768, (
        f"Mock embedding instance dimension mismatch: "
        f"got {test_instance.embedding_dim}, expected 768 (CLIP/SigLIP)"
    )

    yield

    # Post-test: no additional validation needed since pre-test catches issues


class SemanticMockEmbeddingService:
    """Mock embedding service with meaningful semantic similarity.

    Uses predefined "concept clusters" to generate embeddings where:
    - Semantically similar texts produce similar vectors (high cosine similarity)
    - Unrelated texts produce dissimilar vectors (low cosine similarity)
    - Results are deterministic (same input always produces same output)
    - Vectors are L2-normalized (matches production CLIP/SigLIP behavior)

    Dimension: 768 to match production CLIP/SigLIP model output.
    """

    CONCEPT_CLUSTERS = {
        "nature": [
            "sunset",
            "beach",
            "ocean",
            "mountain",
            "forest",
            "lake",
            "sky",
            "sunrise",
            "landscape",
            "outdoors",
            "nature",
            "tree",
            "river",
        ],
        "animal": [
            "dog",
            "cat",
            "bird",
            "fish",
            "animal",
            "puppy",
            "kitten",
            "horse",
            "pet",
            "wildlife",
        ],
        "food": [
            "pizza",
            "burger",
            "sushi",
            "food",
            "meal",
            "restaurant",
            "cooking",
            "kitchen",
            "chef",
            "recipe",
        ],
        "urban": [
            "city",
            "building",
            "street",
            "car",
            "traffic",
            "downtown",
            "skyscraper",
            "road",
            "architecture",
            "bridge",
        ],
        "people": [
            "person",
            "face",
            "portrait",
            "family",
            "group",
            "crowd",
            "smile",
            "child",
            "baby",
            "wedding",
        ],
    }

    EMBEDDING_DIM = 768

    def __init__(self, seed: int = 42):
        self._rng = np.random.RandomState(seed)
        self._cluster_centers = self._generate_cluster_centers()
        self._cache: dict[str, list[float]] = {}

    def _generate_cluster_centers(self) -> dict[str, np.ndarray]:
        """Generate cluster centers for each concept cluster."""
        centers = {}
        for cluster_name in self.CONCEPT_CLUSTERS:
            center = self._rng.randn(self.EMBEDDING_DIM)
            center = center / np.linalg.norm(center)
            centers[cluster_name] = center
        return centers

    def _find_cluster(self, text: str) -> str | None:
        """Find which concept cluster a text belongs to."""
        text_lower = text.lower()
        for cluster_name, keywords in self.CONCEPT_CLUSTERS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return cluster_name
        return None

    @property
    def embedding_dim(self) -> int:
        """Return fixed embedding dimension."""
        return self.EMBEDDING_DIM

    def embed_text(self, text: str) -> list[float]:
        """Generate semantically meaningful vector from text.

        Args:
            text: Text to embed

        Returns:
            Deterministic 768-dim L2-normalized vector
        """
        if text in self._cache:
            return self._cache[text]

        cluster = self._find_cluster(text)

        if cluster:
            # Text belongs to a known cluster - generate vector near cluster center
            center = self._cluster_centers[cluster]
            text_hash = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            noise_rng = np.random.RandomState(text_hash)
            noise = noise_rng.randn(self.EMBEDDING_DIM) * 0.1
            vector = center + noise
        else:
            # Unknown text - generate random vector
            text_hash = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            text_rng = np.random.RandomState(text_hash)
            vector = text_rng.randn(self.EMBEDDING_DIM)

        # L2 normalize
        vector = vector / np.linalg.norm(vector)

        result = vector.tolist()
        self._cache[text] = result
        return result

    def embed_image(self, image_path: str | Path) -> list[float]:
        """Generate vector from image path (uses filename as semantic hint).

        Args:
            image_path: Path to image

        Returns:
            Deterministic 768-dim vector
        """
        filename = Path(image_path).stem
        return self.embed_text(filename)

    def embed_image_from_pil(self, image: Image.Image) -> list[float]:
        """Generate vector from PIL image (uses size/mode as deterministic seed).

        Args:
            image: PIL Image object

        Returns:
            Deterministic 768-dim vector
        """
        size_key = f"pil_{image.size[0]}x{image.size[1]}_{image.mode}"
        return self.embed_text(size_key)

    def embed_images_batch(self, images: list[Image.Image]) -> list[list[float]]:
        """Generate vectors from batch of PIL images.

        Args:
            images: List of PIL Image objects

        Returns:
            List of 768-dim vectors
        """
        return [self.embed_image_from_pil(img) for img in images]


class LegacyMockEmbeddingService:
    """Legacy mock embedding service that returns deterministic vectors without loading OpenCLIP.

    DEPRECATED: Use SemanticMockEmbeddingService for tests requiring semantic similarity.
    This class is kept for backward compatibility during migration.
    """

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


# Alias for backward compatibility during migration
MockEmbeddingService = SemanticMockEmbeddingService


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
def sync_db_engine():
    """Create synchronous test database engine with SQLite in-memory.

    Used for testing synchronous background jobs.
    """
    from sqlalchemy import create_engine

    engine = create_engine(TEST_SYNC_DATABASE_URL, echo=False)

    # Create all tables
    Base.metadata.create_all(engine)

    yield engine

    # Cleanup
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def sync_db_session(sync_db_engine):
    """Create synchronous test database session.

    Used for testing synchronous background jobs.
    """
    from sqlalchemy.orm import Session

    session = Session(sync_db_engine)
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def qdrant_client() -> QdrantClient:
    """Create in-memory Qdrant client for testing.

    No external Qdrant server needed - uses :memory: mode.
    """
    client = QdrantClient(":memory:")

    # Create test collections with 768-dim vectors for images, 512-dim for faces
    settings = get_settings()

    # Main image assets collection (768-dim for CLIP/SigLIP)
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    # Face embeddings collection with payload indexes (mirrors production)
    client.create_collection(
        collection_name=settings.qdrant_face_collection,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )

    # Add payload indexes for face collection (same as production bootstrap)
    client.create_payload_index(
        collection_name=settings.qdrant_face_collection,
        field_name="person_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=settings.qdrant_face_collection,
        field_name="cluster_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=settings.qdrant_face_collection,
        field_name="is_prototype",
        field_schema=PayloadSchemaType.BOOL,
    )
    client.create_payload_index(
        collection_name=settings.qdrant_face_collection,
        field_name="asset_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=settings.qdrant_face_collection,
        field_name="face_instance_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    return client


@pytest.fixture
def mock_embedding_service() -> MockEmbeddingService:
    """Create mock embedding service that doesn't load OpenCLIP."""
    return MockEmbeddingService()


@pytest.fixture
def face_qdrant_client(qdrant_client: QdrantClient):
    """Create face Qdrant client wrapper for testing.

    Uses the same in-memory Qdrant client as the main fixture.
    """
    from image_search_service.vector.face_qdrant import FaceQdrantClient

    # Create instance and inject test client
    face_client = FaceQdrantClient()
    face_client._client = qdrant_client
    return face_client


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
    sync_db_session,
    qdrant_client: QdrantClient,
    face_qdrant_client,
    mock_embedding_service: MockEmbeddingService,
) -> AsyncGenerator[AsyncClient, None]:
    """Create test client for FastAPI application with dependency overrides.

    Overrides:
        - get_db: Uses test database session (SQLite in-memory)
        - get_sync_db: Uses test sync database session (SQLite in-memory)
        - get_qdrant_client: Uses in-memory Qdrant client
        - get_face_qdrant_client: Uses in-memory face Qdrant client
        - get_embedding_service: Uses mock service (no OpenCLIP loading)

    Args:
        db_session: Test database session fixture
        sync_db_session: Test sync database session fixture
        qdrant_client: In-memory Qdrant client fixture
        face_qdrant_client: In-memory face Qdrant client fixture
        mock_embedding_service: Mock embedding service fixture

    Yields:
        AsyncClient for making test requests
    """
    app = create_app()

    # Override dependencies to use test fixtures
    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    def override_get_sync_db():
        yield sync_db_session

    def override_get_qdrant() -> QdrantClient:
        return qdrant_client

    def override_get_face_qdrant():
        return face_qdrant_client

    def override_get_embedding() -> MockEmbeddingService:
        return mock_embedding_service

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_sync_db] = override_get_sync_db
    app.dependency_overrides[get_qdrant_client] = override_get_qdrant
    app.dependency_overrides[get_face_qdrant_client] = override_get_face_qdrant
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
