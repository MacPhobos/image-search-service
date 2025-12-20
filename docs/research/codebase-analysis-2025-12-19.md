# Image Search Service - Codebase Analysis

**Date**: 2025-12-19
**Purpose**: Comprehensive codebase structure analysis for implementing image search vertical slice
**Status**: Complete

---

## Executive Summary

The image-search-service is a well-scaffolded Python 3.12 FastAPI application with **lazy initialization patterns**, async-first architecture, and minimal implementation. The codebase follows modern best practices with strict type checking, and has a frozen API contract that defines the complete feature set.

**Key Finding**: Only the health check endpoint is implemented. All other features exist as API contract specifications only.

---

## 1. Project Structure

### Directory Layout

```
image-search-service/
├── src/image_search_service/
│   ├── api/              # API routes and endpoints
│   │   └── routes.py     # Single router with /health endpoint only
│   ├── core/             # Configuration and logging
│   │   ├── config.py     # Pydantic Settings with .env support
│   │   └── logging.py    # Basic logging configuration
│   ├── db/               # Database layer
│   │   ├── models.py     # SQLAlchemy models (ImageAsset only)
│   │   ├── session.py    # Async session management with lazy init
│   │   └── migrations/   # Alembic migrations
│   │       └── versions/
│   │           └── 001_initial_migration.py
│   ├── queue/            # Background job processing
│   │   ├── jobs.py       # Job definitions (placeholder only)
│   │   └── worker.py     # RQ worker entry point
│   ├── vector/           # Qdrant operations
│   │   └── qdrant.py     # Client wrapper with lazy init
│   ├── scripts/          # CLI utilities
│   │   └── cli.py        # Typer CLI (not examined in detail)
│   └── main.py           # FastAPI application factory
└── tests/
    ├── conftest.py       # Pytest fixtures (test_client)
    └── test_health.py    # Health endpoint tests

docs/
└── api-contract.md       # FROZEN API contract (v1.0.0)
```

### Organization Patterns

✅ **Domain-driven structure**: `api/`, `db/`, `queue/`, `vector/` separate concerns
✅ **Configuration centralized**: All settings in `core/config.py`
✅ **Migrations tracked**: Alembic properly configured with version control
✅ **Tests separated**: Clean test directory with fixtures

---

## 2. Existing SQLAlchemy Models

### Current Models

**Only one model exists:**

```python
# src/image_search_service/db/models.py

class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


class ImageAsset(Base):
    """Image asset model storing metadata and file paths."""

    __tablename__ = "image_assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    path: Mapped[str] = mapped_column(String(500), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
```

### Model Patterns Observed

✅ **SQLAlchemy 2.0+ syntax**: Using `Mapped` type hints
✅ **Declarative base**: Proper ORM inheritance
✅ **Timezone-aware timestamps**: `DateTime(timezone=True)`
✅ **Server defaults**: `server_default=func.now()`
✅ **Type safety**: Explicit `Mapped[...]` annotations

### Missing Models (Per API Contract)

The API contract defines these entities that **do not yet have database models**:

- **Asset** (extended from ImageAsset): Missing `filename`, `url`, `thumbnailUrl`, `mimeType`, `width`, `height`, `fileSize`, `updatedAt`, `metadata` (JSON field)
- **Person**: Missing entirely (`id`, `name`, `thumbnailUrl`, `faceCount`, timestamps)
- **Face**: Missing entirely (`id`, `assetId`, `personId`, `boundingBox`, `confidence`, `thumbnailUrl`, timestamps)
- **Job**: Missing entirely (`id`, `type`, `status`, `progress`, `result`, `error`, timestamps)
- **Correction**: Missing entirely (future feature, can be skipped)

---

## 3. API Structure

### Current Router Pattern

```python
# src/image_search_service/api/routes.py

router = APIRouter()

@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint that works without external dependencies."""
    return {"status": "ok"}
```

**Only endpoint**: `/health`

### Router Registration

```python
# src/image_search_service/main.py

def create_app() -> FastAPI:
    app = FastAPI(
        title="Image Search Service",
        description="Vector similarity search for images",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Single router registered
    app.include_router(router)

    return app
```

### API Contract vs. Implementation Gap

**API Contract Defines** (in `docs/api-contract.md`):

| Endpoint | Method | Status | Prefix |
|----------|--------|--------|--------|
| `/health` | GET | ✅ Implemented | None |
| `/api/v1/assets` | GET | ❌ Missing | `/api/v1` |
| `/api/v1/assets/{id}` | GET | ❌ Missing | `/api/v1` |
| `/api/v1/assets/scan` | POST | ❌ Missing | `/api/v1` |
| `/api/v1/assets/{id}` | DELETE | ❌ Missing | `/api/v1` |
| `/api/v1/search` | GET | ❌ Missing | `/api/v1` |
| `/api/v1/search/similar` | POST | ❌ Missing | `/api/v1` |
| `/api/v1/people` | GET | ❌ Missing | `/api/v1` |
| `/api/v1/people/{id}` | GET/PATCH | ❌ Missing | `/api/v1` |
| `/api/v1/people/merge` | POST | ❌ Missing | `/api/v1` |
| `/api/v1/people/{id}/faces` | GET | ❌ Missing | `/api/v1` |
| `/api/v1/faces/unassigned` | GET | ❌ Missing | `/api/v1` |
| `/api/v1/faces/{faceId}/assign` | POST | ❌ Missing | `/api/v1` |
| `/api/v1/jobs` | GET | ❌ Missing | `/api/v1` |
| `/api/v1/jobs/{id}` | GET | ❌ Missing | `/api/v1` |
| `/api/v1/jobs/{id}/cancel` | POST | ❌ Missing | `/api/v1` |
| `/api/v1/corrections` | GET/POST | ❌ Missing | `/api/v1` |

**Key Observations**:
- All endpoints except `/health` need `/api/v1` prefix
- Contract is frozen (v1.0.0) - changes require version bump
- OpenAPI spec should be generated at `/openapi.json`
- Frontend generates types from OpenAPI using `openapi-typescript`

---

## 4. Configuration Management

### Settings Class Pattern

```python
# src/image_search_service/core/config.py

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignores unknown env vars
    )

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/image_search"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # Application
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
```

### Configuration Patterns

✅ **Pydantic Settings**: Type-safe environment variable loading
✅ **`.env` support**: Auto-loads from `.env` file in root
✅ **LRU cached**: Single instance via `@lru_cache`
✅ **Sensible defaults**: Works out-of-box for local development
✅ **Extra ignore**: Won't error on unexpected env vars

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | `postgresql+asyncpg://...` | Async Postgres connection |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis for RQ |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant vector DB |
| `QDRANT_API_KEY` | `""` (empty) | Optional Qdrant auth |
| `LOG_LEVEL` | `INFO` | Python logging level |

---

## 5. Dependencies (pyproject.toml)

### Production Dependencies

```toml
dependencies = [
    "fastapi>=0.115.0",           # Web framework
    "uvicorn[standard]>=0.32.0",  # ASGI server
    "pydantic-settings>=2.6.0",   # Config management
    "sqlalchemy[asyncio]>=2.0.36",# Async ORM
    "asyncpg>=0.30.0",            # Async Postgres driver
    "alembic>=1.14.0",            # Database migrations
    "redis>=5.2.0",               # Redis client
    "rq>=2.0.0",                  # Background job queue
    "qdrant-client>=1.12.0",      # Vector database client
    "typer[all]>=0.15.0",         # CLI framework
]
```

### Dev Dependencies

```toml
dev = [
    "pytest>=8.3.0",              # Testing framework
    "pytest-asyncio>=0.24.0",     # Async test support
    "httpx>=0.28.0",              # Async HTTP client for tests
    "ruff>=0.8.0",                # Linter + formatter
    "mypy>=1.13.0",               # Type checker
]
```

### Python Version

```toml
requires-python = ">=3.12"
```

### Type Checking Configuration

```toml
[tool.mypy]
python_version = "3.12"
strict = true                    # Strict mode enabled
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true     # All functions must have type hints

[[tool.mypy.overrides]]
module = ["rq.*", "qdrant_client.*"]
ignore_missing_imports = true    # Allow untyped imports
```

### Testing Configuration

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"            # Auto-detect async tests
testpaths = ["tests"]
```

---

## 6. Existing Services

### Database Session Management

```python
# src/image_search_service/db/session.py

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Get or create async database engine (lazy initialization)."""
    global _engine

    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            echo=False,
            pool_pre_ping=True,     # Check connections before use
            pool_size=5,
            max_overflow=10,
        )
        logger.info("Database engine initialized")

    return _engine


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for FastAPI to get database session."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        finally:
            await session.close()
```

**Lazy Initialization Pattern**: Engine/session factory only created on first use

### Qdrant Client Wrapper

```python
# src/image_search_service/vector/qdrant.py

_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Get or create Qdrant client (lazy initialization)."""
    global _client

    if _client is None:
        settings = get_settings()
        _client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key if settings.qdrant_api_key else None,
        )
        logger.info("Qdrant client initialized")

    return _client


def ping() -> bool:
    """Check if Qdrant server is accessible."""
    try:
        client = get_qdrant_client()
        client.get_collections()
        return True
    except Exception as e:
        logger.warning("Qdrant ping failed: %s", e)
        return False
```

**Same Pattern**: Lazy initialization with global singleton

### Background Jobs (Placeholder)

```python
# src/image_search_service/queue/jobs.py

def process_image(image_path: str) -> dict[str, str]:
    """Process image and extract embeddings.

    Note:
        This is a placeholder implementation. In production, this would:
        1. Load image from path
        2. Generate embeddings using vision model
        3. Store embeddings in Qdrant
        4. Update database with processing status
    """
    logger.info("Processing image: %s", image_path)

    return {
        "status": "success",
        "image_path": image_path,
        "message": "Image processing not yet implemented",
    }
```

**Status**: Stub implementation only

---

## 7. Database Setup

### Async SQLAlchemy Configuration

**Key Details**:
- **Driver**: `asyncpg` (fastest async Postgres driver)
- **Engine**: `create_async_engine()` with connection pooling
- **Sessions**: `async_sessionmaker` for session factory
- **Dependency Injection**: `get_db()` yields session for FastAPI routes

**Connection Pool Settings**:
```python
pool_pre_ping=True,    # Validates connections before using
pool_size=5,           # Min 5 connections in pool
max_overflow=10,       # Up to 15 total connections
```

### Migration History

**Current Migrations**:

1. **`001_initial_migration.py`** (Revision: `001`)
   - Creates `image_assets` table
   - Columns: `id`, `path`, `created_at`
   - Constraints: Primary key on `id`, unique constraint on `path`
   - Down migration: `DROP TABLE image_assets`

**Migration Command**:
```bash
make migrate  # Runs: uv run alembic upgrade head
```

---

## 8. Existing Testing Patterns

### Test Fixtures

```python
# tests/conftest.py

@pytest.fixture
async def test_client() -> AsyncGenerator[AsyncClient, None]:
    """Create test client for FastAPI application."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
```

**Pattern**: Uses `httpx.AsyncClient` with `ASGITransport` (no server required)

### Test Structure

```python
# tests/test_health.py

@pytest.mark.asyncio
async def test_health_check_returns_ok(test_client: AsyncClient) -> None:
    """Test that health endpoint returns 200 and correct JSON."""
    response = await test_client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data == {"status": "ok"}
```

**Patterns Observed**:
- ✅ Async tests with `@pytest.mark.asyncio`
- ✅ Type hints on test functions
- ✅ Descriptive docstrings
- ✅ Clear assertions with expected values
- ✅ Tests verify lazy initialization (health works without DB)

---

## 9. Architectural Patterns

### Lazy Initialization

**Why**: Health endpoint must work without external dependencies

**Implementation**:
```python
# Global singletons initialized on first use
_engine: AsyncEngine | None = None
_client: QdrantClient | None = None

def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(...)
    return _engine
```

**Benefits**:
- Health check doesn't connect to DB/Redis/Qdrant
- Fast startup time
- Tests can run without infrastructure

### Async-First Design

**All database operations are async**:
```python
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    factory = get_session_factory()
    async with factory() as session:
        yield session
```

**Route handlers are async**:
```python
@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
```

### Dependency Injection

**FastAPI dependencies for testability**:
```python
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from image_search_service.db.session import get_db

@router.get("/assets")
async def list_assets(
    db: AsyncSession = Depends(get_db)
):
    # Database session injected here
    pass
```

**Pattern**: Use `Depends(get_db)` for database access

---

## 10. API Contract Analysis

### Contract Status: FROZEN (v1.0.0)

**Location**: `/export/workspace/image-search/image-search-service/docs/api-contract.md`

**Key Contract Rules**:

1. **Type Generation**:
   - Backend: FastAPI auto-generates OpenAPI at `/openapi.json`
   - Frontend: Uses `openapi-typescript` to generate `src/lib/api/generated.ts`
   - Workflow: Backend changes → Deploy → UI runs `npm run gen:api`

2. **URL Prefix**: All endpoints (except `/health`) use `/api/v1` prefix

3. **Pagination**: Consistent `PaginatedResponse<T>` wrapper for all list endpoints

4. **Error Format**: Standardized `ErrorResponse` with `code`, `message`, `details`

5. **CORS**: Configured for SvelteKit dev server (`localhost:5173`, `localhost:4173`)

### Pydantic Schema Requirements

**From API Contract** - These Pydantic models are needed:

#### Common Schemas

```python
class PaginationMeta(BaseModel):
    page: int
    page_size: int = Field(alias="pageSize")
    total_items: int = Field(alias="totalItems")
    total_pages: int = Field(alias="totalPages")

class PaginatedResponse(BaseModel, Generic[T]):
    data: list[T]
    pagination: PaginationMeta

class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict | None = None

class ErrorResponse(BaseModel):
    error: ErrorDetail
```

#### Asset Schemas

```python
class LocationMetadata(BaseModel):
    latitude: float
    longitude: float

class AssetMetadata(BaseModel):
    camera: str | None = None
    date_taken: str | None = Field(None, alias="dateTaken")
    location: LocationMetadata | None = None

class Asset(BaseModel):
    id: str  # UUID
    path: str
    filename: str
    url: str
    thumbnail_url: str = Field(alias="thumbnailUrl")
    mime_type: str = Field(alias="mimeType")
    width: int
    height: int
    file_size: int = Field(alias="fileSize")
    created_at: str = Field(alias="createdAt")  # ISO 8601
    updated_at: str = Field(alias="updatedAt")  # ISO 8601
    metadata: AssetMetadata | None = None
```

#### Search Schemas

```python
class SearchResult(BaseModel):
    asset: Asset
    score: float
    highlights: list[str]
```

#### Job Schemas

```python
class JobProgress(BaseModel):
    current: int
    total: int
    percentage: float

class Job(BaseModel):
    id: str
    type: Literal["SCAN", "EMBED", "FACE_DETECT", "FACE_CLUSTER", "THUMBNAIL"]
    status: Literal["PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"]
    progress: JobProgress | None = None
    result: dict | None = None
    error: str | None = None
    created_at: str = Field(alias="createdAt")
    started_at: str | None = Field(None, alias="startedAt")
    completed_at: str | None = Field(None, alias="completedAt")
```

---

## 11. Implementation Recommendations

### For Vertical Slice Implementation

**Recommended Order** (Most Critical First):

1. **Assets API** (`/api/v1/assets`):
   - Extend `ImageAsset` model with missing fields
   - Create Pydantic schemas for request/response
   - Implement `GET /api/v1/assets` with pagination
   - Implement `GET /api/v1/assets/{id}`
   - Add proper error handling (404, validation)

2. **Search API** (`/api/v1/search`):
   - Implement basic text search (vector similarity)
   - Return paginated search results with scores
   - Integrate with Qdrant for vector search

3. **Background Jobs** (`/api/v1/jobs`):
   - Implement job tracking models
   - Connect RQ job queue
   - Create job status endpoints

4. **People/Faces** (Later):
   - Face detection models
   - Person management
   - Face clustering

### File Structure Recommendations

**Create these new files**:

```
src/image_search_service/
├── api/
│   ├── schemas.py        # Pydantic request/response models
│   ├── routes.py         # Split into:
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health.py     # Health endpoint (move from routes.py)
│   │   ├── assets.py     # Asset management endpoints
│   │   ├── search.py     # Search endpoints
│   │   ├── jobs.py       # Job management endpoints
│   │   └── people.py     # People/face endpoints
│   ├── dependencies.py   # Shared FastAPI dependencies
│   └── errors.py         # Custom exceptions and error handlers
├── services/             # Business logic layer (NEW)
│   ├── __init__.py
│   ├── asset_service.py  # Asset operations
│   ├── search_service.py # Search logic
│   └── job_service.py    # Job queue operations
```

### Database Model Extensions Needed

**Extend `ImageAsset` model**:

```python
class ImageAsset(Base):
    __tablename__ = "image_assets"

    # Existing fields
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    path: Mapped[str] = mapped_column(String(500), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Add these fields
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
```

**Create new models**:

- `Job` model for background job tracking
- `Person` model for face grouping
- `Face` model for detected faces

### Router Organization Pattern

**Current**: Single `routes.py` with one endpoint

**Recommended**: Split into domain-specific routers

```python
# src/image_search_service/api/routes/__init__.py

from fastapi import APIRouter
from .health import router as health_router
from .assets import router as assets_router
from .search import router as search_router

api_router = APIRouter()

# No prefix for health
api_router.include_router(health_router)

# /api/v1 prefix for all others
v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(assets_router, prefix="/assets", tags=["assets"])
v1_router.include_router(search_router, prefix="/search", tags=["search"])

api_router.include_router(v1_router)
```

### Testing Strategy

**Create test files matching router structure**:

```
tests/
├── conftest.py
├── test_health.py
├── test_assets.py       # Asset endpoints
├── test_search.py       # Search endpoints
├── test_jobs.py         # Job endpoints
└── fixtures/            # Test data fixtures
    └── sample_assets.py
```

---

## 12. Critical Observations

### ✅ Strengths

1. **Clean Architecture**: Domain separation, dependency injection, lazy initialization
2. **Type Safety**: mypy strict mode, Mapped types, type hints everywhere
3. **Modern Stack**: Python 3.12, SQLAlchemy 2.0+, FastAPI latest, async-first
4. **Good Defaults**: Sensible configuration, connection pooling, logging setup
5. **Contract-Driven**: Frozen API contract ensures UI/backend alignment
6. **Test Foundation**: Proper fixtures, async test support, isolated tests

### ⚠️ Gaps & Missing Pieces

1. **No Pydantic Schemas**: API contract defines schemas, but none exist in code
2. **No Router Structure**: Single `routes.py` will become unwieldy
3. **No Service Layer**: Business logic will need separation from routes
4. **Minimal Models**: Only `ImageAsset` exists, missing 90% of required models
5. **No Error Handling**: Custom exceptions and error handlers not implemented
6. **No CORS**: API contract defines CORS, but not configured in FastAPI app
7. **Placeholder Jobs**: RQ integration exists but jobs are stubs

### 🚀 Immediate Next Steps

**For Image Search Vertical Slice**:

1. Create `api/schemas.py` with Pydantic models from API contract
2. Extend `ImageAsset` model with missing columns (filename, mime_type, etc.)
3. Create Alembic migration for model changes
4. Split `routes.py` into `routes/assets.py`, `routes/search.py`
5. Implement `GET /api/v1/assets` with pagination
6. Add CORS middleware to `main.py`
7. Create service layer for business logic

---

## 13. Code Quality & Standards

### Linting & Formatting

**Ruff Configuration**:
```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
```

**Commands**:
- `make lint`: Check code style
- `make format`: Auto-fix issues and format

### Type Checking

**Mypy Strict Mode**:
```toml
strict = true
disallow_untyped_defs = true
```

**Command**:
- `make typecheck`: Run mypy on `src/`

### Testing

**Pytest + Async**:
```toml
asyncio_mode = "auto"
```

**Command**:
- `make test`: Run full test suite

---

## 14. Development Workflow

### Local Development Setup

```bash
# 1. Install dependencies
uv sync --dev

# 2. Start infrastructure (Postgres, Redis, Qdrant)
make db-up

# 3. Run migrations
make migrate

# 4. Start dev server
make dev  # http://localhost:8000

# 5. Start background worker (separate terminal)
make worker
```

### Available Makefile Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all commands |
| `make dev` | Run uvicorn with reload |
| `make lint` | Check code style |
| `make format` | Auto-format code |
| `make typecheck` | Run mypy |
| `make test` | Run pytest |
| `make db-up` | Start Docker containers |
| `make db-down` | Stop containers |
| `make migrate` | Apply migrations |
| `make makemigrations` | Create new migration |
| `make worker` | Start RQ worker |

---

## 15. API Contract Compliance Checklist

**For Implementing Features, Ensure**:

- [ ] All Pydantic models use `alias` for camelCase fields (e.g., `page_size` → `pageSize`)
- [ ] All list endpoints return `PaginatedResponse<T>` wrapper
- [ ] All errors return `ErrorResponse` structure
- [ ] All endpoints (except `/health`) use `/api/v1` prefix
- [ ] All routes have proper tags for OpenAPI grouping
- [ ] Response models inherit from `BaseModel` for OpenAPI generation
- [ ] Timestamps are ISO 8601 strings (use `.isoformat()`)
- [ ] UUIDs are strings, not UUID objects in responses
- [ ] CORS middleware configured for SvelteKit dev URLs
- [ ] OpenAPI spec accessible at `/openapi.json`

---

## Appendices

### A. Full Dependency Tree

**Production**:
- FastAPI (web framework)
- Uvicorn (ASGI server)
- Pydantic Settings (config)
- SQLAlchemy (async ORM)
- asyncpg (Postgres driver)
- Alembic (migrations)
- Redis (cache/queue)
- RQ (background jobs)
- Qdrant Client (vector DB)
- Typer (CLI)

**Development**:
- Pytest (testing)
- pytest-asyncio (async tests)
- httpx (test client)
- Ruff (linter/formatter)
- mypy (type checker)

### B. Environment File Template

```bash
# .env (copy from .env.example)

DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/image_search
REDIS_URL=redis://localhost:6379/0
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
LOG_LEVEL=INFO
```

### C. Docker Compose Services

```yaml
# docker-compose.dev.yml (inferred from Makefile)
services:
  postgres:
    image: postgres:latest
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: image_search

  redis:
    image: redis:latest
    ports:
      - "6379:6379"

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
```

---

## Conclusion

The image-search-service codebase is a **well-architected foundation** with modern patterns (lazy initialization, async-first, type safety) but **minimal implementation**. Only the health check endpoint exists; all other features are defined in the frozen API contract but not yet built.

**To implement the image search vertical slice**, the primary tasks are:

1. Create Pydantic schemas matching the API contract
2. Extend the `ImageAsset` model with missing fields
3. Implement the Assets and Search APIs with proper pagination
4. Add CORS middleware and error handling
5. Create a service layer for business logic
6. Write comprehensive tests for new endpoints

The existing patterns provide clear guidance: follow the lazy initialization approach, use async everywhere, inject dependencies via FastAPI's `Depends()`, and maintain strict type safety with mypy.

---

**End of Analysis**
