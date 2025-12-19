# Image Search Service

Python 3.12 FastAPI service for image search with vector similarity using Qdrant.

## Tech Stack

- **Web**: FastAPI + uvicorn
- **Database**: PostgreSQL (SQLAlchemy async + asyncpg)
- **Migrations**: Alembic
- **Queue**: Redis + RQ
- **Vectors**: Qdrant + OpenCLIP
- **Package Manager**: uv
- **Quality**: ruff, mypy (strict), pytest

## Local Development

```bash
# Install dependencies
uv sync --dev

# Start Postgres + Redis
make db-up

# Run migrations
make migrate

# Start API (with hot reload)
make dev

# Start background worker
make worker

# Run tests
make test

# Type check
make typecheck

# Lint + format
make format
```

## Project Structure

```
src/image_search_service/
├── api/routes/        # FastAPI endpoints
├── api/schemas.py     # Pydantic request/response models
├── core/config.py     # Settings (pydantic-settings)
├── db/models.py       # SQLAlchemy models
├── db/session.py      # Database connection
├── queue/             # RQ jobs and worker
├── services/          # Business logic
└── vector/            # Qdrant client
```

## Coding Rules

1. **src/ layout only** - all code under `src/image_search_service/`
2. **No import-time side effects** - use lazy initialization for DB, Redis, Qdrant
3. **Config via pydantic-settings** - all settings in `core/config.py`, loaded from env
4. **Async everywhere** - all DB operations must be async
5. **Dependency injection** - use FastAPI `Depends()` for testability

## API Rules

1. **Update contract first** - modify `docs/api-contract.md` before changing endpoints
2. **Backward compatibility** - don't break existing clients unless versioned
3. **All endpoints prefixed** - `/api/v1/` except `/health`
4. **Health stays simple** - `/health` must work without external dependencies
5. **Pydantic models** - all request/response bodies must have schemas

## Database Rules

1. **Always create migrations** - run `make makemigrations` after model changes
2. **Review migrations** - check generated SQL in `db/migrations/versions/`
3. **Test migrations** - run `make migrate` locally before committing
4. **No raw SQL in routes** - use SQLAlchemy ORM in service layer

## Logging Rules

1. **Structured format** - use `core/logging.py` utilities
2. **Avoid hot-path noise** - no per-request debug logs in production paths
3. **Include context** - add asset_id, job_id when relevant
4. **Log at boundaries** - API entry/exit, queue job start/end

## Testing Rules

1. **Tests required** - every feature change must include test updates in the same PR
2. **Test structure** - follow `tests/api/` and `tests/unit/` layout
3. **Naming conventions**:
   - Test files: `test_{module}.py`
   - Test functions: `test_{behavior}_[when_condition]_[expected_result]()`
4. **No external deps** - tests must pass without Postgres/Redis/Qdrant running
5. **Coverage preservation** - never delete tests without replacing coverage
6. **API changes require**:
   - Update `docs/api-contract.md`
   - Note: UI types regeneration needed
7. **Pre-commit checks** - always run before committing:
   ```bash
   make lint && make typecheck && make test
   ```

## Common Tasks

### Add a new endpoint

1. Update `docs/api-contract.md` with endpoint spec
2. Add Pydantic schemas in `api/schemas.py`
3. Create route in `api/routes/`
4. Register router in `main.py`

### Add a new model field

1. Modify model in `db/models.py`
2. Run `make makemigrations`
3. Review generated migration
4. Run `make migrate`

### Add a background job

1. Define job function in `queue/jobs.py`
2. Enqueue via `rq.Queue.enqueue()`
3. Job results stored in Redis, poll via `/api/v1/jobs/{id}`

## Environment Variables

```bash
# Required
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/image_search
REDIS_URL=redis://localhost:6379

# Optional
QDRANT_URL=http://localhost:6333
LOG_LEVEL=INFO
```
