# Image Search Service - AI Assistant Context

## Project Overview
Python 3.12 FastAPI service for image search with vector similarity using Qdrant.

## Architecture Patterns
- **Lazy Initialization**: All external clients (DB, Redis, Qdrant) initialized on-demand
- **Health Endpoint**: Works without dependencies running
- **Async-First**: SQLAlchemy async, asyncpg driver
- **Config Management**: Pydantic Settings with .env support
- **Background Jobs**: Redis Queue (RQ) for async processing

## Key Technologies
- **FastAPI**: Web framework
- **SQLAlchemy 2.0+**: Async ORM
- **Alembic**: Database migrations
- **Redis/RQ**: Task queue
- **Qdrant**: Vector database
- **uv**: Package manager

## Development Workflow
1. Start services: `make db-up`
2. Run migrations: `make migrate`
3. Start dev server: `make dev`
4. Run tests: `make test`

## Important Constraints
- Health endpoint must not require external dependencies
- All database operations must be async
- Use dependency injection for testability
- Type safety with mypy strict mode

## Future Considerations
- Vector embedding strategy (model selection)
- Image preprocessing pipeline
- Batch processing optimization
- Caching strategy for search results
