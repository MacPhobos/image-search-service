# Image Search Service

Python 3.12 image search service with vector similarity using Qdrant.

## Quickstart

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker and Docker Compose

### Setup

1. **Install dependencies:**
   ```bash
   uv sync --dev
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start services:**
   ```bash
   make db-up
   ```

4. **Run migrations:**
   ```bash
   make migrate
   ```

5. **Start development server:**
   ```bash
   make dev
   ```

   API available at http://localhost:8000

## Development Commands

| Command | Description |
|---------|-------------|
| `make help` | Show available commands |
| `make dev` | Run FastAPI development server |
| `make lint` | Run ruff linter |
| `make format` | Format code with ruff |
| `make typecheck` | Run mypy type checker |
| `make test` | Run pytest tests |
| `make db-up` | Start Postgres, Redis, and Qdrant containers |
| `make db-down` | Stop all containers |
| `make migrate` | Run database migrations |
| `make makemigrations` | Create new migration |
| `make worker` | Start RQ background worker |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection URL | `postgresql+asyncpg://postgres:postgres@localhost:5432/image_search` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `QDRANT_API_KEY` | Qdrant API key (optional) | - |
| `LOG_LEVEL` | Logging level | `INFO` |

## Project Structure

```
image-search-service/
├── src/image_search_service/
│   ├── api/              # API routes and endpoints
│   ├── core/             # Configuration and logging
│   ├── db/               # Database models and sessions
│   │   └── migrations/   # Alembic migrations
│   ├── queue/            # Background job workers
│   ├── vector/           # Qdrant vector operations
│   └── scripts/          # CLI utilities
└── tests/                # Test suite
```

## API Endpoints

- `GET /health` - Health check endpoint

## Testing

```bash
make test
```

## License

See LICENSE file.
