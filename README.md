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

## Database creation

### 1) Create role (user) with password
    sudo -u postgres psql -v ON_ERROR_STOP=1 <<'SQL'
    CREATE ROLE "image-search" LOGIN PASSWORD 'somepassword';
    SQL

### 2) Create database owned by that user
    sudo -u postgres psql -v ON_ERROR_STOP=1 <<'SQL'
    CREATE DATABASE "image-search" OWNER "image-search";
    SQL

### 3) Grant “admin-ish” rights on that DB + schema (so the user can create/use objects)
    sudo -u postgres psql -v ON_ERROR_STOP=1 <<'SQL'
    GRANT ALL PRIVILEGES ON DATABASE "image-search" TO "image-search";

    \c "image-search"
    GRANT ALL ON SCHEMA public TO "image-search";
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO "image-search";
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO "image-search";
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO "image-search";
    SQL

### Quick test login:

psql "postgresql://image-search:somepassword@localhost:5432/image-search"


##

Commands to Run

cd /export/workspace/image-search/image-search-service

# 1. Install dependencies
uv sync --dev

# 2. Run database migration
make migrate

# 3. Ensure Qdrant collection exists
make faces-ensure-collection

# 4. Run face backfill on existing assets
make faces-backfill LIMIT=100

# 5. Cluster unlabeled faces
make faces-cluster MAX_FACES=10000

# 6. Label a cluster via API
curl -X POST http://localhost:8000/api/v1/faces/clusters/clu_abc123def456/label \
-H "Content-Type: application/json" \
-d '{"name": "Alice"}'

# 7. Run incremental assignment
make faces-assign MAX_FACES=500

# 8. View statistics
make faces-stats

🔄 Full Pipeline (Single Command)

make faces-pipeline

This runs: ensure-collection → backfill → cluster → assign → centroids → stats


## License

See LICENSE file.
