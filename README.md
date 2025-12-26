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

4. **Bootstrap Qdrant collections:**
   ```bash
   make bootstrap-qdrant
   ```

5. **Run migrations:**
   ```bash
   make migrate
   ```

6. **Start development server:**
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
| `make bootstrap-qdrant` | Initialize Qdrant collections for fresh install |
| `make verify-qdrant` | Verify Qdrant collections are properly configured |
| `make migrate` | Run database migrations |
| `make makemigrations` | Create new migration |
| `make worker` | Start RQ background worker |
| `make faces-pipeline-dual` | Run dual-mode face pipeline (detect → cluster → stats) |
| `make faces-pipeline-full` | Full pipeline with training (detect → cluster → train → re-cluster) |
| `make faces-cluster-dual` | Run dual-mode face clustering |
| `make faces-train-matching` | Train face matching model |

## Qdrant Setup

The service requires two Qdrant collections to be initialized before use. Run the bootstrap command after starting Qdrant:

```bash
# Initialize collections (idempotent - safe to run multiple times)
make bootstrap-qdrant

# Verify collections are configured correctly
make verify-qdrant
```

**What it creates:**

1. **`image_assets` collection**: Stores image embeddings (512-dim, COSINE distance)
2. **`faces` collection**: Stores face embeddings with payload indexes for:
   - `person_id` (KEYWORD)
   - `cluster_id` (KEYWORD)
   - `is_prototype` (BOOL)
   - `asset_id` (KEYWORD)
   - `face_instance_id` (KEYWORD)

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection URL | `postgresql+asyncpg://postgres:postgres@localhost:5432/image_search` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `QDRANT_API_KEY` | Qdrant API key (optional) | - |
| `QDRANT_COLLECTION` | Image assets collection name | `image_assets` |
| `EMBEDDING_DIM` | Embedding vector dimension | `512` |
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
make faces-backfill LIMIT=10000

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

## Face Recognition & Clustering

The service includes advanced face recognition with dual-mode clustering and training capabilities.

### Quick Start - Face Pipeline

**One-command pipelines:**

```bash
# Quick pipeline: detect → cluster → stats (recommended for first run)
make faces-pipeline-dual

# Full pipeline with training: detect → cluster → train → re-cluster → stats
make faces-pipeline-full EPOCHS=20
```

**Step-by-step workflow:**

```bash
# 1. Run dual-mode pipeline (detects faces and clusters them)
make faces-pipeline-dual

# 2. Label clusters via API (identify who's who)
curl -X POST http://localhost:8000/api/v1/faces/clusters/{cluster_id}/label \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice"}'

# 3. Train model to improve accuracy (after labeling 5+ people)
make faces-train-matching EPOCHS=20

# 4. Re-cluster with improved model
make faces-cluster-dual

# 5. View statistics
make faces-stats
```

**Pipeline options:**

```bash
# Custom face detection limit
make faces-pipeline-dual LIMIT=10000

# Custom clustering threshold (higher = more conservative)
make faces-pipeline-dual PERSON_THRESHOLD=0.8

# Full pipeline with custom training
make faces-pipeline-full EPOCHS=50 MARGIN=0.3
```

### Dual-Mode Clustering

Combines two approaches for optimal face organization:

| Mode | Purpose | Output |
|------|---------|--------|
| **Supervised** | Match faces to known people | `person_*` clusters |
| **Unsupervised** | Group unknown faces by similarity | `unknown_cluster_*` groups |

**CLI Usage:**
```bash
# Default settings
make faces-cluster-dual

# Custom thresholds
make faces-cluster-dual PERSON_THRESHOLD=0.8 UNKNOWN_METHOD=hdbscan

# Limit faces processed
make faces-cluster-dual MAX_FACES=1000
```

**API Usage:**
```bash
# Synchronous (small batches)
curl -X POST http://localhost:8000/api/v1/faces/cluster/dual \
  -H "Content-Type: application/json" \
  -d '{"person_threshold": 0.7, "unknown_method": "hdbscan", "queue": false}'

# Background job (large batches)
curl -X POST http://localhost:8000/api/v1/faces/cluster/dual \
  -H "Content-Type: application/json" \
  -d '{"person_threshold": 0.7, "queue": true}'
```

### Training (Triplet Loss)

Improve person separation by training on labeled faces:

```bash
# Default training (20 epochs)
make faces-train-matching

# Custom training
make faces-train-matching EPOCHS=50 MARGIN=0.3 MIN_FACES=10
```

**API Usage:**
```bash
curl -X POST http://localhost:8000/api/v1/faces/train \
  -H "Content-Type: application/json" \
  -d '{"epochs": 20, "margin": 0.2, "queue": true}'
```

### Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `FACE_PERSON_MATCH_THRESHOLD` | Similarity threshold for person assignment | `0.7` |
| `FACE_UNKNOWN_CLUSTERING_METHOD` | Algorithm: hdbscan, dbscan, agglomerative | `hdbscan` |
| `FACE_UNKNOWN_MIN_CLUSTER_SIZE` | Minimum faces per cluster | `3` |
| `FACE_TRIPLET_MARGIN` | Training margin for triplet loss | `0.2` |
| `FACE_TRAINING_EPOCHS` | Default training epochs | `20` |

### Workflow Recommendations

1. **Initial Setup**: Run `faces-backfill` → `faces-cluster-dual`
2. **Labeling**: Use API or UI to label large clusters
3. **Training**: After 5+ people labeled, run `faces-train-matching`
4. **Re-clustering**: Run `faces-cluster-dual` to apply improvements
5. **Iterate**: Label more → Train → Re-cluster for best accuracy

## Troubleshooting

### Orphaned Faces (Face Embedding Not Found)

**Symptom:** When assigning a face to a person or fetching suggestions, you see:
```
Face embedding not found in vector database. The face may need to be re-detected.
```

**Cause:** The face record exists in PostgreSQL but its embedding is missing from Qdrant. This can happen if:
1. Faces were detected before a bug fix that ensured atomic database/Qdrant operations
2. Qdrant was reset or lost data while PostgreSQL retained face records
3. Network errors during face detection caused partial writes

**Diagnosis:** Find orphaned faces using the CLI:
```bash
# Check for orphaned faces (dry run)
uv run python -m image_search_service.scripts.cli faces find-orphans --limit 1000
```

Example output:
```
Checking for orphaned faces...
Checked: 100 faces
Orphaned: 3 faces (3.0%)

Orphaned faces:
  • 5a70e1d9-52f3-440b-b25c-ff2ac644d043 (asset 1234)
  • 7b82f3a1-... (asset 1234)
  • 9c93d4b2-... (asset 5678)

Affected assets:
  • Asset 1234: 2 orphaned faces
  • Asset 5678: 1 orphaned face
```

**Solution:** Re-detect faces on affected assets:
```bash
# Automatically fix orphaned faces by re-running face detection
uv run python -m image_search_service.scripts.cli faces find-orphans --fix
```

This will:
1. Identify all orphaned faces
2. Group them by asset
3. Re-run face detection on each affected asset
4. Regenerate the missing Qdrant embeddings

**Prevention:** This issue was fixed in commit `9e08c23`. The fix ensures:
- Qdrant upsert happens **before** database commit
- If Qdrant fails, the database transaction is rolled back
- No orphaned faces can be created going forward

## License

See LICENSE file.
