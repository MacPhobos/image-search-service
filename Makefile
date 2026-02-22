.PHONY: help dev api lint format typecheck test test-serial test-fast test-failed-first test-profile test-affected test-cov db-up db-down db-status migrate makemigrations worker ingest \
	bootstrap-qdrant verify-qdrant exif-backfill backfill-hashes \
	faces-backfill faces-cluster faces-assign faces-centroids faces-stats faces-ensure-collection \
	faces-cluster-dual faces-train-matching faces-pipeline faces-pipeline-dual faces-pipeline-full

# ---- Load .env file ----
# The `-` prefix makes this non-fatal if .env does not exist.
# This project's .env uses plain KEY=VALUE format (no `export`, no quotes),
# which Make parses natively as recursive variable assignments (VAR = value).
# These assignments override later `?=` conditional defaults below.
-include .env

# ---- Docker Profile Construction ----
# Build comma-separated COMPOSE_PROFILES from DOCKER_* env vars.
# Default: all services enabled (DOCKER_POSTGRES=true, DOCKER_REDIS=true, DOCKER_QDRANT=true)
#
# Variable Precedence (verified empirically with GNU Make):
#   1. Make command-line args (highest) ... make db-up DOCKER_POSTGRES=false
#   2. .env file values (via -include)  ... DOCKER_POSTGRES=false in .env
#   3. Shell-exported env vars          ... export DOCKER_POSTGRES=false
#   4. Makefile ?= defaults (lowest)    ... true
#
# NOTE: -include .env performs a full `=` assignment, which overrides `?=`.
# This means .env values beat shell exports. If you need a one-time override
# while .env has a value set, use command-line args: make db-up DOCKER_POSTGRES=true
DOCKER_POSTGRES ?= true
DOCKER_REDIS    ?= true
DOCKER_QDRANT   ?= true

_PROFILES :=
ifneq ($(DOCKER_POSTGRES),false)
  _PROFILES += postgres
endif
ifneq ($(DOCKER_REDIS),false)
  _PROFILES += redis
endif
ifneq ($(DOCKER_QDRANT),false)
  _PROFILES += qdrant
endif

# Convert space-separated list to comma-separated for COMPOSE_PROFILES
_EMPTY :=
_SPACE := $(_EMPTY) $(_EMPTY)
_COMMA := ,
COMPOSE_PROFILES := $(subst $(_SPACE),$(_COMMA),$(strip $(_PROFILES)))

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

dev: ## Run FastAPI development server
	uv run uvicorn image_search_service.main:app --reload --host 0.0.0.0 --port 8000

lint: ## Run ruff linter
	uv run ruff check .

format: ## Format code with ruff
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## Run mypy type checker
	uv run mypy src/

test: ## Run pytest tests (parallel, randomized, with timeout)
	uv run pytest -n auto

test-serial: ## Run pytest serially (for debugging test isolation issues)
	uv run pytest -n0 -p no:randomly --timeout=60

test-fast: ## Re-run only previously failed tests (serial, for quick iteration)
	uv run pytest --lf -x -n0 --timeout=60

test-failed-first: ## Run failed tests first, then all remaining tests
	uv run pytest --ff

test-profile: ## Show slowest 20 tests (serial, for profiling)
	uv run pytest --durations=20 --durations-min=0.5 -n0 -p no:randomly --timeout=60

test-affected: ## Run only tests affected by recent changes (requires testmon)
	uv run pytest --testmon -n0 --timeout=60

test-cov: ## Run tests with fast coverage collection
	COVERAGE_CORE=sysmon uv run pytest --cov=image_search_service --cov-report=html -n0

test-postgres: ## Run PostgreSQL integration tests (requires Docker)
	@echo "Starting PostgreSQL integration tests (requires Docker)..."
	uv run pytest tests/ -m "postgres" -v --tb=short

test-all: ## Run all tests (SQLite + PostgreSQL)
	@echo "Running all tests..."
	uv run pytest tests/ -m "" -v --tb=short

db-up: ## Start Docker containers (disable with DOCKER_POSTGRES=false, DOCKER_REDIS=false, DOCKER_QDRANT=false)
	@if [ -z "$(COMPOSE_PROFILES)" ]; then \
		echo "All Docker services disabled. Nothing to start."; \
	else \
		echo "Starting Docker services: $(COMPOSE_PROFILES)"; \
		COMPOSE_PROFILES=$(COMPOSE_PROFILES) docker compose -f docker-compose.dev.yml up -d; \
	fi

db-down: ## Stop Docker containers (respects DOCKER_* settings)
	@if [ -z "$(COMPOSE_PROFILES)" ]; then \
		echo "All Docker services disabled. Nothing to stop."; \
	else \
		echo "Stopping Docker services: $(COMPOSE_PROFILES)"; \
		COMPOSE_PROFILES=$(COMPOSE_PROFILES) docker compose -f docker-compose.dev.yml down; \
	fi

db-status: ## Show status of Docker containers
	@docker compose -f docker-compose.dev.yml ps 2>/dev/null || echo "No containers running"

migrate: ## Run database migrations
	uv run alembic upgrade head

makemigrations: ## Create new migration
	@read -p "Enter migration message: " msg; \
	uv run alembic revision --autogenerate -m "$$msg"

worker: ## Start RQ Listener-based worker
	@echo "Starting RQ Listener-based worker..."
	@echo "Platform: Auto-detected (MPS on macOS, CUDA on Linux, CPU fallback)"
	@echo ""
	uv run python -m image_search_service.queue.listener_worker

api: ## Run API without reload (production-like)
	uv run uvicorn image_search_service.main:app --host 0.0.0.0 --port 8000

ingest: ## Ingest images from directory (usage: make ingest DIR=/path/to/images)
	@if [ -z "$(DIR)" ]; then echo "Usage: make ingest DIR=/path/to/images"; exit 1; fi
	curl -X POST http://localhost:8000/api/v1/assets/ingest \
		-H "Content-Type: application/json" \
		-d '{"rootPath": "$(DIR)", "recursive": true}'

# Qdrant bootstrap targets
bootstrap-qdrant: ## Initialize Qdrant collections for fresh install
	@echo "Initializing Qdrant collections..."
	uv run python -m image_search_service.scripts.bootstrap_qdrant init

verify-qdrant: ## Verify Qdrant collections are properly configured
	@echo "Verifying Qdrant collections..."
	uv run python -m image_search_service.scripts.bootstrap_qdrant verify

# EXIF metadata backfill
exif-backfill: ## Backfill EXIF metadata for existing images
	@echo "Running EXIF backfill (limit=$(or $(LIMIT),unlimited), batch-size=$(or $(BATCH_SIZE),100))..."
	uv run python scripts/backfill_exif.py \
		$(if $(LIMIT),--limit $(LIMIT)) \
		--batch-size $(or $(BATCH_SIZE),100) \
		$(if $(DRY_RUN),--dry-run)

backfill-hashes: ## Backfill perceptual hashes for existing assets
	@echo "Backfilling perceptual hashes (batch-size=$(or $(BATCH_SIZE),500), limit=$(or $(LIMIT),all))..."
	uv run python -m image_search_service.queue.hash_backfill_jobs \
		--batch-size $(or $(BATCH_SIZE),500) \
		$(if $(LIMIT),--limit $(LIMIT))

# Face detection and recognition targets
faces-backfill: ## Backfill face detection for assets without faces
	@echo "Running face backfill (limit=$(or $(LIMIT),1000), batch-size=$(or $(BATCH_SIZE),8))..."
	uv run python -m image_search_service.scripts.cli faces backfill --limit $(or $(LIMIT),5000) --batch-size $(or $(BATCH_SIZE),8)

faces-cluster: ## Cluster unlabeled faces using HDBSCAN
	@echo "Running face clustering (max-faces=$(or $(MAX_FACES),50000))..."
	uv run python -m image_search_service.scripts.cli faces cluster --max-faces $(or $(MAX_FACES),50000)

faces-assign: ## Assign new faces to known persons
	@echo "Running face assignment (max-faces=$(or $(MAX_FACES),1000))..."
	uv run python -m image_search_service.scripts.cli faces assign --max-faces $(or $(MAX_FACES),5000)

faces-centroids: ## Compute/update person centroid embeddings
	@echo "Computing person centroids..."
	uv run python -m image_search_service.scripts.cli faces centroids

faces-stats: ## Show face detection and recognition statistics
	@echo "Face pipeline statistics..."
	uv run python -m image_search_service.scripts.cli faces stats

faces-ensure-collection: ## Ensure Qdrant faces collection exists
	@echo "Ensuring Qdrant faces collection..."
	uv run python -m image_search_service.scripts.cli faces ensure-collection

faces-cluster-dual: ## Run dual-mode clustering (supervised + unsupervised)
	@echo "Running dual-mode clustering..."
	uv run python -m image_search_service.scripts.cli faces cluster-dual \
		--person-threshold $(or $(PERSON_THRESHOLD),0.7) \
		--unknown-method $(or $(UNKNOWN_METHOD),hdbscan) \
		--unknown-min-size $(or $(UNKNOWN_MIN_SIZE),3)

faces-train-matching: ## Train face matching model with triplet loss
	@echo "Training face matching model..."
	uv run python -m image_search_service.scripts.cli faces train-matching \
		--epochs $(or $(EPOCHS),20) \
		--margin $(or $(MARGIN),0.2) \
		--batch-size $(or $(BATCH_SIZE),32) \
		--min-faces $(or $(MIN_FACES),5)

faces-pipeline: faces-ensure-collection faces-backfill faces-cluster faces-assign faces-centroids faces-stats ## Run full face detection pipeline (legacy)
	@echo "Face pipeline complete!"

faces-pipeline-dual: ## Run dual-mode face pipeline (detect → cluster → stats)
	@echo "=========================================="
	@echo "  DUAL-MODE FACE PIPELINE"
	@echo "=========================================="
	@echo ""
	@echo "Step 1/4: Ensuring Qdrant collection..."
	@$(MAKE) faces-ensure-collection
	@echo ""
	@echo "Step 2/4: Detecting faces in images..."
	@$(MAKE) faces-backfill LIMIT=$(or $(LIMIT),5000)
	@echo ""
	@echo "Step 3/4: Running dual-mode clustering..."
	@$(MAKE) faces-cluster-dual \
		PERSON_THRESHOLD=$(or $(PERSON_THRESHOLD),0.7) \
		UNKNOWN_METHOD=$(or $(UNKNOWN_METHOD),hdbscan)
	@echo ""
	@echo "Step 4/4: Showing statistics..."
	@$(MAKE) faces-stats
	@echo ""
	@echo "=========================================="
	@echo "  PIPELINE COMPLETE"
	@echo "=========================================="
	@echo ""
	@echo "Next steps:"
	@echo "  1. Label clusters: curl -X POST localhost:8000/api/v1/faces/clusters/{id}/label"
	@echo "  2. Train model:    make faces-train-matching EPOCHS=20"
	@echo "  3. Re-cluster:     make faces-cluster-dual"
	@echo ""

faces-pipeline-full: ## Run complete pipeline with training (detect → cluster → train → re-cluster)
	@echo "=========================================="
	@echo "  FULL FACE PIPELINE WITH TRAINING"
	@echo "=========================================="
	@echo ""
	@time_step() { \
	step_num=$$1; shift; \
	step_name=$$1; shift; \
	echo "Step $$step_num: $$step_name..."; \
	start=$$(date +%s); \
	"$$@"; \
	end=$$(date +%s); \
	echo "✓ Step $$step_num completed in $$((end - start))s"; \
	echo ""; \
	}; \
	time_step "1/6" "Ensuring Qdrant collection" $(MAKE) faces-ensure-collection; \
	time_step "2/6" "Detecting faces in images" $(MAKE) faces-backfill LIMIT=$(or $(LIMIT),5000); \
	time_step "3/6" "Running initial dual-mode clustering" $(MAKE) faces-cluster-dual PERSON_THRESHOLD=$(or $(PERSON_THRESHOLD),0.7) UNKNOWN_METHOD=$(or $(UNKNOWN_METHOD),hdbscan); \
	time_step "4/6" "Training face matching model" $(MAKE) faces-train-matching EPOCHS=$(or $(EPOCHS),20) MARGIN=$(or $(MARGIN),0.2) MIN_FACES=$(or $(MIN_FACES),5); \
	time_step "5/6" "Re-clustering with trained model" $(MAKE) faces-cluster-dual PERSON_THRESHOLD=$(or $(PERSON_THRESHOLD),0.7) UNKNOWN_METHOD=$(or $(UNKNOWN_METHOD),hdbscan); \
	time_step "6/6" "Showing final statistics" $(MAKE) faces-stats
	@echo "=========================================="
	@echo "  FULL PIPELINE COMPLETE"
	@echo "=========================================="
	@echo ""}
