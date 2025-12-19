.PHONY: help dev api lint format typecheck test db-up db-down migrate makemigrations worker ingest

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

test: ## Run pytest tests
	uv run pytest

db-up: ## Start Postgres and Redis containers
	docker compose -f docker-compose.dev.yml up -d

db-down: ## Stop Postgres and Redis containers
	docker compose -f docker-compose.dev.yml down

migrate: ## Run database migrations
	uv run alembic upgrade head

makemigrations: ## Create new migration
	@read -p "Enter migration message: " msg; \
	uv run alembic revision --autogenerate -m "$$msg"

worker: ## Start RQ worker
	uv run python -m image_search_service.queue.worker

api: ## Run API without reload (production-like)
	uv run uvicorn image_search_service.main:app --host 0.0.0.0 --port 8000

ingest: ## Ingest images from directory (usage: make ingest DIR=/path/to/images)
	@if [ -z "$(DIR)" ]; then echo "Usage: make ingest DIR=/path/to/images"; exit 1; fi
	curl -X POST http://localhost:8000/api/v1/assets/ingest \
		-H "Content-Type: application/json" \
		-d '{"rootPath": "$(DIR)", "recursive": true}'
