# Selective Docker Service Startup - Research and Proposal

> **Date**: 2026-02-21
> **Status**: Proposal (research complete, no implementation)
> **Scope**: `image-search-service` Docker service management

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Proposed Approaches](#proposed-approaches)
4. [Recommended Approach](#recommended-approach)
5. [Implementation Specification](#implementation-specification)
6. [Edge Cases and Failure Modes](#edge-cases-and-failure-modes)
7. [Migration Path](#migration-path)

---

## Executive Summary

The project currently uses `make db-up` to start **all three** Docker services (Postgres, Redis, Qdrant) unconditionally. This is inflexible for developers who run one or more services natively (e.g., Postgres installed via system package, Qdrant running as a separate process).

**Proposed solution**: Introduce environment variables (`DOCKER_POSTGRES`, `DOCKER_REDIS`, `DOCKER_QDRANT`) that default to `true` and can be set to `false` to skip individual services. The implementation uses **Docker Compose profiles** (a Docker Compose v2 feature, fully supported by the installed `docker compose v2.37.1`) combined with thin Makefile logic to translate env vars into `--profile` flags.

**Key findings**:
- No existing mechanism for selective service startup exists today.
- All three services are lazily initialized by the application, so a missing Docker container only causes errors when the service is first accessed at runtime -- not at import time.
- Docker Compose profiles are the cleanest approach because they keep all service definitions in one file while allowing fine-grained activation.
- The `make db-down` target also needs corresponding selective behavior.

---

## Current State Analysis

### 2.1 Makefile Targets

**File**: `/export/workspace/image-search/image-search-service/Makefile`

```makefile
db-up: ## Start Postgres and Redis containers
    docker compose -f docker-compose.dev.yml up -d

db-down: ## Stop Postgres and Redis containers
    docker compose -f docker-compose.dev.yml down
```

Observations:
- The `db-up` help text says "Postgres and Redis" but the compose file also includes Qdrant -- the help text is stale.
- Both targets operate unconditionally on the entire compose file.
- No other targets reference `docker compose` directly (all other targets use `uv run` or `make` sub-targets).
- There is no `db-restart` target (though one could be useful).

### 2.2 Docker Compose File

**File**: `/export/workspace/image-search/image-search-service/docker-compose.dev.yml`

Three services are defined:

| Service | Image | Container Name | Host Port | Health Check | Volume |
|---------|-------|---------------|-----------|-------------|--------|
| `postgres` | `postgres:16` | `image-search-postgres` | `15432:5432` | `pg_isready -U postgres` | `postgres_data` (named) |
| `redis` | `redis:7` | `image-search-redis` | `6379:6379` | `redis-cli ping` | None |
| `qdrant` | `qdrant/qdrant:latest` | `image-search-qdrant` | `6333:6333`, `6334:6334` | None | `qdrant_data` (named) |

Observations:
- Postgres uses a **non-standard host port** (`15432`) -- this avoids conflicts with a natively-running Postgres on port `5432`. This is evidence that the developer may already be running Postgres natively and only uses Docker for certain scenarios.
- Redis uses the **standard port** (`6379`) -- a native Redis on the same port would conflict.
- Qdrant uses **standard ports** (`6333`, `6334`).
- Qdrant has no health check defined.
- The compose file uses the deprecated `version: '3.8'` key (harmless but outdated).

### 2.3 Application Configuration

**File**: `/export/workspace/image-search/image-search-service/src/image_search_service/core/config.py`

```python
class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/image_search"
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
```

The application connects to services via URLs configured through environment variables. It does **not** care whether the service is Docker-managed or native -- only the URL matters.

**Current `.env` file** (production/dev):
```
DATABASE_URL=postgresql+asyncpg://image-search:somepassword@localhost:5432/image-search
REDIS_URL=redis://localhost:6379/0
QDRANT_URL=http://localhost:6333
```

Note: The `.env` points to `localhost:5432` (native Postgres), while the Docker container exposes on `15432`. This confirms the developer is **already running Postgres natively** and the Docker Postgres is an alternative.

The `.env.example` file shows:
```
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:15432/image_search
```

This points to the Docker-provided Postgres port (`15432`).

### 2.4 Lazy Initialization Pattern

All three external service clients use **lazy initialization** -- connections are only established on first use, not at import time or app startup:

- **Database** (`db/session.py`): `get_engine()` creates the async engine on first call.
- **Redis** (`queue/worker.py`): `get_redis()` creates the Redis connection on first call.
- **Qdrant** (`vector/qdrant.py`): `get_qdrant_client()` creates the client on first call.

This means the application **starts successfully** even if a service is unreachable. It will fail with a connection error only when the missing service is first accessed. The one exception is Qdrant validation at startup (controlled by `QDRANT_STRICT_STARTUP`, which defaults to `true`).

### 2.5 Startup Validation

The `main.py` lifespan handler performs Qdrant collection validation at startup:
- If `QDRANT_STRICT_STARTUP=true` (default) and Qdrant is unreachable, the app **exits with sys.exit(1)**.
- If `QDRANT_STRICT_STARTUP=false`, it logs a warning and continues.

There is no equivalent strict validation for Postgres or Redis at startup.

### 2.6 Service Dependencies

There are no `depends_on` directives in the compose file. The services are independent at the Docker level:

| Service | Depends On (Application Level) |
|---------|-------------------------------|
| API Server | Postgres (for DB queries), Qdrant (for vector search), Redis (optional, for queue endpoints) |
| RQ Worker | Postgres (for DB writes), Redis (for job queue), Qdrant (for vector upserts) |
| CLI Scripts | Postgres, Qdrant (Redis not needed for most scripts) |

---

## Proposed Approaches

### Approach A: Makefile-Level Conditional Logic

**Mechanism**: Use shell conditionals in the Makefile to selectively call `docker compose up` for individual services.

```makefile
db-up:
    @if [ "$(DOCKER_POSTGRES)" != "false" ]; then \
        docker compose -f docker-compose.dev.yml up -d postgres; \
    fi
    @if [ "$(DOCKER_REDIS)" != "false" ]; then \
        docker compose -f docker-compose.dev.yml up -d redis; \
    fi
    @if [ "$(DOCKER_QDRANT)" != "false" ]; then \
        docker compose -f docker-compose.dev.yml up -d qdrant; \
    fi
```

**Pros**:
- Simple, no compose file changes.
- Works with any Docker Compose version.
- Easy to understand.

**Cons**:
- Verbose Makefile logic.
- Each service started as a separate `docker compose up` invocation (slightly slower, though negligible).
- `docker compose down` behavior is unclear -- it still tears down the entire project regardless.
- Does not leverage Docker Compose's built-in capabilities.
- Hard to extend (adding a new service means updating multiple Makefile targets).

### Approach B: Docker Compose Profiles

**Mechanism**: Assign each service a `profiles` list in the compose file. Services with profiles are only started when their profile is explicitly activated.

```yaml
services:
  postgres:
    profiles: ["postgres"]
    # ... existing config
  redis:
    profiles: ["redis"]
    # ... existing config
  qdrant:
    profiles: ["qdrant"]
    # ... existing config
```

Then activate profiles via `COMPOSE_PROFILES` env var or `--profile` flag:

```makefile
db-up:
    COMPOSE_PROFILES=$(shell echo \
        $(if $(filter-out false,$(or $(DOCKER_POSTGRES),true)),postgres,) \
        $(if $(filter-out false,$(or $(DOCKER_REDIS),true)),redis,) \
        $(if $(filter-out false,$(or $(DOCKER_QDRANT),true)),qdrant,) \
    | tr ' ' ',') \
    docker compose -f docker-compose.dev.yml up -d
```

**Pros**:
- Uses Docker Compose's native feature designed for this purpose.
- `docker compose down` respects profiles -- only tears down active services.
- Declarative: profile membership is defined in the compose file.
- Docker Compose v2.37.1 (installed) fully supports profiles.
- Single `docker compose up` invocation regardless of number of services.

**Cons**:
- Slightly more complex Makefile logic to construct the `COMPOSE_PROFILES` string.
- Developers need to understand the profiles concept.
- Services with profiles are NOT started by default (by design) -- must always provide profile flags.

### Approach C: Multiple Compose Files (Override Pattern)

**Mechanism**: Break services into separate compose files and conditionally include them.

```
docker-compose.dev.yml         # Base (empty or shared config)
docker-compose.postgres.yml    # Postgres service
docker-compose.redis.yml       # Redis service
docker-compose.qdrant.yml      # Qdrant service
```

```makefile
db-up:
    docker compose \
        $(if $(filter-out false,$(or $(DOCKER_POSTGRES),true)),-f docker-compose.postgres.yml) \
        $(if $(filter-out false,$(or $(DOCKER_REDIS),true)),-f docker-compose.redis.yml) \
        $(if $(filter-out false,$(or $(DOCKER_QDRANT),true)),-f docker-compose.qdrant.yml) \
        up -d
```

**Pros**:
- Maximum flexibility and separation of concerns.
- Each file is self-contained and easy to understand.
- Works with all Docker Compose versions.

**Cons**:
- File proliferation (4 files instead of 1).
- Volume definitions need to be in a shared base file or duplicated.
- More complex `docker compose` invocations.
- Harder to see the full picture at a glance.
- Overkill for 3 services.

---

## Recommended Approach

**Approach B: Docker Compose Profiles** is the recommended approach.

### Rationale

1. **Native Feature**: Docker Compose profiles are specifically designed for this use case. Using a purpose-built feature rather than working around it leads to more maintainable code.

2. **Down Behavior**: With profiles, `docker compose down` automatically respects which services are active. Approach A would require additional logic to avoid accidentally stopping natively-running services.

3. **Simplicity**: Despite the Makefile logic to construct the `COMPOSE_PROFILES` string, the overall solution is simpler than alternatives because Docker Compose handles the orchestration logic.

4. **Version Support**: The installed Docker Compose v2.37.1 fully supports profiles (introduced in Docker Compose v2.1).

5. **No File Proliferation**: All services remain in a single compose file, just with profile annotations added.

6. **Future-Proof**: Adding a new service (e.g., MinIO for object storage) means adding one `profiles` entry and one env var check -- minimal changes.

---

## Implementation Specification

### 5.1 Environment Variables

| Variable | Default | Values | Effect |
|----------|---------|--------|--------|
| `DOCKER_POSTGRES` | `true` | `true` / `false` | Controls whether Docker Compose starts the Postgres container |
| `DOCKER_REDIS` | `true` | `true` / `false` | Controls whether Docker Compose starts the Redis container |
| `DOCKER_QDRANT` | `true` | `true` / `false` | Controls whether Docker Compose starts the Qdrant container |

These variables are **only** consumed by the Makefile. They have no effect on the application itself (the app uses `DATABASE_URL`, `REDIS_URL`, `QDRANT_URL` to connect).

These variables can be set in the `.env` file, which the Makefile loads automatically via `-include .env`. This is the recommended approach for persistent configuration. See section 5.3 for precedence rules and section 5.4 for usage examples.

### 5.2 Docker Compose Changes

**File**: `docker-compose.dev.yml`

```yaml
services:
  postgres:
    profiles: ["postgres"]
    image: postgres:16
    container_name: image-search-postgres
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: image_search
    ports:
      - "15432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    profiles: ["redis"]
    image: redis:7
    container_name: image-search-redis
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5

  qdrant:
    profiles: ["qdrant"]
    image: qdrant/qdrant:latest
    container_name: image-search-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:6333/healthz || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 3

volumes:
  postgres_data:
  qdrant_data:
```

Changes from current:
1. Added `profiles: ["postgres"]`, `profiles: ["redis"]`, `profiles: ["qdrant"]` to each service.
2. Removed deprecated `version: '3.8'` key.
3. Added health check for Qdrant (was missing).

### 5.3 Makefile Changes

```makefile
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
COMPOSE_PROFILES := $(subst $(_SPACE),$(_COMMA),$(_PROFILES))

db-up: ## Start Docker containers (disable with DOCKER_POSTGRES=false etc.)
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
```

### 5.4 Usage Examples

**Persistent configuration via `.env` (recommended for daily use):**

```bash
# Add to your .env file (already loaded by the Makefile via -include):
DOCKER_POSTGRES=false
DOCKER_QDRANT=false
# Now every `make db-up` and `make db-down` only manages Redis automatically.

make db-up    # Only starts Redis (reads DOCKER_* from .env)
make db-down  # Only stops Redis
```

**One-time override via command-line args (overrides .env):**

```bash
# Even if .env has DOCKER_POSTGRES=false, this starts all three services:
make db-up DOCKER_POSTGRES=true DOCKER_QDRANT=true

# Or skip a single service for one invocation:
make db-up DOCKER_REDIS=false
```

**Default behavior (no DOCKER_* set anywhere):**

```bash
# Without any DOCKER_* in .env or command line, all default to true:
make db-up    # Starts Postgres, Redis, and Qdrant (same as today)
make db-down  # Stops all three
```

**Check what is running:**

```bash
make db-status
```

**Variable Precedence** (verified with GNU Make):

| Priority | Source | Example | Overrides `.env`? |
|----------|--------|---------|-------------------|
| 1 (highest) | Make command-line arg | `make db-up DOCKER_POSTGRES=true` | Yes |
| 2 | `.env` file (via `-include .env`) | `DOCKER_POSTGRES=false` in `.env` | N/A |
| 3 | Shell-exported env var | `export DOCKER_POSTGRES=false` | No (`.env` wins) |
| 4 (lowest) | Makefile `?=` default | `DOCKER_POSTGRES ?= true` | No |

The key subtlety: `-include .env` performs a full `=` (recursive) assignment in Make, which overrides the later `?=` conditional defaults. Shell-exported env vars only take effect for variables NOT set by the `.env` file or command line. If you need a one-time override while `.env` has a value set, use command-line args.

### 5.5 `.env.example` Updates

Add to the top of `.env.example`. Since the Makefile loads `.env` via `-include`, these variables are automatically picked up without requiring shell `export` or command-line arguments:

```bash
# Docker Service Selection
# Controls which services `make db-up` / `make db-down` manage via Docker.
# Set to 'false' to skip a service (useful when running it natively).
# The Makefile loads this file automatically via `-include .env`.
# To override a .env value for one command, use: make db-up DOCKER_POSTGRES=true
DOCKER_POSTGRES=true
DOCKER_REDIS=true
DOCKER_QDRANT=true
```

Note: The `.env.example` shows uncommented defaults so that copying `.env.example` to `.env` immediately works. Developers then edit `.env` to set any service to `false` for their local setup.

### 5.6 CLAUDE.md Updates

Update the "Daily Development Workflow" section to mention selective service startup and add the new `db-status` target.

---

## Edge Cases and Failure Modes

### 6.1 Service Disabled but App Expects It

**Scenario**: Developer sets `DOCKER_REDIS=false` but does not have Redis running natively.

**Behavior**: The application starts successfully (lazy initialization). When the first Redis-dependent operation occurs (e.g., enqueueing a job), it throws a `ConnectionRefusedError`.

**Mitigation**: No code changes needed. The application's existing error handling surfaces the connection failure clearly. The `QueueService._is_redis_connected()` method already gracefully handles Redis being unavailable (returns empty data instead of crashing).

**Recommendation**: Add a `make check-services` target that pings each expected service URL and reports status. This is informational only, not blocking.

### 6.2 Port Conflicts

**Scenario**: Developer runs native Postgres on port 5432 and Docker Postgres on port 15432. Both can coexist.

**Current state**: This is already handled well. Docker Postgres maps to `15432` while native runs on `5432`. The `.env` file controls which URL the app uses.

**Scenario**: Developer runs native Redis on port 6379 and forgets to set `DOCKER_REDIS=false`.

**Behavior**: `docker compose up` will fail with a port binding error for the Redis container. The error message clearly states the port conflict.

**Mitigation**: Document this in the `.env.example` comments.

### 6.3 `make db-down` with Mixed Services

**Scenario**: Developer started with all services, then later wants to run `make db-down DOCKER_REDIS=false` (only stop Postgres and Qdrant).

**Behavior**: With Docker Compose profiles, `docker compose down` only tears down services whose profiles are active. So `COMPOSE_PROFILES=postgres,qdrant docker compose down` would only stop Postgres and Qdrant containers, leaving Redis running.

**This is correct behavior** -- the developer is saying "I manage Redis outside Docker, so don't touch it."

### 6.4 All Services Disabled

**Scenario**: Developer sets `DOCKER_POSTGRES=false DOCKER_REDIS=false DOCKER_QDRANT=false`.

**Behavior**: The Makefile detects an empty `COMPOSE_PROFILES` and prints "All Docker services disabled. Nothing to start." without invoking Docker Compose.

### 6.5 Volume Persistence

**Scenario**: Developer switches between Docker and native Postgres.

**Behavior**: Docker volumes (`postgres_data`, `qdrant_data`) persist independently. Data in Docker containers is separate from native service data. This is already the case and does not change.

### 6.6 Qdrant Strict Startup

**Scenario**: Developer disables Docker Qdrant (`DOCKER_QDRANT=false`) but does not run Qdrant natively. App starts with `QDRANT_STRICT_STARTUP=true`.

**Behavior**: The app will exit with a fatal error during startup because Qdrant is unreachable.

**Mitigation**: If the developer intentionally wants to run without Qdrant, they should also set `QDRANT_STRICT_STARTUP=false`. This is already documented in the project.

### 6.7 `.env` Format Compatibility with Make `-include`

**Scenario**: The `.env` file format evolves to include shell-isms like `export KEY=VALUE`, quoted values (`KEY="value"`), or variable interpolation (`KEY=${OTHER}`).

**Current state**: The project's `.env` file uses plain `KEY=VALUE` format with `#` comments. Make's `-include` parses this correctly because `KEY=VALUE` is valid Make syntax and `#` is a comment character in both Make and shell.

**What breaks `-include`**:
- `export KEY=VALUE` -- Make does not understand the `export` keyword and will error.
- `KEY="value with spaces"` -- Make preserves the literal quotes in the value.
- `KEY=${OTHER_VAR}` -- Make interprets `${}` differently than shell does.

**Mitigation**: The project's `.env` file has used plain `KEY=VALUE` consistently since project inception. The new `DOCKER_*` variables follow the same format. If the `.env` format ever changes to include shell-isms, the `-include .env` line would need to be replaced with the `grep`-based approach described in the "Alternatives Considered" note below.

**Alternatives Considered**: A more defensive approach using `$(shell grep ...)` to extract only `DOCKER_*` variables was considered. This would handle any `.env` format but adds complexity. Given the project's stable `.env` format, the simpler `-include .env` approach is preferred.

### 6.8 Shell Export Does Not Override `.env`

**Scenario**: Developer has `DOCKER_POSTGRES=false` in `.env` and runs `export DOCKER_POSTGRES=true; make db-up` expecting Postgres to start.

**Behavior**: The `.env` value wins. Postgres is NOT started. This is because `-include .env` performs a full `=` assignment that overrides the `?=` conditional default, and Make's `?=` only checks the environment for variables not already set by the Makefile itself.

**Mitigation**: Use command-line args instead: `make db-up DOCKER_POSTGRES=true`. Command-line args always have the highest priority in Make.

### 6.9 RQ Worker Without Redis

**Scenario**: Developer disables Docker Redis and does not run Redis natively, then runs `make worker`.

**Behavior**: The RQ listener worker immediately crashes with a `ConnectionRefusedError` because it cannot connect to Redis.

**Mitigation**: This is expected behavior. RQ fundamentally requires Redis. The error message is clear.

---

## Migration Path

### Phase 1: Non-Breaking Change

1. Add `-include .env` to the top of the Makefile (loads `.env` variables automatically).
2. Add `profiles` to each service in `docker-compose.dev.yml`.
3. Update Makefile `db-up` and `db-down` targets with profile logic.
4. Add `db-status` target.
5. Update `.env.example` with `DOCKER_*` variables (uncommented, defaults to `true`).
6. Update Makefile help text to be accurate (mentions "Postgres and Redis" but should include Qdrant).

**Zero breaking changes**: By defaulting all `DOCKER_*` variables to `true`, the behavior of `make db-up` and `make db-down` is identical to today. The `-include .env` line is non-fatal if `.env` does not exist, and existing `.env` files without `DOCKER_*` variables fall through to the `?=` defaults.

### Phase 2: Optional Enhancements

1. Add `make check-services` target to verify connectivity to all expected services.
2. Add a Qdrant health check to the compose file (currently missing).
3. Remove the deprecated `version: '3.8'` from the compose file.
4. Consider adding a `make db-restart` target for convenience.

### Phase 3: Documentation

1. Update `CLAUDE.md` root file's "Daily Development Workflow" section.
2. Update `image-search-service/CLAUDE.md` Local Development section.
3. Add example configurations for common scenarios:
   - "I run Postgres natively on macOS"
   - "I run everything in Docker"
   - "I run Postgres and Qdrant natively, only need Redis from Docker"

---

## Appendix: Verification Checklist

Before merging, verify:

- [ ] `make db-up` with no `DOCKER_*` in `.env` starts all 3 services (backward compatible)
- [ ] `make db-up` with `DOCKER_POSTGRES=false` in `.env` skips Postgres (starts Redis + Qdrant)
- [ ] `make db-up DOCKER_POSTGRES=true` overrides `.env` value (command-line wins)
- [ ] `make db-up DOCKER_REDIS=false DOCKER_QDRANT=false` (command-line) starts only Postgres
- [ ] `make db-down` stops only the Docker-managed services (respects same precedence)
- [ ] `make db-up DOCKER_POSTGRES=false DOCKER_REDIS=false DOCKER_QDRANT=false` prints informational message
- [ ] `make db-status` shows running containers
- [ ] `.env` file without `DOCKER_*` variables works (falls through to `?=` defaults)
- [ ] Missing `.env` file works (`-include` is non-fatal)
- [ ] The app starts successfully with any combination of Docker/native services (assuming correct URLs in `.env`)
- [ ] Port conflicts produce clear error messages
- [ ] Existing CI/CD pipelines are unaffected (they should use default `true` values)
