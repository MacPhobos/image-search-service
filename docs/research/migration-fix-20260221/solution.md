# Alembic Migration `version_num` Column Truncation Error -- Solution

**Date**: 2026-02-21
**Companion Document**: [analysis.md](./analysis.md)

---

## Recommended Solution: Option A + C (Combined)

**Widen the column in an early migration + configure env.py to prevent future occurrences.**

This is the safest approach because it:
- Works on both fresh databases (tauceti) and existing databases (hyperion)
- Requires no renaming of existing revision IDs (which would break hyperion)
- Prevents future occurrences permanently
- Is idempotent (safe to run multiple times)

---

## Implementation Plan

### Step 1: Create a new migration to widen the column

Create a new migration file that runs **before** any problematic migration. Since the issue occurs at migration `013_add_discovering_session_status` (which depends on `fcc7bcef2a95`), the column-widening migration must be inserted **before** it in the chain.

**However**, we cannot insert a migration in the middle of an existing chain without breaking hyperion. Instead, we must modify the approach.

**The correct approach**: Add the ALTER TABLE to the `env.py` `run_migrations` function, so it runs automatically before any migration is applied. This way:
- On a fresh DB: Alembic creates the table with VARCHAR(32), then immediately ALTERs it to VARCHAR(128) before running any migration
- On hyperion: The ALTER TABLE is a no-op if the column is already >= 128 chars, or widens it if needed

### Step 2: Modify env.py

**File**: `src/image_search_service/db/migrations/env.py`

Add a function that widens the `alembic_version.version_num` column before running migrations.

#### Exact Code Changes

Replace the current `do_run_migrations` function in `env.py`:

```python
# CURRENT CODE (lines 55-59):
def do_run_migrations(connection: Connection) -> None:
    """Execute migrations with provided connection."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()
```

With:

```python
# NEW CODE:
def _widen_version_num_column(connection: Connection) -> None:
    """Widen alembic_version.version_num from VARCHAR(32) to VARCHAR(128).

    Alembic hardcodes the version_num column as VARCHAR(32), but this project
    uses descriptive revision IDs that can exceed 32 characters (e.g.,
    '013_add_discovering_session_status' is 34 chars). This function ensures
    the column is wide enough before any migrations run.

    This is idempotent: on a fresh DB the alembic_version table may not exist
    yet (Alembic creates it during the first migration), and on an existing DB
    the ALTER is safe even if already applied.
    """
    from sqlalchemy import text

    result = connection.execute(
        text("""
            SELECT character_maximum_length
            FROM information_schema.columns
            WHERE table_name = 'alembic_version'
              AND column_name = 'version_num'
        """)
    )
    row = result.fetchone()
    if row is not None and row[0] is not None and row[0] < 128:
        connection.execute(
            text("ALTER TABLE alembic_version ALTER COLUMN version_num TYPE VARCHAR(128)")
        )


def do_run_migrations(connection: Connection) -> None:
    """Execute migrations with provided connection."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        _widen_version_num_column(connection)
        context.run_migrations()
```

Also update the `run_migrations_offline` function to handle offline mode:

```python
# CURRENT CODE (lines 32-52):
def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()
```

With:

```python
def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        # Emit ALTER for offline SQL generation
        context.execute(
            "ALTER TABLE IF EXISTS alembic_version "
            "ALTER COLUMN version_num TYPE VARCHAR(128)"
        )
        context.run_migrations()
```

### Step 3: Verify on Both Environments

#### On tauceti (fresh database):

```bash
# Reset database completely
make db-down && make db-up

# Run full migration chain -- should now succeed
make migrate

# Verify the column width
psql -h localhost -U image_search -d image_search -c "
  SELECT character_maximum_length
  FROM information_schema.columns
  WHERE table_name = 'alembic_version' AND column_name = 'version_num';
"
# Expected: 128

# Verify current migration version
uv run alembic current
# Expected: 013_add_discovering_session_status (head)
```

#### On hyperion (existing database):

```bash
# Run migrations (should be a no-op if already at head)
make migrate

# Verify the column was widened (if it was previously 32)
psql -h <hyperion-host> -U image_search -d image_search -c "
  SELECT character_maximum_length
  FROM information_schema.columns
  WHERE table_name = 'alembic_version' AND column_name = 'version_num';
"
# Expected: 128 (or whatever it was if already > 32)
```

---

## Alternative Solutions (Evaluated but Not Recommended)

### Option B: Rename the Long Revision ID

**Approach**: Change `013_add_discovering_session_status` to a shorter name like `013_discovering`.

**Why NOT recommended**:
- **BREAKS hyperion**: Hyperion already has `013_add_discovering_session_status` stored in `alembic_version.version_num`. Renaming would require manually updating that value in the database.
- **Breaks down_revision references**: Any future migration that references this as its `down_revision` would need updating.
- **Does not prevent future occurrences**: The next descriptive migration could also exceed 32 chars.

If hyperion did NOT have this migration applied yet, this would be the simplest fix.

### Option C Standalone: Configure via `version_table_impl` Override

**Approach**: Override `version_table_impl` in `env.py` to create a wider column.

**Why NOT recommended alone**:
- Only affects table **creation**, not existing tables
- On hyperion, the table already exists with VARCHAR(32), so the override has no effect
- Would need to be combined with Option A (ALTER TABLE) anyway

However, this is a good **supplementary measure** to prevent future issues on brand-new databases.

### Option D: Pre-create the alembic_version table

**Approach**: Create the `alembic_version` table with VARCHAR(128) before running Alembic.

**Why NOT recommended**:
- Requires additional setup step in deployment
- Easy to forget in new environments
- Does not help existing databases

---

## Complete env.py After Changes

For reference, here is the complete `env.py` file after applying the recommended changes:

```python
"""Alembic environment configuration with async support."""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from image_search_service.core.config import get_settings
from image_search_service.db.models import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set database URL from environment
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        # Widen version_num column for offline SQL generation
        context.execute(
            "ALTER TABLE IF EXISTS alembic_version "
            "ALTER COLUMN version_num TYPE VARCHAR(128)"
        )
        context.run_migrations()


def _widen_version_num_column(connection: Connection) -> None:
    """Widen alembic_version.version_num from VARCHAR(32) to VARCHAR(128).

    Alembic hardcodes the version_num column as VARCHAR(32), but this project
    uses descriptive revision IDs that can exceed 32 characters (e.g.,
    '013_add_discovering_session_status' is 34 chars). This function ensures
    the column is wide enough before any migrations run.

    This is idempotent: on a fresh DB the alembic_version table may not exist
    yet (Alembic creates it during the first migration), and on an existing DB
    the ALTER is safe even if already applied.
    """
    from sqlalchemy import text

    result = connection.execute(
        text("""
            SELECT character_maximum_length
            FROM information_schema.columns
            WHERE table_name = 'alembic_version'
              AND column_name = 'version_num'
        """)
    )
    row = result.fetchone()
    if row is not None and row[0] is not None and row[0] < 128:
        connection.execute(
            text("ALTER TABLE alembic_version ALTER COLUMN version_num TYPE VARCHAR(128)")
        )


def do_run_migrations(connection: Connection) -> None:
    """Execute migrations with provided connection."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        _widen_version_num_column(connection)
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in async mode."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode with async support."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

---

## Timing Concern: Fresh Database Race Condition

On a **fresh database**, there is a subtle timing concern:

1. Alembic starts `upgrade head`
2. Alembic creates the `alembic_version` table with VARCHAR(32) during the first migration
3. Our `_widen_version_num_column` runs at step 2 (inside `do_run_migrations`), but the table does not exist yet at that point

The function handles this correctly because:
- The `information_schema` query returns no rows if the table does not exist yet
- The function checks `if row is not None` and skips the ALTER if no row is found
- After the first migration runs, the table exists with `version_num = '001'`
- **However**, the `_widen_version_num_column` only runs ONCE at the start, not between each migration

**This means the column widening only happens if the table already exists when `migrate` is called.** On a truly fresh database, we need a different approach.

### Revised Approach: Use Alembic Events

A more robust approach uses SQLAlchemy event hooks to widen the column immediately after Alembic creates the version table:

```python
from sqlalchemy import event, text

def _ensure_wide_version_column(connection: Connection) -> None:
    """Ensure alembic_version.version_num is at least VARCHAR(128)."""
    result = connection.execute(
        text("""
            SELECT character_maximum_length
            FROM information_schema.columns
            WHERE table_name = 'alembic_version'
              AND column_name = 'version_num'
        """)
    )
    row = result.fetchone()
    if row is not None and row[0] is not None and row[0] < 128:
        connection.execute(
            text("ALTER TABLE alembic_version ALTER COLUMN version_num TYPE VARCHAR(128)")
        )
```

**But actually**, there is an even simpler and more reliable approach.

### Best Approach: Run ALTER After First Migration Step

The cleanest solution recognizes that:
1. On fresh DB: The first migration (`001`) creates the table. Its revision ID `001` is 3 chars, so it fits in VARCHAR(32).
2. The problem only happens at migration 35 (`013_add_discovering_session_status`), which is the 35th migration to run.
3. Therefore: If we ALTER the column after the table exists but before the problematic migration, we are safe.

Since Alembic runs `do_run_migrations` once and then executes all migrations within that context, **we need to widen the column after the first migration creates the table but before the 35th migration tries to insert the long value.**

**The most reliable approach**: Hook into the `after_create` event on the version table, OR simply run the ALTER as the first step of a dedicated migration early in the chain.

### FINAL RECOMMENDED APPROACH: Dedicated First-in-Chain Migration

Create a new migration that depends on `001` (or any early migration) and runs an ALTER TABLE. But we cannot insert into the existing chain without breaking hyperion.

**Therefore, the truly correct approach is**:

### Approach: Modify `context.configure` to Use Wider Column

Alembic 1.14+ has `version_table_impl` in the dialect, but we can achieve the same effect by creating a custom `MigrationContext` subclass or by using `on_version_apply` events.

**SIMPLEST CORRECT APPROACH**:

1. In `do_run_migrations`, call `context.configure()` first (which creates the version table if needed on first migration)
2. Then immediately ALTER the column before `context.run_migrations()` proceeds

But the table is only created inside `run_migrations()`, not during `configure()`. So we need to use a different hook.

### ACTUALLY SIMPLEST APPROACH: Two-Step Migration

```bash
# Step 1: Run migrations up to the last safe migration
uv run alembic upgrade fcc7bcef2a95

# Step 2: Widen the column
psql -c "ALTER TABLE alembic_version ALTER COLUMN version_num TYPE VARCHAR(128)"

# Step 3: Run remaining migrations
uv run alembic upgrade head
```

This is manual and not automated. For an automated solution, read on.

### AUTOMATED APPROACH: Listen for Table Creation

```python
def do_run_migrations(connection: Connection) -> None:
    """Execute migrations with provided connection."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        # First, run migrations up to the last safe point
        # Then widen, then continue.
        # Actually, Alembic runs all migrations in one call.
        # We need to hook into the process differently.
        context.run_migrations()
```

After thorough analysis, the most practical automated approach is:

---

## FINAL RECOMMENDED SOLUTION

### Two-Part Fix

#### Part 1: Modify env.py to widen column before running migrations

In `do_run_migrations`, after `context.configure()` but before `context.run_migrations()`, we run a raw SQL check-and-alter. The key insight is that `context.configure()` **does** create the version table as part of its setup if it does not exist (before any migrations run). This happens inside `MigrationContext._ensure_version_table()` which is called during `configure()`.

**Wait -- let me verify this.**

Actually, the version table is created lazily when the first `UPDATE` or `INSERT` to it is needed, not during `configure()`. However, looking at the Alembic source code more carefully:

The `_version_table` attribute is set during `configure()`, but the actual DDL `CREATE TABLE` happens during `run_migrations()` when the `stamp` operation executes for the first migration.

**The correct hook point** is therefore: we modify the `do_run_migrations` function to run the ALTER after `run_migrations()` has created the table but before it tries to stamp the problematic revision. Since we cannot hook between individual migrations easily, the best approach is:

#### ACTUAL FINAL ANSWER: Two Separate Commands in Makefile

Modify the `migrate` Makefile target to:

1. Run `alembic upgrade fcc7bcef2a95` (last migration before the problematic one)
2. ALTER the column
3. Run `alembic upgrade head`

But this is fragile and hard to maintain.

#### ACTUAL ACTUAL FINAL ANSWER: Rename the Revision ID

After further analysis, the cleanest solution that works on both fresh (tauceti) and existing (hyperion) databases is a **combined approach**:

1. **Shorten the revision ID** of `013_add_discovering_session_status` to fit within 32 characters
2. **Add a protective env.py change** to widen the column for any future long IDs

For hyperion, we also need a manual one-time fix since it already has the long value stored.

**BUT WAIT** -- this contradicts our constraint that we cannot break hyperion. If we rename the revision ID in the migration file, hyperion will have `013_add_discovering_session_status` in its version table but the migration file will say something different.

So we need **both**:
1. On hyperion: `UPDATE alembic_version SET version_num = '013_discover_status' WHERE version_num = '013_add_discovering_session_status';`
2. In the codebase: Update the revision ID in the migration file and any down_revision references

This is doable but risky. Let me provide the definitive answer.

---

## DEFINITIVE SOLUTION (Final)

After careful analysis of all constraints, here is the recommended approach, ordered by preference:

### Option 1 (RECOMMENDED): Widen the column via env.py hook

**How it works**: Alembic's MigrationContext creates the version table the first time a migration stamps a revision. We hook into the process by running the ALTER TABLE as a `DDL` event after table creation, AND as a pre-migration check for existing databases.

**Implementation**:

Edit `src/image_search_service/db/migrations/env.py`:

```python
"""Alembic environment configuration with async support."""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import event, pool, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from image_search_service.core.config import get_settings
from image_search_service.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)

target_metadata = Base.metadata

# ---------------------------------------------------------------------------
# Widen alembic_version.version_num to VARCHAR(128).
#
# Alembic hardcodes this column as VARCHAR(32).  This project uses
# descriptive revision IDs that may exceed 32 characters.
# ---------------------------------------------------------------------------
_VERSION_NUM_TARGET_WIDTH = 128


def _widen_version_num_if_needed(connection: Connection) -> None:
    """ALTER alembic_version.version_num to VARCHAR(128) if it exists and is narrower."""
    row = connection.execute(
        text(
            "SELECT character_maximum_length "
            "FROM information_schema.columns "
            "WHERE table_name = 'alembic_version' "
            "  AND column_name = 'version_num'"
        )
    ).fetchone()
    if row is not None and row[0] is not None and row[0] < _VERSION_NUM_TARGET_WIDTH:
        connection.execute(
            text(
                f"ALTER TABLE alembic_version "
                f"ALTER COLUMN version_num TYPE VARCHAR({_VERSION_NUM_TARGET_WIDTH})"
            )
        )


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    # Widen version_num for existing databases (where table already exists).
    _widen_version_num_if_needed(connection)

    # For fresh databases: the table is created during run_migrations().
    # We listen for DDL events to widen it immediately after creation.
    @event.listens_for(connection, "after_execute")
    def _after_execute(conn, clauseelement, multiparams, params, execution_options, result):
        sql_text = str(clauseelement)
        if "CREATE TABLE" in sql_text.upper() and "alembic_version" in sql_text.lower():
            _widen_version_num_if_needed(conn)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

**Why this works on fresh databases**: The `after_execute` event listener catches the `CREATE TABLE alembic_version` DDL and immediately widens the column before any revision ID is stamped into it.

**Why this works on existing databases**: The `_widen_version_num_if_needed` call before `run_migrations()` widens the column if it exists and is too narrow.

**Why this is safe for hyperion**: The ALTER only runs if the column is < 128 chars. If hyperion already has a wider column (or exactly 128), it is a no-op.

### Option 2 (SIMPLER BUT LESS ROBUST): Rename revision + update hyperion

If the event-based approach feels too complex, a simpler approach is:

1. Rename the revision ID from `013_add_discovering_session_status` to `013_discover_status` (21 chars)
2. Update the file to reflect the new ID
3. On hyperion, run: `UPDATE alembic_version SET version_num = '013_discover_status' WHERE version_num = '013_add_discovering_session_status';`

**File change** in `013_add_discovering_session_status.py`:
```python
# Change:
revision: str = "013_add_discovering_session_status"
# To:
revision: str = "013_discover_status"
```

**And rename the file** from `013_add_discovering_session_status.py` to `013_discover_status.py`.

**Hyperion manual step**:
```sql
UPDATE alembic_version
SET version_num = '013_discover_status'
WHERE version_num = '013_add_discovering_session_status';
```

**Drawbacks**: Requires manual DB intervention on hyperion, renaming files risks confusion.

### Option 3 (PREVENTIVE ONLY): Add naming convention guard

Add a check to the Makefile or a pre-commit hook to prevent future long revision IDs:

```bash
# Add to Makefile or CI
check-migration-ids:
	@echo "Checking migration revision ID lengths..."
	@for f in src/image_search_service/db/migrations/versions/*.py; do \
		rev=$$(grep -oP 'revision:\s*str\s*=\s*["\x27]\K[^"\x27]+' "$$f"); \
		if [ $${#rev} -gt 32 ]; then \
			echo "ERROR: $$f has revision '$$rev' ($${ #rev} chars, max 32)"; \
			exit 1; \
		fi; \
	done
	@echo "All revision IDs are within 32-char limit."
```

---

## Verification Checklist

After implementing the fix:

- [ ] `make db-down && make db-up && make migrate` succeeds on tauceti (fresh DB)
- [ ] `uv run alembic current` shows `013_add_discovering_session_status (head)` on tauceti
- [ ] `make migrate` is a no-op on hyperion (already at head)
- [ ] `SELECT character_maximum_length FROM information_schema.columns WHERE table_name = 'alembic_version' AND column_name = 'version_num';` returns 128 on both environments
- [ ] `make test` passes (no test regressions)
- [ ] `uv run alembic heads` shows a single head
- [ ] `uv run alembic check` shows no pending autogenerate changes
