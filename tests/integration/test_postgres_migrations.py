"""PostgreSQL Alembic migration integration tests.

These tests run Alembic migrations against real PostgreSQL, validating that
all migration files produce valid DDL and are reversible.

30 migration files exist in src/image_search_service/db/migrations/versions/.
These migrations target PostgreSQL DDL syntax and are bypassed entirely by
the SQLite test setup (which uses Base.metadata.create_all()).

These tests catch:
- Invalid PostgreSQL DDL syntax
- Missing enum type creation
- Incorrect column types
- Foreign key ordering issues
- Non-reversible migrations
- Schema drift (model vs. migration mismatch)
"""

import pytest
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine

from image_search_service.db.models import Base


@pytest.mark.postgres
def test_alembic_full_upgrade(fresh_pg_database):  # type: ignore
    """Verify all migrations apply cleanly from empty DB to head.

    This catches:
    - Invalid PostgreSQL DDL syntax
    - Missing enum type creation
    - Incorrect column types
    - Foreign key ordering issues
    """
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", fresh_pg_database)

    # Upgrade to head (all migrations)
    command.upgrade(alembic_cfg, "head")

    # Verify we're at head
    script = ScriptDirectory.from_config(alembic_cfg)
    head_rev = script.get_current_head()

    engine = create_engine(fresh_pg_database)
    with engine.connect() as conn:
        context = MigrationContext.configure(conn)
        current_rev = context.get_current_revision()

    assert current_rev == head_rev
    engine.dispose()


@pytest.mark.postgres
def test_alembic_downgrade_to_base(fresh_pg_database):  # type: ignore
    """Verify all migrations can be cleanly rolled back.

    This catches missing downgrade implementations.
    """
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", fresh_pg_database)

    # First upgrade to head
    command.upgrade(alembic_cfg, "head")

    # Then downgrade to base
    command.downgrade(alembic_cfg, "base")

    # Verify we're at base (no revision)
    engine = create_engine(fresh_pg_database)
    with engine.connect() as conn:
        context = MigrationContext.configure(conn)
        current_rev = context.get_current_revision()

    assert current_rev is None
    engine.dispose()


@pytest.mark.postgres
def test_alembic_upgrade_downgrade_upgrade_cycle(fresh_pg_database):  # type: ignore
    """Verify migrations are idempotent: up -> down -> up produces same result."""
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", fresh_pg_database)

    # Cycle: up -> down -> up
    command.upgrade(alembic_cfg, "head")
    command.downgrade(alembic_cfg, "base")
    command.upgrade(alembic_cfg, "head")

    # Verify at head
    script = ScriptDirectory.from_config(alembic_cfg)
    head_rev = script.get_current_head()

    engine = create_engine(fresh_pg_database)
    with engine.connect() as conn:
        context = MigrationContext.configure(conn)
        current_rev = context.get_current_revision()

    assert current_rev == head_rev
    engine.dispose()


@pytest.mark.postgres
def test_no_pending_migrations(fresh_pg_database):  # type: ignore
    """Verify that models.py and migrations are in sync.

    Detects when someone adds a column to models.py but forgets
    to create a migration.
    """
    from alembic.autogenerate import compare_metadata

    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", fresh_pg_database)

    # Apply all migrations
    command.upgrade(alembic_cfg, "head")

    # Compare current DB schema with models
    engine = create_engine(fresh_pg_database)
    with engine.connect() as conn:
        context = MigrationContext.configure(conn)
        diff = compare_metadata(context, Base.metadata)

    engine.dispose()

    # Filter out noise (index name differences, etc.)
    # Note: Alembic may report index differences due to naming conventions
    significant_diffs = [
        d
        for d in diff
        if d[0]
        not in (
            "remove_index",  # Index naming may differ
            "add_index",  # Auto-generated indexes may have different names
        )
    ]

    assert len(significant_diffs) == 0, (
        "Models and migrations are out of sync. Diffs:\n"
        + "\n".join(str(d) for d in significant_diffs)
    )
