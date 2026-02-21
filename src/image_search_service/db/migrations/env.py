"""Alembic environment configuration with async support."""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool, text
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
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Execute migrations with provided connection.

    Before running migrations we ensure the alembic_version table exists with
    a VARCHAR(128) version_num column.  Alembic hardcodes VARCHAR(32) which is
    too short for descriptive revision IDs such as
    '013_add_discovering_session_status' (34 chars).  We handle two cases:

    * Fresh database: CREATE TABLE IF NOT EXISTS creates the table with the
      wide column before Alembic tries to INSERT any revision ID.
    * Existing database: ALTER TABLE widens the column non-destructively if it
      is still the old VARCHAR(32) default.

    Both statements are committed in their own transaction so they take effect
    before Alembic opens its own migration transaction.
    """
    # Step 1: Ensure alembic_version exists with a wide version_num column.
    # If the table already exists this is a no-op.
    connection.execute(
        text(
            "CREATE TABLE IF NOT EXISTS alembic_version ("
            "    version_num VARCHAR(128) NOT NULL, "
            "    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)"
            ")"
        )
    )
    # Step 2: Widen the column on existing databases that still have
    # the Alembic-default VARCHAR(32).  On PostgreSQL this is always safe
    # and instant (no table rewrite needed for varchar widening).
    connection.execute(
        text(
            "ALTER TABLE alembic_version "
            "ALTER COLUMN version_num TYPE VARCHAR(128)"
        )
    )
    connection.commit()

    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
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
