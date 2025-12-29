"""add temporal prototype fields

Revision ID: temporal_proto_001
Revises: 9511886120f4
Create Date: 2025-12-29 00:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "temporal_proto_001"
down_revision: str | Sequence[str] | None = "9511886120f4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema to support temporal prototype system."""

    # 1. Add new values to prototype_role enum
    op.execute("""
        ALTER TYPE prototype_role ADD VALUE IF NOT EXISTS 'primary';
    """)
    op.execute("""
        ALTER TYPE prototype_role ADD VALUE IF NOT EXISTS 'temporal';
    """)
    op.execute("""
        ALTER TYPE prototype_role ADD VALUE IF NOT EXISTS 'fallback';
    """)

    # 2. Create age_era_bucket enum type
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE age_era_bucket AS ENUM (
                'infant',
                'child',
                'teen',
                'young_adult',
                'adult',
                'senior'
            );
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)

    # 3. Add temporal metadata columns to person_prototypes
    op.execute("""
        ALTER TABLE person_prototypes
        ADD COLUMN IF NOT EXISTS age_era_bucket VARCHAR(50),
        ADD COLUMN IF NOT EXISTS decade_bucket VARCHAR(10);
    """)

    # 4. Add pinning metadata columns to person_prototypes
    op.execute("""
        ALTER TABLE person_prototypes
        ADD COLUMN IF NOT EXISTS is_pinned BOOLEAN DEFAULT FALSE,
        ADD COLUMN IF NOT EXISTS pinned_by VARCHAR(255),
        ADD COLUMN IF NOT EXISTS pinned_at TIMESTAMPTZ;
    """)

    # 5. Create performance indexes
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_person_prototypes_role
        ON person_prototypes (person_id, role);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_person_prototypes_era
        ON person_prototypes (person_id, age_era_bucket);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_person_prototypes_pinned
        ON person_prototypes (person_id, is_pinned);
    """)


def downgrade() -> None:
    """Downgrade schema - remove temporal prototype features."""

    # Drop indexes
    op.execute("DROP INDEX IF EXISTS ix_person_prototypes_pinned;")
    op.execute("DROP INDEX IF EXISTS ix_person_prototypes_era;")
    op.execute("DROP INDEX IF EXISTS ix_person_prototypes_role;")

    # Drop columns
    op.execute("""
        ALTER TABLE person_prototypes
        DROP COLUMN IF EXISTS pinned_at,
        DROP COLUMN IF EXISTS pinned_by,
        DROP COLUMN IF EXISTS is_pinned,
        DROP COLUMN IF EXISTS decade_bucket,
        DROP COLUMN IF EXISTS age_era_bucket;
    """)

    # Drop enum type (only if no other tables use it)
    op.execute("DROP TYPE IF EXISTS age_era_bucket;")

    # Note: Cannot remove enum values from prototype_role in PostgreSQL
    # They will remain but unused after downgrade
