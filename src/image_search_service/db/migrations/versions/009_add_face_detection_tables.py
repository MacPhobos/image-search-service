"""add face detection tables

Revision ID: 009
Revises: 008
Create Date: 2025-12-23 12:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "009"
down_revision: str | Sequence[str] | None = "008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema using raw SQL to avoid SQLAlchemy enum auto-creation issues."""

    # Create enum types (idempotent with IF NOT EXISTS approach)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE person_status AS ENUM ('active', 'merged', 'hidden');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)

    op.execute("""
        DO $$ BEGIN
            CREATE TYPE prototype_role AS ENUM ('centroid', 'exemplar');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)

    # Create persons table
    op.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(255) NOT NULL,
            status person_status NOT NULL DEFAULT 'active',
            merged_into_id UUID REFERENCES persons(id),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)

    # Create indexes on persons (with IF NOT EXISTS)
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ix_persons_name_lower
        ON persons (LOWER(name));
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_persons_status
        ON persons (status);
    """)

    # Create face_instances table
    op.execute("""
        CREATE TABLE IF NOT EXISTS face_instances (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            asset_id INTEGER NOT NULL REFERENCES image_assets(id) ON DELETE CASCADE,
            bbox_x INTEGER NOT NULL,
            bbox_y INTEGER NOT NULL,
            bbox_w INTEGER NOT NULL,
            bbox_h INTEGER NOT NULL,
            landmarks JSONB,
            detection_confidence REAL NOT NULL,
            quality_score REAL,
            qdrant_point_id UUID NOT NULL UNIQUE,
            cluster_id VARCHAR(100),
            person_id UUID REFERENCES persons(id) ON DELETE SET NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_face_instance_location
                UNIQUE (asset_id, bbox_x, bbox_y, bbox_w, bbox_h)
        );
    """)

    # Create indexes on face_instances
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_face_instances_asset_id
        ON face_instances (asset_id);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_face_instances_cluster_id
        ON face_instances (cluster_id);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_face_instances_person_id
        ON face_instances (person_id);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_face_instances_quality
        ON face_instances (quality_score);
    """)

    # Create person_prototypes table
    op.execute("""
        CREATE TABLE IF NOT EXISTS person_prototypes (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            person_id UUID NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
            face_instance_id UUID REFERENCES face_instances(id) ON DELETE SET NULL,
            qdrant_point_id UUID NOT NULL,
            role prototype_role NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)

    # Create index on person_prototypes
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_person_prototypes_person_id
        ON person_prototypes (person_id);
    """)


def downgrade() -> None:
    """Downgrade schema."""
    # Drop tables in reverse order
    op.execute("DROP TABLE IF EXISTS person_prototypes CASCADE;")
    op.execute("DROP TABLE IF EXISTS face_instances CASCADE;")
    op.execute("DROP TABLE IF EXISTS persons CASCADE;")

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS prototype_role;")
    op.execute("DROP TYPE IF EXISTS person_status;")
