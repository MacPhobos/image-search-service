"""add face detection tables

Revision ID: 009
Revises: 008
Create Date: 2025-12-23 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

# revision identifiers, used by Alembic.
revision: str = "009"
down_revision: str | Sequence[str] | None = "008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create person_status enum (idempotent)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE person_status AS ENUM ('active', 'merged', 'hidden');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)

    # Create prototype_role enum (idempotent)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE prototype_role AS ENUM ('centroid', 'exemplar');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)

    # Create persons table
    op.create_table(
        "persons",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column(
            "status",
            sa.Enum("active", "merged", "hidden", name="person_status"),
            nullable=False,
            server_default="active",
        ),
        sa.Column("merged_into_id", UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.ForeignKeyConstraint(
            ["merged_into_id"],
            ["persons.id"],
            name="fk_persons_merged_into_id",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes on persons
    op.create_index("ix_persons_name_lower", "persons", [sa.text("LOWER(name)")], unique=True)
    op.create_index("ix_persons_status", "persons", ["status"])

    # Create face_instances table
    op.create_table(
        "face_instances",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("asset_id", sa.Integer(), nullable=False),
        sa.Column("bbox_x", sa.Integer(), nullable=False),
        sa.Column("bbox_y", sa.Integer(), nullable=False),
        sa.Column("bbox_w", sa.Integer(), nullable=False),
        sa.Column("bbox_h", sa.Integer(), nullable=False),
        sa.Column("landmarks", JSONB, nullable=True),
        sa.Column("detection_confidence", sa.Float(), nullable=False),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column("qdrant_point_id", UUID(as_uuid=True), nullable=False),
        sa.Column("cluster_id", sa.String(100), nullable=True),
        sa.Column("person_id", UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.ForeignKeyConstraint(
            ["asset_id"],
            ["image_assets.id"],
            name="fk_face_instances_asset_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["person_id"],
            ["persons.id"],
            name="fk_face_instances_person_id",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "asset_id",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            name="uq_face_instance_location",
        ),
        sa.UniqueConstraint("qdrant_point_id", name="uq_face_instances_qdrant_point_id"),
    )

    # Create indexes on face_instances
    op.create_index("ix_face_instances_asset_id", "face_instances", ["asset_id"])
    op.create_index("ix_face_instances_cluster_id", "face_instances", ["cluster_id"])
    op.create_index("ix_face_instances_person_id", "face_instances", ["person_id"])
    op.create_index("ix_face_instances_quality", "face_instances", ["quality_score"])

    # Create person_prototypes table
    op.create_table(
        "person_prototypes",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("person_id", UUID(as_uuid=True), nullable=False),
        sa.Column("face_instance_id", UUID(as_uuid=True), nullable=True),
        sa.Column("qdrant_point_id", UUID(as_uuid=True), nullable=False),
        sa.Column(
            "role",
            sa.Enum("centroid", "exemplar", name="prototype_role"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.ForeignKeyConstraint(
            ["person_id"],
            ["persons.id"],
            name="fk_person_prototypes_person_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["face_instance_id"],
            ["face_instances.id"],
            name="fk_person_prototypes_face_instance_id",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes on person_prototypes
    op.create_index(
        "ix_person_prototypes_person_id", "person_prototypes", ["person_id"]
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes on person_prototypes
    op.drop_index("ix_person_prototypes_person_id", table_name="person_prototypes")

    # Drop person_prototypes table
    op.drop_table("person_prototypes")

    # Drop indexes on face_instances
    op.drop_index("ix_face_instances_quality", table_name="face_instances")
    op.drop_index("ix_face_instances_person_id", table_name="face_instances")
    op.drop_index("ix_face_instances_cluster_id", table_name="face_instances")
    op.drop_index("ix_face_instances_asset_id", table_name="face_instances")

    # Drop face_instances table
    op.drop_table("face_instances")

    # Drop indexes on persons
    op.drop_index("ix_persons_status", table_name="persons")
    op.drop_index("ix_persons_name_lower", table_name="persons")

    # Drop persons table
    op.drop_table("persons")

    # Drop enums (idempotent)
    op.execute("DROP TYPE IF EXISTS prototype_role")
    op.execute("DROP TYPE IF EXISTS person_status")
