"""add_person_centroid_table

Revision ID: d1e2f3g4h5i6
Revises: 012_post_train_suggestions
Create Date: 2026-01-16 00:00:00.000000

This migration creates the person_centroid table for storing computed
person centroid embeddings. Centroids are robust average face embeddings
used for improved face→person suggestion generation.

Key features:
- Versioning support (model_version, centroid_version)
- Staleness detection (source_face_ids_hash)
- Multi-centroid support (global + cluster modes)
- 1:1 mapping with Qdrant person_centroids collection
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import JSONB, UUID

# revision identifiers, used by Alembic.
revision: str = 'd1e2f3g4h5i6'
down_revision: str | Sequence[str] | None = '012_post_train_suggestions'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create person_centroid table with versioning and staleness detection."""

    # Create enum types first (required by SQLAlchemy model)
    centroid_type_enum = postgresql.ENUM(
        'global', 'cluster',
        name='centroid_type_enum',
        create_type=False
    )
    centroid_status_enum = postgresql.ENUM(
        'active', 'deprecated', 'building', 'failed',
        name='centroid_status_enum',
        create_type=False
    )

    # Create the enum types in database (idempotent)
    centroid_type_enum.create(op.get_bind(), checkfirst=True)
    centroid_status_enum.create(op.get_bind(), checkfirst=True)

    # Create person_centroid table using the enum types
    op.create_table(
        'person_centroid',
        sa.Column('centroid_id', UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('person_id', UUID(as_uuid=True), nullable=False),
        sa.Column('qdrant_point_id', UUID(as_uuid=True), nullable=False, unique=True),
        sa.Column('model_version', sa.String(64), nullable=False),
        sa.Column('centroid_version', sa.Integer(), nullable=False),
        sa.Column('centroid_type', centroid_type_enum, nullable=False, server_default='global'),
        sa.Column('cluster_label', sa.String(32), nullable=True, server_default='global'),
        sa.Column('n_faces', sa.Integer(), nullable=False),
        sa.Column('status', centroid_status_enum, nullable=False, server_default='active'),
        sa.Column('source_face_ids_hash', sa.String(64), nullable=True),
        sa.Column('build_params', JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('NOW()')),
        sa.ForeignKeyConstraint(['person_id'], ['persons.id'], ondelete='CASCADE'),
    )

    # Create indexes for efficient queries
    op.create_index('ix_person_centroid_person_id', 'person_centroid', ['person_id'])

    op.create_index('ix_person_centroid_version', 'person_centroid',
                    ['person_id', 'model_version', 'centroid_version'])

    # Partial index for active centroids only
    op.create_index('ix_person_centroid_status', 'person_centroid', ['status'],
                    postgresql_where=sa.text("status = 'active'"))

    # Unique constraint: prevent duplicate active centroids for same
    # person/version/type/label
    op.create_index(
        'ix_person_centroid_unique_active',
        'person_centroid',
        ['person_id', 'model_version', 'centroid_version',
         'centroid_type', 'cluster_label'],
        unique=True,
        postgresql_where=sa.text("status = 'active'")
    )


def downgrade() -> None:
    """Drop person_centroid table, indexes, and enum types."""
    # Drop indexes first
    op.drop_index('ix_person_centroid_unique_active', table_name='person_centroid')
    op.drop_index('ix_person_centroid_status', table_name='person_centroid')
    op.drop_index('ix_person_centroid_version', table_name='person_centroid')
    op.drop_index('ix_person_centroid_person_id', table_name='person_centroid')

    # Drop table
    op.drop_table('person_centroid')

    # Drop enum types (order matters: drop types after table)
    op.execute("DROP TYPE IF EXISTS centroid_status_enum")
    op.execute("DROP TYPE IF EXISTS centroid_type_enum")
