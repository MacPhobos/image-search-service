"""Add dismissed unknown person groups table.

This table tracks dismissed unknown person candidate groups across re-clustering runs.

Groups are identified by membership_hash (SHA-256 of sorted face_instance_ids),
ensuring the same group of faces remains dismissed regardless of cluster_id changes.

Revision ID: 5b9f75181a7d
Revises: df737c255b50
Create Date: 2026-02-11 15:38:10.666921

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '5b9f75181a7d'
down_revision: str | Sequence[str] | None = 'df737c255b50'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create dismissed_unknown_person_groups table with membership hash tracking."""
    op.create_table(
        'dismissed_unknown_person_groups',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('membership_hash', sa.String(length=64), nullable=False),
        sa.Column('cluster_id', sa.String(length=50), nullable=True),
        sa.Column('face_count', sa.Integer(), nullable=False),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column('marked_as_noise', sa.Boolean(), nullable=False),
        sa.Column('dismissed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('face_instance_ids', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('membership_hash')
    )
    # Only create index on cluster_id (membership_hash index created by unique constraint)
    op.create_index(
        'ix_dismissed_unknown_person_groups_cluster_id',
        'dismissed_unknown_person_groups',
        ['cluster_id'],
        unique=False
    )


def downgrade() -> None:
    """Drop dismissed_unknown_person_groups table."""
    op.drop_index('ix_dismissed_unknown_person_groups_cluster_id', table_name='dismissed_unknown_person_groups')
    op.drop_table('dismissed_unknown_person_groups')
