"""add ignored_directories table

Revision ID: c3198aefbaa4
Revises: d1e2f3g4h5i6
Create Date: 2026-01-19 00:00:00.000000

This migration adds the ignored_directories table to store directories that should be
excluded from directory listings during training session setup. This allows users to
hide unwanted directories (e.g., system folders, temporary directories) from the UI
without deleting them from the filesystem.

Business context: Users need to mark directories as "ignored" so they don't clutter
the directory selection UI when creating training sessions. Ignored directories can
be unignored later if needed.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c3198aefbaa4'
down_revision: str | Sequence[str] | None = 'd1e2f3g4h5i6'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create ignored_directories table."""
    op.create_table(
        'ignored_directories',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('path', sa.String(length=1024), nullable=False),
        sa.Column('reason', sa.String(length=255), nullable=True),
        sa.Column('ignored_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('ignored_by', sa.String(length=255), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('path', name='uq_ignored_directories_path')
    )
    op.create_index('idx_ignored_directories_path', 'ignored_directories', ['path'], unique=False)


def downgrade() -> None:
    """Drop ignored_directories table."""
    op.drop_index('idx_ignored_directories_path', table_name='ignored_directories')
    op.drop_table('ignored_directories')
