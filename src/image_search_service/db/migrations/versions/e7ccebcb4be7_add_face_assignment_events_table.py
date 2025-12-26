"""add face_assignment_events table

Revision ID: e7ccebcb4be7
Revises: 009
Create Date: 2025-12-24 11:46:54.290514

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'e7ccebcb4be7'
down_revision: str | Sequence[str] | None = '009'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'face_assignment_events',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('actor', sa.String(length=255), nullable=True),
        sa.Column('operation', sa.String(length=50), nullable=False),
        sa.Column('from_person_id', sa.UUID(), nullable=True),
        sa.Column('to_person_id', sa.UUID(), nullable=True),
        sa.Column('affected_photo_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('affected_face_instance_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('face_count', sa.Integer(), nullable=False),
        sa.Column('photo_count', sa.Integer(), nullable=False),
        sa.Column('note', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['from_person_id'], ['persons.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['to_person_id'], ['persons.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_face_assignment_events_created_at', 'face_assignment_events', ['created_at'], unique=False)
    op.create_index('ix_face_assignment_events_operation', 'face_assignment_events', ['operation'], unique=False)
    op.create_index('ix_face_assignment_events_from_person_id', 'face_assignment_events', ['from_person_id'], unique=False)
    op.create_index('ix_face_assignment_events_to_person_id', 'face_assignment_events', ['to_person_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_face_assignment_events_to_person_id', table_name='face_assignment_events')
    op.drop_index('ix_face_assignment_events_from_person_id', table_name='face_assignment_events')
    op.drop_index('ix_face_assignment_events_operation', table_name='face_assignment_events')
    op.drop_index('ix_face_assignment_events_created_at', table_name='face_assignment_events')
    op.drop_table('face_assignment_events')
