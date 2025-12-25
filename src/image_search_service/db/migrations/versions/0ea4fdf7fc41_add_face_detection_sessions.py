"""add_face_detection_sessions

Revision ID: 0ea4fdf7fc41
Revises: e7ccebcb4be7
Create Date: 2025-12-25 16:20:59.671965

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0ea4fdf7fc41'
down_revision: str | Sequence[str] | None = 'e7ccebcb4be7'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('face_detection_sessions',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('training_session_id', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('total_images', sa.Integer(), nullable=False),
        sa.Column('processed_images', sa.Integer(), nullable=False),
        sa.Column('failed_images', sa.Integer(), nullable=False),
        sa.Column('faces_detected', sa.Integer(), nullable=False),
        sa.Column('faces_assigned', sa.Integer(), nullable=False),
        sa.Column('min_confidence', sa.Float(), nullable=False),
        sa.Column('min_face_size', sa.Integer(), nullable=False),
        sa.Column('batch_size', sa.Integer(), nullable=False),
        sa.Column('last_error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('job_id', sa.String(length=100), nullable=True),
        sa.ForeignKeyConstraint(['training_session_id'], ['training_sessions.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_face_detection_sessions_status', 'face_detection_sessions', ['status'], unique=False)
    op.create_index('ix_face_detection_sessions_training_session_id', 'face_detection_sessions', ['training_session_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_face_detection_sessions_training_session_id', table_name='face_detection_sessions')
    op.drop_index('ix_face_detection_sessions_status', table_name='face_detection_sessions')
    op.drop_table('face_detection_sessions')
