"""add_resume_fields_to_face_detection_sessions

Revision ID: 0d2febc7f1d5
Revises: 974bfe0f68ed
Create Date: 2025-12-27 16:49:39.669594

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0d2febc7f1d5'
down_revision: Union[str, Sequence[str], None] = '974bfe0f68ed'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add resume support fields to face_detection_sessions
    op.add_column('face_detection_sessions', sa.Column('asset_ids_json', sa.Text(), nullable=True))
    op.add_column('face_detection_sessions', sa.Column('current_asset_index', sa.Integer(), nullable=False, server_default='0'))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('face_detection_sessions', 'current_asset_index')
    op.drop_column('face_detection_sessions', 'asset_ids_json')
