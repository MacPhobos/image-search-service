"""add_pending_suggestion_unique_index

Revision ID: c1d2e3f4g5h6
Revises: f6a668d072bb
Create Date: 2026-01-10 12:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c1d2e3f4g5h6'
down_revision: str | Sequence[str] | None = 'f6a668d072bb'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add partial unique index to prevent duplicate pending face suggestions.

    This index prevents race conditions when multiple find-more jobs might
    try to create the same suggestion simultaneously. The partial index only
    applies to pending suggestions, allowing the same face-person pair to
    exist multiple times in accepted/rejected states (for audit trail).
    """
    op.create_index(
        'uq_face_suggestion_pending',
        'face_suggestions',
        ['face_instance_id', 'suggested_person_id'],
        unique=True,
        postgresql_where=sa.text("status = 'pending'")
    )


def downgrade() -> None:
    """Remove pending suggestion unique index."""
    op.drop_index(
        'uq_face_suggestion_pending',
        table_name='face_suggestions'
    )
