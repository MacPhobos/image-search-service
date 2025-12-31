"""add face prototype config keys

Revision ID: a5e555cf5477
Revises: 0d2febc7f1d5
Create Date: 2025-12-28 14:43:20.374599

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a5e555cf5477'
down_revision: str | Sequence[str] | None = '0d2febc7f1d5'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add face prototype configuration keys."""
    op.execute(
        """
        INSERT INTO system_configs (key, value, data_type, description, min_value, max_value, category)
        VALUES
        (
            'face_prototype_min_quality',
            '0.5',
            'float',
            'Minimum quality score required for a face to be used as an exemplar prototype. Higher values ensure only high-quality faces are selected as representatives.',
            '0.0',
            '1.0',
            'face_matching'
        ),
        (
            'face_prototype_max_exemplars',
            '5',
            'int',
            'Maximum number of exemplar faces to store per person. More exemplars improve matching accuracy but increase storage and computation costs.',
            '1',
            '20',
            'face_matching'
        )
        ON CONFLICT (key) DO NOTHING
        """
    )


def downgrade() -> None:
    """Remove face prototype configuration keys."""
    op.execute(
        """
        DELETE FROM system_configs
        WHERE key IN ('face_prototype_min_quality', 'face_prototype_max_exemplars')
        """
    )
