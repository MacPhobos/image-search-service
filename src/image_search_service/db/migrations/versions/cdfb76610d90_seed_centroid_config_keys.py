"""seed centroid config keys

Add centroid-based suggestion configuration keys:
- post_training_use_centroids: Enable/disable centroid-based suggestions
  (faster alternative to prototype-based suggestions)
- centroid_min_faces_for_suggestions: Minimum number of faces required
  for a person to be included in centroid-based suggestions

These settings control the centroid-based post-training suggestion system,
which provides faster suggestion generation by using person centroids instead
of individual face prototypes.

Revision ID: cdfb76610d90
Revises: c3198aefbaa4
Create Date: 2026-02-06 20:40:06.668846

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'cdfb76610d90'
down_revision: Union[str, Sequence[str], None] = 'c3198aefbaa4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add centroid-based suggestion configuration keys."""
    # Insert post_training_use_centroids (boolean: true|false)
    op.execute("""
        INSERT INTO system_configs (key, value, data_type, category, description)
        VALUES (
            'post_training_use_centroids',
            'true',
            'boolean',
            'face_matching',
            'Enable centroid-based post-training suggestions (faster alternative to prototype-based)'
        )
        ON CONFLICT (key) DO NOTHING;
    """)

    # Insert centroid_min_faces_for_suggestions (int: 1-100)
    op.execute("""
        INSERT INTO system_configs (
            key, value, data_type, min_value, max_value, category, description
        )
        VALUES (
            'centroid_min_faces_for_suggestions',
            '5',
            'int',
            '1',
            '100',
            'face_matching',
            'Minimum number of faces required for a person to be included in centroid-based suggestions'
        )
        ON CONFLICT (key) DO NOTHING;
    """)


def downgrade() -> None:
    """Remove centroid-based suggestion configuration keys."""
    op.execute("DELETE FROM system_configs WHERE key = 'post_training_use_centroids'")
    op.execute("DELETE FROM system_configs WHERE key = 'centroid_min_faces_for_suggestions'")
