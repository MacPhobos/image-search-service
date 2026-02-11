"""seed unknown person config defaults

Add configuration keys for unknown person discovery feature:
- unknown_person_min_display_count: Minimum faces to show unknown person group
- unknown_person_default_threshold: Default similarity threshold for clustering
- unknown_person_max_faces: Maximum unlabeled faces to process
- unknown_person_chunk_size: Batch size for embedding retrieval

These settings control the unknown person discovery system, which helps users
identify and label clusters of similar unlabeled faces.

Revision ID: 34633924ee16
Revises: 5b9f75181a7d
Create Date: 2026-02-11 15:48:56.902313

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '34633924ee16'
down_revision: str | Sequence[str] | None = '5b9f75181a7d'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add unknown person discovery configuration keys."""
    # Insert unknown_person_min_display_count (int: 1-1000)
    op.execute("""
        INSERT INTO system_configs (
            key, value, data_type, min_value, max_value, category, description
        )
        VALUES (
            'unknown_person_min_display_count',
            '5',
            'int',
            '1',
            '1000',
            'face_matching',
            'Minimum number of faces required to display an unknown person group'
        )
        ON CONFLICT (key) DO NOTHING;
    """)

    # Insert unknown_person_default_threshold (float: 0.0-1.0)
    op.execute("""
        INSERT INTO system_configs (
            key, value, data_type, min_value, max_value, category, description
        )
        VALUES (
            'unknown_person_default_threshold',
            '0.70',
            'float',
            '0.0',
            '1.0',
            'face_matching',
            'Default similarity threshold for unknown person clustering (0.0 = loose, 1.0 = strict)'
        )
        ON CONFLICT (key) DO NOTHING;
    """)

    # Insert unknown_person_max_faces (int: 1000-1000000)
    op.execute("""
        INSERT INTO system_configs (
            key, value, data_type, min_value, max_value, category, description
        )
        VALUES (
            'unknown_person_max_faces',
            '50000',
            'int',
            '1000',
            '1000000',
            'face_matching',
            'Maximum number of unlabeled faces to process for unknown person discovery'
        )
        ON CONFLICT (key) DO NOTHING;
    """)

    # Insert unknown_person_chunk_size (int: 100-50000)
    op.execute("""
        INSERT INTO system_configs (
            key, value, data_type, min_value, max_value, category, description
        )
        VALUES (
            'unknown_person_chunk_size',
            '10000',
            'int',
            '100',
            '50000',
            'face_matching',
            'Batch size for retrieving embeddings during unknown person discovery'
        )
        ON CONFLICT (key) DO NOTHING;
    """)


def downgrade() -> None:
    """Remove unknown person discovery configuration keys."""
    op.execute("DELETE FROM system_configs WHERE key = 'unknown_person_min_display_count'")
    op.execute("DELETE FROM system_configs WHERE key = 'unknown_person_default_threshold'")
    op.execute("DELETE FROM system_configs WHERE key = 'unknown_person_max_faces'")
    op.execute("DELETE FROM system_configs WHERE key = 'unknown_person_chunk_size'")
