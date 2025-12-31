"""backfill_training_subdirectory_trained_count

Revision ID: 8d46a4ba4167
Revises: 010
Create Date: 2025-12-31 16:30:03.682948

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '8d46a4ba4167'
down_revision: str | Sequence[str] | None = '010'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Backfill trained_count from completed training jobs.

    Recalculates trained_count for each TrainingSubdirectory by counting
    completed TrainingJob records where the asset's parent directory
    matches the subdirectory's path.
    """
    # Use raw SQL for performance when processing thousands of records
    # This uses regexp_replace to extract the parent directory from the file path
    # and matches it against the subdirectory path
    op.execute("""
        UPDATE training_subdirectories ts
        SET trained_count = (
            SELECT COUNT(tj.id)
            FROM training_jobs tj
            JOIN image_assets ia ON tj.asset_id = ia.id
            WHERE tj.session_id = ts.session_id
              AND tj.status = 'completed'
              AND regexp_replace(ia.path, '/[^/]+$', '') = ts.path
        )
    """)


def downgrade() -> None:
    """Reset trained_count to 0.

    This is acceptable data loss for this field since it can be recalculated.
    """
    op.execute("UPDATE training_subdirectories SET trained_count = 0")
