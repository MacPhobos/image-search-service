"""add training_subdirectories path index

Revision ID: 010
Revises: temporal_proto_001
Create Date: 2025-12-30 00:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "010"
down_revision: str | Sequence[str] | None = "temporal_proto_001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add index on path column for performance optimization.

    This index improves query performance when enriching directories
    with training status metadata.
    """
    op.create_index(
        "idx_training_subdirectories_path",
        "training_subdirectories",
        ["path"],
    )


def downgrade() -> None:
    """Remove path index."""
    op.drop_index("idx_training_subdirectories_path", table_name="training_subdirectories")
