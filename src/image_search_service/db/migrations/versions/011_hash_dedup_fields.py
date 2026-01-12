"""add_hash_dedup_fields

Add perceptual hash and deduplication tracking fields:
- image_assets.perceptual_hash: dHash for duplicate detection
- training_sessions.skipped_images: Count of skipped duplicates
- training_jobs.image_path: Full absolute path for audit trail
- training_jobs.skip_reason: Why job was skipped

Revision ID: 011_hash_dedup_fields
Revises: 737ef70e7bab
Create Date: 2026-01-11 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "011_hash_dedup_fields"
down_revision: str | Sequence[str] | None = "737ef70e7bab"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add hash deduplication fields."""
    # Add perceptual_hash to image_assets
    op.add_column(
        "image_assets",
        sa.Column("perceptual_hash", sa.String(64), nullable=True),
    )

    # Index for fast duplicate detection
    op.create_index(
        "idx_image_assets_perceptual_hash",
        "image_assets",
        ["perceptual_hash"],
    )

    # Add skipped_images counter to training_sessions
    op.add_column(
        "training_sessions",
        sa.Column("skipped_images", sa.Integer(), nullable=False, server_default="0"),
    )

    # Add image_path to training_jobs for audit trail
    op.add_column(
        "training_jobs",
        sa.Column("image_path", sa.String(500), nullable=True),
    )

    # Add skip_reason to training_jobs
    op.add_column(
        "training_jobs",
        sa.Column("skip_reason", sa.String(100), nullable=True),
    )


def downgrade() -> None:
    """Remove hash deduplication fields."""
    op.drop_column("training_jobs", "skip_reason")
    op.drop_column("training_jobs", "image_path")
    op.drop_column("training_sessions", "skipped_images")
    op.drop_index("idx_image_assets_perceptual_hash", table_name="image_assets")
    op.drop_column("image_assets", "perceptual_hash")
