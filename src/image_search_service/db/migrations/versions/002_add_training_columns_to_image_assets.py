"""add training columns to image_assets

Revision ID: 002
Revises: ce719ca53e7b
Create Date: 2025-12-19 14:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: str | Sequence[str] | None = "ce719ca53e7b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add new columns to image_assets table
    op.add_column(
        "image_assets", sa.Column("thumbnail_path", sa.String(500), nullable=True)
    )
    op.add_column("image_assets", sa.Column("width", sa.Integer(), nullable=True))
    op.add_column("image_assets", sa.Column("height", sa.Integer(), nullable=True))
    op.add_column("image_assets", sa.Column("file_size", sa.BigInteger(), nullable=True))
    op.add_column("image_assets", sa.Column("mime_type", sa.String(50), nullable=True))
    op.add_column(
        "image_assets",
        sa.Column("file_modified_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "image_assets",
        sa.Column(
            "training_status", sa.String(20), nullable=False, server_default="pending"
        ),
    )

    # Create indexes
    op.create_index(
        "idx_image_assets_training_status", "image_assets", ["training_status"]
    )
    op.create_index(
        "idx_image_assets_file_modified_at", "image_assets", ["file_modified_at"]
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes
    op.drop_index("idx_image_assets_file_modified_at", table_name="image_assets")
    op.drop_index("idx_image_assets_training_status", table_name="image_assets")

    # Drop columns
    op.drop_column("image_assets", "training_status")
    op.drop_column("image_assets", "file_modified_at")
    op.drop_column("image_assets", "mime_type")
    op.drop_column("image_assets", "file_size")
    op.drop_column("image_assets", "height")
    op.drop_column("image_assets", "width")
    op.drop_column("image_assets", "thumbnail_path")
