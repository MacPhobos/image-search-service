"""add_exif_metadata_to_image_assets

Add EXIF metadata fields to image_assets table:
- taken_at: Photo capture date from EXIF DateTimeOriginal
- camera_make/model: Camera identification
- gps_latitude/longitude: GPS coordinates
- exif_metadata: Full EXIF blob for extensibility

Revision ID: a8b9c0d1e2f3
Revises: f6a668d072bb
Create Date: 2026-01-09 16:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "a8b9c0d1e2f3"
down_revision: str | Sequence[str] | None = "f6a668d072bb"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add EXIF metadata columns to image_assets."""
    # Core EXIF field: when photo was taken
    # CRITICAL: Only populated from EXIF DateTimeOriginal, NEVER from file dates
    op.add_column(
        "image_assets",
        sa.Column("taken_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Camera identification
    op.add_column(
        "image_assets",
        sa.Column("camera_make", sa.String(100), nullable=True),
    )
    op.add_column(
        "image_assets",
        sa.Column("camera_model", sa.String(100), nullable=True),
    )

    # GPS location (decimal degrees)
    op.add_column(
        "image_assets",
        sa.Column("gps_latitude", sa.Float(), nullable=True),
    )
    op.add_column(
        "image_assets",
        sa.Column("gps_longitude", sa.Float(), nullable=True),
    )

    # Full EXIF blob for extensibility (stores all readable EXIF tags)
    op.add_column(
        "image_assets",
        sa.Column(
            "exif_metadata",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=True,
        ),
    )

    # Indexes for common queries
    op.create_index(
        "idx_image_assets_taken_at",
        "image_assets",
        ["taken_at"],
    )

    # Partial index for GPS queries (only index rows with GPS data)
    op.execute(
        """
        CREATE INDEX idx_image_assets_gps
        ON image_assets (gps_latitude, gps_longitude)
        WHERE gps_latitude IS NOT NULL AND gps_longitude IS NOT NULL
        """
    )


def downgrade() -> None:
    """Remove EXIF metadata columns from image_assets."""
    op.drop_index("idx_image_assets_gps", table_name="image_assets")
    op.drop_index("idx_image_assets_taken_at", table_name="image_assets")
    op.drop_column("image_assets", "exif_metadata")
    op.drop_column("image_assets", "gps_longitude")
    op.drop_column("image_assets", "gps_latitude")
    op.drop_column("image_assets", "camera_model")
    op.drop_column("image_assets", "camera_make")
    op.drop_column("image_assets", "taken_at")
