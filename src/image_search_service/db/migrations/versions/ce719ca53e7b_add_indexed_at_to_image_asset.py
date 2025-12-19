"""add indexed_at to image_asset

Revision ID: ce719ca53e7b
Revises: 001
Create Date: 2025-12-19 08:56:01.421278

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "ce719ca53e7b"
down_revision: str | Sequence[str] | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "image_assets", sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True)
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("image_assets", "indexed_at")
