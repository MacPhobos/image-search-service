"""add_birth_date_to_persons

Add optional birth_date field to persons table for exact age calculation.

Revision ID: b9c0d1e2f3g4
Revises: a8b9c0d1e2f3
Create Date: 2026-01-09 17:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b9c0d1e2f3g4"
down_revision: str | Sequence[str] | None = "a8b9c0d1e2f3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add birth_date column to persons table."""
    # Add birth_date column (nullable, no default)
    op.add_column(
        "persons",
        sa.Column("birth_date", sa.Date(), nullable=True),
    )

    # Index for querying by birth_date (e.g., finding people by age)
    op.create_index(
        "idx_persons_birth_date",
        "persons",
        ["birth_date"],
    )


def downgrade() -> None:
    """Remove birth_date column from persons table."""
    op.drop_index("idx_persons_birth_date", table_name="persons")
    op.drop_column("persons", "birth_date")
