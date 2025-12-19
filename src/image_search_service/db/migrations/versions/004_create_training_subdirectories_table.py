"""create training_subdirectories table

Revision ID: 004
Revises: 003
Create Date: 2025-12-19 14:02:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "004"
down_revision: str | Sequence[str] | None = "003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "training_subdirectories",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("path", sa.String(500), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("selected", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("image_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("trained_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "status", sa.String(20), nullable=False, server_default="pending"
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["training_sessions.id"],
            name="fk_training_subdirectories_session_id",
            ondelete="CASCADE",
        ),
    )

    # Create indexes
    op.create_index(
        "idx_training_subdirectories_session", "training_subdirectories", ["session_id"]
    )
    op.create_index(
        "idx_training_subdirectories_status", "training_subdirectories", ["status"]
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "idx_training_subdirectories_status", table_name="training_subdirectories"
    )
    op.drop_index(
        "idx_training_subdirectories_session", table_name="training_subdirectories"
    )
    op.drop_table("training_subdirectories")
