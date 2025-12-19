"""create training_sessions table

Revision ID: 003
Revises: 002
Create Date: 2025-12-19 14:01:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: str | Sequence[str] | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "training_sessions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column(
            "status", sa.String(20), nullable=False, server_default="pending"
        ),
        sa.Column("root_path", sa.String(500), nullable=False),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("total_images", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "processed_images", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("failed_images", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("paused_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes
    op.create_index("idx_training_sessions_status", "training_sessions", ["status"])
    op.create_index(
        "idx_training_sessions_created_at", "training_sessions", ["created_at"]
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("idx_training_sessions_created_at", table_name="training_sessions")
    op.drop_index("idx_training_sessions_status", table_name="training_sessions")
    op.drop_table("training_sessions")
