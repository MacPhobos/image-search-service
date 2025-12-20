"""add vector management tables and fields

Revision ID: 008
Revises: 007
Create Date: 2025-12-19 20:43:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "008"
down_revision: str | Sequence[str] | None = "007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create vector_deletion_logs table
    op.create_table(
        "vector_deletion_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("deletion_type", sa.String(50), nullable=False),
        sa.Column("deletion_target", sa.Text(), nullable=False),
        sa.Column("vector_count", sa.Integer(), nullable=False),
        sa.Column("deletion_reason", sa.Text(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_vector_deletion_logs_deletion_type",
        "vector_deletion_logs",
        ["deletion_type"],
    )
    op.create_index(
        "ix_vector_deletion_logs_created_at", "vector_deletion_logs", ["created_at"]
    )

    # Add reset fields to training_sessions
    op.add_column(
        "training_sessions",
        sa.Column("reset_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "training_sessions", sa.Column("reset_reason", sa.Text(), nullable=True)
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("training_sessions", "reset_reason")
    op.drop_column("training_sessions", "reset_at")
    op.drop_index(
        "ix_vector_deletion_logs_created_at", table_name="vector_deletion_logs"
    )
    op.drop_index(
        "ix_vector_deletion_logs_deletion_type", table_name="vector_deletion_logs"
    )
    op.drop_table("vector_deletion_logs")
