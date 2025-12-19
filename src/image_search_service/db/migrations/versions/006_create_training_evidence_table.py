"""create training_evidence table

Revision ID: 006
Revises: 005
Create Date: 2025-12-19 14:04:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "006"
down_revision: str | Sequence[str] | None = "005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "training_evidence",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("asset_id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("model_version", sa.String(50), nullable=False),
        sa.Column("embedding_checksum", sa.String(64), nullable=True),
        sa.Column("device", sa.String(20), nullable=False),
        sa.Column("processing_time_ms", sa.Integer(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["asset_id"],
            ["image_assets.id"],
            name="fk_training_evidence_asset_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["training_sessions.id"],
            name="fk_training_evidence_session_id",
            ondelete="CASCADE",
        ),
    )

    # Create indexes
    op.create_index("idx_training_evidence_asset", "training_evidence", ["asset_id"])
    op.create_index("idx_training_evidence_session", "training_evidence", ["session_id"])
    op.create_index(
        "idx_training_evidence_created_at", "training_evidence", ["created_at"]
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("idx_training_evidence_created_at", table_name="training_evidence")
    op.drop_index("idx_training_evidence_session", table_name="training_evidence")
    op.drop_index("idx_training_evidence_asset", table_name="training_evidence")
    op.drop_table("training_evidence")
