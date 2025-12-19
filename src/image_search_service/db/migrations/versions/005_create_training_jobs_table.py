"""create training_jobs table

Revision ID: 005
Revises: 004
Create Date: 2025-12-19 14:03:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "005"
down_revision: str | Sequence[str] | None = "004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "training_jobs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("asset_id", sa.Integer(), nullable=False),
        sa.Column(
            "status", sa.String(20), nullable=False, server_default="pending"
        ),
        sa.Column("rq_job_id", sa.String(100), nullable=True),
        sa.Column("progress", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("processing_time_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["training_sessions.id"],
            name="fk_training_jobs_session_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["asset_id"],
            ["image_assets.id"],
            name="fk_training_jobs_asset_id",
            ondelete="CASCADE",
        ),
    )

    # Create indexes
    op.create_index("idx_training_jobs_session", "training_jobs", ["session_id"])
    op.create_index("idx_training_jobs_asset", "training_jobs", ["asset_id"])
    op.create_index("idx_training_jobs_status", "training_jobs", ["status"])
    op.create_index("idx_training_jobs_rq_job_id", "training_jobs", ["rq_job_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("idx_training_jobs_rq_job_id", table_name="training_jobs")
    op.drop_index("idx_training_jobs_status", table_name="training_jobs")
    op.drop_index("idx_training_jobs_asset", table_name="training_jobs")
    op.drop_index("idx_training_jobs_session", table_name="training_jobs")
    op.drop_table("training_jobs")
