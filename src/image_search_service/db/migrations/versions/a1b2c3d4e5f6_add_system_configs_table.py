"""add_system_configs_table

Revision ID: a1b2c3d4e5f6
Revises: 56f6544da217
Create Date: 2025-12-27 00:30:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "56f6544da217"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create system_configs table with seed data for face matching settings."""
    # Create the system_configs table
    op.create_table(
        "system_configs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("key", sa.String(length=100), nullable=False),
        sa.Column("value", sa.String(length=500), nullable=False),
        sa.Column("data_type", sa.String(length=20), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("min_value", sa.String(length=50), nullable=True),
        sa.Column("max_value", sa.String(length=50), nullable=True),
        sa.Column("allowed_values", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("category", sa.String(length=50), nullable=False, server_default="general"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key"),
    )
    op.create_index("ix_system_configs_key", "system_configs", ["key"], unique=False)
    op.create_index("ix_system_configs_category", "system_configs", ["category"], unique=False)

    # Seed data for face matching configuration
    op.execute(
        """
        INSERT INTO system_configs (key, value, data_type, description, min_value, max_value, category)
        VALUES
        (
            'face_auto_assign_threshold',
            '0.85',
            'float',
            'Confidence threshold for automatic face-to-person assignment. Faces with similarity scores above this threshold are automatically assigned to the matching person.',
            '0.5',
            '1.0',
            'face_matching'
        ),
        (
            'face_suggestion_threshold',
            '0.70',
            'float',
            'Minimum confidence threshold to create face-to-person suggestions. Faces with similarity scores between this and the auto-assign threshold create suggestions for user review.',
            '0.3',
            '0.95',
            'face_matching'
        ),
        (
            'face_suggestion_max_results',
            '50',
            'int',
            'Maximum number of suggestions to create when a face is manually labeled to a person. Higher values find more similar faces but may increase processing time.',
            '1',
            '200',
            'face_matching'
        ),
        (
            'face_suggestion_expiry_days',
            '30',
            'int',
            'Number of days after which pending face suggestions expire automatically. Expired suggestions are marked as expired and excluded from review.',
            '1',
            '365',
            'face_matching'
        )
        """
    )


def downgrade() -> None:
    """Drop system_configs table."""
    op.drop_index("ix_system_configs_category", table_name="system_configs")
    op.drop_index("ix_system_configs_key", table_name="system_configs")
    op.drop_table("system_configs")
