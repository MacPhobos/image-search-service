"""create categories table and add category_id to training_sessions

Revision ID: 007
Revises: 006
Create Date: 2025-12-19 15:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "007"
down_revision: str | Sequence[str] | None = "006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create categories table
    op.create_table(
        "categories",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("color", sa.String(7), nullable=True),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_categories_name"),
    )

    # Create indexes on categories
    op.create_index("idx_categories_name", "categories", ["name"])
    op.create_index("idx_categories_is_default", "categories", ["is_default"])

    # Insert default "General" category
    op.execute(
        """
        INSERT INTO categories (name, description, is_default, created_at, updated_at)
        VALUES ('General', 'Default category for all training sessions', true, NOW(), NOW())
        """
    )

    # Add category_id column to training_sessions
    op.add_column(
        "training_sessions",
        sa.Column("category_id", sa.Integer(), nullable=True),
    )

    # Add foreign key constraint
    op.create_foreign_key(
        "fk_training_sessions_category_id",
        "training_sessions",
        "categories",
        ["category_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # Create index on category_id
    op.create_index(
        "idx_training_sessions_category_id", "training_sessions", ["category_id"]
    )

    # Update existing training sessions to use the default category
    op.execute(
        """
        UPDATE training_sessions
        SET category_id = (SELECT id FROM categories WHERE is_default = true)
        WHERE category_id IS NULL
        """
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Drop foreign key constraint
    op.drop_constraint(
        "fk_training_sessions_category_id",
        "training_sessions",
        type_="foreignkey",
    )

    # Drop index on category_id
    op.drop_index("idx_training_sessions_category_id", table_name="training_sessions")

    # Drop category_id column from training_sessions
    op.drop_column("training_sessions", "category_id")

    # Drop indexes on categories
    op.drop_index("idx_categories_is_default", table_name="categories")
    op.drop_index("idx_categories_name", table_name="categories")

    # Drop categories table
    op.drop_table("categories")
