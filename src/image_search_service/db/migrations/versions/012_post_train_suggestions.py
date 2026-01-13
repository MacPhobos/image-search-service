"""add_post_training_suggestion_settings

Add post-training suggestion generation settings:
- post_training_suggestions_mode: Mode for generating suggestions
  (all persons or top N by face count)
- post_training_suggestions_top_n_count: Number of top persons when mode is top_n

These settings control automatic suggestion generation after training sessions complete.

Revision ID: 012_post_train_suggestions
Revises: 011_hash_dedup_fields
Create Date: 2026-01-12 10:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "012_post_train_suggestions"
down_revision: str | Sequence[str] | None = "011_hash_dedup_fields"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add post-training suggestion settings to system_configs."""
    # Insert mode setting (enum: all | top_n)
    op.execute("""
        INSERT INTO system_configs (key, value, data_type, allowed_values, category, description)
        VALUES (
            'post_training_suggestions_mode',
            'all',
            'string',
            '["all", "top_n"]',
            'face_matching',
            'Mode for post-training suggestion generation: all persons or top N by face count'
        )
        ON CONFLICT (key) DO NOTHING;
    """)

    # Insert top N count setting (int: 1-100)
    op.execute("""
        INSERT INTO system_configs (
            key, value, data_type, min_value, max_value, category, description
        )
        VALUES (
            'post_training_suggestions_top_n_count',
            '10',
            'int',
            '1',
            '100',
            'face_matching',
            'Number of top persons to generate suggestions for (only used when mode is top_n)'
        )
        ON CONFLICT (key) DO NOTHING;
    """)


def downgrade() -> None:
    """Remove post-training suggestion settings from system_configs."""
    op.execute("DELETE FROM system_configs WHERE key = 'post_training_suggestions_mode'")
    op.execute("DELETE FROM system_configs WHERE key = 'post_training_suggestions_top_n_count'")
