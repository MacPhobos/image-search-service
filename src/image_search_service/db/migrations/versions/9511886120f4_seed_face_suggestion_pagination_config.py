"""seed_face_suggestion_pagination_config

Revision ID: 9511886120f4
Revises: a5e555cf5477
Create Date: 2025-12-28 19:16:27.752706

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '9511886120f4'
down_revision: str | Sequence[str] | None = 'a5e555cf5477'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add face suggestion pagination configuration keys."""
    op.execute(
        """
        INSERT INTO system_configs (key, value, data_type, description, min_value, max_value, category)  # noqa: E501
        VALUES
        (
            'face_suggestion_groups_per_page',
            '10',
            'int',
            'Number of person groups to display per page in face suggestion review UI. Controls how many different people with suggestions are shown at once.',  # noqa: E501
            '1',
            '50',
            'face_matching'
        ),
        (
            'face_suggestion_items_per_group',
            '20',
            'int',
            'Maximum number of face suggestions to display per person group. Controls how many suggested faces are shown for each person.',  # noqa: E501
            '1',
            '50',
            'face_matching'
        )
        ON CONFLICT (key) DO NOTHING
        """
    )


def downgrade() -> None:
    """Remove face suggestion pagination configuration keys."""
    op.execute(
        """
        DELETE FROM system_configs
        WHERE key IN ('face_suggestion_groups_per_page', 'face_suggestion_items_per_group')
        """
    )
