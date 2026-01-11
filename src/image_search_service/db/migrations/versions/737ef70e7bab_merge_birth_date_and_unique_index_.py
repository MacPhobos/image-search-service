"""merge_birth_date_and_unique_index_migrations

Revision ID: 737ef70e7bab
Revises: b9c0d1e2f3g4, c1d2e3f4g5h6
Create Date: 2026-01-10 18:21:32.978882

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '737ef70e7bab'
down_revision: Union[str, Sequence[str], None] = ('b9c0d1e2f3g4', 'c1d2e3f4g5h6')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
