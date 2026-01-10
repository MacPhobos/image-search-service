"""Merge parallel migrations

Revision ID: 34534acb6ed1
Revises: 98ab620aa6f9, f6a668d072bb
Create Date: 2026-01-08 08:34:29.110580

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '34534acb6ed1'
down_revision: Union[str, Sequence[str], None] = ('98ab620aa6f9', 'f6a668d072bb')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
