"""add DISCOVERING session status

Introduces the 'discovering' status value for training sessions.

Background:
  Before this change, starting a training session (POST /start) would block
  the HTTP response while performing synchronous filesystem scanning, asset
  DB inserts, and perceptual-hash computation — potentially minutes for large
  libraries.

  With this change the API handler:
  1. Sets session.status = 'discovering'
  2. Enqueues the RQ job (milliseconds)
  3. Returns immediately

  The RQ worker then performs:
  1. Filesystem scan + ImageAsset DB inserts
  2. Perceptual-hash computation + TrainingJob creation
  3. Transition discovering → running
  4. GPU embedding generation (existing logic)

Status lifecycle:
  pending → discovering → running → completed
                       ↘ cancelled (if user cancels during discovery)
                       ↘ failed    (if no assets found)

No schema change required:
  training_sessions.status is a VARCHAR(20) column with no DB-level enum
  constraint, so 'discovering' (11 chars) fits without altering the column.

Revision ID: 013_add_discovering_session_status
Revises: fcc7bcef2a95
Create Date: 2026-02-19 00:00:00.000000
"""

from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "013_add_discovering_session_status"
down_revision: str | Sequence[str] | None = "fcc7bcef2a95"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """No schema change needed — status column is VARCHAR(20), not an enum type."""
    pass


def downgrade() -> None:
    """No schema change to revert."""
    pass
