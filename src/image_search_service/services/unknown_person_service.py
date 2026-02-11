"""Service for unknown person discovery utilities.

Provides membership hash computation, dismissal tracking, and shared
utilities for the unknown persons feature.
"""

import hashlib
import json
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.db.models import DismissedUnknownPersonGroup


def compute_membership_hash(face_instance_ids: list[uuid.UUID]) -> str:
    """Compute stable hash of face instance IDs.

    Same set of faces always produces the same hash, regardless of
    order or cluster_id. Uses SHA-256 for collision resistance.

    Args:
        face_instance_ids: List of face instance UUIDs in the group.

    Returns:
        64-character hex string (SHA-256 digest).

    Example:
        >>> ids = [uuid.uuid4(), uuid.uuid4()]
        >>> hash1 = compute_membership_hash(ids)
        >>> hash2 = compute_membership_hash(list(reversed(ids)))
        >>> hash1 == hash2  # Same regardless of order
        True
    """
    sorted_ids = sorted(str(fid) for fid in face_instance_ids)
    hash_input = json.dumps(sorted_ids, sort_keys=True)
    return hashlib.sha256(hash_input.encode()).hexdigest()


async def is_group_dismissed(
    db: AsyncSession, membership_hash: str
) -> bool:
    """Check if a group with the given membership hash has been dismissed.

    Args:
        db: Database session.
        membership_hash: SHA-256 hash of sorted face instance IDs.

    Returns:
        True if group has been dismissed, False otherwise.
    """
    result = await db.execute(
        select(DismissedUnknownPersonGroup.id).where(
            DismissedUnknownPersonGroup.membership_hash == membership_hash
        )
    )
    return result.scalar_one_or_none() is not None


async def dismiss_group(
    db: AsyncSession,
    membership_hash: str,
    cluster_id: str | None,
    face_count: int,
    face_instance_ids: list[uuid.UUID],
    reason: str | None = None,
    marked_as_noise: bool = False,
) -> DismissedUnknownPersonGroup:
    """Record a group dismissal in the database.

    Args:
        db: Database session.
        membership_hash: SHA-256 hash of sorted face instance IDs.
        cluster_id: Cluster ID from clustering algorithm (may change across runs).
        face_count: Number of faces in the group.
        face_instance_ids: List of face instance UUIDs.
        reason: Optional user-provided reason for dismissal.
        marked_as_noise: Whether group was marked as noise/outliers.

    Returns:
        Created DismissedUnknownPersonGroup record.
    """
    dismissal = DismissedUnknownPersonGroup(
        membership_hash=membership_hash,
        cluster_id=cluster_id,
        face_count=face_count,
        reason=reason,
        marked_as_noise=marked_as_noise,
        face_instance_ids=[str(fid) for fid in face_instance_ids],
    )
    db.add(dismissal)
    await db.flush()
    return dismissal


async def get_dismissed_hashes(db: AsyncSession) -> set[str]:
    """Get all dismissed membership hashes for fast filtering.

    Args:
        db: Database session.

    Returns:
        Set of membership hashes for dismissed groups.

    Example:
        >>> dismissed = await get_dismissed_hashes(db)
        >>> if compute_membership_hash(face_ids) in dismissed:
        ...     # Skip this group, already dismissed
        ...     pass
    """
    result = await db.execute(
        select(DismissedUnknownPersonGroup.membership_hash)
    )
    return {row[0] for row in result.all()}
