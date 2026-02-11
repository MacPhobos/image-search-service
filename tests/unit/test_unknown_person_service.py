"""Tests for unknown person discovery service."""

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.db.models import DismissedUnknownPersonGroup
from image_search_service.services.unknown_person_service import (
    compute_membership_hash,
    dismiss_group,
    get_dismissed_hashes,
    is_group_dismissed,
)


class TestComputeMembershipHash:
    """Tests for membership hash computation."""

    def test_membership_hash_same_ids_different_order(self):
        """Hash is order-independent for same set of IDs."""
        id1 = uuid.uuid4()
        id2 = uuid.uuid4()
        id3 = uuid.uuid4()

        hash_forward = compute_membership_hash([id1, id2, id3])
        hash_reverse = compute_membership_hash([id3, id2, id1])
        hash_mixed = compute_membership_hash([id2, id3, id1])

        assert hash_forward == hash_reverse == hash_mixed

    def test_membership_hash_different_ids(self):
        """Different sets of IDs produce different hashes."""
        ids1 = [uuid.uuid4(), uuid.uuid4()]
        ids2 = [uuid.uuid4(), uuid.uuid4()]

        hash1 = compute_membership_hash(ids1)
        hash2 = compute_membership_hash(ids2)

        assert hash1 != hash2

    def test_membership_hash_empty_list(self):
        """Empty list produces deterministic hash."""
        hash1 = compute_membership_hash([])
        hash2 = compute_membership_hash([])

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest

    def test_membership_hash_single_id(self):
        """Single ID produces deterministic hash."""
        face_id = uuid.uuid4()

        hash1 = compute_membership_hash([face_id])
        hash2 = compute_membership_hash([face_id])

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest

    def test_membership_hash_deterministic(self):
        """Same IDs always produce same hash (deterministic)."""
        id1 = uuid.UUID("12345678-1234-5678-1234-567812345678")
        id2 = uuid.UUID("87654321-4321-8765-4321-876543218765")

        hash1 = compute_membership_hash([id1, id2])
        hash2 = compute_membership_hash([id1, id2])
        hash3 = compute_membership_hash([id2, id1])  # Different order

        assert hash1 == hash2 == hash3

    def test_membership_hash_length(self):
        """Hash is always 64 characters (SHA-256 hex)."""
        ids = [uuid.uuid4() for _ in range(10)]
        hash_result = compute_membership_hash(ids)
        assert len(hash_result) == 64

    def test_membership_hash_subset_different(self):
        """Adding/removing IDs changes the hash."""
        id1 = uuid.uuid4()
        id2 = uuid.uuid4()
        id3 = uuid.uuid4()

        hash_two = compute_membership_hash([id1, id2])
        hash_three = compute_membership_hash([id1, id2, id3])

        assert hash_two != hash_three


class TestIsGroupDismissed:
    """Tests for checking if a group is dismissed."""

    @pytest.mark.asyncio
    async def test_is_group_dismissed_not_found(self, db_session: AsyncSession):
        """Returns False when group not dismissed."""
        fake_hash = "a" * 64
        result = await is_group_dismissed(db_session, fake_hash)
        assert result is False

    @pytest.mark.asyncio
    async def test_is_group_dismissed_found(self, db_session: AsyncSession):
        """Returns True when group has been dismissed."""
        # Create dismissed group
        face_ids = [uuid.uuid4(), uuid.uuid4()]
        membership_hash = compute_membership_hash(face_ids)

        dismissal = DismissedUnknownPersonGroup(
            membership_hash=membership_hash,
            cluster_id="cluster_1",
            face_count=2,
            face_instance_ids=[str(fid) for fid in face_ids],
        )
        db_session.add(dismissal)
        await db_session.commit()

        # Check if dismissed
        result = await is_group_dismissed(db_session, membership_hash)
        assert result is True


class TestDismissGroup:
    """Tests for dismissing groups."""

    @pytest.mark.asyncio
    async def test_dismiss_group_creates_record(self, db_session: AsyncSession):
        """dismiss_group creates record with correct fields."""
        face_ids = [uuid.uuid4(), uuid.uuid4(), uuid.uuid4()]
        membership_hash = compute_membership_hash(face_ids)

        dismissal = await dismiss_group(
            db=db_session,
            membership_hash=membership_hash,
            cluster_id="cluster_42",
            face_count=3,
            face_instance_ids=face_ids,
            reason="Duplicate person",
            marked_as_noise=False,
        )
        await db_session.commit()

        # Verify record fields
        assert dismissal.membership_hash == membership_hash
        assert dismissal.cluster_id == "cluster_42"
        assert dismissal.face_count == 3
        assert dismissal.reason == "Duplicate person"
        assert dismissal.marked_as_noise is False
        assert len(dismissal.face_instance_ids) == 3

        # Verify persisted to database
        is_dismissed = await is_group_dismissed(db_session, membership_hash)
        assert is_dismissed is True

    @pytest.mark.asyncio
    async def test_dismiss_group_marked_as_noise(self, db_session: AsyncSession):
        """dismiss_group can mark group as noise."""
        face_ids = [uuid.uuid4()]
        membership_hash = compute_membership_hash(face_ids)

        dismissal = await dismiss_group(
            db=db_session,
            membership_hash=membership_hash,
            cluster_id=None,
            face_count=1,
            face_instance_ids=face_ids,
            marked_as_noise=True,
        )
        await db_session.commit()

        assert dismissal.marked_as_noise is True
        assert dismissal.cluster_id is None

    @pytest.mark.asyncio
    async def test_dismiss_group_optional_fields(self, db_session: AsyncSession):
        """dismiss_group handles optional fields correctly."""
        face_ids = [uuid.uuid4(), uuid.uuid4()]
        membership_hash = compute_membership_hash(face_ids)

        dismissal = await dismiss_group(
            db=db_session,
            membership_hash=membership_hash,
            cluster_id="cluster_5",
            face_count=2,
            face_instance_ids=face_ids,
            # reason omitted
            # marked_as_noise defaults to False
        )
        await db_session.commit()

        assert dismissal.reason is None
        assert dismissal.marked_as_noise is False


class TestGetDismissedHashes:
    """Tests for retrieving all dismissed hashes."""

    @pytest.mark.asyncio
    async def test_get_dismissed_hashes_empty(self, db_session: AsyncSession):
        """Returns empty set when no dismissals."""
        result = await get_dismissed_hashes(db_session)
        assert result == set()

    @pytest.mark.asyncio
    async def test_get_dismissed_hashes_multiple(self, db_session: AsyncSession):
        """Returns all dismissed hashes."""
        # Create multiple dismissals
        dismissals = []
        expected_hashes = set()

        for i in range(3):
            face_ids = [uuid.uuid4() for _ in range(i + 2)]
            membership_hash = compute_membership_hash(face_ids)
            expected_hashes.add(membership_hash)

            dismissal = DismissedUnknownPersonGroup(
                membership_hash=membership_hash,
                cluster_id=f"cluster_{i}",
                face_count=len(face_ids),
                face_instance_ids=[str(fid) for fid in face_ids],
            )
            dismissals.append(dismissal)

        db_session.add_all(dismissals)
        await db_session.commit()

        # Retrieve all hashes
        result = await get_dismissed_hashes(db_session)
        assert result == expected_hashes
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_get_dismissed_hashes_for_filtering(self, db_session: AsyncSession):
        """Dismissed hashes can be used to filter groups."""
        # Dismiss one group
        dismissed_ids = [uuid.uuid4(), uuid.uuid4()]
        dismissed_hash = compute_membership_hash(dismissed_ids)
        await dismiss_group(
            db=db_session,
            membership_hash=dismissed_hash,
            cluster_id="cluster_1",
            face_count=2,
            face_instance_ids=dismissed_ids,
        )
        await db_session.commit()

        # Get dismissed hashes
        dismissed = await get_dismissed_hashes(db_session)

        # Check filtering logic
        new_group_ids = [uuid.uuid4(), uuid.uuid4()]
        new_hash = compute_membership_hash(new_group_ids)

        assert dismissed_hash in dismissed
        assert new_hash not in dismissed
