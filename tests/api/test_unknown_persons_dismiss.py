"""Tests for POST /api/v1/faces/unknown-persons/candidates/{group_id}/dismiss endpoint."""

from __future__ import annotations

from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.db.models import (
    DismissedUnknownPersonGroup,
    FaceInstance,
    ImageAsset,
)


@pytest.fixture
async def sample_asset(db_session: AsyncSession) -> ImageAsset:
    """Create sample image asset."""
    asset = ImageAsset(path="/test/image.jpg", training_status="pending")
    db_session.add(asset)
    await db_session.commit()
    await db_session.refresh(asset)
    return asset


@pytest.fixture
async def test_cluster(db_session: AsyncSession, sample_asset: ImageAsset) -> list[FaceInstance]:
    """Create test cluster with 5 faces."""
    faces = []
    for i in range(5):
        face = FaceInstance(
            id=uuid4(),
            asset_id=sample_asset.id,
            bbox_x=100 + i * 50,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=0.85,
            qdrant_point_id=uuid4(),
            cluster_id="unknown_dismiss_test",
            person_id=None,
        )
        db_session.add(face)
        faces.append(face)

    await db_session.commit()
    return faces


@pytest.mark.asyncio
async def test_dismiss_basic(
    test_client: AsyncClient,
    db_session: AsyncSession,
    test_cluster: list[FaceInstance],
) -> None:
    """Test dismissing a group records dismissal."""
    response = await test_client.post(
        "/api/v1/faces/unknown-persons/candidates/unknown_dismiss_test/dismiss",
        json={},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["groupId"] == "unknown_dismiss_test"
    assert data["facesAffected"] == 5
    assert data["markedAsNoise"] is False
    assert "membershipHash" in data

    # Verify dismissal recorded in database
    dismissed_result = await db_session.execute(
        select(DismissedUnknownPersonGroup).where(
            DismissedUnknownPersonGroup.cluster_id == "unknown_dismiss_test"
        )
    )
    dismissed = dismissed_result.scalar_one()
    assert dismissed is not None
    assert dismissed.face_count == 5
    assert dismissed.marked_as_noise is False


@pytest.mark.asyncio
async def test_dismiss_with_reason(
    test_client: AsyncClient,
    db_session: AsyncSession,
    test_cluster: list[FaceInstance],
) -> None:
    """Test dismissing group with reason."""
    response = await test_client.post(
        "/api/v1/faces/unknown-persons/candidates/unknown_dismiss_test/dismiss",
        json={"reason": "Not a real person, just shadows"},
    )

    assert response.status_code == 200

    # Verify reason stored
    dismissed_result = await db_session.execute(
        select(DismissedUnknownPersonGroup).where(
            DismissedUnknownPersonGroup.cluster_id == "unknown_dismiss_test"
        )
    )
    dismissed = dismissed_result.scalar_one()
    assert dismissed.reason == "Not a real person, just shadows"


@pytest.mark.asyncio
async def test_dismiss_with_mark_as_noise(
    test_client: AsyncClient,
    db_session: AsyncSession,
    test_cluster: list[FaceInstance],
) -> None:
    """Test dismissing group and marking faces as noise."""
    response = await test_client.post(
        "/api/v1/faces/unknown-persons/candidates/unknown_dismiss_test/dismiss",
        json={"markAsNoise": True},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["markedAsNoise"] is True

    # Verify faces marked as noise (cluster_id = '-1')
    faces_result = await db_session.execute(
        select(FaceInstance).where(FaceInstance.id.in_([f.id for f in test_cluster]))
    )
    faces = list(faces_result.scalars().all())
    assert all(f.cluster_id == "-1" for f in faces)

    # Verify dismissal recorded with noise flag
    dismissed_result = await db_session.execute(
        select(DismissedUnknownPersonGroup).where(
            DismissedUnknownPersonGroup.cluster_id == "unknown_dismiss_test"
        )
    )
    dismissed = dismissed_result.scalar_one()
    assert dismissed.marked_as_noise is True


@pytest.mark.asyncio
async def test_dismiss_group_not_found(test_client: AsyncClient) -> None:
    """Test dismissing non-existent group returns 404."""
    response = await test_client.post(
        "/api/v1/faces/unknown-persons/candidates/unknown_nonexistent/dismiss",
        json={},
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_dismiss_membership_hash_consistency(
    test_client: AsyncClient,
    db_session: AsyncSession,
    test_cluster: list[FaceInstance],
) -> None:
    """Test membership hash is consistent for same group."""
    response = await test_client.post(
        "/api/v1/faces/unknown-persons/candidates/unknown_dismiss_test/dismiss",
        json={},
    )

    assert response.status_code == 200
    data = response.json()

    membership_hash = data["membershipHash"]

    # Verify hash stored in database
    dismissed_result = await db_session.execute(
        select(DismissedUnknownPersonGroup).where(
            DismissedUnknownPersonGroup.membership_hash == membership_hash
        )
    )
    dismissed = dismissed_result.scalar_one()
    assert dismissed is not None
    assert len(dismissed.membership_hash) == 64  # SHA-256 hex digest


@pytest.mark.asyncio
async def test_dismiss_face_instance_ids_stored(
    test_client: AsyncClient,
    db_session: AsyncSession,
    test_cluster: list[FaceInstance],
) -> None:
    """Test face instance IDs are stored in dismissal record."""
    response = await test_client.post(
        "/api/v1/faces/unknown-persons/candidates/unknown_dismiss_test/dismiss",
        json={},
    )

    assert response.status_code == 200

    # Verify face IDs stored
    dismissed_result = await db_session.execute(
        select(DismissedUnknownPersonGroup).where(
            DismissedUnknownPersonGroup.cluster_id == "unknown_dismiss_test"
        )
    )
    dismissed = dismissed_result.scalar_one()
    assert dismissed.face_instance_ids is not None
    assert len(dismissed.face_instance_ids) == 5

    # Verify stored IDs match actual faces
    from uuid import UUID

    stored_ids = {UUID(id_str) for id_str in dismissed.face_instance_ids}
    actual_ids = {f.id for f in test_cluster}
    assert stored_ids == actual_ids


@pytest.mark.asyncio
async def test_dismiss_without_noise_preserves_cluster_id(
    test_client: AsyncClient,
    db_session: AsyncSession,
    test_cluster: list[FaceInstance],
) -> None:
    """Test dismissing without noise flag preserves original cluster_id."""
    response = await test_client.post(
        "/api/v1/faces/unknown-persons/candidates/unknown_dismiss_test/dismiss",
        json={"markAsNoise": False},
    )

    assert response.status_code == 200

    # Verify cluster_id unchanged
    faces_result = await db_session.execute(
        select(FaceInstance).where(FaceInstance.id.in_([f.id for f in test_cluster]))
    )
    faces = list(faces_result.scalars().all())
    assert all(f.cluster_id == "unknown_dismiss_test" for f in faces)


@pytest.mark.asyncio
async def test_dismiss_idempotent(
    test_client: AsyncClient,
    db_session: AsyncSession,
    test_cluster: list[FaceInstance],
) -> None:
    """Test dismissing same group multiple times (edge case)."""
    # First dismissal
    response1 = await test_client.post(
        "/api/v1/faces/unknown-persons/candidates/unknown_dismiss_test/dismiss",
        json={"reason": "First dismissal"},
    )

    assert response1.status_code == 200

    # Second dismissal (should still work, creates duplicate record currently)
    # NOTE: In production, may want to check for existing dismissal
    response2 = await test_client.post(
        "/api/v1/faces/unknown-persons/candidates/unknown_dismiss_test/dismiss",
        json={"reason": "Second dismissal"},
    )

    # Currently allows duplicate dismissals (could be enhanced)
    assert response2.status_code == 200


@pytest.mark.asyncio
async def test_dismiss_with_long_reason(
    test_client: AsyncClient,
    db_session: AsyncSession,
    test_cluster: list[FaceInstance],
) -> None:
    """Test dismissing with long reason text."""
    long_reason = "This group contains faces that are not real people. " * 20

    response = await test_client.post(
        "/api/v1/faces/unknown-persons/candidates/unknown_dismiss_test/dismiss",
        json={"reason": long_reason},
    )

    assert response.status_code == 200

    # Verify long reason stored
    dismissed_result = await db_session.execute(
        select(DismissedUnknownPersonGroup).where(
            DismissedUnknownPersonGroup.cluster_id == "unknown_dismiss_test"
        )
    )
    dismissed = dismissed_result.scalar_one()
    assert dismissed.reason == long_reason


@pytest.mark.asyncio
async def test_dismiss_combined_reason_and_noise(
    test_client: AsyncClient,
    db_session: AsyncSession,
    test_cluster: list[FaceInstance],
) -> None:
    """Test dismissing with both reason and noise marking."""
    response = await test_client.post(
        "/api/v1/faces/unknown-persons/candidates/unknown_dismiss_test/dismiss",
        json={"reason": "Blurry shadows", "markAsNoise": True},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["markedAsNoise"] is True

    # Verify both reason and noise flag stored
    dismissed_result = await db_session.execute(
        select(DismissedUnknownPersonGroup).where(
            DismissedUnknownPersonGroup.cluster_id == "unknown_dismiss_test"
        )
    )
    dismissed = dismissed_result.scalar_one()
    assert dismissed.reason == "Blurry shadows"
    assert dismissed.marked_as_noise is True

    # Verify faces marked as noise
    faces_result = await db_session.execute(
        select(FaceInstance).where(FaceInstance.id.in_([f.id for f in test_cluster]))
    )
    faces = list(faces_result.scalars().all())
    assert all(f.cluster_id == "-1" for f in faces)
