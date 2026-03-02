"""Tests for DELETE /api/v1/gdrive/upload/{batch_id} endpoint."""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.db.models import StorageUpload, StorageUploadStatus

from .conftest import _enable_drive


class TestCancelUpload:
    """Tests for DELETE /api/v1/gdrive/upload/{batch_id}."""

    @pytest.mark.asyncio
    async def test_returns_503_when_drive_disabled(self, test_client: AsyncClient) -> None:
        """Cancel must return 503 when Google Drive is disabled."""
        response = await test_client.delete(f"/api/v1/gdrive/upload/{uuid.uuid4()}")
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_batch(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cancel must return 404 for a batch that does not exist."""
        _enable_drive(monkeypatch)

        response = await test_client.delete(f"/api/v1/gdrive/upload/{uuid.uuid4()}")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_409_when_batch_already_completed(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cancel must return 409 when no pending uploads remain."""
        _enable_drive(monkeypatch)
        batch_id = str(uuid.uuid4())

        db_session.add(
            StorageUpload(
                batch_id=batch_id,
                asset_id=500,
                status=StorageUploadStatus.COMPLETED.value,
            )
        )
        await db_session.commit()

        response = await test_client.delete(f"/api/v1/gdrive/upload/{batch_id}")
        assert response.status_code == 409

    @pytest.mark.asyncio
    async def test_cancel_marks_pending_records_as_cancelled(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cancel must mark pending uploads as cancelled and return correct counts."""
        _enable_drive(monkeypatch)
        batch_id = str(uuid.uuid4())

        # 2 completed, 2 pending
        for i in range(4):
            db_session.add(
                StorageUpload(
                    batch_id=batch_id,
                    asset_id=600 + i,
                    status=(
                        StorageUploadStatus.COMPLETED.value
                        if i < 2
                        else StorageUploadStatus.PENDING.value
                    ),
                )
            )
        await db_session.commit()

        response = await test_client.delete(f"/api/v1/gdrive/upload/{batch_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["batchId"] == batch_id
        assert data["status"] == "cancelled"
        assert data["completedBeforeCancel"] == 2
        assert data["cancelledCount"] == 2
        assert data["message"] is not None

    @pytest.mark.asyncio
    async def test_cancel_response_has_all_contract_fields(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cancel response shape must match the API contract."""
        _enable_drive(monkeypatch)
        batch_id = str(uuid.uuid4())

        db_session.add(
            StorageUpload(
                batch_id=batch_id,
                asset_id=700,
                status=StorageUploadStatus.PENDING.value,
            )
        )
        await db_session.commit()

        response = await test_client.delete(f"/api/v1/gdrive/upload/{batch_id}")
        assert response.status_code == 200
        data = response.json()

        required_fields = {
            "batchId",
            "status",
            "completedBeforeCancel",
            "cancelledCount",
            "message",
        }
        missing = required_fields - set(data.keys())
        assert not missing, f"CancelUploadResponse missing fields: {missing}"
