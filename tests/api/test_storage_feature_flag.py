"""Tests for GOOGLE_DRIVE_ENABLED feature flag behavior."""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.db.models import StorageUpload, StorageUploadStatus


class TestFeatureFlag:
    """Tests verifying GOOGLE_DRIVE_ENABLED feature flag behavior."""

    @pytest.mark.asyncio
    async def test_upload_returns_503_when_disabled(self, test_client: AsyncClient) -> None:
        """All upload endpoints must return 503 when Drive is disabled."""
        response = await test_client.post(
            "/api/v1/gdrive/upload",
            json={"personId": str(uuid.uuid4()), "photoIds": [1], "folderId": "f"},
        )
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_folders_post_returns_503_when_disabled(self, test_client: AsyncClient) -> None:
        """Folder creation must return 503 when Drive is disabled."""
        response = await test_client.post(
            "/api/v1/gdrive/folders",
            json={"name": "test"},
        )
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_folders_get_returns_503_when_disabled(self, test_client: AsyncClient) -> None:
        """Folder listing must return 503 when Drive is disabled."""
        response = await test_client.get("/api/v1/gdrive/folders")
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_cancel_returns_503_when_disabled(self, test_client: AsyncClient) -> None:
        """Cancel endpoint must return 503 when Drive is disabled."""
        response = await test_client.delete(f"/api/v1/gdrive/upload/{uuid.uuid4()}")
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_health_works_when_disabled(self, test_client: AsyncClient) -> None:
        """Health endpoint MUST return 200 even when Drive is disabled."""
        response = await test_client.get("/api/v1/gdrive/health")
        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is False
        assert data["connected"] is False

    @pytest.mark.asyncio
    async def test_status_endpoint_works_without_drive_enabled(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
    ) -> None:
        """Status polling endpoint does NOT require Drive enabled — it only queries DB."""
        batch_id = str(uuid.uuid4())
        db_session.add(
            StorageUpload(
                batch_id=batch_id,
                asset_id=999,
                status=StorageUploadStatus.PENDING.value,
            )
        )
        await db_session.commit()

        # No _require_drive_enabled() call in the status endpoint
        response = await test_client.get(f"/api/v1/gdrive/upload/{batch_id}/status")
        assert response.status_code == 200
