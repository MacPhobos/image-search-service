"""Tests for GET /api/v1/gdrive/health endpoint."""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient

from image_search_service.storage.base import StorageEntry

from .conftest import FakeAsyncStorageWrapper, _enable_drive, _mock_async_storage


class TestDriveHealth:
    """Tests for GET /api/v1/gdrive/health."""

    @pytest.mark.asyncio
    async def test_health_returns_200_when_disabled(self, test_client: AsyncClient) -> None:
        """Health endpoint must return 200 even when Drive is disabled."""
        response = await test_client.get("/api/v1/gdrive/health")
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is False
        assert data["enabled"] is False

    @pytest.mark.asyncio
    async def test_health_returns_200_when_enabled_and_connected(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        fake_storage: FakeAsyncStorageWrapper,
    ) -> None:
        """Health endpoint should show connected=true when Drive works."""
        _enable_drive(monkeypatch)
        _mock_async_storage(monkeypatch, fake_storage)

        response = await test_client.get("/api/v1/gdrive/health")
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is True
        assert data["enabled"] is True
        assert data["error"] is None

    @pytest.mark.asyncio
    async def test_health_returns_200_when_drive_raises(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Health endpoint must return 200 even if Drive connection fails."""
        _enable_drive(monkeypatch)

        # Fake storage that raises on list_folder
        class FailingStorage(FakeAsyncStorageWrapper):
            async def list_folder(self, folder_id: str) -> list[StorageEntry]:
                from image_search_service.storage.exceptions import NotFoundError

                raise NotFoundError(folder_id)

        _mock_async_storage(monkeypatch, FailingStorage())

        response = await test_client.get("/api/v1/gdrive/health")
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is False
        assert data["enabled"] is True
        assert data["error"] is not None

    @pytest.mark.asyncio
    async def test_health_response_has_all_contract_fields(self, test_client: AsyncClient) -> None:
        """Health response must include all fields defined in the API contract."""
        response = await test_client.get("/api/v1/gdrive/health")
        data = response.json()

        required_fields = {
            "connected",
            "enabled",
            "serviceAccountEmail",
            "rootFolderId",
            "rootFolderName",
            "storageUsedBytes",
            "storageTotalBytes",
            "storageUsagePercentage",
            "lastUploadAt",
            "error",
        }
        missing = required_fields - set(data.keys())
        assert not missing, f"Health response missing fields: {missing}"

    @pytest.mark.asyncio
    async def test_health_uses_get_user_email_for_oauth_backend(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        fake_storage: FakeAsyncStorageWrapper,
    ) -> None:
        """CRITICAL-1: Health endpoint calls get_user_email() for OAuth backends.

        When the storage backend is a GoogleDriveOAuthV3Storage instance,
        the health endpoint must call get_user_email() (not _get_sa_email())
        so that OAuth users can confirm their authentication is working.
        """
        from unittest.mock import MagicMock

        from image_search_service.storage.google_drive_oauth_v3 import (
            GoogleDriveOAuthV3Storage,
        )

        _enable_drive(monkeypatch)

        # Create a real GoogleDriveOAuthV3Storage subclass instance so that
        # isinstance(storage, GoogleDriveOAuthV3Storage) returns True inside
        # the route. We override _build_service to avoid Drive API calls and
        # provide get_user_email() and an async list_folder() mock.
        class _FakeOAuthStorage(GoogleDriveOAuthV3Storage):
            def _build_service(self) -> MagicMock:  # type: ignore[override]
                return MagicMock()

            def get_user_email(self) -> str:
                return "alice@example.com"

            async def list_folder(self, folder_id: str) -> list[Any]:  # type: ignore[override]
                return []

        oauth_storage = _FakeOAuthStorage(
            client_id="test-id",
            client_secret="test-secret",
            refresh_token="test-token",
            root_folder_id="test-root",
        )

        monkeypatch.setattr(
            "image_search_service.storage.get_async_storage",
            lambda: oauth_storage,
        )

        response = await test_client.get("/api/v1/gdrive/health")
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is True
        assert data["serviceAccountEmail"] == "alice@example.com"

    @pytest.mark.asyncio
    async def test_health_uses_sa_email_for_service_account_backend(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        fake_storage: FakeAsyncStorageWrapper,
    ) -> None:
        """CRITICAL-1: Health endpoint calls _get_sa_email() for SA backends.

        When the backend is NOT a GoogleDriveOAuthV3Storage instance (i.e. the
        standard SA backend), _get_sa_email() must be used as before, so
        existing deployments are not broken.
        """
        _enable_drive(monkeypatch)
        _mock_async_storage(monkeypatch, fake_storage)

        # Patch _get_sa_email to return a known value
        monkeypatch.setattr(
            "image_search_service.api.routes.storage._get_sa_email",
            lambda: "sa-robot@my-project.iam.gserviceaccount.com",
        )

        response = await test_client.get("/api/v1/gdrive/health")
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is True
        assert data["serviceAccountEmail"] == "sa-robot@my-project.iam.gserviceaccount.com"
