"""Tests for POST /api/v1/gdrive/folders endpoint."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from .conftest import FakeAsyncStorageWrapper, _enable_drive, _mock_async_storage


class TestCreateFolder:
    """Tests for POST /api/v1/gdrive/folders."""

    @pytest.mark.asyncio
    async def test_returns_503_when_drive_disabled(self, test_client: AsyncClient) -> None:
        """Folder creation must return 503 when Drive is not enabled."""
        response = await test_client.post(
            "/api/v1/gdrive/folders",
            json={"name": "New Folder"},
        )
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_returns_422_for_empty_name(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Empty folder name must fail Pydantic validation (422 Unprocessable Entity)."""
        _enable_drive(monkeypatch)

        response = await test_client.post(
            "/api/v1/gdrive/folders",
            json={"name": "", "parentId": "some-parent"},
        )
        assert response.status_code == 422  # Pydantic min_length=1

    @pytest.mark.asyncio
    async def test_creates_folder_and_returns_201(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        fake_storage: FakeAsyncStorageWrapper,
    ) -> None:
        """Valid create folder request must return 201 with the new folder ID."""
        _enable_drive(monkeypatch)
        _mock_async_storage(monkeypatch, fake_storage)

        response = await test_client.post(
            "/api/v1/gdrive/folders",
            json={"name": "New Album", "parentId": "parent-folder-id"},
        )

        assert response.status_code == 201
        data = response.json()
        assert "folderId" in data
        assert data["name"] == "New Album"
        assert data["parentId"] == "parent-folder-id"

    @pytest.mark.asyncio
    async def test_uses_root_when_no_parent_provided(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        fake_storage: FakeAsyncStorageWrapper,
    ) -> None:
        """Without parentId, the configured root folder ID is used as parent."""
        _enable_drive(monkeypatch)
        _mock_async_storage(monkeypatch, fake_storage)

        response = await test_client.post(
            "/api/v1/gdrive/folders",
            json={"name": "Root Level Folder"},
        )

        assert response.status_code == 201
        data = response.json()
        # parent_id should default to the configured root
        assert data["parentId"] == "fake-root-folder-id"

    @pytest.mark.asyncio
    async def test_create_folder_response_has_all_contract_fields(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        fake_storage: FakeAsyncStorageWrapper,
    ) -> None:
        """Create folder response shape must match the API contract."""
        _enable_drive(monkeypatch)
        _mock_async_storage(monkeypatch, fake_storage)

        response = await test_client.post(
            "/api/v1/gdrive/folders",
            json={"name": "Contract Test Folder"},
        )
        assert response.status_code == 201
        data = response.json()

        required_fields = {"folderId", "name", "parentId", "path"}
        missing = required_fields - set(data.keys())
        assert not missing, f"CreateFolderResponse missing fields: {missing}"

    @pytest.mark.asyncio
    async def test_storage_error_translated_to_http_status(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """StorageError from Drive should be translated to an HTTP error response."""
        _enable_drive(monkeypatch)

        class QuotaExceededStorage(FakeAsyncStorageWrapper):
            async def create_folder(self, name: str, parent_id: str | None = None) -> str:
                from image_search_service.storage.exceptions import StorageQuotaError

                raise StorageQuotaError(
                    quota_bytes=15_000_000_000,
                    used_bytes=15_000_000_000,
                )

        _mock_async_storage(monkeypatch, QuotaExceededStorage())

        response = await test_client.post(
            "/api/v1/gdrive/folders",
            json={"name": "Will Fail"},
        )
        assert response.status_code == 507
