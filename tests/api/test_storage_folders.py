"""Tests for GET /api/v1/gdrive/folders endpoint."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from image_search_service.storage.base import StorageEntry

from .conftest import FakeAsyncStorageWrapper, _enable_drive, _mock_async_storage


class TestListFolders:
    """Tests for GET /api/v1/gdrive/folders."""

    @pytest.mark.asyncio
    async def test_returns_503_when_drive_disabled(self, test_client: AsyncClient) -> None:
        """Folder listing must return 503 when Drive is not enabled."""
        response = await test_client.get("/api/v1/gdrive/folders")
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_returns_only_folders_not_files(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        fake_storage: FakeAsyncStorageWrapper,
    ) -> None:
        """Folder listing must filter out file entries, returning only folders."""
        _enable_drive(monkeypatch)
        _mock_async_storage(monkeypatch, fake_storage)

        response = await test_client.get("/api/v1/gdrive/folders")

        assert response.status_code == 200
        data = response.json()
        # fake_storage has 2 folders and 1 file; only folders should appear
        assert data["total"] == 2
        assert len(data["folders"]) == 2
        folder_names = {f["name"] for f in data["folders"]}
        assert "Family Photos" in folder_names
        assert "Vacation 2025" in folder_names

    @pytest.mark.asyncio
    async def test_uses_root_id_when_no_parent_provided(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        fake_storage: FakeAsyncStorageWrapper,
    ) -> None:
        """Without parentId query param, the configured root folder ID is used."""
        _enable_drive(monkeypatch)
        _mock_async_storage(monkeypatch, fake_storage)

        # Track which folder_id was passed to list_folder
        listed_ids: list[str] = []
        original_list = fake_storage.list_folder

        async def tracking_list(folder_id: str) -> list[StorageEntry]:
            listed_ids.append(folder_id)
            return await original_list(folder_id)

        fake_storage.list_folder = tracking_list  # type: ignore[method-assign]

        response = await test_client.get("/api/v1/gdrive/folders")
        assert response.status_code == 200
        assert listed_ids == ["fake-root-folder-id"]

    @pytest.mark.asyncio
    async def test_uses_provided_parent_id(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        fake_storage: FakeAsyncStorageWrapper,
    ) -> None:
        """When parentId is provided, it is passed directly to list_folder."""
        _enable_drive(monkeypatch)
        _mock_async_storage(monkeypatch, fake_storage)

        listed_ids: list[str] = []
        original_list = fake_storage.list_folder

        async def tracking_list(folder_id: str) -> list[StorageEntry]:
            listed_ids.append(folder_id)
            return await original_list(folder_id)

        fake_storage.list_folder = tracking_list  # type: ignore[method-assign]

        response = await test_client.get("/api/v1/gdrive/folders?parentId=custom-parent-folder-id")
        assert response.status_code == 200
        assert listed_ids == ["custom-parent-folder-id"]

    @pytest.mark.asyncio
    async def test_folder_list_response_has_all_contract_fields(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        fake_storage: FakeAsyncStorageWrapper,
    ) -> None:
        """Folder list response shape must match the API contract."""
        _enable_drive(monkeypatch)
        _mock_async_storage(monkeypatch, fake_storage)

        response = await test_client.get("/api/v1/gdrive/folders")
        assert response.status_code == 200
        data = response.json()

        # Top-level response fields
        assert "parentId" in data
        assert "folders" in data
        assert "total" in data

        # Per-folder entry fields
        if data["folders"]:
            folder = data["folders"][0]
            folder_fields = {"id", "name", "parentId", "hasChildren", "createdAt"}
            missing = folder_fields - set(folder.keys())
            assert not missing, f"DriveFolderInfo missing fields: {missing}"

    @pytest.mark.asyncio
    async def test_returns_storage_error_as_http_status(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """StorageError from Drive should be translated to an HTTP error response."""
        _enable_drive(monkeypatch)

        class PermissionDeniedStorage(FakeAsyncStorageWrapper):
            async def list_folder(self, folder_id: str) -> list[StorageEntry]:
                from image_search_service.storage.exceptions import StoragePermissionError

                raise StoragePermissionError(folder_id, "Not shared with service account")

        _mock_async_storage(monkeypatch, PermissionDeniedStorage())

        response = await test_client.get("/api/v1/gdrive/folders")
        assert response.status_code == 403
