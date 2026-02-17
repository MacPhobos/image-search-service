"""Unit tests for GoogleDriveV3Storage.

All tests use mocked Drive API — zero real Google API calls.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from googleapiclient.errors import HttpError  # type: ignore[import-untyped]

from image_search_service.storage.base import EntryType, UploadResult
from image_search_service.storage.exceptions import (
    NotFoundError,
    PathAmbiguousError,
    RateLimitError,
    StorageError,
    StoragePermissionError,
    StorageQuotaError,
    UploadError,
)
from image_search_service.storage.google_drive import GoogleDriveV3Storage

# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_http_error(
    status: int,
    reason: str = "",
    message: str = "Drive API Error",
) -> HttpError:
    """Create a mock HttpError with controlled status and reason."""
    resp = MagicMock()
    resp.status = status
    resp.get = MagicMock(return_value=None)

    # Build error content
    content_dict: dict[str, Any] = {
        "error": {
            "message": message,
            "errors": [{"reason": reason, "message": message}] if reason else [],
        }
    }
    content = json.dumps(content_dict).encode("utf-8")
    return HttpError(resp=resp, content=content)


# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_drive_service() -> MagicMock:
    """Create a mock Drive API service with common response patterns."""
    service = MagicMock()

    # Default: files().list() returns empty
    service.files.return_value.list.return_value.execute.return_value = {
        "files": [],
        "nextPageToken": None,
    }

    # Default: files().create() returns a file ID (non-resumable)
    service.files.return_value.create.return_value.execute.return_value = {
        "id": "fake-file-id-001",
        "name": "test.jpg",
        "size": "1024",
        "mimeType": "image/jpeg",
    }

    # Default: files().get() returns file metadata (not trashed)
    service.files.return_value.get.return_value.execute.return_value = {
        "id": "fake-file-id-001",
        "name": "test.jpg",
        "trashed": False,
    }

    # Default: files().update() returns empty dict
    service.files.return_value.update.return_value.execute.return_value = {}

    # Default: files().delete() returns None (204 No Content)
    service.files.return_value.delete.return_value.execute.return_value = None

    return service


@pytest.fixture
def storage(mock_drive_service: MagicMock) -> GoogleDriveV3Storage:
    """Create a GoogleDriveV3Storage with mocked Drive service.

    Bypasses _build_service() to inject mock directly.
    """
    s = GoogleDriveV3Storage(
        service_account_json_path="/fake/sa.json",
        root_folder_id="root-folder-id",
    )
    # Inject mock service directly (bypass _build_service)
    s._service = mock_drive_service
    return s


# ─── TestUploadFile ────────────────────────────────────────────────────────────


class TestUploadFile:
    """Tests for GoogleDriveV3Storage.upload_file()"""

    def test_upload_file_returns_upload_result(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """upload_file returns UploadResult with correct field names."""
        # Configure mock for resumable upload
        mock_request = MagicMock()
        mock_request.next_chunk.return_value = (
            None,
            {
                "id": "uploaded-id-123",
                "name": "photo.jpg",
                "size": "2048",
                "mimeType": "image/jpeg",
            },
        )
        mock_drive_service.files.return_value.create.return_value = mock_request

        result = storage.upload_file(
            content=b"fake-image-data",
            filename="photo.jpg",
            mime_type="image/jpeg",
            folder_id="target-folder-id",
        )

        assert isinstance(result, UploadResult)
        assert result.file_id == "uploaded-id-123"
        assert result.name == "photo.jpg"  # NOT filename=
        assert result.size == 2048  # NOT size_bytes=
        assert result.mime_type == "image/jpeg"

    def test_upload_file_uses_root_folder_when_no_folder_id(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """upload_file defaults to root folder when folder_id is None."""
        mock_request = MagicMock()
        mock_request.next_chunk.return_value = (
            None,
            {"id": "file-id", "name": "photo.jpg", "size": "100", "mimeType": "image/jpeg"},
        )
        mock_drive_service.files.return_value.create.return_value = mock_request

        storage.upload_file(content=b"data", filename="photo.jpg", mime_type="image/jpeg")

        # Verify parents includes root folder
        call_kwargs = mock_drive_service.files.return_value.create.call_args[1]
        assert call_kwargs["body"]["parents"] == ["root-folder-id"]

    def test_upload_file_raises_upload_error_on_http_error(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """upload_file raises UploadError when Drive API returns persistent error."""
        mock_request = MagicMock()
        # Persistent 500 — retried MAX_RETRIES times then raises UploadError
        mock_request.next_chunk.side_effect = make_http_error(500, message="Internal")
        mock_drive_service.files.return_value.create.return_value = mock_request

        with patch("time.sleep"):  # Don't actually sleep during retry backoff
            with pytest.raises(UploadError) as exc_info:
                storage.upload_file(b"data", "test.jpg", "image/jpeg")

        assert exc_info.value.filename == "test.jpg"

    def test_upload_file_retries_transient_chunk_errors(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """upload_file retries 500 errors per chunk before raising UploadError."""
        mock_request = MagicMock()
        # First call raises 500, second succeeds — retry logic kicks in
        mock_request.next_chunk.side_effect = [
            make_http_error(500, message="Transient"),
            (None, {"id": "f1", "name": "photo.jpg", "size": "100", "mimeType": "image/jpeg"}),
        ]
        mock_drive_service.files.return_value.create.return_value = mock_request

        with patch("time.sleep"):
            result = storage.upload_file(b"data", "photo.jpg", "image/jpeg")

        assert result.file_id == "f1"
        assert mock_request.next_chunk.call_count == 2

    def test_upload_file_rejects_empty_content(self, storage: GoogleDriveV3Storage) -> None:
        """upload_file raises UploadError immediately for empty content."""
        with pytest.raises(UploadError, match="empty"):
            storage.upload_file(b"", "empty.jpg", "image/jpeg")

    def test_upload_file_sets_explicit_mime_type(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """upload_file passes MIME type to MediaIoBaseUpload to prevent Drive conversion."""
        with patch("image_search_service.storage.google_drive.MediaIoBaseUpload") as mock_media:
            mock_request = MagicMock()
            mock_request.next_chunk.return_value = (
                None,
                {"id": "f1", "name": "x.jpg", "size": "100", "mimeType": "image/jpeg"},
            )
            mock_drive_service.files.return_value.create.return_value = mock_request

            storage.upload_file(b"data", "x.jpg", "image/jpeg")

            mock_media.assert_called_once()
            # Verify mimetype kwarg is passed correctly
            call_kwargs = mock_media.call_args[1]
            assert call_kwargs.get("mimetype") == "image/jpeg"

    def test_upload_file_with_explicit_folder_id(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """upload_file uses the provided folder_id as parent."""
        mock_request = MagicMock()
        mock_request.next_chunk.return_value = (
            None,
            {"id": "f1", "name": "x.jpg", "size": "50", "mimeType": "image/jpeg"},
        )
        mock_drive_service.files.return_value.create.return_value = mock_request

        storage.upload_file(
            content=b"data",
            filename="x.jpg",
            mime_type="image/jpeg",
            folder_id="custom-folder-id",
        )

        call_kwargs = mock_drive_service.files.return_value.create.call_args[1]
        assert call_kwargs["body"]["parents"] == ["custom-folder-id"]

    def test_upload_file_handles_progress_chunks(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """upload_file loops until next_chunk returns (None, response)."""
        mock_status = MagicMock()
        mock_status.progress.return_value = 0.5

        mock_request = MagicMock()
        mock_request.next_chunk.side_effect = [
            (mock_status, None),  # First chunk: in progress
            (None, {"id": "f1", "name": "big.jpg", "size": "1000", "mimeType": "image/jpeg"}),
        ]
        mock_drive_service.files.return_value.create.return_value = mock_request

        result = storage.upload_file(b"data" * 1000, "big.jpg", "image/jpeg")
        assert result.file_id == "f1"
        assert mock_request.next_chunk.call_count == 2


# ─── TestCreateFolder ──────────────────────────────────────────────────────────


class TestCreateFolder:
    """Tests for GoogleDriveV3Storage.create_folder()"""

    def test_create_folder_returns_new_folder_id(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """create_folder creates folder and returns Drive ID."""
        mock_drive_service.files.return_value.list.return_value.execute.return_value = {"files": []}
        mock_drive_service.files.return_value.create.return_value.execute.return_value = {
            "id": "new-folder-id"
        }

        result = storage.create_folder("John Doe")
        assert result == "new-folder-id"

    def test_create_folder_returns_existing_id_when_found(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """create_folder returns existing folder ID instead of creating duplicate."""
        mock_drive_service.files.return_value.list.return_value.execute.return_value = {
            "files": [{"id": "existing-folder-id", "name": "John Doe"}]
        }

        result = storage.create_folder("John Doe")
        assert result == "existing-folder-id"
        # Verify create was NOT called
        mock_drive_service.files.return_value.create.assert_not_called()

    def test_create_folder_raises_ambiguous_error_on_duplicates(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """create_folder raises PathAmbiguousError when multiple folders match."""
        mock_drive_service.files.return_value.list.return_value.execute.return_value = {
            "files": [
                {"id": "folder-1", "name": "John Doe"},
                {"id": "folder-2", "name": "John Doe"},
            ]
        }

        with pytest.raises(PathAmbiguousError) as exc_info:
            storage.create_folder("John Doe")
        assert exc_info.value.count == 2

    def test_create_folder_uses_root_as_default_parent(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """create_folder uses root folder as parent when parent_id is None."""
        mock_drive_service.files.return_value.list.return_value.execute.return_value = {"files": []}
        mock_drive_service.files.return_value.create.return_value.execute.return_value = {
            "id": "new-id"
        }

        storage.create_folder("TestFolder")

        # Verify list query contains root folder ID
        list_call = mock_drive_service.files.return_value.list.call_args[1]
        assert "root-folder-id" in list_call["q"]

    def test_create_folder_escapes_single_quotes_in_name(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """create_folder escapes quotes in folder names for Drive query."""
        mock_drive_service.files.return_value.list.return_value.execute.return_value = {"files": []}
        mock_drive_service.files.return_value.create.return_value.execute.return_value = {
            "id": "new-id"
        }

        storage.create_folder("O'Brien")

        # Verify the query string properly escaped the single quote.
        # _escape_query_value("O'Brien") -> "O\\'Brien"
        # So the query should contain: name='O\'Brien'
        call_kwargs = mock_drive_service.files.return_value.list.call_args[1]
        assert "O\\'Brien" in call_kwargs["q"]

    def test_create_folder_invalidates_parent_cache(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """create_folder invalidates PathResolver cache for parent folder."""
        mock_drive_service.files.return_value.list.return_value.execute.return_value = {"files": []}
        mock_drive_service.files.return_value.create.return_value.execute.return_value = {
            "id": "new-folder-id"
        }

        # Seed the resolver cache with an entry for the parent
        storage.resolver.cache_put("root-folder-id", "SomeChild", "old-id", True)
        assert storage.resolver.cache_size == 1

        storage.create_folder("NewFolder")

        # Cache for the parent should be cleared
        assert storage.resolver.cache_size == 0

    def test_create_folder_with_explicit_parent_id(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """create_folder uses provided parent_id."""
        mock_drive_service.files.return_value.list.return_value.execute.return_value = {"files": []}
        mock_drive_service.files.return_value.create.return_value.execute.return_value = {
            "id": "child-folder-id"
        }

        result = storage.create_folder("Child", parent_id="parent-folder-id")
        assert result == "child-folder-id"

        list_call = mock_drive_service.files.return_value.list.call_args[1]
        assert "parent-folder-id" in list_call["q"]


# ─── TestFileExists ────────────────────────────────────────────────────────────


class TestFileExists:
    """Tests for GoogleDriveV3Storage.file_exists()"""

    def test_file_exists_returns_true_for_existing_file(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """file_exists returns True when Drive API returns file metadata."""
        mock_drive_service.files.return_value.get.return_value.execute.return_value = {
            "id": "file-id",
            "trashed": False,
        }
        assert storage.file_exists("file-id") is True

    def test_file_exists_returns_false_for_trashed_file(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """file_exists returns False when file is in trash."""
        mock_drive_service.files.return_value.get.return_value.execute.return_value = {
            "id": "file-id",
            "trashed": True,
        }
        assert storage.file_exists("file-id") is False

    def test_file_exists_returns_false_for_404(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """file_exists returns False when file not found (404)."""
        mock_drive_service.files.return_value.get.return_value.execute.side_effect = (
            make_http_error(404)
        )
        assert storage.file_exists("nonexistent-id") is False


# ─── TestListFolder ────────────────────────────────────────────────────────────


class TestListFolder:
    """Tests for GoogleDriveV3Storage.list_folder()"""

    def test_list_folder_returns_entries(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """list_folder returns StorageEntry objects from Drive API response."""
        mock_drive_service.files.return_value.list.return_value.execute.return_value = {
            "files": [
                {
                    "id": "f1",
                    "name": "photo.jpg",
                    "mimeType": "image/jpeg",
                    "size": "1024",
                    "modifiedTime": "2026-01-01T00:00:00Z",
                },
                {
                    "id": "f2",
                    "name": "subfolder",
                    "mimeType": "application/vnd.google-apps.folder",
                    "size": None,
                    "modifiedTime": "2026-01-02T00:00:00Z",
                },
            ],
            "nextPageToken": None,
        }

        entries = storage.list_folder("folder-id")
        assert len(entries) == 2
        assert entries[0].entry_type == EntryType.FILE
        assert entries[0].name == "photo.jpg"
        assert entries[0].size == 1024
        assert entries[1].entry_type == EntryType.FOLDER
        assert entries[1].size is None

    def test_list_folder_handles_pagination(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """list_folder fetches all pages when nextPageToken is present."""
        mock_drive_service.files.return_value.list.return_value.execute.side_effect = [
            {
                "files": [
                    {
                        "id": "f1",
                        "name": "a.jpg",
                        "mimeType": "image/jpeg",
                        "size": "100",
                        "modifiedTime": "2026-01-01T00:00:00Z",
                    }
                ],
                "nextPageToken": "token2",
            },
            {
                "files": [
                    {
                        "id": "f2",
                        "name": "b.jpg",
                        "mimeType": "image/jpeg",
                        "size": "200",
                        "modifiedTime": "2026-01-02T00:00:00Z",
                    }
                ],
                "nextPageToken": None,
            },
        ]

        entries = storage.list_folder("folder-id")
        assert len(entries) == 2
        assert entries[0].id == "f1"
        assert entries[1].id == "f2"

    def test_list_folder_returns_empty_for_empty_folder(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """list_folder returns empty list for folder with no children."""
        mock_drive_service.files.return_value.list.return_value.execute.return_value = {
            "files": [],
            "nextPageToken": None,
        }
        assert storage.list_folder("empty-folder") == []

    def test_list_folder_sets_none_size_for_folders(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """list_folder sets size=None for folder entries."""
        mock_drive_service.files.return_value.list.return_value.execute.return_value = {
            "files": [
                {
                    "id": "dir1",
                    "name": "MyFolder",
                    "mimeType": "application/vnd.google-apps.folder",
                    "modifiedTime": "2026-01-01T00:00:00Z",
                }
            ],
            "nextPageToken": None,
        }

        entries = storage.list_folder("parent-id")
        assert len(entries) == 1
        assert entries[0].size is None
        assert entries[0].entry_type == EntryType.FOLDER


# ─── TestDeleteFile ────────────────────────────────────────────────────────────


class TestDeleteFile:
    """Tests for GoogleDriveV3Storage.delete_file()"""

    def test_delete_file_trashes_file_by_default(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """delete_file moves file to trash (not permanent delete) by default."""
        storage.delete_file("file-id")

        mock_drive_service.files.return_value.update.assert_called_once()
        call_kwargs = mock_drive_service.files.return_value.update.call_args[1]
        assert call_kwargs["body"] == {"trashed": True}
        assert call_kwargs["fileId"] == "file-id"
        # Verify permanent delete was NOT called
        mock_drive_service.files.return_value.delete.assert_not_called()

    def test_delete_file_permanently_deletes_when_trash_false(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """delete_file permanently deletes when trash=False."""
        storage.delete_file("file-id", trash=False)

        mock_drive_service.files.return_value.delete.assert_called_once()
        call_kwargs = mock_drive_service.files.return_value.delete.call_args[1]
        assert call_kwargs["fileId"] == "file-id"
        # Verify update (trash) was NOT called
        mock_drive_service.files.return_value.update.assert_not_called()

    def test_delete_file_raises_not_found(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """delete_file raises NotFoundError for nonexistent file."""
        # get() for parent lookup raises 404 — file already gone
        # update() (trash) also raises 404 — file already gone
        mock_drive_service.files.return_value.get.return_value.execute.side_effect = (
            make_http_error(404)
        )
        mock_drive_service.files.return_value.update.return_value.execute.side_effect = (
            make_http_error(404)
        )
        with pytest.raises(NotFoundError):
            storage.delete_file("nonexistent-id")

    def test_delete_file_invalidates_parent_cache(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """delete_file invalidates PathResolver cache for parent folders after deletion."""
        # Setup: get() returns file metadata with parent info
        mock_drive_service.files.return_value.get.return_value.execute.return_value = {
            "parents": ["parent-folder-id"],
        }
        mock_drive_service.files.return_value.update.return_value.execute.return_value = {}

        # Pre-populate the resolver cache with a stale entry for the parent
        storage.resolver.cache_put("parent-folder-id", "old-file", "old-id", False)
        assert storage.resolver.cache_size == 1

        storage.delete_file("file-id")

        # Cache for the parent folder should be cleared after deletion
        assert storage.resolver.cache_size == 0

    def test_delete_file_skips_cache_invalidation_when_file_already_gone(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """delete_file skips cache invalidation gracefully when parent get returns 404."""
        # get() raises 404 for parent lookup — silently ignored (file may have been
        # deleted already between our lookup and the actual delete call).
        mock_drive_service.files.return_value.get.return_value.execute.side_effect = (
            make_http_error(404)
        )
        # update() (trash) also raises 404 since file is gone
        mock_drive_service.files.return_value.update.return_value.execute.side_effect = (
            make_http_error(404)
        )

        # NotFoundError from the actual trash operation should propagate
        with pytest.raises(NotFoundError):
            storage.delete_file("file-id")


# ─── TestErrorTranslation ──────────────────────────────────────────────────────


class TestErrorTranslation:
    """Tests for _execute_with_backoff error translation."""

    def test_translates_404_to_not_found(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """404 HttpError becomes NotFoundError."""
        request = MagicMock()
        request.execute.side_effect = make_http_error(404)

        with pytest.raises(NotFoundError):
            storage._execute_with_backoff(request, path="/test")

    def test_translates_403_quota_to_storage_quota_error(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """403 with storageQuotaExceeded becomes StorageQuotaError."""
        request = MagicMock()
        request.execute.side_effect = make_http_error(403, reason="storageQuotaExceeded")

        with pytest.raises(StorageQuotaError):
            storage._execute_with_backoff(request, path="/test")

    def test_translates_403_permission_to_storage_permission_error(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """403 with insufficientFilePermissions becomes StoragePermissionError."""
        request = MagicMock()
        request.execute.side_effect = make_http_error(403, reason="insufficientFilePermissions")

        with pytest.raises(StoragePermissionError):
            storage._execute_with_backoff(request, path="/test")

    def test_translates_400_to_storage_error(self, storage: GoogleDriveV3Storage) -> None:
        """400 HttpError becomes generic StorageError."""
        request = MagicMock()
        request.execute.side_effect = make_http_error(400, message="Bad request")

        with pytest.raises(StorageError):
            storage._execute_with_backoff(request, path="/test")

    def test_retries_on_500_and_eventually_succeeds(self, storage: GoogleDriveV3Storage) -> None:
        """500 errors trigger exponential backoff retry, success on second attempt."""
        request = MagicMock()
        request.execute.side_effect = [
            make_http_error(500, message="Internal Server Error"),
            {"id": "success"},
        ]

        with patch("time.sleep"):  # Don't actually sleep in tests
            result = storage._execute_with_backoff(request)

        assert result == {"id": "success"}
        assert request.execute.call_count == 2

    def test_retries_on_503_and_eventually_succeeds(self, storage: GoogleDriveV3Storage) -> None:
        """503 errors trigger retry."""
        request = MagicMock()
        request.execute.side_effect = [
            make_http_error(503, message="Service Unavailable"),
            {"result": "ok"},
        ]

        with patch("time.sleep"):
            result = storage._execute_with_backoff(request)

        assert result == {"result": "ok"}

    def test_raises_rate_limit_after_max_retries_on_429(
        self, storage: GoogleDriveV3Storage
    ) -> None:
        """429 after max retries becomes RateLimitError."""
        request = MagicMock()
        request.execute.side_effect = make_http_error(429, message="Rate Limited")

        with patch("time.sleep"):
            with pytest.raises(RateLimitError):
                storage._execute_with_backoff(request)

    def test_raises_rate_limit_for_403_rate_limit_reason_after_max_retries(
        self, storage: GoogleDriveV3Storage
    ) -> None:
        """403 with rateLimitExceeded after max retries becomes RateLimitError."""
        request = MagicMock()
        request.execute.side_effect = make_http_error(403, reason="rateLimitExceeded")

        with patch("time.sleep"):
            with pytest.raises(RateLimitError):
                storage._execute_with_backoff(request)

    def test_raises_storage_error_after_max_retries_on_500(
        self, storage: GoogleDriveV3Storage
    ) -> None:
        """500 after max retries becomes StorageError."""
        request = MagicMock()
        request.execute.side_effect = make_http_error(500, message="Server Error")

        with patch("time.sleep"):
            with pytest.raises(StorageError):
                storage._execute_with_backoff(request)

    def test_translates_unexpected_status_to_storage_error(
        self, storage: GoogleDriveV3Storage
    ) -> None:
        """Unexpected HTTP status becomes StorageError."""
        request = MagicMock()
        request.execute.side_effect = make_http_error(422, message="Unprocessable")

        with pytest.raises(StorageError):
            storage._execute_with_backoff(request)


# ─── TestLazyInitialization ────────────────────────────────────────────────────


class TestLazyInitialization:
    """Tests for lazy initialization pattern."""

    def test_no_api_calls_on_init(self) -> None:
        """Constructor does not trigger Drive API calls."""
        with patch("image_search_service.storage.google_drive.build") as mock_build:
            GoogleDriveV3Storage(
                service_account_json_path="/fake/path.json",
                root_folder_id="root-id",
            )
            mock_build.assert_not_called()

    def test_service_built_on_first_access(self) -> None:
        """Drive service is built only when .service property is accessed."""
        with patch("image_search_service.storage.google_drive.build") as mock_build:
            with patch("image_search_service.storage.google_drive.service_account") as mock_sa:
                mock_sa.Credentials.from_service_account_file.return_value = MagicMock()
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("pathlib.Path.stat") as mock_stat:
                        mock_stat.return_value.st_mode = 0o600  # Safe permissions
                        storage = GoogleDriveV3Storage("/fake.json", "root-id")
                        mock_build.assert_not_called()
                        _ = storage.service
                        mock_build.assert_called_once()

    def test_service_is_cached_after_first_access(self) -> None:
        """Drive service is not rebuilt on subsequent accesses."""
        with patch("image_search_service.storage.google_drive.build") as mock_build:
            with patch("image_search_service.storage.google_drive.service_account") as mock_sa:
                mock_sa.Credentials.from_service_account_file.return_value = MagicMock()
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("pathlib.Path.stat") as mock_stat:
                        mock_stat.return_value.st_mode = 0o600
                        storage = GoogleDriveV3Storage("/fake.json", "root-id")
                        _ = storage.service
                        _ = storage.service
                        mock_build.assert_called_once()

    def test_resolver_is_lazy(self) -> None:
        """PathResolver is not created until .resolver property is accessed."""
        storage = GoogleDriveV3Storage(
            service_account_json_path="/fake/path.json",
            root_folder_id="root-id",
        )
        assert storage._resolver is None
        # Access the property — injects a mock service first
        storage._service = MagicMock()
        _ = storage.resolver
        assert storage._resolver is not None


# ─── TestBuildServiceError ─────────────────────────────────────────────────────


class TestBuildServiceError:
    """Tests for _build_service() error handling."""

    def test_raises_storage_error_when_sa_file_missing(self) -> None:
        """_build_service raises StorageError when SA key file doesn't exist."""
        storage = GoogleDriveV3Storage("/nonexistent/path.json", "root-id")

        with pytest.raises(StorageError, match="Service account key file not found"):
            storage._build_service()


# ─── TestRootFolderValidation ──────────────────────────────────────────────────


class TestRootFolderValidation:
    """Tests for _validate_root_access()"""

    def test_validate_root_returns_true_when_accessible(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """_validate_root_access returns True for accessible folder."""
        mock_drive_service.files.return_value.get.return_value.execute.return_value = {
            "id": "root-id",
            "name": "Photos",
            "mimeType": "application/vnd.google-apps.folder",
            "trashed": False,
        }
        assert storage._validate_root_access() is True

    def test_validate_root_returns_false_when_trashed(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """_validate_root_access returns False for trashed root folder."""
        mock_drive_service.files.return_value.get.return_value.execute.return_value = {
            "id": "root-id",
            "trashed": True,
            "mimeType": "application/vnd.google-apps.folder",
        }
        assert storage._validate_root_access() is False

    def test_validate_root_returns_false_when_not_a_folder(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """_validate_root_access returns False when root ID points to a file."""
        mock_drive_service.files.return_value.get.return_value.execute.return_value = {
            "id": "root-id",
            "trashed": False,
            "mimeType": "image/jpeg",  # Not a folder!
        }
        assert storage._validate_root_access() is False

    def test_validate_root_returns_false_on_storage_error(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """_validate_root_access returns False when API call fails with 403."""
        mock_drive_service.files.return_value.get.return_value.execute.side_effect = (
            make_http_error(403, reason="insufficientFilePermissions")
        )
        assert storage._validate_root_access() is False

    def test_validate_root_returns_false_on_404(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """_validate_root_access returns False when root folder not found."""
        mock_drive_service.files.return_value.get.return_value.execute.side_effect = (
            make_http_error(404)
        )
        assert storage._validate_root_access() is False


# ─── TestMkdirp ───────────────────────────────────────────────────────────────


class TestMkdirp:
    """Tests for mkdirp() convenience method."""

    def test_mkdirp_root_returns_root_folder_id(self, storage: GoogleDriveV3Storage) -> None:
        """mkdirp('/') returns root folder ID without any API calls."""
        result = storage.mkdirp("/")
        assert result == "root-folder-id"

    def test_mkdirp_creates_nested_folders(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """mkdirp creates folders for each path segment."""
        # First call (people): no existing
        # Second call (John Doe): no existing
        mock_drive_service.files.return_value.list.return_value.execute.side_effect = [
            {"files": []},  # check for "people"
            {"files": []},  # check for "John Doe"
        ]
        mock_drive_service.files.return_value.create.return_value.execute.side_effect = [
            {"id": "people-id"},  # create "people"
            {"id": "john-id"},  # create "John Doe"
        ]

        result = storage.mkdirp("/people/John Doe")
        assert result == "john-id"

    def test_mkdirp_single_segment(
        self, storage: GoogleDriveV3Storage, mock_drive_service: MagicMock
    ) -> None:
        """mkdirp with single segment creates one folder."""
        mock_drive_service.files.return_value.list.return_value.execute.return_value = {"files": []}
        mock_drive_service.files.return_value.create.return_value.execute.return_value = {
            "id": "folder-id"
        }

        result = storage.mkdirp("/people")
        assert result == "folder-id"


# ─── TestEscapeQueryValue ─────────────────────────────────────────────────────


class TestEscapeQueryValue:
    """Tests for _escape_query_value static method."""

    def test_escapes_single_quotes(self) -> None:
        """Single quotes are escaped with backslash."""
        result = GoogleDriveV3Storage._escape_query_value("O'Brien")
        assert result == "O\\'Brien"

    def test_escapes_backslashes(self) -> None:
        """Backslashes are escaped first."""
        result = GoogleDriveV3Storage._escape_query_value("path\\to")
        assert result == "path\\\\to"

    def test_safe_string_unchanged(self) -> None:
        """String without special chars is returned unchanged."""
        result = GoogleDriveV3Storage._escape_query_value("John Doe")
        assert result == "John Doe"


# ─── TestGetServiceAccountEmail ───────────────────────────────────────────────


class TestGetServiceAccountEmail:
    """Tests for get_service_account_email()."""

    def test_returns_email_from_json(self, tmp_path: Any) -> None:
        """get_service_account_email extracts email from SA JSON."""
        sa_json = tmp_path / "sa.json"
        sa_json.write_text(json.dumps({"client_email": "sa@project.iam.gserviceaccount.com"}))

        storage = GoogleDriveV3Storage(str(sa_json), "root-id")
        assert storage.get_service_account_email() == "sa@project.iam.gserviceaccount.com"

    def test_returns_none_when_file_missing(self) -> None:
        """get_service_account_email returns None for missing file."""
        storage = GoogleDriveV3Storage("/nonexistent/sa.json", "root-id")
        assert storage.get_service_account_email() is None
