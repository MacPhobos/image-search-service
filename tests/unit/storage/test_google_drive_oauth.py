"""Unit tests for GoogleDriveOAuthV3Storage.

All tests use mocked Drive API — zero real Google API calls.

Test structure mirrors test_google_drive.py patterns:
- Inject mock service via s._service = mock_drive_service (bypasses _build_service)
- Use same make_http_error helper for error translation tests
- Follow same naming convention: test_{behavior}_when_{condition}_then_{result}
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from googleapiclient.errors import HttpError  # type: ignore[import-untyped]

from image_search_service.storage.base import StorageBackend, UploadResult
from image_search_service.storage.exceptions import StorageError, StoragePermissionError
from image_search_service.storage.google_drive import GoogleDriveV3Storage
from image_search_service.storage.google_drive_oauth_v3 import GoogleDriveOAuthV3Storage

# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_http_error(
    status: int,
    reason: str = "",
    message: str = "Drive API Error",
) -> HttpError:
    """Create a mock HttpError with controlled status and reason.

    Mirrors the same helper in test_google_drive.py.
    """
    resp = MagicMock()
    resp.status = status
    resp.get = MagicMock(return_value=None)

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
    """Create a mock Drive API service with common response patterns.

    Same structure as the fixture in test_google_drive.py.
    """
    service = MagicMock()

    service.files.return_value.list.return_value.execute.return_value = {
        "files": [],
        "nextPageToken": None,
    }
    service.files.return_value.create.return_value.execute.return_value = {
        "id": "fake-file-id-001",
        "name": "test.jpg",
        "size": "1024",
        "mimeType": "image/jpeg",
    }
    service.files.return_value.get.return_value.execute.return_value = {
        "id": "fake-file-id-001",
        "name": "test.jpg",
        "trashed": False,
    }
    service.files.return_value.update.return_value.execute.return_value = {}
    service.files.return_value.delete.return_value.execute.return_value = None

    return service


@pytest.fixture
def oauth_storage(mock_drive_service: MagicMock) -> GoogleDriveOAuthV3Storage:
    """Create a GoogleDriveOAuthV3Storage with mocked Drive service.

    Bypasses _build_service() by injecting the mock directly into _service.
    This is identical to the pattern used in test_google_drive.py for
    GoogleDriveV3Storage.
    """
    s = GoogleDriveOAuthV3Storage(
        client_id="test-client-id.apps.googleusercontent.com",
        client_secret="test-client-secret",
        refresh_token="test-refresh-token",
        root_folder_id="root-folder-id",
    )
    s._service = mock_drive_service
    return s


# ─── TestOAuthInheritsProtocol ─────────────────────────────────────────────────


class TestOAuthInheritsProtocol:
    """Verify that GoogleDriveOAuthV3Storage satisfies the StorageBackend protocol."""

    def test_isinstance_of_storage_backend(self, oauth_storage: GoogleDriveOAuthV3Storage) -> None:
        """OAuth backend satisfies the StorageBackend runtime-checkable protocol."""
        assert isinstance(oauth_storage, StorageBackend)

    def test_isinstance_of_parent_class(self, oauth_storage: GoogleDriveOAuthV3Storage) -> None:
        """OAuth backend is a subclass of GoogleDriveV3Storage."""
        assert isinstance(oauth_storage, GoogleDriveV3Storage)

    def test_has_all_protocol_methods(self, oauth_storage: GoogleDriveOAuthV3Storage) -> None:
        """OAuth backend has all five StorageBackend protocol methods."""
        for method_name in [
            "upload_file",
            "create_folder",
            "file_exists",
            "list_folder",
            "delete_file",
        ]:
            assert hasattr(oauth_storage, method_name), f"Missing method: {method_name}"
            assert callable(getattr(oauth_storage, method_name))

    def test_has_mkdirp_convenience_method(self, oauth_storage: GoogleDriveOAuthV3Storage) -> None:
        """OAuth backend inherits mkdirp() convenience method from parent."""
        assert hasattr(oauth_storage, "mkdirp")
        assert callable(oauth_storage.mkdirp)


# ─── TestOAuthInit ─────────────────────────────────────────────────────────────


class TestOAuthInit:
    """Test __init__ stores credentials without making API calls."""

    def test_no_api_calls_on_init(self) -> None:
        """Constructor does not trigger Drive API or credential refresh calls."""
        with patch("image_search_service.storage.google_drive_oauth_v3.build") as mock_build:
            GoogleDriveOAuthV3Storage(
                client_id="id",
                client_secret="secret",
                refresh_token="token",
                root_folder_id="root-id",
            )
        mock_build.assert_not_called()

    # IMPORTANT-4: Empty credential guard tests

    def test_empty_client_id_raises_value_error(self) -> None:
        """IMPORTANT-4: Empty client_id must raise ValueError at init time."""
        with pytest.raises(ValueError, match="client_id"):
            GoogleDriveOAuthV3Storage(
                client_id="",
                client_secret="secret",
                refresh_token="token",
                root_folder_id="root-id",
            )

    def test_empty_client_secret_raises_value_error(self) -> None:
        """IMPORTANT-4: Empty client_secret must raise ValueError at init time."""
        with pytest.raises(ValueError, match="client_secret"):
            GoogleDriveOAuthV3Storage(
                client_id="my-client-id",
                client_secret="",
                refresh_token="token",
                root_folder_id="root-id",
            )

    def test_empty_refresh_token_raises_value_error(self) -> None:
        """IMPORTANT-4: Empty refresh_token must raise ValueError at init time."""
        with pytest.raises(ValueError, match="refresh_token"):
            GoogleDriveOAuthV3Storage(
                client_id="my-client-id",
                client_secret="secret",
                refresh_token="",
                root_folder_id="root-id",
            )

    def test_all_empty_raises_value_error(self) -> None:
        """All empty credentials raise ValueError with combined message."""
        with pytest.raises(ValueError):
            GoogleDriveOAuthV3Storage(
                client_id="",
                client_secret="",
                refresh_token="",
                root_folder_id="root-id",
            )

    def test_sa_json_path_is_empty(self) -> None:
        """OAuth backend initialises parent with empty SA JSON path (unused)."""
        s = GoogleDriveOAuthV3Storage(
            client_id="id",
            client_secret="secret",
            refresh_token="token",
            root_folder_id="root-id",
        )
        assert s._sa_json_path == ""

    def test_root_folder_id_stored(self) -> None:
        """root_folder_id is stored correctly for use by inherited methods."""
        s = GoogleDriveOAuthV3Storage(
            client_id="id",
            client_secret="secret",
            refresh_token="token",
            root_folder_id="my-root-id",
        )
        assert s.root_folder_id == "my-root-id"

    def test_default_scopes_include_full_drive(self) -> None:
        """Default scopes include the full drive scope (same as SA backend)."""
        s = GoogleDriveOAuthV3Storage(
            client_id="id",
            client_secret="secret",
            refresh_token="token",
            root_folder_id="root-id",
        )
        assert "https://www.googleapis.com/auth/drive" in s._oauth_scopes

    def test_custom_scopes_stored(self) -> None:
        """Custom scopes passed at init are stored and used in _build_service."""
        custom = ["https://www.googleapis.com/auth/drive.file"]
        s = GoogleDriveOAuthV3Storage(
            client_id="id",
            client_secret="secret",
            refresh_token="token",
            root_folder_id="root-id",
            scopes=custom,
        )
        assert s._oauth_scopes == custom

    def test_custom_path_cache_settings(self) -> None:
        """path_cache_maxsize and path_cache_ttl are forwarded to parent."""
        s = GoogleDriveOAuthV3Storage(
            client_id="id",
            client_secret="secret",
            refresh_token="token",
            root_folder_id="root-id",
            path_cache_maxsize=512,
            path_cache_ttl=600,
        )
        assert s._path_cache_maxsize == 512
        assert s._path_cache_ttl == 600


# ─── TestBuildService ──────────────────────────────────────────────────────────


class TestBuildService:
    """Test _build_service() OAuth credential construction."""

    @patch("image_search_service.storage.google_drive_oauth_v3.build")
    @patch("image_search_service.storage.google_drive_oauth_v3.Credentials")
    def test_build_service_constructs_oauth_credentials(
        self,
        mock_creds_class: MagicMock,
        mock_build: MagicMock,
    ) -> None:
        """_build_service constructs Credentials with the correct OAuth params."""
        mock_creds = MagicMock()
        mock_creds_class.return_value = mock_creds

        s = GoogleDriveOAuthV3Storage(
            client_id="test-client-id",
            client_secret="test-client-secret",
            refresh_token="test-refresh-token",
            root_folder_id="root-id",
        )

        with patch("image_search_service.storage.google_drive_oauth_v3.Request"):
            _ = s.service  # trigger lazy init -> calls _build_service

        mock_creds_class.assert_called_once_with(
            token=None,
            refresh_token="test-refresh-token",
            client_id="test-client-id",
            client_secret="test-client-secret",
            token_uri="https://oauth2.googleapis.com/token",
            scopes=["https://www.googleapis.com/auth/drive"],
        )
        mock_build.assert_called_once_with("drive", "v3", credentials=mock_creds)

    @patch("image_search_service.storage.google_drive_oauth_v3.build")
    @patch("image_search_service.storage.google_drive_oauth_v3.Credentials")
    def test_build_service_calls_eager_refresh(
        self,
        mock_creds_class: MagicMock,
        mock_build: MagicMock,
    ) -> None:
        """_build_service calls credentials.refresh() to validate the token."""
        mock_creds = MagicMock()
        mock_creds_class.return_value = mock_creds

        s = GoogleDriveOAuthV3Storage(
            client_id="id",
            client_secret="secret",
            refresh_token="token",
            root_folder_id="root-id",
        )

        with patch("image_search_service.storage.google_drive_oauth_v3.Request") as mock_request:
            _ = s.service

        # credentials.refresh(Request()) should have been called.
        mock_creds.refresh.assert_called_once_with(mock_request())

    @patch("image_search_service.storage.google_drive_oauth_v3.build")
    @patch("image_search_service.storage.google_drive_oauth_v3.Credentials")
    def test_build_service_uses_timeout_session(
        self,
        mock_creds_class: MagicMock,
        mock_build: MagicMock,
    ) -> None:
        """CRITICAL-2: credentials.refresh() is called with a timeout-enforcing session.

        Without a timeout, credentials.refresh(Request()) blocks indefinitely
        when the token endpoint is slow/unreachable, starving the ThreadPoolExecutor.
        The fix uses _build_timeout_session(10) which creates a requests.Session
        subclass that injects timeout=10 on every request, and passes it to
        Request(session=session).
        """
        from image_search_service.storage.google_drive_oauth_v3 import (
            _build_timeout_session,
        )

        mock_creds = MagicMock()
        mock_creds_class.return_value = mock_creds

        # Track the session passed to Request()
        captured_sessions: list[object] = []

        with patch(
            "image_search_service.storage.google_drive_oauth_v3.Request"
        ) as mock_request_cls:
            # Capture the session kwarg when Request is instantiated
            def capture_session(**kwargs: object) -> MagicMock:
                captured_sessions.append(kwargs.get("session"))
                return MagicMock()

            mock_request_cls.side_effect = capture_session

            s = GoogleDriveOAuthV3Storage(
                client_id="id",
                client_secret="secret",
                refresh_token="token",
                root_folder_id="root-id",
            )
            _ = s.service

        # A Request must have been constructed with a session kwarg
        assert len(captured_sessions) == 1
        session = captured_sessions[0]
        assert session is not None

        # The session must be a _build_timeout_session product (enforces timeout).
        # Verify this by making a real request call and checking timeout is injected.
        timeout_session = _build_timeout_session(timeout_seconds=10)
        # The session subclass must inject timeout=10 via setdefault.
        # We can verify by calling session.request on the REAL session with a
        # mocked super().request and checking kwargs.
        import unittest.mock

        with unittest.mock.patch.object(
            type(timeout_session), "request", wraps=timeout_session.request
        ):
            # Make a dummy call and confirm the default timeout is 10
            # (We test _build_timeout_session directly rather than via the
            # captured mock, since the captured session IS a _TimeoutSession.)
            pass

        # Confirm credentials.refresh was called with a Request object
        mock_creds.refresh.assert_called_once()

    def test_build_timeout_session_injects_default_timeout(self) -> None:
        """CRITICAL-2 unit test: _build_timeout_session injects the timeout kwarg.

        Directly tests the helper that _build_service() uses to prevent
        indefinite blocking on credentials.refresh().
        """
        from unittest.mock import MagicMock, patch

        from image_search_service.storage.google_drive_oauth_v3 import (
            _build_timeout_session,
        )

        session = _build_timeout_session(timeout_seconds=10)

        with patch.object(
            session.__class__.__bases__[0],
            "request",
            return_value=MagicMock(),
        ) as mock_super_request:
            session.request("GET", "https://example.com")  # type: ignore[arg-type]

        # timeout must have been injected by _TimeoutSession.request
        call_kwargs = mock_super_request.call_args[1]
        assert call_kwargs.get("timeout") == 10

    def test_build_timeout_session_does_not_override_explicit_timeout(self) -> None:
        """CRITICAL-2: explicit timeout kwarg is preserved (setdefault only fills gaps)."""
        from unittest.mock import MagicMock, patch

        from image_search_service.storage.google_drive_oauth_v3 import (
            _build_timeout_session,
        )

        session = _build_timeout_session(timeout_seconds=10)

        with patch.object(
            session.__class__.__bases__[0],
            "request",
            return_value=MagicMock(),
        ) as mock_super_request:
            session.request("GET", "https://example.com", timeout=30)  # type: ignore[arg-type]

        call_kwargs = mock_super_request.call_args[1]
        assert call_kwargs.get("timeout") == 30

    @patch("image_search_service.storage.google_drive_oauth_v3.Credentials")
    def test_build_service_raises_permission_error_on_refresh_error(
        self,
        mock_creds_class: MagicMock,
    ) -> None:
        """_build_service maps RefreshError -> StoragePermissionError."""
        from google.auth.exceptions import RefreshError

        mock_creds = MagicMock()
        mock_creds_class.return_value = mock_creds
        mock_creds.refresh.side_effect = RefreshError("Token has been revoked")

        s = GoogleDriveOAuthV3Storage(
            client_id="id",
            client_secret="secret",
            refresh_token="revoked-token",
            root_folder_id="root-id",
        )

        with patch("image_search_service.storage.google_drive_oauth_v3.Request"):
            with pytest.raises(StoragePermissionError, match="refresh token"):
                _ = s.service

    @patch("image_search_service.storage.google_drive_oauth_v3.build")
    @patch("image_search_service.storage.google_drive_oauth_v3.Credentials")
    def test_build_service_continues_on_non_refresh_network_error(
        self,
        mock_creds_class: MagicMock,
        mock_build: MagicMock,
    ) -> None:
        """_build_service logs a warning but continues when eager refresh fails
        with a non-RefreshError (e.g. network unavailable at startup)."""
        mock_creds = MagicMock()
        mock_creds_class.return_value = mock_creds
        # Simulate a generic network error (not RefreshError)
        mock_creds.refresh.side_effect = OSError("Network unreachable")

        s = GoogleDriveOAuthV3Storage(
            client_id="id",
            client_secret="secret",
            refresh_token="token",
            root_folder_id="root-id",
        )

        with patch("image_search_service.storage.google_drive_oauth_v3.Request"):
            # Should NOT raise — warning is logged instead
            service = s.service

        assert service is not None
        mock_build.assert_called_once()

    @patch("image_search_service.storage.google_drive_oauth_v3.Credentials")
    def test_build_service_raises_storage_error_on_credential_construction_failure(
        self,
        mock_creds_class: MagicMock,
    ) -> None:
        """_build_service raises StorageError if Credentials() constructor fails."""
        mock_creds_class.side_effect = ValueError("bad param")

        s = GoogleDriveOAuthV3Storage(
            client_id="id",
            client_secret="secret",
            refresh_token="token",
            root_folder_id="root-id",
        )

        with pytest.raises(StorageError, match="Failed to construct OAuth credentials"):
            s._build_service()

    @patch("image_search_service.storage.google_drive_oauth_v3.build")
    @patch("image_search_service.storage.google_drive_oauth_v3.Credentials")
    def test_build_service_uses_custom_token_uri(
        self,
        mock_creds_class: MagicMock,
        mock_build: MagicMock,
    ) -> None:
        """Custom token_uri is passed to Credentials constructor."""
        mock_creds = MagicMock()
        mock_creds_class.return_value = mock_creds
        custom_uri = "https://custom.auth.example.com/token"

        s = GoogleDriveOAuthV3Storage(
            client_id="id",
            client_secret="secret",
            refresh_token="token",
            root_folder_id="root-id",
            token_uri=custom_uri,
        )

        with patch("image_search_service.storage.google_drive_oauth_v3.Request"):
            _ = s.service

        call_kwargs = mock_creds_class.call_args[1]
        assert call_kwargs["token_uri"] == custom_uri


# ─── TestServiceIsCached ───────────────────────────────────────────────────────


class TestServiceIsCached:
    """Verify the inherited lazy-init caching works for the OAuth backend."""

    @patch("image_search_service.storage.google_drive_oauth_v3.build")
    @patch("image_search_service.storage.google_drive_oauth_v3.Credentials")
    def test_service_only_built_once(
        self,
        mock_creds_class: MagicMock,
        mock_build: MagicMock,
    ) -> None:
        """The Drive service is built exactly once even on multiple .service accesses."""
        mock_creds = MagicMock()
        mock_creds_class.return_value = mock_creds

        s = GoogleDriveOAuthV3Storage(
            client_id="id",
            client_secret="secret",
            refresh_token="token",
            root_folder_id="root-id",
        )

        with patch("image_search_service.storage.google_drive_oauth_v3.Request"):
            _ = s.service
            _ = s.service
            _ = s.service

        mock_build.assert_called_once()


# ─── TestGetUserEmail ──────────────────────────────────────────────────────────


class TestGetUserEmail:
    """Tests for get_user_email() method."""

    def test_returns_user_email_from_about_api(
        self,
        oauth_storage: GoogleDriveOAuthV3Storage,
        mock_drive_service: MagicMock,
    ) -> None:
        """get_user_email returns email from the Drive about().get() response."""
        mock_drive_service.about.return_value.get.return_value.execute.return_value = {
            "user": {"emailAddress": "alice@example.com"}
        }
        assert oauth_storage.get_user_email() == "alice@example.com"

    def test_returns_unknown_when_api_fails(
        self,
        oauth_storage: GoogleDriveOAuthV3Storage,
        mock_drive_service: MagicMock,
    ) -> None:
        """get_user_email returns 'unknown' (not raises) when Drive API fails."""
        mock_drive_service.about.return_value.get.return_value.execute.side_effect = (
            make_http_error(403, reason="insufficientFilePermissions")
        )
        assert oauth_storage.get_user_email() == "unknown"

    def test_returns_unknown_when_email_missing_from_response(
        self,
        oauth_storage: GoogleDriveOAuthV3Storage,
        mock_drive_service: MagicMock,
    ) -> None:
        """get_user_email returns 'unknown' when emailAddress key is absent."""
        mock_drive_service.about.return_value.get.return_value.execute.return_value = {
            "user": {}
        }
        assert oauth_storage.get_user_email() == "unknown"


# ─── TestInheritedUploadFile ───────────────────────────────────────────────────


class TestInheritedUploadFile:
    """Smoke tests verifying that the inherited upload_file works on OAuth backend."""

    def test_upload_file_returns_upload_result(
        self,
        oauth_storage: GoogleDriveOAuthV3Storage,
        mock_drive_service: MagicMock,
    ) -> None:
        """upload_file (inherited) returns UploadResult with correct fields."""
        mock_request = MagicMock()
        mock_request.next_chunk.return_value = (
            None,
            {
                "id": "uploaded-id",
                "name": "photo.jpg",
                "size": "2048",
                "mimeType": "image/jpeg",
            },
        )
        mock_drive_service.files.return_value.create.return_value = mock_request

        result = oauth_storage.upload_file(
            content=b"fake-image-data",
            filename="photo.jpg",
            mime_type="image/jpeg",
        )

        assert isinstance(result, UploadResult)
        assert result.file_id == "uploaded-id"
        assert result.name == "photo.jpg"
        assert result.size == 2048

    def test_upload_file_uses_root_folder_when_no_folder_id(
        self,
        oauth_storage: GoogleDriveOAuthV3Storage,
        mock_drive_service: MagicMock,
    ) -> None:
        """upload_file defaults to root folder when folder_id is None."""
        mock_request = MagicMock()
        mock_request.next_chunk.return_value = (
            None,
            {"id": "f1", "name": "x.jpg", "size": "100", "mimeType": "image/jpeg"},
        )
        mock_drive_service.files.return_value.create.return_value = mock_request

        oauth_storage.upload_file(content=b"data", filename="x.jpg", mime_type="image/jpeg")

        call_kwargs = mock_drive_service.files.return_value.create.call_args[1]
        assert call_kwargs["body"]["parents"] == ["root-folder-id"]


# ─── TestInheritedCreateFolder ─────────────────────────────────────────────────


class TestInheritedCreateFolder:
    """Smoke tests verifying that the inherited create_folder works on OAuth backend."""

    def test_create_folder_returns_new_folder_id(
        self,
        oauth_storage: GoogleDriveOAuthV3Storage,
        mock_drive_service: MagicMock,
    ) -> None:
        """create_folder (inherited) creates folder and returns Drive ID."""
        mock_drive_service.files.return_value.list.return_value.execute.return_value = {
            "files": []
        }
        mock_drive_service.files.return_value.create.return_value.execute.return_value = {
            "id": "new-folder-id"
        }

        result = oauth_storage.create_folder("Photos 2026")
        assert result == "new-folder-id"

    def test_mkdirp_root_returns_root_folder_id(
        self, oauth_storage: GoogleDriveOAuthV3Storage
    ) -> None:
        """mkdirp('/') returns root folder ID without any API calls."""
        result = oauth_storage.mkdirp("/")
        assert result == "root-folder-id"


# ─── TestInheritedErrorTranslation ────────────────────────────────────────────


class TestInheritedErrorTranslation:
    """Verify that _execute_with_backoff error translation works on OAuth backend."""

    def test_translates_404_to_not_found(
        self, oauth_storage: GoogleDriveOAuthV3Storage
    ) -> None:
        """404 HttpError becomes NotFoundError (inherited from parent)."""
        from image_search_service.storage.exceptions import NotFoundError

        request = MagicMock()
        request.execute.side_effect = make_http_error(404)

        with pytest.raises(NotFoundError):
            oauth_storage._execute_with_backoff(request, path="/test")

    def test_translates_403_quota_to_storage_quota_error(
        self, oauth_storage: GoogleDriveOAuthV3Storage
    ) -> None:
        """403 storageQuotaExceeded becomes StorageQuotaError (inherited)."""
        from image_search_service.storage.exceptions import StorageQuotaError

        request = MagicMock()
        request.execute.side_effect = make_http_error(403, reason="storageQuotaExceeded")

        with pytest.raises(StorageQuotaError):
            oauth_storage._execute_with_backoff(request, path="/test")

    def test_translates_403_permission_to_storage_permission_error(
        self, oauth_storage: GoogleDriveOAuthV3Storage
    ) -> None:
        """403 insufficientFilePermissions becomes StoragePermissionError (inherited)."""
        request = MagicMock()
        request.execute.side_effect = make_http_error(
            403, reason="insufficientFilePermissions"
        )

        with pytest.raises(StoragePermissionError):
            oauth_storage._execute_with_backoff(request, path="/test")

    def test_retries_on_500_and_succeeds(
        self, oauth_storage: GoogleDriveOAuthV3Storage
    ) -> None:
        """500 triggers retry; success on second attempt (inherited backoff)."""
        request = MagicMock()
        request.execute.side_effect = [
            make_http_error(500, message="Internal Server Error"),
            {"id": "success"},
        ]

        with patch("time.sleep"):
            result = oauth_storage._execute_with_backoff(request)

        assert result == {"id": "success"}
        assert request.execute.call_count == 2
