"""Google Drive v3 implementation using OAuth 2.0 user credentials.

Subclasses GoogleDriveV3Storage, overriding only _build_service() to use
OAuth user credentials (refresh_token + client_id + client_secret) instead
of a service account JSON key file.

All operational logic (upload, folder operations, retry, path resolution,
error translation) is inherited from the parent class unchanged.

This is NOT a re-implementation — it is a single-method override (~50 lines)
that swaps the credential construction path while keeping every other behaviour
identical to the existing service-account backend.

Usage:
    storage = GoogleDriveOAuthV3Storage(
        client_id="xxxx.apps.googleusercontent.com",
        client_secret="GOCSPX-xxxx",
        refresh_token="1//xxxx",
        root_folder_id="1ABC_your_root_folder_id",
    )
    result = storage.upload_file(b"...", "photo.jpg", "image/jpeg")

Thread Safety:
    Same as parent class — double-checked locking for lazy init of the
    Drive service. OAuth token refresh is idempotent and safe under
    concurrent access from multiple ThreadPoolExecutor workers.

IMPORTANT: _build_service() is the ONLY method overridden here.
    GoogleDriveV3Storage._build_service() is the SA credential path.
    This subclass replaces that with the OAuth credential path.
"""

from __future__ import annotations

from typing import Any

import requests as requests_lib
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from image_search_service.core.logging import get_logger
from image_search_service.storage.exceptions import StorageError, StoragePermissionError
from image_search_service.storage.google_drive import GoogleDriveV3Storage

_logger = get_logger(__name__)

# Token endpoint used for refreshing access tokens.
_TOKEN_URI = "https://oauth2.googleapis.com/token"


def _build_timeout_session(timeout_seconds: int) -> requests_lib.Session:
    """Create a requests.Session that enforces a default timeout.

    requests.Session has no built-in default-timeout attribute, so we
    subclass it and inject the timeout via kwargs.setdefault inside the
    overridden request() method.

    Args:
        timeout_seconds: Default timeout to apply to every request.

    Returns:
        A requests.Session subclass instance with the timeout pre-set.
    """

    class _TimeoutSession(requests_lib.Session):
        def request(  # type: ignore[override]
            self,
            method: str | bytes,
            url: str | bytes,
            **kwargs: object,
        ) -> requests_lib.Response:
            kwargs.setdefault("timeout", timeout_seconds)
            return super().request(method, url, **kwargs)  # type: ignore[arg-type]

    return _TimeoutSession()


class GoogleDriveOAuthV3Storage(GoogleDriveV3Storage):
    """Google Drive API v3 using OAuth 2.0 user credentials.

    Overrides _build_service() to construct credentials from
    refresh_token + client_id + client_secret instead of a
    service account JSON key file.

    All operational methods (upload_file, create_folder, file_exists,
    list_folder, delete_file, mkdirp) and retry/error-handling logic are
    inherited from GoogleDriveV3Storage without modification.

    Attributes:
        _client_id: OAuth 2.0 client ID.
        _client_secret: OAuth 2.0 client secret.
        _refresh_token: OAuth 2.0 offline refresh token.
        _token_uri: Token endpoint URL (default: Google's OAuth2 endpoint).
        _oauth_scopes: Requested Drive API scopes.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        root_folder_id: str,
        token_uri: str = _TOKEN_URI,
        scopes: list[str] | None = None,
        path_cache_maxsize: int = 1024,
        path_cache_ttl: int = 300,
    ) -> None:
        """Initialize OAuth storage backend.

        No API calls are made here. The Drive service is lazily initialized
        on first use via the inherited .service property.

        Args:
            client_id: OAuth 2.0 client ID from Google Cloud Console.
            client_secret: OAuth 2.0 client secret (treat as sensitive).
            refresh_token: Offline refresh token from the bootstrap script.
            root_folder_id: Google Drive folder ID for uploads (app root).
            token_uri: Token exchange endpoint. Defaults to Google's OAuth2 URL.
            scopes: Drive API scopes. Defaults to full drive access.
            path_cache_maxsize: Maximum entries in path-to-ID cache.
            path_cache_ttl: TTL in seconds for path-to-ID cache entries.
        """
        if not client_id or not client_secret or not refresh_token:
            raise ValueError(
                "client_id, client_secret, and refresh_token must all be non-empty"
            )

        # Store OAuth credentials BEFORE calling super().__init__ so they
        # are available when _build_service() is eventually called.
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._token_uri = token_uri
        self._oauth_scopes = scopes if scopes is not None else list(self.SCOPES)

        # Call parent with an empty SA path — it is stored but never used,
        # because _build_service() is overridden here.
        super().__init__(
            service_account_json_path="",  # unused — overridden by _build_service
            root_folder_id=root_folder_id,
            path_cache_maxsize=path_cache_maxsize,
            path_cache_ttl=path_cache_ttl,
        )

    # ─── Override: credential construction ────────────────────────────────────

    def _build_service(self) -> Any:
        """Build Drive v3 service using OAuth 2.0 user credentials.

        Constructs google.oauth2.credentials.Credentials from the stored
        refresh_token + client_id + client_secret. The access token is
        obtained automatically on first API call via a token refresh against
        the token_uri.

        An eager credential refresh is attempted immediately so that a
        revoked or expired refresh token is detected at initialization time
        (rather than on the first user-facing API call).

        Returns:
            googleapiclient Resource for Drive v3 API.

        Raises:
            StoragePermissionError: If the refresh token is expired or revoked.
            StorageError: If service construction fails for any other reason.
        """
        # Log client_id prefix only — never log secrets or tokens.
        client_id_prefix = (self._client_id[:20] + "...") if self._client_id else "<empty>"
        _logger.info(
            "Building Drive v3 service with OAuth user credentials",
            extra={"client_id_prefix": client_id_prefix},
        )

        try:
            credentials = Credentials(  # type: ignore[no-untyped-call]
                token=None,  # Will be populated automatically on first refresh
                refresh_token=self._refresh_token,
                client_id=self._client_id,
                client_secret=self._client_secret,
                token_uri=self._token_uri,
                scopes=self._oauth_scopes,
            )
        except Exception as exc:
            raise StorageError(f"Failed to construct OAuth credentials: {exc}") from exc

        # Eagerly validate the refresh token before the service is used.
        # This surfaces expired/revoked tokens at startup rather than
        # failing silently on the first real API call.
        #
        # Use an explicit 10-second timeout on the token-endpoint POST to
        # prevent ThreadPoolExecutor workers from blocking indefinitely when
        # the token endpoint is slow or unreachable.
        #
        # requests.Session has no built-in default timeout attribute, so we
        # subclass it to inject the timeout via kwargs.setdefault.
        session = _build_timeout_session(timeout_seconds=10)
        try:
            credentials.refresh(Request(session=session))
        except RefreshError as exc:
            raise StoragePermissionError(
                path="<oauth-credential-refresh>",
                detail=(
                    f"OAuth refresh token is invalid or revoked: {exc}. "
                    "Re-run the bootstrap script: "
                    "python scripts/gdrive_oauth_bootstrap.py"
                ),
            ) from exc
        except Exception as exc:
            # Non-fatal: network may be unavailable during init.
            # The credential will attempt refresh on the first real API call.
            _logger.warning(
                "Could not eagerly refresh OAuth token (will retry on first API call): %s",
                exc,
            )

        return build("drive", "v3", credentials=credentials)

    # ─── Additional method: user identity ─────────────────────────────────────

    def get_user_email(self) -> str:
        """Return the authenticated user's email address.

        Uses the Drive API about().get() call to retrieve the user's profile
        information. Replaces get_service_account_email() for the OAuth
        authentication context.

        Returns:
            User email string, or "unknown" if retrieval fails.
        """
        try:
            about = self._execute_with_backoff(
                self.service.about().get(fields="user(emailAddress)"),
                path="<about>",
            )
            email: str = about.get("user", {}).get("emailAddress", "unknown")
            return email
        except Exception as exc:
            _logger.warning("Could not retrieve authenticated user email: %s", exc)
            return "unknown"
