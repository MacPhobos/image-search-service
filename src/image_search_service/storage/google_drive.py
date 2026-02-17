"""Google Drive v3 implementation of the StorageBackend protocol.

Provides cloud storage via Google Drive using service account authentication.
All operations are confined within a configured root folder ID.

IMPORTANT — Thread Safety:
- The google-api-python-client service object is NOT thread-safe per call.
- Each GoogleDriveV3Storage instance creates its own Drive service.
- Lazy initialization of .service and .resolver uses double-checked locking
  (threading.Lock) to prevent races when multiple ThreadPoolExecutor workers
  access the instance concurrently for the first time.
- The PathResolver cache IS thread-safe (uses threading.Lock internally).

Lazy Initialization:
- No import-time side effects (per CLAUDE.md rule #2).
- Drive service is built on first use via @property.
- Credential validation happens on first API call, not __init__.
"""

from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path
from typing import Any

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseUpload

from image_search_service.core.logging import get_logger
from image_search_service.storage.base import (
    EntryType,
    StorageEntry,
    UploadResult,
)
from image_search_service.storage.exceptions import (
    NotFoundError,
    PathAmbiguousError,
    RateLimitError,
    StorageError,
    StoragePermissionError,
    StorageQuotaError,
    UploadError,
)
from image_search_service.storage.path_resolver import (
    LookupFn,
    PathResolver,
    normalize_path,
)


class GoogleDriveV3Storage:
    """Google Drive API v3 implementation of StorageBackend.

    Uses service account authentication to interact with a shared
    Google Drive folder. All operations are confined within a
    configured root folder ID.

    Lazy Initialization:
        No API calls at import or __init__ time. The Drive service
        is built on first access via the .service property.

    Usage:
        storage = GoogleDriveV3Storage(
            service_account_json_path="/secrets/sa.json",
            root_folder_id="1ABC_your_root_folder_id",
        )
        result = storage.upload_file(b"...", "photo.jpg", "image/jpeg")
    """

    DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"
    SCOPES = ["https://www.googleapis.com/auth/drive"]
    MAX_RETRIES = 5
    MAX_BACKOFF_SECONDS = 64
    RESUMABLE_CHUNK_SIZE = 256 * 1024 * 4  # 1 MB (multiple of 256 KB)

    def __init__(
        self,
        service_account_json_path: str,
        root_folder_id: str,
        path_cache_maxsize: int = 1024,
        path_cache_ttl: int = 300,
    ) -> None:
        """Initialize storage backend.

        No API calls are made here. The Drive service is lazily initialized
        on first use.

        Args:
            service_account_json_path: Path to the Google service account
                                       JSON key file.
            root_folder_id: Google Drive folder ID that serves as the root.
                            Must be shared with the service account.
            path_cache_maxsize: Maximum entries in the path-to-ID cache.
            path_cache_ttl: TTL in seconds for path-to-ID cache entries.
        """
        self._sa_json_path = service_account_json_path
        self._root_folder_id = root_folder_id
        self._path_cache_maxsize = path_cache_maxsize
        self._path_cache_ttl = path_cache_ttl

        # Lazy-initialized (no import-time side effects)
        # _service typed as Any because googleapiclient.Resource is dynamic
        # (`.files()` and other method calls are not statically typed).
        self._service: Any = None
        self._resolver: PathResolver | None = None
        self._logger = get_logger(__name__)

        # Thread-safety lock for lazy initialization.
        # Multiple threads from the ThreadPoolExecutor can race on first access.
        # Double-checked locking pattern used in .service and .resolver properties.
        self._init_lock = threading.Lock()

    # ─── Lazy-initialization properties ──────────────────────────────────────

    @property
    def service(self) -> Any:
        """Lazily build and return the Drive v3 service.

        Returns Any because googleapiclient.Resource is a dynamic object
        whose methods (.files(), etc.) are not statically typed.
        Creates credentials from the SA JSON key file on first access.

        Thread-safe via double-checked locking: multiple threads from
        ThreadPoolExecutor can race on first access.
        """
        if self._service is None:
            with self._init_lock:
                if self._service is None:  # double-check after acquiring lock
                    self._service = self._build_service()
        return self._service

    @property
    def resolver(self) -> PathResolver:
        """Lazily build and return the PathResolver.

        PathResolver takes lookup_fn (a callable), NOT drive_service.
        See INTERFACES.md §9 for the canonical construction pattern.

        Thread-safe via double-checked locking: shares the same lock as
        .service to avoid two separate locks racing on first access.
        """
        if self._resolver is None:
            with self._init_lock:
                if self._resolver is None:  # double-check after acquiring lock
                    self._resolver = PathResolver(
                        root_folder_id=self._root_folder_id,
                        lookup_fn=self._make_lookup_fn(),
                        cache_maxsize=self._path_cache_maxsize,
                        cache_ttl=self._path_cache_ttl,
                    )
        return self._resolver

    @property
    def root_folder_id(self) -> str:
        """Return the root folder ID for external consumers."""
        return self._root_folder_id

    # ─── StorageBackend protocol methods ─────────────────────────────────────

    def upload_file(
        self,
        content: bytes,
        filename: str,
        mime_type: str,
        folder_id: str | None = None,
    ) -> UploadResult:
        """Upload file content to Google Drive.

        Uses resumable upload protocol for robustness (even for small files).
        Always specifies explicit MIME type to prevent Drive from converting
        images into Google Docs/Sheets.

        Drive API mapping:
            files().create(body=metadata, media_body=media, fields="id,name,size,mimeType")

        Args:
            content: File content as bytes.
            filename: Name for the uploaded file in Drive.
            mime_type: MIME type (e.g., "image/jpeg").
            folder_id: Parent folder Drive ID. None = upload to root folder.

        Returns:
            UploadResult with file_id, name, size, mime_type.

        Raises:
            UploadError: If upload fails.
            StorageQuotaError: If Drive storage quota exceeded.
            RateLimitError: If rate limited after max retries.
        """
        import io

        if not content:
            raise UploadError(filename, detail="Cannot upload empty file content")

        target_folder = folder_id or self._root_folder_id

        metadata: dict[str, Any] = {
            "name": filename,
            "parents": [target_folder],
        }

        # Use MediaIoBaseUpload with BytesIO to avoid temp files
        fh = io.BytesIO(content)
        media = MediaIoBaseUpload(
            fh,
            mimetype=mime_type,
            resumable=True,
            chunksize=self.RESUMABLE_CHUNK_SIZE,
        )

        try:
            request = self.service.files().create(
                body=metadata,
                media_body=media,
                fields="id,name,size,mimeType",
                supportsAllDrives=True,
            )

            # Execute resumable upload with per-chunk retry on transient errors.
            # Each chunk is retried independently with exponential backoff to
            # handle 429/500/502/503/504 without restarting the whole upload.
            response = None
            while response is None:
                for attempt in range(self.MAX_RETRIES):
                    try:
                        status, response = request.next_chunk()
                        if status:
                            self._logger.debug(
                                "Upload progress for '%s': %d%%",
                                filename,
                                int(status.progress() * 100),
                            )
                        break  # Chunk succeeded — exit retry loop
                    except HttpError as e:
                        status_code = int(e.resp.status)
                        is_transient = status_code in (429, 500, 502, 503, 504)
                        if is_transient and attempt < self.MAX_RETRIES - 1:
                            wait = min(
                                (2**attempt) + random.random(),
                                self.MAX_BACKOFF_SECONDS,
                            )
                            self._logger.warning(
                                "Upload chunk error %d for '%s' (attempt %d/%d), retrying in %.1fs",
                                status_code,
                                filename,
                                attempt + 1,
                                self.MAX_RETRIES,
                                wait,
                            )
                            time.sleep(wait)
                        else:
                            raise UploadError(filename, detail=str(e)) from e

        except HttpError as e:
            # CANONICAL: UploadError(filename: str, detail: str = "")
            # See INTERFACES.md §3
            raise UploadError(filename, detail=str(e)) from e

        # CANONICAL: UploadResult(file_id, name, size, mime_type)
        # NOT: filename=, size_bytes= — see INTERFACES.md §2.1
        return UploadResult(
            file_id=response["id"],
            name=response["name"],
            size=int(response.get("size", 0)),
            mime_type=response.get("mimeType", mime_type),
        )

    def create_folder(self, name: str, parent_id: str | None = None) -> str:
        """Create a folder in Google Drive.

        Idempotent: checks for existing folder with same name first.
        If found, returns existing folder ID instead of creating a duplicate.
        Drive allows multiple folders with the same name, so we must check.

        Drive API mapping:
            1. files().list(q=...) — Check for existing folder.
            2. files().create(body=metadata) — Create if not found.

        Args:
            name: Folder name (single segment, not a path).
            parent_id: Parent folder Drive ID. None = create under root.

        Returns:
            Drive folder ID (existing or newly created).

        Raises:
            PathAmbiguousError: If multiple folders with same name exist.
            StoragePermissionError: If insufficient permissions.
        """
        target_parent = parent_id or self._root_folder_id

        # Check for existing folder with same name (prevent duplicates)
        escaped_name = self._escape_query_value(name)
        query = (
            f"'{target_parent}' in parents "
            f"and name='{escaped_name}' "
            f"and mimeType='{self.DRIVE_FOLDER_MIME}' "
            f"and trashed=false"
        )

        response = self._execute_with_backoff(
            self.service.files().list(
                q=query,
                fields="files(id, name)",
                pageSize=2,  # Only need to detect ambiguity
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ),
            path=name,
        )

        existing = response.get("files", [])

        if len(existing) == 1:
            self._logger.debug(
                "Folder '%s' already exists (id=%s), reusing",
                name,
                existing[0]["id"],
            )
            return str(existing[0]["id"])

        if len(existing) > 1:
            raise PathAmbiguousError(name, len(existing))

        # Create new folder
        folder_metadata: dict[str, Any] = {
            "name": name,
            "mimeType": self.DRIVE_FOLDER_MIME,
            "parents": [target_parent],
        }

        result = self._execute_with_backoff(
            self.service.files().create(
                body=folder_metadata,
                fields="id",
                supportsAllDrives=True,
            ),
            path=name,
        )

        new_folder_id = str(result["id"])
        self._logger.info(
            "Created folder '%s' (id=%s) under parent %s",
            name,
            new_folder_id,
            target_parent,
        )

        # Invalidate parent cache entry since children changed
        self.resolver.invalidate_parent(target_parent)

        return new_folder_id

    def file_exists(self, file_id: str) -> bool:
        """Check if a file/folder exists by its Drive ID.

        Returns False (not raises) for missing or trashed files.

        Drive API mapping:
            files().get(fileId=file_id, fields="id,trashed")

        Args:
            file_id: Google Drive file ID.

        Returns:
            True if file exists and is not trashed.
        """
        try:
            result = self._execute_with_backoff(
                self.service.files().get(
                    fileId=file_id,
                    fields="id,trashed",
                    supportsAllDrives=True,
                ),
                path=f"id:{file_id}",
            )
            return not bool(result.get("trashed", False))
        except NotFoundError:
            return False

    def list_folder(self, folder_id: str) -> list[StorageEntry]:
        """List all files and subfolders within a folder.

        Handles pagination transparently (pageSize=100 per request).

        Drive API mapping:
            files().list(q="'folder_id' in parents and trashed=false")

        Args:
            folder_id: Google Drive folder ID.

        Returns:
            List of StorageEntry objects (files and subfolders).

        Raises:
            NotFoundError: If folder_id doesn't exist.
            StoragePermissionError: If permission is denied.
        """
        from datetime import datetime

        entries: list[StorageEntry] = []
        page_token: str | None = None

        while True:
            kwargs: dict[str, Any] = dict(
                q=f"'{folder_id}' in parents and trashed=false",
                fields="nextPageToken, files(id, name, mimeType, size, modifiedTime)",
                pageSize=100,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            if page_token is not None:
                kwargs["pageToken"] = page_token

            response = self._execute_with_backoff(
                self.service.files().list(**kwargs),
                path=f"id:{folder_id}",
            )

            for item in response.get("files", []):
                is_folder = item.get("mimeType") == self.DRIVE_FOLDER_MIME
                modified_at: datetime | None = None
                if item.get("modifiedTime"):
                    modified_at = datetime.fromisoformat(
                        item["modifiedTime"].replace("Z", "+00:00")
                    )

                entries.append(
                    StorageEntry(
                        name=str(item["name"]),
                        entry_type=EntryType.FOLDER if is_folder else EntryType.FILE,
                        id=str(item["id"]),
                        size=int(item["size"]) if item.get("size") else None,
                        modified_at=modified_at,
                        mime_type=item.get("mimeType"),
                    )
                )

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        return entries

    def delete_file(self, file_id: str, *, trash: bool = True) -> None:
        """Delete or trash a file/folder.

        CANONICAL signature matches StorageBackend protocol: trash is keyword-only.
        See INTERFACES.md §1 and §10.

        Drive API mapping:
            trash=True:  files().update(body={"trashed": True}) — recoverable
            trash=False: files().delete()                        — permanent

        Args:
            file_id: Google Drive file ID.
            trash: If True (default), move to trash. If False, permanently delete.

        Raises:
            NotFoundError: If file doesn't exist.
            StoragePermissionError: If insufficient permissions.
        """
        # Fetch parent folder IDs before deletion so we can invalidate the
        # PathResolver cache after. Without this, stale cache entries cause
        # 404 errors for subsequent operations on the parent folder.
        parents: list[str] = []
        try:
            file_info = self._execute_with_backoff(
                self.service.files().get(
                    fileId=file_id,
                    fields="parents",
                    supportsAllDrives=True,
                ),
                path=f"id:{file_id}",
            )
            parents = file_info.get("parents", [])
        except NotFoundError:
            pass  # File already gone — no cache to invalidate

        if trash:
            self._execute_with_backoff(
                self.service.files().update(
                    fileId=file_id,
                    body={"trashed": True},
                    supportsAllDrives=True,
                ),
                path=f"id:{file_id}",
            )
            self._logger.info("Trashed file id=%s", file_id)
        else:
            self._execute_with_backoff(
                self.service.files().delete(
                    fileId=file_id,
                    supportsAllDrives=True,
                ),
                path=f"id:{file_id}",
            )
            self._logger.info("Permanently deleted file id=%s", file_id)

        # Invalidate PathResolver cache for each parent folder so subsequent
        # lookups re-query the Drive API instead of using stale entries.
        for parent_id in parents:
            self.resolver.invalidate_parent(parent_id)

    # ─── Convenience methods (NOT part of StorageBackend protocol) ────────────

    def mkdirp(self, path: str) -> str:
        """Create directory and all parent directories.

        NOT part of StorageBackend protocol — convenience method for
        batch job setup where path-based access is simpler.

        Args:
            path: Virtual path like "/people/John Doe".

        Returns:
            Drive folder ID of the deepest folder.

        Raises:
            RootBoundaryError: If path contains "..".
            PathAmbiguousError: If duplicate folder names found.
        """
        normalized = normalize_path(path)
        if normalized == "/":
            return self._root_folder_id

        segments = normalized.strip("/").split("/")
        current_id = self._root_folder_id

        for segment in segments:
            current_id = self.create_folder(segment, parent_id=current_id)

        return current_id

    def _validate_root_access(self) -> bool:
        """Verify that the service account can access the root folder.

        Used by health check endpoints to confirm Drive connectivity.

        Returns:
            True if root folder is accessible and is a non-trashed folder.
        """
        try:
            result = self._execute_with_backoff(
                self.service.files().get(
                    fileId=self._root_folder_id,
                    fields="id,name,mimeType,trashed",
                    supportsAllDrives=True,
                ),
                path="/",
            )
            if result.get("trashed"):
                self._logger.error("Root folder is trashed!")
                return False
            if result.get("mimeType") != self.DRIVE_FOLDER_MIME:
                self._logger.error(
                    "Root folder ID does not point to a folder (mimeType=%s)",
                    result.get("mimeType"),
                )
                return False
            return True
        except StorageError as e:
            self._logger.error("Cannot access root folder: %s", e)
            return False

    def get_service_account_email(self) -> str | None:
        """Extract the service account email from the SA JSON key file.

        Returns:
            Service account email address, or None if not readable.
        """
        try:
            with open(self._sa_json_path) as f:
                data = json.load(f)
            return str(data.get("client_email", "")) or None
        except Exception:
            return None

    # ─── Private helpers ──────────────────────────────────────────────────────

    def _build_service(self) -> Any:
        """Build the Google Drive v3 API service from SA credentials.

        Raises:
            StorageError: If SA key file is missing or unreadable.
        """
        path = Path(self._sa_json_path)
        if not path.exists():
            raise StorageError(f"Service account key file not found: {self._sa_json_path}")

        # Warn about overly permissive file permissions
        if path.stat().st_mode & 0o077:
            self._logger.warning(
                "Service account key file has group/other permissions. Consider: chmod 600 %s",
                self._sa_json_path,
            )

        credentials = service_account.Credentials.from_service_account_file(  # type: ignore[no-untyped-call]
            self._sa_json_path,
            scopes=self.SCOPES,
        )
        return build("drive", "v3", credentials=credentials)

    def _execute_with_backoff(self, request: Any, *, path: str = "") -> Any:
        """Execute a Drive API request with exponential backoff.

        Translates HttpError to the custom StorageError hierarchy.
        See INTERFACES.md §3 for the HTTP error translation table.

        NOTE: This method retries by calling request.execute() multiple times.
        This works for standard google-api-python-client HttpRequest objects
        (files().list(), files().get(), files().update(), files().delete(), etc.)
        which construct a new HTTP request on each execute() call. Do NOT use
        this for resumable uploads — those have their own per-chunk retry logic
        in upload_file() to avoid restarting partially completed uploads.

        Args:
            request: A Drive API request object (from service.files().xxx()).
            path: Virtual path or ID for error context messages.

        Returns:
            The API response dict.

        Raises:
            StorageError subclass: Translated from HttpError.
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                return request.execute()
            except HttpError as e:
                status_code = int(e.resp.status)

                # ── Non-retryable: translate and raise immediately ──────────

                if status_code == 404:
                    raise NotFoundError(path) from e

                if status_code == 403:
                    error_reason = self._extract_error_reason(e)
                    if error_reason == "storageQuotaExceeded":
                        # CANONICAL: StorageQuotaError() — no positional string arg
                        # See INTERFACES.md §3
                        raise StorageQuotaError() from e
                    if error_reason in ("rateLimitExceeded", "userRateLimitExceeded"):
                        # Fall through to retry logic below
                        pass
                    else:
                        # insufficientFilePermissions or other 403
                        raise StoragePermissionError(path, str(e)) from e

                if status_code == 400:
                    raise StorageError(f"Bad request for '{path}': {e}") from e

                # ── Retryable: 429, 403 rate limit, 500/502/503/504 ─────────

                if status_code in (429, 403, 500, 502, 503, 504):
                    if attempt == self.MAX_RETRIES - 1:
                        # Last attempt exhausted
                        if status_code in (429, 403):
                            # Use lowercase header name — HTTP/2 and many proxies
                            # normalize headers to lowercase.
                            retry_after_raw = e.resp.get("retry-after")
                            raise RateLimitError(
                                int(retry_after_raw) if retry_after_raw else None
                            ) from e
                        raise StorageError(
                            f"Drive API error after {self.MAX_RETRIES} retries ({status_code}): {e}"
                        ) from e

                    wait = min(
                        (2**attempt) + random.random(),
                        self.MAX_BACKOFF_SECONDS,
                    )
                    self._logger.warning(
                        "Drive API %d error (attempt %d/%d), retrying in %.1fs: %s",
                        status_code,
                        attempt + 1,
                        self.MAX_RETRIES,
                        wait,
                        e,
                    )
                    time.sleep(wait)
                else:
                    raise StorageError(f"Unexpected Drive API error ({status_code}): {e}") from e

        # Should not be reached in normal execution
        raise StorageError("Max retries exceeded")

    def _make_lookup_fn(self) -> LookupFn:
        """Create a lookup_fn closure that queries Drive for (parent_id, child_name).

        Returns a callable matching LookupFn = Callable[[str, str], tuple[str, bool]]
        that queries the Drive API to resolve a (parent_id, child_name) pair into
        a (child_id, is_dir) pair.

        See INTERFACES.md §9 for canonical construction pattern.
        """

        def lookup(parent_id: str, child_name: str) -> tuple[str, bool]:
            escaped = self._escape_query_value(child_name)
            query = f"'{parent_id}' in parents and name='{escaped}' and trashed=false"
            response = self._execute_with_backoff(
                self.service.files().list(
                    q=query,
                    fields="files(id, name, mimeType)",
                    pageSize=2,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                ),
                path=child_name,
            )
            files = response.get("files", [])
            if not files:
                raise NotFoundError(child_name)
            if len(files) > 1:
                raise PathAmbiguousError(child_name, len(files))
            f = files[0]
            is_dir = f.get("mimeType") == self.DRIVE_FOLDER_MIME
            return str(f["id"]), is_dir

        return lookup

    @staticmethod
    def _extract_error_reason(error: HttpError) -> str:
        """Extract the error reason string from an HttpError.

        Returns:
            Error reason like 'storageQuotaExceeded' or '' if not extractable.
        """
        try:
            content = json.loads(error.content.decode("utf-8"))
            errors = content.get("error", {}).get("errors", [])
            if errors:
                return str(errors[0].get("reason", ""))
        except Exception:
            pass
        return ""

    @staticmethod
    def _escape_query_value(value: str) -> str:
        """Escape single quotes for Drive API query strings.

        Drive API query syntax uses single quotes for string values.
        Single quotes within values must be escaped with backslash.

        Args:
            value: The string value to escape.

        Returns:
            Escaped string safe for use in Drive API query.
        """
        return value.replace("\\", "\\\\").replace("'", "\\'")
