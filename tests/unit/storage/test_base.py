"""Tests for storage protocol and data types."""

from __future__ import annotations

from datetime import datetime

import pytest

from image_search_service.storage.base import (
    EntryType,
    StorageBackend,
    StorageEntry,
    UploadResult,
)


class TestEntryType:
    def test_file_value(self) -> None:
        assert EntryType.FILE.value == "file"

    def test_folder_value(self) -> None:
        assert EntryType.FOLDER.value == "folder"

    def test_is_string_enum(self) -> None:
        assert isinstance(EntryType.FILE, str)

    def test_has_exactly_two_values(self) -> None:
        assert set(EntryType) == {EntryType.FILE, EntryType.FOLDER}


class TestStorageEntry:
    def test_create_file_entry(self) -> None:
        entry = StorageEntry(
            name="photo.jpg",
            entry_type=EntryType.FILE,
            id="file123",
            size=1024,
            modified_at=datetime(2026, 1, 1),
            mime_type="image/jpeg",
        )
        assert entry.name == "photo.jpg"
        assert entry.entry_type == EntryType.FILE
        assert entry.id == "file123"
        assert entry.size == 1024
        assert entry.mime_type == "image/jpeg"

    def test_create_folder_entry(self) -> None:
        entry = StorageEntry(
            name="photos",
            entry_type=EntryType.FOLDER,
            id="folder456",
            size=None,
            modified_at=None,
        )
        assert entry.size is None
        assert entry.mime_type is None

    def test_mime_type_defaults_to_none(self) -> None:
        entry = StorageEntry(
            name="photo.jpg",
            entry_type=EntryType.FILE,
            id="file123",
            size=1024,
            modified_at=None,
        )
        assert entry.mime_type is None

    def test_frozen_immutable(self) -> None:
        entry = StorageEntry(
            name="photo.jpg",
            entry_type=EntryType.FILE,
            id="file123",
            size=1024,
            modified_at=None,
        )
        with pytest.raises(AttributeError):
            entry.name = "other.jpg"  # type: ignore[misc]

    def test_hashable(self) -> None:
        entry = StorageEntry(
            name="photo.jpg",
            entry_type=EntryType.FILE,
            id="file123",
            size=1024,
            modified_at=None,
        )
        assert hash(entry) is not None
        assert entry in {entry}

    def test_equality(self) -> None:
        entry1 = StorageEntry(
            name="photo.jpg",
            entry_type=EntryType.FILE,
            id="file123",
            size=1024,
            modified_at=None,
        )
        entry2 = StorageEntry(
            name="photo.jpg",
            entry_type=EntryType.FILE,
            id="file123",
            size=1024,
            modified_at=None,
        )
        assert entry1 == entry2


class TestUploadResult:
    def test_create_result(self) -> None:
        result = UploadResult(
            file_id="abc123",
            name="photo.jpg",
            size=2048,
            mime_type="image/jpeg",
        )
        assert result.file_id == "abc123"
        assert result.name == "photo.jpg"
        assert result.size == 2048
        assert result.mime_type == "image/jpeg"

    def test_frozen_immutable(self) -> None:
        result = UploadResult(
            file_id="abc123",
            name="photo.jpg",
            size=2048,
            mime_type="image/jpeg",
        )
        with pytest.raises(AttributeError):
            result.file_id = "other"  # type: ignore[misc]

    def test_hashable(self) -> None:
        result = UploadResult(
            file_id="abc123",
            name="photo.jpg",
            size=2048,
            mime_type="image/jpeg",
        )
        assert hash(result) is not None
        assert result in {result}


class TestStorageBackendProtocol:
    def test_runtime_checkable(self) -> None:
        """Verify @runtime_checkable allows isinstance checks."""

        class FakeStorage:
            def upload_file(
                self,
                content: bytes,
                filename: str,
                mime_type: str,
                folder_id: str | None = None,
            ) -> UploadResult:
                return UploadResult("id", "name", 0, "mime")

            def create_folder(self, name: str, parent_id: str | None = None) -> str:
                return "folder_id"

            def file_exists(self, file_id: str) -> bool:
                return True

            def list_folder(self, folder_id: str) -> list[StorageEntry]:
                return []

            def delete_file(self, file_id: str, *, trash: bool = True) -> None:
                pass

        fake = FakeStorage()
        assert isinstance(fake, StorageBackend)

    def test_non_conforming_class_fails_check(self) -> None:
        """Class missing methods should not pass isinstance check."""

        class Incomplete:
            def upload_file(
                self,
                content: bytes,
                filename: str,
                mime_type: str,
                folder_id: str | None = None,
            ) -> UploadResult:
                return UploadResult("id", "name", 0, "mime")

        incomplete = Incomplete()
        assert not isinstance(incomplete, StorageBackend)

    def test_protocol_has_five_methods(self) -> None:
        """StorageBackend defines exactly 5 methods."""
        protocol_methods = {
            name
            for name in dir(StorageBackend)
            if not name.startswith("_")
        }
        expected = {"upload_file", "create_folder", "file_exists", "list_folder", "delete_file"}
        assert protocol_methods == expected
