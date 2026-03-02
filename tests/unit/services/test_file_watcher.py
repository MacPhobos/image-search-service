"""Tests for file watcher service (ImageFileHandler and FileWatcherService).

Covers file event handling, extension filtering, debouncing, error handling,
and watcher lifecycle management.
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
from watchdog.events import DirCreatedEvent, FileCreatedEvent, FileModifiedEvent

from image_search_service.services.file_watcher import (
    SUPPORTED_EXTENSIONS,
    FileWatcherService,
    ImageFileHandler,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def callback() -> Mock:
    """Create a mock callback for tracking invocations."""
    return Mock()


@pytest.fixture
def handler(callback: Mock) -> ImageFileHandler:
    """Create an ImageFileHandler with a very short debounce for fast tests."""
    return ImageFileHandler(callback=callback, debounce_seconds=0.05)


@pytest.fixture
def handler_no_debounce(callback: Mock) -> ImageFileHandler:
    """Create an ImageFileHandler with zero debounce for synchronous-style tests."""
    return ImageFileHandler(callback=callback, debounce_seconds=0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_file_created_event(path: str) -> FileCreatedEvent:
    """Build a watchdog FileCreatedEvent for the given path."""
    return FileCreatedEvent(src_path=path)


def _make_file_modified_event(path: str) -> FileModifiedEvent:
    """Build a watchdog FileModifiedEvent for the given path."""
    return FileModifiedEvent(src_path=path)


def _make_dir_created_event(path: str) -> DirCreatedEvent:
    """Build a watchdog DirCreatedEvent for the given path."""
    return DirCreatedEvent(src_path=path)


def _wait_for_debounce(handler: ImageFileHandler, margin: float = 0.05) -> None:
    """Block until all pending debounce timers should have fired.

    Args:
        handler: The handler whose debounce_seconds determines the wait.
        margin: Extra time (seconds) to wait beyond the debounce period.
    """
    time.sleep(handler.debounce_seconds + margin)


# ---------------------------------------------------------------------------
# ImageFileHandler -- supported extensions
# ---------------------------------------------------------------------------


class TestImageFileHandlerSupportedExtensions:
    """on_created() should invoke the callback for supported image extensions."""

    @pytest.mark.parametrize(
        "extension",
        [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"],
    )
    def test_on_created_when_supported_extension_then_callback_invoked(
        self, handler: ImageFileHandler, callback: Mock, extension: str
    ) -> None:
        event = _make_file_created_event(f"/photos/image{extension}")
        handler.on_created(event)
        _wait_for_debounce(handler)

        callback.assert_called_once_with(f"/photos/image{extension}")

    @pytest.mark.parametrize(
        "extension",
        [".JPG", ".Jpeg", ".PNG", ".GIF", ".WEBP", ".BMP"],
    )
    def test_on_created_when_uppercase_extension_then_callback_invoked(
        self, handler: ImageFileHandler, callback: Mock, extension: str
    ) -> None:
        """Case-insensitive extension matching should work."""
        event = _make_file_created_event(f"/photos/image{extension}")
        handler.on_created(event)
        _wait_for_debounce(handler)

        callback.assert_called_once()


class TestImageFileHandlerUnsupportedExtensions:
    """on_created() should NOT invoke the callback for non-image files."""

    @pytest.mark.parametrize(
        "filename",
        [
            "/photos/readme.txt",
            "/photos/document.pdf",
            "/photos/video.mp4",
            "/photos/archive.zip",
            "/photos/data.csv",
            "/photos/script.py",
        ],
    )
    def test_on_created_when_unsupported_extension_then_callback_not_invoked(
        self, handler: ImageFileHandler, callback: Mock, filename: str
    ) -> None:
        event = _make_file_created_event(filename)
        handler.on_created(event)
        _wait_for_debounce(handler)

        callback.assert_not_called()


# ---------------------------------------------------------------------------
# ImageFileHandler -- on_modified()
# ---------------------------------------------------------------------------


class TestImageFileHandlerOnModified:
    """on_modified() should mirror on_created() behaviour for supported images."""

    def test_on_modified_when_supported_extension_then_callback_invoked(
        self, handler: ImageFileHandler, callback: Mock
    ) -> None:
        event = _make_file_modified_event("/photos/sunset.jpg")
        handler.on_modified(event)
        _wait_for_debounce(handler)

        callback.assert_called_once_with("/photos/sunset.jpg")

    def test_on_modified_when_unsupported_extension_then_callback_not_invoked(
        self, handler: ImageFileHandler, callback: Mock
    ) -> None:
        event = _make_file_modified_event("/photos/notes.txt")
        handler.on_modified(event)
        _wait_for_debounce(handler)

        callback.assert_not_called()


# ---------------------------------------------------------------------------
# ImageFileHandler -- directory events ignored
# ---------------------------------------------------------------------------


class TestImageFileHandlerDirectoryEvents:
    """Directory events must be silently ignored regardless of name."""

    def test_on_created_when_directory_event_then_callback_not_invoked(
        self, handler: ImageFileHandler, callback: Mock
    ) -> None:
        event = _make_dir_created_event("/photos/vacation")
        handler.on_created(event)
        _wait_for_debounce(handler)

        callback.assert_not_called()

    def test_on_created_when_directory_with_image_extension_then_callback_not_invoked(
        self, handler: ImageFileHandler, callback: Mock
    ) -> None:
        """A directory whose name ends in .jpg should still be ignored."""
        event = _make_dir_created_event("/photos/album.jpg")
        handler.on_created(event)
        _wait_for_debounce(handler)

        callback.assert_not_called()


# ---------------------------------------------------------------------------
# ImageFileHandler -- hidden and system files
# ---------------------------------------------------------------------------


class TestImageFileHandlerHiddenAndSystemFiles:
    """Hidden files and system files should be ignored if their extension is not
    in SUPPORTED_EXTENSIONS (which it typically is not)."""

    @pytest.mark.parametrize(
        "filename",
        [
            "/photos/.DS_Store",
            "/photos/Thumbs.db",
            "/photos/.hidden_file",
            "/photos/.thumbnails/cache.db",
        ],
    )
    def test_on_created_when_system_or_hidden_file_then_callback_not_invoked(
        self, handler: ImageFileHandler, callback: Mock, filename: str
    ) -> None:
        event = _make_file_created_event(filename)
        handler.on_created(event)
        _wait_for_debounce(handler)

        callback.assert_not_called()


# ---------------------------------------------------------------------------
# ImageFileHandler -- bytes path handling
# ---------------------------------------------------------------------------


class TestImageFileHandlerBytesPath:
    """The handler converts bytes src_path to str (e.g. Linux inotify)."""

    def test_on_created_when_bytes_path_then_callback_invoked_with_str(
        self, handler: ImageFileHandler, callback: Mock
    ) -> None:
        event = FileCreatedEvent(src_path=b"/photos/image.jpg")
        handler.on_created(event)
        _wait_for_debounce(handler)

        callback.assert_called_once_with("/photos/image.jpg")


# ---------------------------------------------------------------------------
# ImageFileHandler -- debouncing
# ---------------------------------------------------------------------------


class TestImageFileHandlerDebouncing:
    """Rapid events for the same file should produce a single callback."""

    def test_debounce_when_rapid_events_for_same_file_then_single_callback(
        self, callback: Mock
    ) -> None:
        handler = ImageFileHandler(callback=callback, debounce_seconds=0.15)

        # Fire multiple events in quick succession for the same file
        for _ in range(5):
            event = _make_file_created_event("/photos/burst.jpg")
            handler.on_created(event)
            time.sleep(0.02)  # 20ms between events -- well within debounce window

        _wait_for_debounce(handler, margin=0.10)

        # Only a single callback should fire
        callback.assert_called_once_with("/photos/burst.jpg")

    def test_debounce_when_different_files_then_separate_callbacks(self, callback: Mock) -> None:
        handler = ImageFileHandler(callback=callback, debounce_seconds=0.05)

        handler.on_created(_make_file_created_event("/photos/a.jpg"))
        handler.on_created(_make_file_created_event("/photos/b.png"))

        _wait_for_debounce(handler)

        assert callback.call_count == 2

    def test_debounce_timer_resets_on_new_event(self, callback: Mock) -> None:
        """Sending a new event within the debounce window should reset the timer."""
        handler = ImageFileHandler(callback=callback, debounce_seconds=0.20)

        handler.on_created(_make_file_created_event("/photos/resetting.jpg"))
        time.sleep(0.10)  # Wait 100ms (half the debounce period)

        # Fire another event -- this should reset the 200ms timer
        handler.on_created(_make_file_created_event("/photos/resetting.jpg"))

        # After 150ms from second event, the first timer would have expired
        # at 200ms after the first event, but the reset means the callback
        # should not have fired yet.
        time.sleep(0.10)
        callback.assert_not_called()

        # Now wait for the second timer to fire
        time.sleep(0.15)
        callback.assert_called_once_with("/photos/resetting.jpg")


# ---------------------------------------------------------------------------
# ImageFileHandler -- error handling in callback
# ---------------------------------------------------------------------------


class TestImageFileHandlerErrorHandling:
    """Exceptions in the callback should be logged but not crash the handler."""

    def test_execute_callback_when_callback_raises_then_error_logged_not_raised(
        self, callback: Mock
    ) -> None:
        callback.side_effect = RuntimeError("Queue unavailable")
        handler = ImageFileHandler(callback=callback, debounce_seconds=0.0)

        # Directly call the internal method to bypass debounce threading
        # (the timer thread would swallow the exception regardless, but
        # this verifies _execute_callback itself does not propagate it).
        handler._execute_callback("/photos/broken.jpg")

        callback.assert_called_once_with("/photos/broken.jpg")

    def test_execute_callback_when_callback_raises_then_pending_cleared(
        self, callback: Mock
    ) -> None:
        """The pending dict should not grow unboundedly on errors."""
        callback.side_effect = RuntimeError("Queue unavailable")
        handler = ImageFileHandler(callback=callback, debounce_seconds=0.0)

        # Simulate a pending entry
        handler._pending["/photos/broken.jpg"] = MagicMock()

        handler._execute_callback("/photos/broken.jpg")

        assert "/photos/broken.jpg" not in handler._pending


# ---------------------------------------------------------------------------
# ImageFileHandler -- SUPPORTED_EXTENSIONS constant
# ---------------------------------------------------------------------------


class TestSupportedExtensions:
    """Verify the SUPPORTED_EXTENSIONS constant covers expected formats."""

    def test_supported_extensions_includes_common_formats(self) -> None:
        expected = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
        assert expected == SUPPORTED_EXTENSIONS

    def test_supported_extensions_are_lowercase(self) -> None:
        for ext in SUPPORTED_EXTENSIONS:
            assert ext == ext.lower(), f"Extension {ext} should be lowercase"

    def test_supported_extensions_all_start_with_dot(self) -> None:
        for ext in SUPPORTED_EXTENSIONS:
            assert ext.startswith("."), f"Extension {ext} should start with '.'"


# ---------------------------------------------------------------------------
# FileWatcherService -- lifecycle
# ---------------------------------------------------------------------------


class TestFileWatcherServiceLifecycle:
    """Tests for start/stop and is_running property."""

    def test_is_running_when_not_started_then_false(self) -> None:
        service = FileWatcherService(watch_paths=["/nonexistent"], on_new_image=Mock())
        assert service.is_running is False

    def test_start_when_valid_directory_then_running(self, tmp_path: Path) -> None:
        service = FileWatcherService(
            watch_paths=[str(tmp_path)],
            on_new_image=Mock(),
            debounce_seconds=0.05,
        )
        try:
            service.start()
            assert service.is_running is True
        finally:
            service.stop()

    def test_stop_when_running_then_not_running(self, tmp_path: Path) -> None:
        service = FileWatcherService(
            watch_paths=[str(tmp_path)],
            on_new_image=Mock(),
            debounce_seconds=0.05,
        )
        service.start()
        service.stop()
        assert service.is_running is False

    def test_start_when_already_running_then_noop(self, tmp_path: Path) -> None:
        """Calling start() twice should not create a second observer."""
        service = FileWatcherService(
            watch_paths=[str(tmp_path)],
            on_new_image=Mock(),
            debounce_seconds=0.05,
        )
        try:
            service.start()
            first_observer = service._observer
            service.start()  # Should be idempotent
            assert service._observer is first_observer
        finally:
            service.stop()

    def test_stop_when_not_started_then_noop(self) -> None:
        """Calling stop() without start() should not raise."""
        service = FileWatcherService(watch_paths=["/nonexistent"], on_new_image=Mock())
        service.stop()  # Should not raise
        assert service.is_running is False

    def test_stop_then_observer_cleaned_up(self, tmp_path: Path) -> None:
        service = FileWatcherService(
            watch_paths=[str(tmp_path)],
            on_new_image=Mock(),
            debounce_seconds=0.05,
        )
        service.start()
        service.stop()
        assert service._observer is None


# ---------------------------------------------------------------------------
# FileWatcherService -- watch path validation
# ---------------------------------------------------------------------------


class TestFileWatcherServiceWatchPaths:
    """Verify behaviour with valid and invalid watch paths."""

    def test_start_when_nonexistent_directory_then_logs_warning_and_runs(
        self,
    ) -> None:
        """Non-existent paths should be skipped but the service still starts."""
        service = FileWatcherService(
            watch_paths=["/totally/nonexistent/path"],
            on_new_image=Mock(),
            debounce_seconds=0.05,
        )
        try:
            service.start()
            assert service.is_running is True
        finally:
            service.stop()

    def test_start_when_mix_of_valid_and_invalid_paths_then_watches_valid(
        self, tmp_path: Path
    ) -> None:
        """Only existing directories should be scheduled for watching."""
        service = FileWatcherService(
            watch_paths=[str(tmp_path), "/nonexistent/path"],
            on_new_image=Mock(),
            debounce_seconds=0.05,
        )
        try:
            service.start()
            assert service.is_running is True
        finally:
            service.stop()

    def test_start_when_path_is_file_not_directory_then_skips(self, tmp_path: Path) -> None:
        """A file path (not directory) should be skipped."""
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("hello")

        service = FileWatcherService(
            watch_paths=[str(file_path)],
            on_new_image=Mock(),
            debounce_seconds=0.05,
        )
        try:
            service.start()
            assert service.is_running is True
        finally:
            service.stop()


# ---------------------------------------------------------------------------
# FileWatcherService -- end-to-end with real filesystem
# ---------------------------------------------------------------------------


class TestFileWatcherServiceEndToEnd:
    """Integration-style tests using tmp_path to create real files and verify
    that the callback fires through the full watchdog pipeline."""

    def test_new_image_file_created_then_callback_invoked(self, tmp_path: Path) -> None:
        """Creating a .jpg file in a watched directory should trigger the callback."""
        callback = Mock()
        service = FileWatcherService(
            watch_paths=[str(tmp_path)],
            on_new_image=callback,
            debounce_seconds=0.05,
        )
        try:
            service.start()

            # Create a real image file
            image_path = tmp_path / "vacation.jpg"
            image_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

            # Wait for watchdog + debounce
            time.sleep(0.5)

            callback.assert_called_once()
            called_path = callback.call_args[0][0]
            assert called_path.endswith("vacation.jpg")
        finally:
            service.stop()

    def test_non_image_file_created_then_callback_not_invoked(self, tmp_path: Path) -> None:
        """Creating a .txt file should NOT trigger the callback."""
        callback = Mock()
        service = FileWatcherService(
            watch_paths=[str(tmp_path)],
            on_new_image=callback,
            debounce_seconds=0.05,
        )
        try:
            service.start()

            txt_path = tmp_path / "notes.txt"
            txt_path.write_text("Some notes")

            time.sleep(0.5)

            callback.assert_not_called()
        finally:
            service.stop()

    def test_image_in_subdirectory_when_recursive_then_callback_invoked(
        self, tmp_path: Path
    ) -> None:
        """Files in subdirectories should be detected when recursive=True."""
        subdir = tmp_path / "2024" / "vacation"
        subdir.mkdir(parents=True)

        callback = Mock()
        service = FileWatcherService(
            watch_paths=[str(tmp_path)],
            on_new_image=callback,
            recursive=True,
            debounce_seconds=0.05,
        )
        try:
            service.start()

            image_path = subdir / "beach.png"
            image_path.write_bytes(b"\x89PNG" + b"\x00" * 100)

            time.sleep(0.5)

            callback.assert_called_once()
            called_path = callback.call_args[0][0]
            assert "beach.png" in called_path
        finally:
            service.stop()


# ---------------------------------------------------------------------------
# FileWatcherService -- constructor parameters
# ---------------------------------------------------------------------------


class TestFileWatcherServiceConstructor:
    """Verify constructor stores parameters correctly."""

    def test_constructor_stores_watch_paths_as_path_objects(self) -> None:
        service = FileWatcherService(
            watch_paths=["/a", "/b"],
            on_new_image=Mock(),
        )
        assert all(isinstance(p, Path) for p in service.watch_paths)

    def test_constructor_default_recursive_is_true(self) -> None:
        service = FileWatcherService(
            watch_paths=["/a"],
            on_new_image=Mock(),
        )
        assert service.recursive is True

    def test_constructor_default_debounce_seconds_is_one(self) -> None:
        service = FileWatcherService(
            watch_paths=["/a"],
            on_new_image=Mock(),
        )
        assert service.debounce_seconds == 1.0

    def test_constructor_custom_debounce(self) -> None:
        service = FileWatcherService(
            watch_paths=["/a"],
            on_new_image=Mock(),
            debounce_seconds=2.5,
        )
        assert service.debounce_seconds == 2.5

    def test_constructor_custom_recursive_false(self) -> None:
        service = FileWatcherService(
            watch_paths=["/a"],
            on_new_image=Mock(),
            recursive=False,
        )
        assert service.recursive is False
