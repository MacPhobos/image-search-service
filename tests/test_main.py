"""Tests for main.py FastAPI application."""

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI

from image_search_service.main import lifespan


@pytest.fixture
def lifespan_mocks(monkeypatch: pytest.MonkeyPatch) -> dict[str, Mock | AsyncMock]:
    """Set up all mocks required by the lifespan function.

    Uses monkeypatch.setattr instead of stacked @patch decorators so that
    each test can refer to mocks by name rather than positional parameter.

    Returns a dict keyed by descriptive names for easy per-test configuration.
    """
    mocks: dict[str, Mock | AsyncMock] = {}

    # --- validate_qdrant_collections ---
    mocks["validate"] = Mock(return_value=[])
    monkeypatch.setattr(
        "image_search_service.vector.qdrant.validate_qdrant_collections",
        mocks["validate"],
    )

    # --- get_settings (module-level import in main.py) ---
    mock_settings = Mock()
    mock_settings.qdrant_strict_startup = True
    mock_settings.qdrant_url = "http://localhost:6333"
    mocks["settings"] = mock_settings
    mocks["get_settings"] = Mock(return_value=mock_settings)
    monkeypatch.setattr(
        "image_search_service.main.get_settings",
        mocks["get_settings"],
    )

    # --- preload_embedding_model ---
    mocks["preload"] = Mock()
    monkeypatch.setattr(
        "image_search_service.services.embedding.preload_embedding_model",
        mocks["preload"],
    )

    # --- WatcherManager ---
    mock_watcher = Mock()
    mocks["watcher"] = mock_watcher
    mocks["watcher_manager"] = Mock()
    mocks["watcher_manager"].get_instance.return_value = mock_watcher
    monkeypatch.setattr(
        "image_search_service.services.watcher_manager.WatcherManager",
        mocks["watcher_manager"],
    )

    # --- close_db (async) ---
    mocks["close_db"] = AsyncMock()
    monkeypatch.setattr(
        "image_search_service.main.close_db",
        mocks["close_db"],
    )

    # --- close_qdrant ---
    mocks["close_qdrant"] = Mock()
    monkeypatch.setattr(
        "image_search_service.main.close_qdrant",
        mocks["close_qdrant"],
    )

    return mocks


class TestLifespan:
    """Tests for FastAPI lifespan handler."""

    @pytest.mark.asyncio
    async def test_startup_success_all_collections_exist(
        self,
        lifespan_mocks: dict[str, Mock],
    ) -> None:
        """Should start successfully when all collections exist."""
        # Setup: all collections exist (default from fixture is already [])
        lifespan_mocks["settings"].qdrant_strict_startup = True
        lifespan_mocks["validate"].return_value = []

        # Execute: should not raise
        app = FastAPI()
        async with lifespan(app):
            pass  # Success

        lifespan_mocks["validate"].assert_called_once()
        lifespan_mocks["watcher"].start.assert_called_once()
        lifespan_mocks["watcher"].stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_startup_exits_on_missing_collections_strict_mode(
        self,
        lifespan_mocks: dict[str, Mock],
    ) -> None:
        """Should exit with code 1 when collections missing in strict mode."""
        # Setup: missing collections, strict mode
        lifespan_mocks["settings"].qdrant_strict_startup = True
        lifespan_mocks["validate"].return_value = ["faces", "person_centroids"]

        # Execute: should raise SystemExit
        app = FastAPI()
        with pytest.raises(SystemExit) as exc_info:
            async with lifespan(app):
                pass

        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_startup_warns_on_missing_collections_non_strict(
        self,
        lifespan_mocks: dict[str, Mock],
    ) -> None:
        """Should warn but continue when collections missing in non-strict mode."""
        # Setup: missing collections, non-strict mode
        lifespan_mocks["settings"].qdrant_strict_startup = False
        lifespan_mocks["validate"].return_value = ["faces", "person_centroids"]

        # Execute: should NOT raise
        app = FastAPI()
        async with lifespan(app):
            pass  # Success with warning

        lifespan_mocks["validate"].assert_called_once()
        lifespan_mocks["watcher"].start.assert_called_once()
        lifespan_mocks["watcher"].stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_startup_exits_on_validation_exception_strict_mode(
        self,
        lifespan_mocks: dict[str, Mock],
    ) -> None:
        """Should exit when validation raises exception in strict mode."""
        # Setup: validation fails with exception
        lifespan_mocks["settings"].qdrant_strict_startup = True
        lifespan_mocks["settings"].qdrant_url = "http://localhost:6333"
        lifespan_mocks["validate"].side_effect = Exception("Qdrant connection failed")

        # Execute: should raise SystemExit
        app = FastAPI()
        with pytest.raises(SystemExit) as exc_info:
            async with lifespan(app):
                pass

        assert exc_info.value.code == 1
