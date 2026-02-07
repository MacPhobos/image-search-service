"""Tests for main.py FastAPI application."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from image_search_service.main import lifespan


class TestLifespan:
    """Tests for FastAPI lifespan handler."""

    @pytest.mark.asyncio
    @patch("image_search_service.vector.qdrant.validate_qdrant_collections")
    @patch("image_search_service.core.config.get_settings")
    @patch("image_search_service.services.embedding.preload_embedding_model")
    @patch("image_search_service.services.watcher_manager.WatcherManager")
    @patch("image_search_service.main.close_db", new_callable=AsyncMock)
    @patch("image_search_service.main.close_qdrant")
    async def test_startup_success_all_collections_exist(
        self,
        mock_close_qdrant: Mock,
        mock_close_db: AsyncMock,
        mock_watcher_manager: Mock,
        mock_preload: Mock,
        mock_get_settings: Mock,
        mock_validate: Mock,
    ) -> None:
        """Should start successfully when all collections exist."""
        # Setup: all collections exist
        mock_settings = Mock()
        mock_settings.qdrant_strict_startup = True
        mock_get_settings.return_value = mock_settings

        mock_validate.return_value = []  # No missing collections

        # Setup watcher mock
        mock_watcher = Mock()
        mock_watcher_manager.get_instance.return_value = mock_watcher

        # Execute: should not raise
        from fastapi import FastAPI

        app = FastAPI()
        async with lifespan(app):
            pass  # Success

        mock_validate.assert_called_once()
        mock_watcher.start.assert_called_once()
        mock_watcher.stop.assert_called_once()

    @pytest.mark.asyncio
    @patch("image_search_service.vector.qdrant.validate_qdrant_collections")
    @patch("image_search_service.core.config.get_settings")
    @patch("image_search_service.services.embedding.preload_embedding_model")
    @patch("image_search_service.services.watcher_manager.WatcherManager")
    async def test_startup_exits_on_missing_collections_strict_mode(
        self,
        mock_watcher_manager: Mock,
        mock_preload: Mock,
        mock_get_settings: Mock,
        mock_validate: Mock,
    ) -> None:
        """Should exit with code 1 when collections missing in strict mode."""
        # Setup: missing collections, strict mode
        mock_settings = Mock()
        mock_settings.qdrant_strict_startup = True
        mock_get_settings.return_value = mock_settings

        mock_validate.return_value = ["faces", "person_centroids"]

        # Setup watcher mock
        mock_watcher = Mock()
        mock_watcher_manager.get_instance.return_value = mock_watcher

        # Execute: should raise SystemExit
        from fastapi import FastAPI

        app = FastAPI()
        with pytest.raises(SystemExit) as exc_info:
            async with lifespan(app):
                pass

        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    @patch("image_search_service.vector.qdrant.validate_qdrant_collections")
    @patch("image_search_service.main.get_settings")
    @patch("image_search_service.services.embedding.preload_embedding_model")
    @patch("image_search_service.services.watcher_manager.WatcherManager")
    @patch("image_search_service.main.close_db", new_callable=AsyncMock)
    @patch("image_search_service.main.close_qdrant")
    async def test_startup_warns_on_missing_collections_non_strict(
        self,
        mock_close_qdrant: Mock,
        mock_close_db: AsyncMock,
        mock_watcher_manager: Mock,
        mock_preload: Mock,
        mock_get_settings: Mock,
        mock_validate: Mock,
    ) -> None:
        """Should warn but continue when collections missing in non-strict mode."""
        # Setup: missing collections, non-strict mode
        mock_settings = Mock()
        mock_settings.qdrant_strict_startup = False
        mock_get_settings.return_value = mock_settings

        mock_validate.return_value = ["faces", "person_centroids"]

        # Setup watcher mock
        mock_watcher = Mock()
        mock_watcher_manager.get_instance.return_value = mock_watcher

        # Execute: should NOT raise
        from fastapi import FastAPI

        app = FastAPI()
        async with lifespan(app):
            pass  # Success with warning

        mock_validate.assert_called_once()
        mock_watcher.start.assert_called_once()
        mock_watcher.stop.assert_called_once()

    @pytest.mark.asyncio
    @patch("image_search_service.vector.qdrant.validate_qdrant_collections")
    @patch("image_search_service.core.config.get_settings")
    @patch("image_search_service.services.embedding.preload_embedding_model")
    @patch("image_search_service.services.watcher_manager.WatcherManager")
    async def test_startup_exits_on_validation_exception_strict_mode(
        self,
        mock_watcher_manager: Mock,
        mock_preload: Mock,
        mock_get_settings: Mock,
        mock_validate: Mock,
    ) -> None:
        """Should exit when validation raises exception in strict mode."""
        # Setup: validation fails
        mock_settings = Mock()
        mock_settings.qdrant_strict_startup = True
        mock_settings.qdrant_url = "http://localhost:6333"
        mock_get_settings.return_value = mock_settings

        mock_validate.side_effect = Exception("Qdrant connection failed")

        # Setup watcher mock
        mock_watcher = Mock()
        mock_watcher_manager.get_instance.return_value = mock_watcher

        # Execute: should raise SystemExit
        from fastapi import FastAPI

        app = FastAPI()
        with pytest.raises(SystemExit) as exc_info:
            async with lifespan(app):
                pass

        assert exc_info.value.code == 1
