"""Unit tests for enqueue_person_ids_update helper."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from image_search_service.queue.worker import (
    QUEUE_DEFAULT,
    enqueue_person_ids_update,
)


class TestEnqueuePersonIdsUpdate:
    """Tests for the enqueue_person_ids_update helper function."""

    def test_enqueues_one_job_per_asset_id(self) -> None:
        """Each asset ID should produce exactly one enqueued job."""
        mock_queue = MagicMock()

        with patch(
            "image_search_service.queue.worker.get_queue",
            return_value=mock_queue,
        ):
            result = enqueue_person_ids_update({10, 20, 30})

        assert result == 3
        assert mock_queue.enqueue.call_count == 3

        # Verify each call used the right kwargs
        enqueued_asset_ids = set()
        for call in mock_queue.enqueue.call_args_list:
            kwargs = call[1]
            assert kwargs["job_timeout"] == "5m"
            enqueued_asset_ids.add(kwargs["asset_id"])
            # DA improvement: verify the correct job function is enqueued
            assert call[0][0].__name__ == "update_asset_person_ids_job"

        assert enqueued_asset_ids == {10, 20, 30}

    def test_returns_zero_for_empty_input(self) -> None:
        """Empty input should short-circuit without touching Redis."""
        with patch(
            "image_search_service.queue.worker.get_queue",
        ) as mock_get_queue:
            result = enqueue_person_ids_update(set())

        assert result == 0
        mock_get_queue.assert_not_called()

    def test_returns_zero_on_redis_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        """Queue failure should be swallowed and return 0, with warning logged."""
        with patch(
            "image_search_service.queue.worker.get_queue",
            side_effect=ConnectionError("Redis unavailable"),
        ):
            with caplog.at_level(logging.WARNING, logger="image_search_service.queue.worker"):
                result = enqueue_person_ids_update({1})

        assert result == 0
        assert any("Failed to enqueue person_ids update jobs" in r.message for r in caplog.records)

    def test_returns_zero_on_enqueue_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        """Individual enqueue failure should be swallowed and return 0, with warning logged."""
        mock_queue = MagicMock()
        mock_queue.enqueue.side_effect = RuntimeError("queue full")

        with patch(
            "image_search_service.queue.worker.get_queue",
            return_value=mock_queue,
        ):
            with caplog.at_level(logging.WARNING, logger="image_search_service.queue.worker"):
                result = enqueue_person_ids_update({1, 2})

        assert result == 0
        assert any("Failed to enqueue person_ids update jobs" in r.message for r in caplog.records)

    def test_single_asset_id(self) -> None:
        """Single-element set should enqueue exactly one job."""
        mock_queue = MagicMock()

        with patch(
            "image_search_service.queue.worker.get_queue",
            return_value=mock_queue,
        ):
            result = enqueue_person_ids_update({42})

        assert result == 1
        mock_queue.enqueue.assert_called_once()
        kwargs = mock_queue.enqueue.call_args[1]
        assert kwargs["asset_id"] == 42

    def test_uses_default_queue(self) -> None:
        """Helper must use the QUEUE_DEFAULT priority."""
        mock_queue = MagicMock()

        with patch(
            "image_search_service.queue.worker.get_queue",
            return_value=mock_queue,
        ) as mock_get_queue:
            enqueue_person_ids_update({1})

        mock_get_queue.assert_called_once_with(QUEUE_DEFAULT)

    def test_accepts_list_input(self) -> None:
        """Helper should accept any Collection[int], not just sets."""
        mock_queue = MagicMock()

        with patch(
            "image_search_service.queue.worker.get_queue",
            return_value=mock_queue,
        ):
            result = enqueue_person_ids_update([5, 10])

        assert result == 2
