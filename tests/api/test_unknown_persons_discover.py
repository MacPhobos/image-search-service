"""Tests for POST /api/v1/faces/unknown-persons/discover endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_discover_minimal_params(test_client: AsyncClient) -> None:
    """Test discovery with default parameters."""
    # The route pre-generates a UUID via uuid.uuid4() and returns it directly —
    # it never reads job.id from the enqueue return value. Patch uuid.uuid4 to
    # get a predictable job ID. Also patch get_redis because the route calls it
    # before get_queue to pre-seed the progress key in Redis.
    fixed_uuid = "test-job-123"
    with (
        patch("image_search_service.api.routes.unknown_persons.get_queue") as mock_get_queue,
        patch("image_search_service.api.routes.unknown_persons.get_redis") as mock_get_redis,
        patch(
            "image_search_service.api.routes.unknown_persons.uuid.uuid4",
            return_value=fixed_uuid,
        ),
    ):
        mock_queue = MagicMock()
        mock_queue.enqueue.return_value = MagicMock()
        mock_get_queue.return_value = mock_queue

        mock_redis_conn = MagicMock()
        mock_get_redis.return_value = mock_redis_conn

        response = await test_client.post("/api/v1/faces/unknown-persons/discover", json={})

        assert response.status_code == 200
        data = response.json()

        assert data["jobId"] == fixed_uuid
        assert data["status"] == "queued"
        assert data["progressKey"] == f"job:{fixed_uuid}:progress"
        assert "params" in data

        # Verify job was enqueued with defaults
        mock_queue.enqueue.assert_called_once()
        call_kwargs = mock_queue.enqueue.call_args[1]
        assert call_kwargs["clustering_method"] == "hdbscan"
        assert call_kwargs["min_cluster_size"] == 5
        assert call_kwargs["min_quality"] == 0.3
        assert call_kwargs["max_faces"] == 50000
        assert call_kwargs["min_cluster_confidence"] == 0.70
        assert call_kwargs["eps"] == 0.5


@pytest.mark.asyncio
async def test_discover_custom_params(test_client: AsyncClient) -> None:
    """Test discovery with custom parameters."""
    # Same wiring issue as test_discover_minimal_params: the route uses a
    # pre-generated uuid.uuid4() (not job.id) and calls get_redis() before
    # get_queue(). Patch all three to get deterministic, isolated behavior.
    fixed_uuid = "test-job-456"
    with (
        patch("image_search_service.api.routes.unknown_persons.get_queue") as mock_get_queue,
        patch("image_search_service.api.routes.unknown_persons.get_redis") as mock_get_redis,
        patch(
            "image_search_service.api.routes.unknown_persons.uuid.uuid4",
            return_value=fixed_uuid,
        ),
    ):
        mock_queue = MagicMock()
        mock_queue.enqueue.return_value = MagicMock()
        mock_get_queue.return_value = mock_queue

        mock_redis_conn = MagicMock()
        mock_get_redis.return_value = mock_redis_conn

        response = await test_client.post(
            "/api/v1/faces/unknown-persons/discover",
            json={
                "clusteringMethod": "dbscan",
                "minClusterSize": 10,
                "minQuality": 0.5,
                "maxFaces": 10000,
                "minClusterConfidence": 0.80,
                "eps": 0.3,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["jobId"] == fixed_uuid

        # Verify custom params passed to job
        call_kwargs = mock_queue.enqueue.call_args[1]
        assert call_kwargs["clustering_method"] == "dbscan"
        assert call_kwargs["min_cluster_size"] == 10
        assert call_kwargs["min_quality"] == 0.5
        assert call_kwargs["max_faces"] == 10000
        assert call_kwargs["min_cluster_confidence"] == 0.80
        assert call_kwargs["eps"] == 0.3


@pytest.mark.asyncio
async def test_discover_invalid_clustering_method(test_client: AsyncClient) -> None:
    """Test discovery rejects invalid clustering method."""
    response = await test_client.post(
        "/api/v1/faces/unknown-persons/discover",
        json={"clusteringMethod": "invalid_method"},
    )

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_discover_min_cluster_size_validation(test_client: AsyncClient) -> None:
    """Test min_cluster_size validation."""
    # Too small
    response = await test_client.post(
        "/api/v1/faces/unknown-persons/discover",
        json={"minClusterSize": 1},
    )
    assert response.status_code == 422

    # Too large
    response = await test_client.post(
        "/api/v1/faces/unknown-persons/discover",
        json={"minClusterSize": 100},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_discover_min_quality_validation(test_client: AsyncClient) -> None:
    """Test min_quality validation."""
    # Negative
    response = await test_client.post(
        "/api/v1/faces/unknown-persons/discover",
        json={"minQuality": -0.1},
    )
    assert response.status_code == 422

    # Above 1.0
    response = await test_client.post(
        "/api/v1/faces/unknown-persons/discover",
        json={"minQuality": 1.5},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_discover_max_faces_validation(test_client: AsyncClient) -> None:
    """Test max_faces validation."""
    # Too small
    response = await test_client.post(
        "/api/v1/faces/unknown-persons/discover",
        json={"maxFaces": 50},
    )
    assert response.status_code == 422

    # Too large
    response = await test_client.post(
        "/api/v1/faces/unknown-persons/discover",
        json={"maxFaces": 200000},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_discover_job_enqueue_failure(test_client: AsyncClient) -> None:
    """Test discovery handles job enqueue failure."""
    with patch("image_search_service.api.routes.unknown_persons.get_queue") as mock_get_queue:
        mock_queue = MagicMock()
        mock_queue.enqueue.side_effect = Exception("Queue connection failed")
        mock_get_queue.return_value = mock_queue

        response = await test_client.post("/api/v1/faces/unknown-persons/discover", json={})

        assert response.status_code == 500
        data = response.json()
        assert "Failed to enqueue job" in data["detail"]


@pytest.mark.asyncio
async def test_discover_dbscan_method(test_client: AsyncClient) -> None:
    """Test discovery with dbscan clustering."""
    with patch("image_search_service.api.routes.unknown_persons.get_queue") as mock_get_queue:
        mock_queue = MagicMock()
        mock_job = MagicMock()
        mock_job.id = "test-job-789"
        mock_queue.enqueue.return_value = mock_job
        mock_get_queue.return_value = mock_queue

        response = await test_client.post(
            "/api/v1/faces/unknown-persons/discover",
            json={"clusteringMethod": "dbscan"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["params"]["clusteringMethod"] == "dbscan"
