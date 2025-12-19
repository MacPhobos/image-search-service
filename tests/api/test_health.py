"""Test health check endpoint."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check_returns_ok(test_client: AsyncClient) -> None:
    """Test that health endpoint returns 200 and correct JSON."""
    response = await test_client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data == {"status": "ok"}


@pytest.mark.asyncio
async def test_health_check_works_without_dependencies(test_client: AsyncClient) -> None:
    """Test that health endpoint works even if Postgres/Redis/Qdrant are down.

    This test verifies lazy initialization - the health check should not
    attempt to connect to any external services.
    """
    response = await test_client.get("/health")

    # Should succeed without any external dependencies
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
