"""Test image serving endpoints."""

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.db.models import ImageAsset


class TestBatchThumbnails:
    """Test batch thumbnail endpoint."""

    @pytest.mark.asyncio
    async def test_batch_thumbnails_success(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
        temp_image_factory: Callable[..., Path],
    ) -> None:
        """Test successful batch thumbnail retrieval with valid asset IDs."""
        # Create test images
        image1_path = temp_image_factory("test1.jpg")
        image2_path = temp_image_factory("test2.jpg")
        image3_path = temp_image_factory("test3.jpg")

        # Create test assets in database
        asset1 = ImageAsset(path=str(image1_path))
        asset2 = ImageAsset(path=str(image2_path))
        asset3 = ImageAsset(path=str(image3_path))
        db_session.add(asset1)
        db_session.add(asset2)
        db_session.add(asset3)
        await db_session.flush()

        # Mock thumbnail service to return predictable data URIs
        with patch("image_search_service.api.routes.images.ThumbnailService") as mock_service_class:
            mock_service = MagicMock()
            mock_service_class.return_value = mock_service

            # Mock returns data URIs for each asset
            def mock_get_thumbnail(asset_id: int, path: str) -> str:
                return f"data:image/jpeg;base64,fake_base64_data_{asset_id}"

            mock_service.get_thumbnail_as_base64.side_effect = mock_get_thumbnail

            # Make request
            response = await test_client.post(
                "/api/v1/images/thumbnails/batch",
                json={"assetIds": [asset1.id, asset2.id, asset3.id]},
            )

            assert response.status_code == 200
            data = response.json()

            # Verify response structure
            assert "thumbnails" in data
            assert "found" in data
            assert "notFound" in data

            # All three assets should be found
            assert data["found"] == 3
            assert len(data["notFound"]) == 0

            # Verify thumbnail data URIs (keys are strings in response)
            assert str(asset1.id) in data["thumbnails"]
            assert str(asset2.id) in data["thumbnails"]
            assert str(asset3.id) in data["thumbnails"]

            # Verify data URI format
            assert data["thumbnails"][str(asset1.id)].startswith("data:image/jpeg;base64,")
            assert data["thumbnails"][str(asset2.id)].startswith("data:image/jpeg;base64,")
            assert data["thumbnails"][str(asset3.id)].startswith("data:image/jpeg;base64,")

    @pytest.mark.asyncio
    async def test_batch_thumbnails_partial_not_found(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
        temp_image_factory: Callable[..., Path],
    ) -> None:
        """Test batch request where some assets are found and some are not."""
        # Create only two test assets
        image1_path = temp_image_factory("exists1.jpg")
        image2_path = temp_image_factory("exists2.jpg")

        asset1 = ImageAsset(path=str(image1_path))
        asset2 = ImageAsset(path=str(image2_path))
        db_session.add(asset1)
        db_session.add(asset2)
        await db_session.flush()

        # Request includes two existing assets and two non-existent ones
        nonexistent_id_1 = 999
        nonexistent_id_2 = 1000

        # Mock thumbnail service
        with patch("image_search_service.api.routes.images.ThumbnailService") as mock_service_class:
            mock_service = MagicMock()
            mock_service_class.return_value = mock_service

            def mock_get_thumbnail(asset_id: int, path: str) -> str:
                return f"data:image/jpeg;base64,data_for_{asset_id}"

            mock_service.get_thumbnail_as_base64.side_effect = mock_get_thumbnail

            # Request mix of existing and non-existing asset IDs
            response = await test_client.post(
                "/api/v1/images/thumbnails/batch",
                json={
                    "assetIds": [
                        asset1.id,
                        nonexistent_id_1,
                        asset2.id,
                        nonexistent_id_2,
                    ]
                },
            )

            assert response.status_code == 200
            data = response.json()

            # Should find 2, missing 2
            assert data["found"] == 2
            assert len(data["notFound"]) == 2
            assert set(data["notFound"]) == {nonexistent_id_1, nonexistent_id_2}

            # Existing assets should have data URIs
            assert data["thumbnails"][str(asset1.id)].startswith("data:image/jpeg;base64,")
            assert data["thumbnails"][str(asset2.id)].startswith("data:image/jpeg;base64,")

            # Non-existent assets should be null
            assert data["thumbnails"][str(nonexistent_id_1)] is None
            assert data["thumbnails"][str(nonexistent_id_2)] is None

    @pytest.mark.asyncio
    async def test_batch_thumbnails_all_not_found(self, test_client: AsyncClient) -> None:
        """Test batch request where all requested assets don't exist."""
        # Request only non-existent asset IDs
        response = await test_client.post(
            "/api/v1/images/thumbnails/batch",
            json={"assetIds": [999, 1000, 1001]},
        )

        assert response.status_code == 200
        data = response.json()

        # None should be found
        assert data["found"] == 0
        assert len(data["notFound"]) == 3
        assert set(data["notFound"]) == {999, 1000, 1001}

        # All thumbnails should be null
        assert data["thumbnails"]["999"] is None
        assert data["thumbnails"]["1000"] is None
        assert data["thumbnails"]["1001"] is None

    @pytest.mark.asyncio
    async def test_batch_thumbnails_empty_array_rejected(self, test_client: AsyncClient) -> None:
        """Test that empty assetIds array returns 422 validation error."""
        response = await test_client.post(
            "/api/v1/images/thumbnails/batch",
            json={"assetIds": []},
        )

        # Should fail validation
        assert response.status_code == 422
        data = response.json()

        # Pydantic validation error for min_length constraint
        assert "detail" in data
        errors = data["detail"]
        assert len(errors) > 0

        # Check first error has correct field location and type
        first_error = errors[0]
        assert first_error["type"] == "too_short"
        assert "assetIds" in first_error["loc"]
        assert "at least 1 item" in first_error["msg"]

    @pytest.mark.asyncio
    async def test_batch_thumbnails_exceeds_limit(self, test_client: AsyncClient) -> None:
        """Test that more than 100 asset IDs returns 422 validation error."""
        # Create array of 101 asset IDs (exceeds limit of 100)
        too_many_ids = list(range(1, 102))

        response = await test_client.post(
            "/api/v1/images/thumbnails/batch",
            json={"assetIds": too_many_ids},
        )

        # Should fail validation
        assert response.status_code == 422
        data = response.json()

        # Pydantic validation error for max_length constraint
        assert "detail" in data
        errors = data["detail"]
        assert len(errors) > 0

        # Check first error has correct field location and type
        first_error = errors[0]
        assert first_error["type"] == "too_long"
        assert "assetIds" in first_error["loc"]
        assert "at most 100 items" in first_error["msg"]

    @pytest.mark.asyncio
    async def test_batch_thumbnails_thumbnail_generation_failure(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
        temp_image_factory: Callable[..., Path],
    ) -> None:
        """Test graceful handling when thumbnail generation fails for some assets."""
        # Create test asset
        image_path = temp_image_factory("test.jpg")
        asset = ImageAsset(path=str(image_path))
        db_session.add(asset)
        await db_session.flush()

        # Mock thumbnail service to return None (generation failed)
        with patch("image_search_service.api.routes.images.ThumbnailService") as mock_service_class:
            mock_service = MagicMock()
            mock_service_class.return_value = mock_service

            # Simulate thumbnail generation failure
            mock_service.get_thumbnail_as_base64.return_value = None

            response = await test_client.post(
                "/api/v1/images/thumbnails/batch",
                json={"assetIds": [asset.id]},
            )

            assert response.status_code == 200
            data = response.json()

            # Asset exists but thumbnail failed, should be in notFound
            assert data["found"] == 0
            assert asset.id in data["notFound"]
            assert data["thumbnails"][str(asset.id)] is None

    @pytest.mark.asyncio
    async def test_batch_thumbnails_duplicate_ids_handled(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
        temp_image_factory: Callable[..., Path],
    ) -> None:
        """Test that duplicate asset IDs in request are handled correctly."""
        # Create test asset
        image_path = temp_image_factory("test.jpg")
        asset = ImageAsset(path=str(image_path))
        db_session.add(asset)
        await db_session.flush()

        # Mock thumbnail service
        with patch("image_search_service.api.routes.images.ThumbnailService") as mock_service_class:
            mock_service = MagicMock()
            mock_service_class.return_value = mock_service
            mock_service.get_thumbnail_as_base64.return_value = "data:image/jpeg;base64,fake_data"

            # Request same asset ID multiple times
            response = await test_client.post(
                "/api/v1/images/thumbnails/batch",
                json={"assetIds": [asset.id, asset.id, asset.id]},
            )

            assert response.status_code == 200
            data = response.json()

            # Should process all requested IDs (even duplicates)
            # found counts unique successful retrievals
            assert data["found"] >= 1  # At least one successful retrieval
            assert str(asset.id) in data["thumbnails"]
            assert data["thumbnails"][str(asset.id)].startswith("data:image/jpeg;base64,")
