"""Tests for unknown face clustering configuration endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from image_search_service.main import app


client = TestClient(app)


class TestUnknownClusteringConfig:
    """Tests for /api/v1/config/face-clustering-unknown endpoints."""

    @patch("image_search_service.api.routes.config.get_settings")
    def test_get_unknown_clustering_config_returns_defaults(self, mock_get_settings):
        """GET should return current configuration."""
        # Given: settings with default values
        mock_settings = mock_get_settings.return_value
        mock_settings.unknown_face_cluster_min_confidence = 0.85
        mock_settings.unknown_face_cluster_min_size = 5

        # When: get config
        response = client.get("/api/v1/config/face-clustering-unknown")

        # Then: returns 200 with config
        assert response.status_code == 200
        data = response.json()
        assert data["minConfidence"] == 0.85
        assert data["minClusterSize"] == 5

    @patch("image_search_service.api.routes.config.get_settings")
    def test_get_unknown_clustering_config_uses_camel_case(self, mock_get_settings):
        """Response should use camelCase field names."""
        # Given: settings
        mock_settings = mock_get_settings.return_value
        mock_settings.unknown_face_cluster_min_confidence = 0.90
        mock_settings.unknown_face_cluster_min_size = 10

        # When: get config
        response = client.get("/api/v1/config/face-clustering-unknown")

        # Then: uses camelCase
        data = response.json()
        assert "minConfidence" in data
        assert "minClusterSize" in data
        assert "min_confidence" not in data  # No snake_case
        assert "min_cluster_size" not in data

    def test_put_unknown_clustering_config_accepts_valid_values(self):
        """PUT should accept valid configuration."""
        # Given: valid config update
        payload = {"minConfidence": 0.90, "minClusterSize": 10}

        # When: update config
        response = client.put("/api/v1/config/face-clustering-unknown", json=payload)

        # Then: returns 200 with updated values
        assert response.status_code == 200
        data = response.json()
        assert data["minConfidence"] == 0.90
        assert data["minClusterSize"] == 10

    def test_put_unknown_clustering_config_validates_confidence_range(self):
        """PUT should reject confidence outside 0.0-1.0 range."""
        # Given: invalid confidence (too high)
        payload = {"minConfidence": 1.5, "minClusterSize": 5}

        # When: update config
        response = client.put("/api/v1/config/face-clustering-unknown", json=payload)

        # Then: returns 422 validation error
        assert response.status_code == 422

    def test_put_unknown_clustering_config_validates_cluster_size(self):
        """PUT should reject cluster_size outside 1-100 range."""
        # Given: invalid cluster size (too large)
        payload = {"minConfidence": 0.85, "minClusterSize": 150}

        # When: update config
        response = client.put("/api/v1/config/face-clustering-unknown", json=payload)

        # Then: returns 422 validation error
        assert response.status_code == 422

    def test_put_unknown_clustering_config_rejects_missing_fields(self):
        """PUT should reject request with missing required fields."""
        # Given: incomplete payload
        payload = {"minConfidence": 0.90}  # Missing minClusterSize

        # When: update config
        response = client.put("/api/v1/config/face-clustering-unknown", json=payload)

        # Then: returns 422 validation error
        assert response.status_code == 422

    def test_put_unknown_clustering_config_accepts_snake_case_input(self):
        """PUT should accept snake_case field names (populate_by_name=True)."""
        # Given: payload with snake_case
        payload = {"min_confidence": 0.88, "min_cluster_size": 7}

        # When: update config
        response = client.put("/api/v1/config/face-clustering-unknown", json=payload)

        # Then: accepts and returns camelCase
        assert response.status_code == 200
        data = response.json()
        assert data["minConfidence"] == 0.88
        assert data["minClusterSize"] == 7

    def test_put_unknown_clustering_config_accepts_boundary_values(self):
        """PUT should accept boundary values (0.0, 1.0, 1, 100)."""
        # Given: boundary values
        payload = {"minConfidence": 0.0, "minClusterSize": 1}

        # When: update config
        response = client.put("/api/v1/config/face-clustering-unknown", json=payload)

        # Then: accepts values
        assert response.status_code == 200
        data = response.json()
        assert data["minConfidence"] == 0.0
        assert data["minClusterSize"] == 1

        # Test upper boundaries
        payload2 = {"minConfidence": 1.0, "minClusterSize": 100}
        response2 = client.put("/api/v1/config/face-clustering-unknown", json=payload2)
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["minConfidence"] == 1.0
        assert data2["minClusterSize"] == 100
