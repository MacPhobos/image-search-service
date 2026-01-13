"""Tests for post-training suggestions configuration endpoints.

Tests the new configuration options added to the face-matching config endpoint:
- post_training_suggestions_mode (all | top_n)
- post_training_suggestions_top_n_count (1-100)
"""

import pytest
from fastapi.testclient import TestClient

from image_search_service.main import app

client = TestClient(app)


class TestPostTrainingSuggestionsConfig:
    """Tests for post-training suggestions settings in /api/v1/config/face-matching."""

    def test_get_face_matching_config_includes_post_training_fields(self):
        """GET should return post-training suggestion settings."""
        # When: get face-matching config
        response = client.get("/api/v1/config/face-matching")

        # Then: returns 200 with all config including post-training settings
        assert response.status_code == 200
        data = response.json()

        # Verify post-training settings present (don't assume defaults due to test isolation)
        assert "post_training_suggestions_mode" in data
        assert "post_training_suggestions_top_n_count" in data
        # Values should be valid (mode is "all" or "top_n", count is 1-100)
        assert data["post_training_suggestions_mode"] in ["all", "top_n"]
        assert 1 <= data["post_training_suggestions_top_n_count"] <= 100

    def test_update_post_training_suggestions_mode_to_top_n(self):
        """PUT should accept switching from 'all' to 'top_n' mode."""
        # Given: update request with top_n mode and count
        payload = {
            "auto_assign_threshold": 0.85,  # Required field
            "suggestion_threshold": 0.65,  # Required field
            "post_training_suggestions_mode": "top_n",
            "post_training_suggestions_top_n_count": 15,
        }

        # When: update config
        response = client.put("/api/v1/config/face-matching", json=payload)

        # Then: returns 200 with updated values
        assert response.status_code == 200
        data = response.json()
        assert data["post_training_suggestions_mode"] == "top_n"
        assert data["post_training_suggestions_top_n_count"] == 15

    def test_update_post_training_suggestions_mode_to_all(self):
        """PUT should accept 'all' mode."""
        # Given: update request with 'all' mode
        payload = {
            "auto_assign_threshold": 0.85,
            "suggestion_threshold": 0.65,
            "post_training_suggestions_mode": "all",
            "post_training_suggestions_top_n_count": 10,  # Value ignored in 'all' mode
        }

        # When: update config
        response = client.put("/api/v1/config/face-matching", json=payload)

        # Then: returns 200 with updated mode
        assert response.status_code == 200
        data = response.json()
        assert data["post_training_suggestions_mode"] == "all"

    def test_invalid_mode_rejected(self):
        """PUT should reject invalid mode values."""
        # Given: invalid mode
        payload = {
            "auto_assign_threshold": 0.85,
            "suggestion_threshold": 0.65,
            "post_training_suggestions_mode": "invalid_mode",  # Not in Literal["all", "top_n"]
        }

        # When: update config
        response = client.put("/api/v1/config/face-matching", json=payload)

        # Then: returns 422 validation error
        assert response.status_code == 422

    def test_top_n_count_too_high_rejected(self):
        """PUT should reject top_n_count > 100."""
        # Given: value exceeds maximum (100)
        payload = {
            "auto_assign_threshold": 0.85,
            "suggestion_threshold": 0.65,
            "post_training_suggestions_top_n_count": 200,  # > max 100
        }

        # When: update config
        response = client.put("/api/v1/config/face-matching", json=payload)

        # Then: returns 422 validation error
        assert response.status_code == 422

    def test_top_n_count_too_low_rejected(self):
        """PUT should reject top_n_count < 1."""
        # Given: value below minimum (1)
        payload = {
            "auto_assign_threshold": 0.85,
            "suggestion_threshold": 0.65,
            "post_training_suggestions_top_n_count": 0,  # < min 1
        }

        # When: update config
        response = client.put("/api/v1/config/face-matching", json=payload)

        # Then: returns 422 validation error
        assert response.status_code == 422

    def test_top_n_count_negative_rejected(self):
        """PUT should reject negative top_n_count values."""
        # Given: negative value
        payload = {
            "auto_assign_threshold": 0.85,
            "suggestion_threshold": 0.65,
            "post_training_suggestions_top_n_count": -5,  # Negative
        }

        # When: update config
        response = client.put("/api/v1/config/face-matching", json=payload)

        # Then: returns 422 validation error
        assert response.status_code == 422

    def test_boundary_value_1_accepted(self):
        """PUT should accept minimum boundary value (1)."""
        # Given: minimum boundary value
        payload = {
            "auto_assign_threshold": 0.85,
            "suggestion_threshold": 0.65,
            "post_training_suggestions_top_n_count": 1,  # Min boundary
        }

        # When: update config
        response = client.put("/api/v1/config/face-matching", json=payload)

        # Then: accepts value
        assert response.status_code == 200
        data = response.json()
        assert data["post_training_suggestions_top_n_count"] == 1

    def test_boundary_value_100_accepted(self):
        """PUT should accept maximum boundary value (100)."""
        # Given: maximum boundary value
        payload = {
            "auto_assign_threshold": 0.85,
            "suggestion_threshold": 0.65,
            "post_training_suggestions_top_n_count": 100,  # Max boundary
        }

        # When: update config
        response = client.put("/api/v1/config/face-matching", json=payload)

        # Then: accepts value
        assert response.status_code == 200
        data = response.json()
        assert data["post_training_suggestions_top_n_count"] == 100

    def test_common_value_25_accepted(self):
        """PUT should accept common middle value (25)."""
        # Given: typical use case value
        payload = {
            "auto_assign_threshold": 0.85,
            "suggestion_threshold": 0.65,
            "post_training_suggestions_mode": "top_n",
            "post_training_suggestions_top_n_count": 25,
        }

        # When: update config
        response = client.put("/api/v1/config/face-matching", json=payload)

        # Then: accepts value
        assert response.status_code == 200
        data = response.json()
        assert data["post_training_suggestions_mode"] == "top_n"
        assert data["post_training_suggestions_top_n_count"] == 25

    def test_update_only_mode_without_count(self):
        """PUT should accept updating only mode (count is optional)."""
        # Given: update only mode, not count
        payload = {
            "auto_assign_threshold": 0.85,
            "suggestion_threshold": 0.65,
            "post_training_suggestions_mode": "all",  # Only mode, no count
        }

        # When: update config
        response = client.put("/api/v1/config/face-matching", json=payload)

        # Then: accepts and returns with existing count
        assert response.status_code == 200
        data = response.json()
        assert data["post_training_suggestions_mode"] == "all"
        # Count should retain previous value or default

    def test_update_only_count_without_mode(self):
        """PUT should accept updating only count (mode is optional)."""
        # Given: update only count, not mode
        payload = {
            "auto_assign_threshold": 0.85,
            "suggestion_threshold": 0.65,
            "post_training_suggestions_top_n_count": 50,  # Only count, no mode
        }

        # When: update config
        response = client.put("/api/v1/config/face-matching", json=payload)

        # Then: accepts and returns with existing mode
        assert response.status_code == 200
        data = response.json()
        assert data["post_training_suggestions_top_n_count"] == 50
        # Mode should retain previous value or default

    def test_update_preserves_other_config_values(self):
        """PUT should not affect other face-matching config when updating post-training settings."""
        # When: update with all fields including post-training settings
        update_payload = {
            "auto_assign_threshold": 0.92,
            "suggestion_threshold": 0.68,
            "max_suggestions": 75,
            "suggestion_expiry_days": 45,
            "post_training_suggestions_mode": "top_n",
            "post_training_suggestions_top_n_count": 30,
        }
        response = client.put("/api/v1/config/face-matching", json=update_payload)

        # Then: all fields set correctly
        assert response.status_code == 200
        data = response.json()
        assert data["auto_assign_threshold"] == 0.92
        assert data["suggestion_threshold"] == 0.68
        assert data["max_suggestions"] == 75
        assert data["suggestion_expiry_days"] == 45
        assert data["post_training_suggestions_mode"] == "top_n"
        assert data["post_training_suggestions_top_n_count"] == 30
