"""Test embedding router for CLIP/SigLIP selection."""

from unittest.mock import MagicMock, patch

import pytest


def test_router_returns_clip_by_default() -> None:
    """Test that router returns CLIP service by default (use_siglip=False)."""
    from image_search_service.services.embedding_router import (
        get_search_embedding_service,
    )

    with patch(
        "image_search_service.services.embedding_router.get_settings"
    ) as mock_settings:
        # Mock settings: SigLIP disabled
        settings_obj = MagicMock()
        settings_obj.use_siglip = False
        settings_obj.siglip_rollout_percentage = 0
        settings_obj.qdrant_collection = "image_assets"
        settings_obj.siglip_collection = "image_assets_siglip"
        mock_settings.return_value = settings_obj

        with patch(
            "image_search_service.services.embedding_router.get_embedding_service"
        ) as mock_clip, patch(
            "image_search_service.services.embedding_router.get_siglip_service"
        ) as mock_siglip:
            mock_clip_service = MagicMock()
            mock_clip.return_value = mock_clip_service

            service, collection = get_search_embedding_service()

            # Should return CLIP service
            assert service is mock_clip_service
            assert collection == "image_assets"
            # CLIP should be called, SigLIP should not
            assert mock_clip.called
            assert not mock_siglip.called


def test_router_returns_siglip_when_use_siglip_true() -> None:
    """Test that router returns SigLIP service when use_siglip=True."""
    from image_search_service.services.embedding_router import (
        get_search_embedding_service,
    )

    with patch(
        "image_search_service.services.embedding_router.get_settings"
    ) as mock_settings:
        # Mock settings: SigLIP enabled
        settings_obj = MagicMock()
        settings_obj.use_siglip = True
        settings_obj.siglip_rollout_percentage = 0
        settings_obj.qdrant_collection = "image_assets"
        settings_obj.siglip_collection = "image_assets_siglip"
        mock_settings.return_value = settings_obj

        with patch(
            "image_search_service.services.embedding_router.get_embedding_service"
        ) as mock_clip, patch(
            "image_search_service.services.embedding_router.get_siglip_service"
        ) as mock_siglip:
            mock_siglip_service = MagicMock()
            mock_siglip.return_value = mock_siglip_service

            service, collection = get_search_embedding_service()

            # Should return SigLIP service
            assert service is mock_siglip_service
            assert collection == "image_assets_siglip"
            # SigLIP should be called, CLIP should not
            assert mock_siglip.called
            assert not mock_clip.called


def test_router_gradual_rollout_deterministic_with_user_id() -> None:
    """Test that router uses deterministic bucketing with user_id."""
    from image_search_service.services.embedding_router import (
        get_search_embedding_service,
    )

    with patch(
        "image_search_service.services.embedding_router.get_settings"
    ) as mock_settings:
        # Mock settings: 50% rollout
        settings_obj = MagicMock()
        settings_obj.use_siglip = False
        settings_obj.siglip_rollout_percentage = 50
        settings_obj.qdrant_collection = "image_assets"
        settings_obj.siglip_collection = "image_assets_siglip"
        mock_settings.return_value = settings_obj

        with patch(
            "image_search_service.services.embedding_router.get_embedding_service"
        ) as mock_clip, patch(
            "image_search_service.services.embedding_router.get_siglip_service"
        ) as mock_siglip:
            mock_clip_service = MagicMock()
            mock_siglip_service = MagicMock()
            mock_clip.return_value = mock_clip_service
            mock_siglip.return_value = mock_siglip_service

            # user_id=23 → bucket=23 → 23 < 50 → SigLIP
            service1, collection1 = get_search_embedding_service(user_id=23)
            assert service1 is mock_siglip_service
            assert collection1 == "image_assets_siglip"

            # user_id=76 → bucket=76 → 76 >= 50 → CLIP
            service2, collection2 = get_search_embedding_service(user_id=76)
            assert service2 is mock_clip_service
            assert collection2 == "image_assets"

            # Same user_id should be deterministic
            service3, collection3 = get_search_embedding_service(user_id=23)
            assert service3 is mock_siglip_service
            assert collection3 == "image_assets_siglip"


def test_router_gradual_rollout_0_percent_uses_clip() -> None:
    """Test that 0% rollout always uses CLIP."""
    from image_search_service.services.embedding_router import (
        get_search_embedding_service,
    )

    with patch(
        "image_search_service.services.embedding_router.get_settings"
    ) as mock_settings:
        # Mock settings: 0% rollout
        settings_obj = MagicMock()
        settings_obj.use_siglip = False
        settings_obj.siglip_rollout_percentage = 0
        settings_obj.qdrant_collection = "image_assets"
        settings_obj.siglip_collection = "image_assets_siglip"
        mock_settings.return_value = settings_obj

        with patch(
            "image_search_service.services.embedding_router.get_embedding_service"
        ) as mock_clip, patch(
            "image_search_service.services.embedding_router.get_siglip_service"
        ) as mock_siglip:
            mock_clip_service = MagicMock()
            mock_clip.return_value = mock_clip_service

            # Try various user IDs - all should get CLIP
            for user_id in [0, 25, 50, 75, 99]:
                service, collection = get_search_embedding_service(user_id=user_id)
                assert service is mock_clip_service
                assert collection == "image_assets"


def test_router_gradual_rollout_100_percent_uses_siglip() -> None:
    """Test that 100% rollout always uses SigLIP."""
    from image_search_service.services.embedding_router import (
        get_search_embedding_service,
    )

    with patch(
        "image_search_service.services.embedding_router.get_settings"
    ) as mock_settings:
        # Mock settings: 100% rollout
        settings_obj = MagicMock()
        settings_obj.use_siglip = False
        settings_obj.siglip_rollout_percentage = 100
        settings_obj.qdrant_collection = "image_assets"
        settings_obj.siglip_collection = "image_assets_siglip"
        mock_settings.return_value = settings_obj

        with patch(
            "image_search_service.services.embedding_router.get_embedding_service"
        ) as mock_clip, patch(
            "image_search_service.services.embedding_router.get_siglip_service"
        ) as mock_siglip:
            mock_siglip_service = MagicMock()
            mock_siglip.return_value = mock_siglip_service

            # Try various user IDs - all should get SigLIP
            for user_id in [0, 25, 50, 75, 99]:
                service, collection = get_search_embedding_service(user_id=user_id)
                assert service is mock_siglip_service
                assert collection == "image_assets_siglip"


def test_router_use_siglip_overrides_rollout_percentage() -> None:
    """Test that use_siglip=True overrides rollout percentage."""
    from image_search_service.services.embedding_router import (
        get_search_embedding_service,
    )

    with patch(
        "image_search_service.services.embedding_router.get_settings"
    ) as mock_settings:
        # Mock settings: use_siglip=True, rollout=0 (use_siglip takes precedence)
        settings_obj = MagicMock()
        settings_obj.use_siglip = True
        settings_obj.siglip_rollout_percentage = 0  # Should be ignored
        settings_obj.qdrant_collection = "image_assets"
        settings_obj.siglip_collection = "image_assets_siglip"
        mock_settings.return_value = settings_obj

        with patch(
            "image_search_service.services.embedding_router.get_embedding_service"
        ) as mock_clip, patch(
            "image_search_service.services.embedding_router.get_siglip_service"
        ) as mock_siglip:
            mock_siglip_service = MagicMock()
            mock_siglip.return_value = mock_siglip_service

            service, collection = get_search_embedding_service()

            # Should use SigLIP despite rollout=0
            assert service is mock_siglip_service
            assert collection == "image_assets_siglip"


def test_router_without_user_id_uses_random_bucketing() -> None:
    """Test that router uses random bucketing when user_id is None."""
    from image_search_service.services.embedding_router import (
        get_search_embedding_service,
    )

    with patch(
        "image_search_service.services.embedding_router.get_settings"
    ) as mock_settings:
        # Mock settings: 50% rollout
        settings_obj = MagicMock()
        settings_obj.use_siglip = False
        settings_obj.siglip_rollout_percentage = 50
        settings_obj.qdrant_collection = "image_assets"
        settings_obj.siglip_collection = "image_assets_siglip"
        mock_settings.return_value = settings_obj

        with patch(
            "image_search_service.services.embedding_router.get_embedding_service"
        ) as mock_clip, patch(
            "image_search_service.services.embedding_router.get_siglip_service"
        ) as mock_siglip, patch(
            "image_search_service.services.embedding_router.random.randint"
        ) as mock_random:
            mock_clip_service = MagicMock()
            mock_siglip_service = MagicMock()
            mock_clip.return_value = mock_clip_service
            mock_siglip.return_value = mock_siglip_service

            # Mock random to return 25 (< 50, should get SigLIP)
            mock_random.return_value = 25

            service, collection = get_search_embedding_service(user_id=None)

            # Should use random bucket
            assert mock_random.called
            assert service is mock_siglip_service
            assert collection == "image_assets_siglip"
