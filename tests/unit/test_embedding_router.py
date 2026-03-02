"""Test embedding router for CLIP/SigLIP selection."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest


def _make_settings(
    *,
    use_siglip: bool = False,
    rollout_pct: int = 0,
    qdrant_collection: str = "image_assets",
    siglip_collection: str = "image_assets_siglip",
) -> MagicMock:
    """Create a mock Settings object for embedding router tests."""
    settings_obj = MagicMock()
    settings_obj.use_siglip = use_siglip
    settings_obj.siglip_rollout_percentage = rollout_pct
    settings_obj.qdrant_collection = qdrant_collection
    settings_obj.siglip_collection = siglip_collection
    return settings_obj


@dataclass
class RouterMocks:
    """Container for all patched mocks used by embedding router tests."""

    get_settings: MagicMock
    get_clip: MagicMock
    get_siglip: MagicMock
    clip_service: MagicMock
    siglip_service: MagicMock


@pytest.fixture()
def router_mocks(monkeypatch: pytest.MonkeyPatch) -> RouterMocks:
    """Monkeypatch all embedding router dependencies and return mock handles."""
    mock_get_settings = MagicMock()
    mock_get_clip = MagicMock()
    mock_get_siglip = MagicMock()

    monkeypatch.setattr(
        "image_search_service.services.embedding_router.get_settings",
        mock_get_settings,
    )
    monkeypatch.setattr(
        "image_search_service.services.embedding_router.get_embedding_service",
        mock_get_clip,
    )
    monkeypatch.setattr(
        "image_search_service.services.embedding_router.get_siglip_service",
        mock_get_siglip,
    )

    clip_service = MagicMock()
    siglip_service = MagicMock()
    mock_get_clip.return_value = clip_service
    mock_get_siglip.return_value = siglip_service

    return RouterMocks(
        get_settings=mock_get_settings,
        get_clip=mock_get_clip,
        get_siglip=mock_get_siglip,
        clip_service=clip_service,
        siglip_service=siglip_service,
    )


@pytest.mark.parametrize(
    "use_siglip,rollout_pct,expected_service,expected_collection",
    [
        pytest.param(False, 0, "clip", "image_assets", id="default-clip"),
        pytest.param(True, 0, "siglip", "image_assets_siglip", id="use-siglip-flag"),
        pytest.param(True, 0, "siglip", "image_assets_siglip", id="use-siglip-overrides-rollout"),
    ],
)
def test_router_basic_selection(
    router_mocks: RouterMocks,
    use_siglip: bool,
    rollout_pct: int,
    expected_service: str,
    expected_collection: str,
) -> None:
    """Test basic router selection based on use_siglip flag."""
    from image_search_service.services.embedding_router import (
        get_search_embedding_service,
    )

    router_mocks.get_settings.return_value = _make_settings(
        use_siglip=use_siglip, rollout_pct=rollout_pct
    )

    service, collection = get_search_embedding_service()

    if expected_service == "clip":
        assert service is router_mocks.clip_service
        assert router_mocks.get_clip.called
        assert not router_mocks.get_siglip.called
    else:
        assert service is router_mocks.siglip_service
        assert router_mocks.get_siglip.called
        assert not router_mocks.get_clip.called
    assert collection == expected_collection


def test_router_gradual_rollout_deterministic_with_user_id(
    router_mocks: RouterMocks,
) -> None:
    """Test that router uses deterministic bucketing with user_id."""
    from image_search_service.services.embedding_router import (
        get_search_embedding_service,
    )

    router_mocks.get_settings.return_value = _make_settings(rollout_pct=50)

    # user_id=23 -> bucket=23 -> 23 < 50 -> SigLIP
    service1, collection1 = get_search_embedding_service(user_id=23)
    assert service1 is router_mocks.siglip_service
    assert collection1 == "image_assets_siglip"

    # user_id=76 -> bucket=76 -> 76 >= 50 -> CLIP
    service2, collection2 = get_search_embedding_service(user_id=76)
    assert service2 is router_mocks.clip_service
    assert collection2 == "image_assets"

    # Same user_id should be deterministic
    service3, collection3 = get_search_embedding_service(user_id=23)
    assert service3 is router_mocks.siglip_service
    assert collection3 == "image_assets_siglip"


@pytest.mark.parametrize(
    "rollout_pct,expected_service,expected_collection",
    [
        pytest.param(0, "clip", "image_assets", id="0pct-always-clip"),
        pytest.param(100, "siglip", "image_assets_siglip", id="100pct-always-siglip"),
    ],
)
def test_router_rollout_boundary(
    router_mocks: RouterMocks,
    rollout_pct: int,
    expected_service: str,
    expected_collection: str,
) -> None:
    """Test that 0% rollout always uses CLIP and 100% always uses SigLIP."""
    from image_search_service.services.embedding_router import (
        get_search_embedding_service,
    )

    router_mocks.get_settings.return_value = _make_settings(rollout_pct=rollout_pct)

    # Try various user IDs - all should get same result
    for user_id in [0, 25, 50, 75, 99]:
        service, collection = get_search_embedding_service(user_id=user_id)
        if expected_service == "clip":
            assert service is router_mocks.clip_service
        else:
            assert service is router_mocks.siglip_service
        assert collection == expected_collection


def test_router_use_siglip_overrides_rollout_percentage(
    router_mocks: RouterMocks,
) -> None:
    """Test that use_siglip=True overrides rollout percentage."""
    from image_search_service.services.embedding_router import (
        get_search_embedding_service,
    )

    # use_siglip=True, rollout=0 (use_siglip takes precedence)
    router_mocks.get_settings.return_value = _make_settings(use_siglip=True, rollout_pct=0)

    service, collection = get_search_embedding_service()

    # Should use SigLIP despite rollout=0
    assert service is router_mocks.siglip_service
    assert collection == "image_assets_siglip"


def test_router_without_user_id_uses_random_bucketing(
    monkeypatch: pytest.MonkeyPatch,
    router_mocks: RouterMocks,
) -> None:
    """Test that router uses random bucketing when user_id is None."""
    from image_search_service.services.embedding_router import (
        get_search_embedding_service,
    )

    router_mocks.get_settings.return_value = _make_settings(rollout_pct=50)

    mock_random = MagicMock(return_value=25)
    monkeypatch.setattr(
        "image_search_service.services.embedding_router.random.randint",
        mock_random,
    )

    # Mock random to return 25 (< 50, should get SigLIP)
    service, collection = get_search_embedding_service(user_id=None)

    # Should use random bucket
    assert mock_random.called
    assert service is router_mocks.siglip_service
    assert collection == "image_assets_siglip"
