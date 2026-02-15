"""Unit tests for device abstraction module."""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestGetDevice:
    """Tests for get_device() function."""

    def setup_method(self) -> None:
        """Clear device cache before each test."""
        from image_search_service.core.device import clear_device_cache

        clear_device_cache()

    @pytest.mark.parametrize(
        "cuda_available,mps_available,expected",
        [
            pytest.param(True, True, "cuda", id="cuda-wins-over-mps"),
            pytest.param(False, True, "mps", id="mps-fallback"),
            pytest.param(False, False, "cpu", id="cpu-fallback"),
        ],
    )
    def test_auto_detect_device(
        self,
        cuda_available: bool,
        mps_available: bool,
        expected: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Should auto-detect best available device."""
        from image_search_service.core.device import clear_device_cache, get_device

        monkeypatch.delenv("DEVICE", raising=False)
        monkeypatch.delenv("FORCE_CPU", raising=False)
        clear_device_cache()

        with (
            patch("torch.cuda.is_available", return_value=cuda_available),
            patch("torch.backends.mps.is_available", return_value=mps_available),
        ):
            assert get_device() == expected

    @patch.dict(os.environ, {"DEVICE": "cpu"})
    def test_device_env_override(self) -> None:
        """Should respect DEVICE environment variable."""
        from image_search_service.core.device import clear_device_cache, get_device

        clear_device_cache()
        assert get_device() == "cpu"

    @patch.dict(os.environ, {"DEVICE": "cuda:0"})
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=2)
    def test_device_env_cuda_with_id(
        self, mock_count: MagicMock, mock_cuda: MagicMock
    ) -> None:
        """Should respect DEVICE environment variable with CUDA device ID."""
        from image_search_service.core.device import clear_device_cache, get_device

        clear_device_cache()
        assert get_device() == "cuda:0"

    @patch.dict(os.environ, {"DEVICE": "auto"})
    @patch("torch.cuda.is_available", return_value=True)
    def test_device_env_auto(self, mock_cuda: MagicMock) -> None:
        """Should auto-detect when DEVICE is 'auto'."""
        from image_search_service.core.device import clear_device_cache, get_device

        clear_device_cache()
        assert get_device() == "cuda"

    @pytest.mark.parametrize(
        "force_cpu_value",
        [
            pytest.param("true", id="true"),
            pytest.param("1", id="one"),
            pytest.param("yes", id="yes"),
        ],
    )
    def test_force_cpu(
        self, force_cpu_value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should force CPU when FORCE_CPU is set to truthy value."""
        from image_search_service.core.device import clear_device_cache, get_device

        monkeypatch.setenv("FORCE_CPU", force_cpu_value)
        clear_device_cache()

        with patch("torch.cuda.is_available", return_value=True):
            assert get_device() == "cpu"

    @pytest.mark.parametrize(
        "device,error_match,cuda_available,device_count",
        [
            pytest.param("invalid_device", "Invalid DEVICE", False, 0, id="bad-prefix"),
            pytest.param("cuda:5", "Invalid CUDA device ID", True, 2, id="bad-cuda-id"),
            pytest.param("cuda:abc", "Invalid CUDA device ID", False, 0, id="bad-cuda-format"),
        ],
    )
    def test_invalid_device_raises(
        self,
        device: str,
        error_match: str,
        cuda_available: bool,
        device_count: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Should raise ValueError for invalid device specifications."""
        from image_search_service.core.device import clear_device_cache, get_device

        monkeypatch.setenv("DEVICE", device)
        clear_device_cache()

        with (
            patch("torch.cuda.is_available", return_value=cuda_available),
            patch("torch.cuda.device_count", return_value=device_count),
            pytest.raises(ValueError, match=error_match),
        ):
            get_device()

    @patch.dict(os.environ, {}, clear=True)
    @patch("torch.cuda.is_available", return_value=True)
    def test_device_cached(self, mock_cuda: MagicMock) -> None:
        """Should cache device selection across multiple calls."""
        from image_search_service.core.device import clear_device_cache, get_device

        clear_device_cache()

        # First call
        result1 = get_device()
        call_count_1 = mock_cuda.call_count

        # Second call should use cache
        result2 = get_device()
        call_count_2 = mock_cuda.call_count

        assert result1 == result2 == "cuda"
        # Cache should prevent additional torch.cuda.is_available() calls
        assert call_count_1 == call_count_2


class TestGetDeviceInfo:
    """Tests for get_device_info() function."""

    def setup_method(self) -> None:
        """Clear device cache before each test."""
        from image_search_service.core.device import clear_device_cache

        clear_device_cache()

    @patch.dict(os.environ, {"FORCE_CPU": "true"})
    @patch("torch.cuda.is_available", return_value=False)
    def test_returns_expected_keys(self, mock_cuda: MagicMock) -> None:
        """Should return dict with expected keys."""
        from image_search_service.core.device import clear_device_cache, get_device_info

        clear_device_cache()

        info = get_device_info()

        # Required keys
        assert "platform" in info
        assert "machine" in info
        assert "python_version" in info
        assert "pytorch_version" in info
        assert "selected_device" in info
        assert "cuda_available" in info

        # MPS keys (at least one should be present)
        assert "mps_available" in info or "mps_built" in info

    @patch.dict(os.environ, {"FORCE_CPU": "true"})
    @patch("torch.cuda.is_available", return_value=False)
    def test_cpu_mode_info(self, mock_cuda: MagicMock) -> None:
        """Should return correct info when in CPU mode."""
        from image_search_service.core.device import clear_device_cache, get_device_info

        clear_device_cache()

        info = get_device_info()

        assert info["selected_device"] == "cpu"
        assert info["cuda_available"] is False
        assert isinstance(info["platform"], str)
        assert isinstance(info["machine"], str)
        assert isinstance(info["python_version"], str)
        assert isinstance(info["pytorch_version"], str)

    @patch.dict(os.environ, {}, clear=True)
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=2)
    @patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 3090")
    @patch("torch.version.cuda", "11.8")
    def test_cuda_info(
        self,
        mock_name: MagicMock,
        mock_count: MagicMock,
        mock_cuda: MagicMock,
    ) -> None:
        """Should return CUDA info when CUDA is available."""
        from image_search_service.core.device import clear_device_cache, get_device_info

        clear_device_cache()

        info = get_device_info()

        assert info["selected_device"] == "cuda"
        assert info["cuda_available"] is True
        assert info["cuda_version"] == "11.8"
        assert info["cuda_device_count"] == 2
        assert info["cuda_device_name"] == "NVIDIA GeForce RTX 3090"

    @patch.dict(os.environ, {}, clear=True)
    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    @patch("torch.backends.mps.is_built", return_value=True)
    def test_mps_info(
        self,
        mock_built: MagicMock,
        mock_mps: MagicMock,
        mock_cuda: MagicMock,
    ) -> None:
        """Should return MPS info when MPS is available."""
        from image_search_service.core.device import clear_device_cache, get_device_info

        clear_device_cache()

        info = get_device_info()

        assert info["selected_device"] == "mps"
        assert info["cuda_available"] is False
        assert info["mps_available"] is True
        assert info["mps_built"] is True

    @patch.dict(os.environ, {"FORCE_CPU": "true"})
    def test_info_cached(self) -> None:
        """Should cache device info across multiple calls."""
        from image_search_service.core.device import clear_device_cache, get_device_info

        clear_device_cache()

        # First call
        info1 = get_device_info()

        # Second call should use cache
        info2 = get_device_info()

        # Should return same object from cache
        assert info1 is info2


class TestGetOnnxProviders:
    """Tests for get_onnx_providers() function."""

    def test_returns_list(self) -> None:
        """Should return a list of providers."""
        from image_search_service.core.device import get_onnx_providers

        providers = get_onnx_providers()

        assert isinstance(providers, list)
        # Should at least have CPU provider
        assert "CPUExecutionProvider" in providers

    @pytest.mark.parametrize(
        "available,expected",
        [
            pytest.param(
                ["CPUExecutionProvider", "CUDAExecutionProvider", "CoreMLExecutionProvider"],
                ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"],
                id="cuda-coreml-cpu",
            ),
            pytest.param(
                ["CPUExecutionProvider", "CUDAExecutionProvider"],
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
                id="cuda-cpu",
            ),
            pytest.param(
                ["CPUExecutionProvider", "CoreMLExecutionProvider"],
                ["CoreMLExecutionProvider", "CPUExecutionProvider"],
                id="coreml-cpu",
            ),
            pytest.param(
                ["CPUExecutionProvider"],
                ["CPUExecutionProvider"],
                id="cpu-only",
            ),
        ],
    )
    def test_provider_priority_order(
        self, available: list[str], expected: list[str]
    ) -> None:
        """Should return providers in priority order (CUDA > CoreML > CPU)."""
        from image_search_service.core.device import get_onnx_providers

        with patch("onnxruntime.get_available_providers", return_value=available):
            assert get_onnx_providers() == expected

    def test_onnxruntime_not_installed(self) -> None:
        """Should return CPU provider when onnxruntime not installed."""
        from image_search_service.core.device import get_onnx_providers

        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'onnxruntime'")
        ):
            providers = get_onnx_providers()
            assert providers == ["CPUExecutionProvider"]


class TestIsAppleSilicon:
    """Tests for is_apple_silicon() function."""

    @pytest.mark.parametrize(
        "system,machine,expected",
        [
            pytest.param("Darwin", "arm64", True, id="apple-silicon"),
            pytest.param("Darwin", "x86_64", False, id="intel-mac"),
            pytest.param("Linux", "x86_64", False, id="linux-x86"),
            pytest.param("Linux", "arm64", False, id="linux-arm64"),
            pytest.param("Windows", "AMD64", False, id="windows"),
        ],
    )
    def test_is_apple_silicon(
        self, system: str, machine: str, expected: bool
    ) -> None:
        """Should detect Apple Silicon based on platform and architecture."""
        from image_search_service.core.device import is_apple_silicon

        with (
            patch("platform.system", return_value=system),
            patch("platform.machine", return_value=machine),
        ):
            assert is_apple_silicon() is expected


class TestClearDeviceCache:
    """Tests for clear_device_cache() function."""

    @patch.dict(os.environ, {"FORCE_CPU": "true"})
    def test_clears_get_device_cache(self) -> None:
        """Should clear get_device() cache."""
        from image_search_service.core.device import clear_device_cache, get_device

        # Prime the cache
        clear_device_cache()
        device1 = get_device()

        # Clear cache
        clear_device_cache()

        # Get device again - should be cpu due to FORCE_CPU
        device2 = get_device()

        # Both should return cpu
        assert device1 == "cpu"
        assert device2 == "cpu"

    @patch.dict(os.environ, {"FORCE_CPU": "true"})
    def test_clears_get_device_info_cache(self) -> None:
        """Should clear get_device_info() cache."""
        from image_search_service.core.device import (
            clear_device_cache,
            get_device_info,
        )

        # Prime the cache
        clear_device_cache()
        info1 = get_device_info()

        # Clear cache
        clear_device_cache()

        # Get info again
        info2 = get_device_info()

        # Should be different objects (cache was cleared)
        assert info1 is not info2
