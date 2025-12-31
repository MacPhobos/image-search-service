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

    @patch.dict(os.environ, {}, clear=True)
    @patch("torch.cuda.is_available", return_value=True)
    def test_cuda_available(self, mock_cuda: MagicMock) -> None:
        """Should return cuda when CUDA is available."""
        from image_search_service.core.device import clear_device_cache, get_device

        clear_device_cache()
        assert get_device() == "cuda"

    @patch.dict(os.environ, {}, clear=True)
    @patch("torch.cuda.is_available", return_value=False)
    def test_mps_fallback(self, mock_cuda: MagicMock) -> None:
        """Should return mps when CUDA unavailable but MPS available."""
        from image_search_service.core.device import clear_device_cache, get_device

        clear_device_cache()

        # Mock MPS availability
        with patch("torch.backends.mps.is_available", return_value=True):
            assert get_device() == "mps"

    @patch.dict(os.environ, {}, clear=True)
    @patch("torch.cuda.is_available", return_value=False)
    def test_cpu_fallback(self, mock_cuda: MagicMock) -> None:
        """Should return cpu when no GPU available."""
        from image_search_service.core.device import clear_device_cache, get_device

        clear_device_cache()

        # Mock MPS as unavailable
        with patch("torch.backends.mps.is_available", return_value=False):
            assert get_device() == "cpu"

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

    @patch.dict(os.environ, {"FORCE_CPU": "true"})
    @patch("torch.cuda.is_available", return_value=True)
    def test_force_cpu_true(self, mock_cuda: MagicMock) -> None:
        """Should force CPU when FORCE_CPU=true even if CUDA available."""
        from image_search_service.core.device import clear_device_cache, get_device

        clear_device_cache()
        assert get_device() == "cpu"

    @patch.dict(os.environ, {"FORCE_CPU": "1"})
    def test_force_cpu_one(self) -> None:
        """Should force CPU when FORCE_CPU=1."""
        from image_search_service.core.device import clear_device_cache, get_device

        clear_device_cache()
        assert get_device() == "cpu"

    @patch.dict(os.environ, {"FORCE_CPU": "yes"})
    def test_force_cpu_yes(self) -> None:
        """Should force CPU when FORCE_CPU=yes."""
        from image_search_service.core.device import clear_device_cache, get_device

        clear_device_cache()
        assert get_device() == "cpu"

    @patch.dict(os.environ, {"DEVICE": "invalid_device"})
    def test_invalid_device_raises(self) -> None:
        """Should raise ValueError for invalid device."""
        from image_search_service.core.device import clear_device_cache, get_device

        clear_device_cache()
        with pytest.raises(ValueError, match="Invalid DEVICE"):
            get_device()

    @patch.dict(os.environ, {"DEVICE": "cuda:5"})
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=2)
    def test_invalid_cuda_device_id_raises(
        self, mock_count: MagicMock, mock_cuda: MagicMock
    ) -> None:
        """Should raise ValueError for invalid CUDA device ID."""
        from image_search_service.core.device import clear_device_cache, get_device

        clear_device_cache()
        with pytest.raises(ValueError, match="Invalid CUDA device ID"):
            get_device()

    @patch.dict(os.environ, {"DEVICE": "cuda:abc"})
    def test_invalid_cuda_device_format_raises(self) -> None:
        """Should raise ValueError for invalid CUDA device format."""
        from image_search_service.core.device import clear_device_cache, get_device

        clear_device_cache()
        with pytest.raises(ValueError, match="Invalid CUDA device ID"):
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

    @patch("onnxruntime.get_available_providers")
    def test_priority_order_cuda_coreml(self, mock_providers: MagicMock) -> None:
        """Should return providers in priority order (CUDA > CoreML > CPU)."""
        from image_search_service.core.device import get_onnx_providers

        mock_providers.return_value = [
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
        ]

        providers = get_onnx_providers()

        # CUDA should come before CoreML, CoreML before CPU
        cuda_idx = providers.index("CUDAExecutionProvider")
        coreml_idx = providers.index("CoreMLExecutionProvider")
        cpu_idx = providers.index("CPUExecutionProvider")

        assert cuda_idx < coreml_idx < cpu_idx

    @patch("onnxruntime.get_available_providers")
    def test_priority_order_cuda_only(self, mock_providers: MagicMock) -> None:
        """Should return CUDA before CPU when CoreML unavailable."""
        from image_search_service.core.device import get_onnx_providers

        mock_providers.return_value = [
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
        ]

        providers = get_onnx_providers()

        assert providers == [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    @patch("onnxruntime.get_available_providers")
    def test_priority_order_coreml_only(self, mock_providers: MagicMock) -> None:
        """Should return CoreML before CPU when CUDA unavailable."""
        from image_search_service.core.device import get_onnx_providers

        mock_providers.return_value = [
            "CPUExecutionProvider",
            "CoreMLExecutionProvider",
        ]

        providers = get_onnx_providers()

        assert providers == [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]

    @patch("onnxruntime.get_available_providers")
    def test_cpu_only(self, mock_providers: MagicMock) -> None:
        """Should return only CPU when no GPU providers available."""
        from image_search_service.core.device import get_onnx_providers

        mock_providers.return_value = ["CPUExecutionProvider"]

        providers = get_onnx_providers()

        assert providers == ["CPUExecutionProvider"]

    def test_onnxruntime_not_installed(self) -> None:
        """Should return CPU provider when onnxruntime not installed."""
        from image_search_service.core.device import get_onnx_providers

        # Mock ImportError by patching the import inside get_onnx_providers
        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'onnxruntime'")
        ):
            providers = get_onnx_providers()
            assert providers == ["CPUExecutionProvider"]


class TestIsAppleSilicon:
    """Tests for is_apple_silicon() function."""

    @patch("platform.system", return_value="Darwin")
    @patch("platform.machine", return_value="arm64")
    def test_apple_silicon_detected(
        self, mock_machine: MagicMock, mock_system: MagicMock
    ) -> None:
        """Should return True on Apple Silicon."""
        from image_search_service.core.device import is_apple_silicon

        assert is_apple_silicon() is True

    @patch("platform.system", return_value="Darwin")
    @patch("platform.machine", return_value="x86_64")
    def test_macos_intel_not_apple_silicon(
        self, mock_machine: MagicMock, mock_system: MagicMock
    ) -> None:
        """Should return False on Intel Mac."""
        from image_search_service.core.device import is_apple_silicon

        assert is_apple_silicon() is False

    @patch("platform.system", return_value="Linux")
    @patch("platform.machine", return_value="x86_64")
    def test_linux_not_apple_silicon(
        self, mock_machine: MagicMock, mock_system: MagicMock
    ) -> None:
        """Should return False on Linux."""
        from image_search_service.core.device import is_apple_silicon

        assert is_apple_silicon() is False

    @patch("platform.system", return_value="Linux")
    @patch("platform.machine", return_value="arm64")
    def test_linux_arm64_not_apple_silicon(
        self, mock_machine: MagicMock, mock_system: MagicMock
    ) -> None:
        """Should return False on Linux ARM64 (not macOS)."""
        from image_search_service.core.device import is_apple_silicon

        assert is_apple_silicon() is False

    @patch("platform.system", return_value="Windows")
    @patch("platform.machine", return_value="AMD64")
    def test_windows_not_apple_silicon(
        self, mock_machine: MagicMock, mock_system: MagicMock
    ) -> None:
        """Should return False on Windows."""
        from image_search_service.core.device import is_apple_silicon

        assert is_apple_silicon() is False


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
