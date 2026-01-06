"""Centralized device management for ML inference.

Device Selection Priority:
1. DEVICE environment variable (explicit override)
2. FORCE_CPU=true (force CPU mode)
3. Auto-detect: CUDA > MPS > CPU

This module provides device abstraction for PyTorch and ONNX Runtime,
supporting CUDA (NVIDIA), MPS (Apple Silicon), and CPU fallback.

MPS Known Issues & Workarounds:
- PyTorch MPS has broken memory accounting on macOS (reports exabytes when only GBs used)
- Solution: Disable high watermark check (official PyTorch workaround)
- See: https://github.com/pytorch/pytorch/issues
"""

import os
import platform
from functools import lru_cache
from typing import Any

import torch


def _initialize_mps_workarounds() -> None:
    """Initialize MPS workarounds for known PyTorch bugs on macOS.

    PyTorch MPS backend has broken memory accounting that incorrectly reports
    "other allocations" in the exabytes range when only gigabytes are actually
    used. This causes the high watermark check to fail even when memory is
    available.

    The official PyTorch workaround is to disable the high watermark check
    by setting PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0. This does not remove
    memory limits - the OS still enforces hard memory caps.

    This must be called before any GPU operations occur.

    See: https://github.com/pytorch/pytorch/issues (MPS memory accounting)
    """
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return

    # Disable MPS high watermark to work around broken memory accounting
    # The high watermark check was failing due to incorrect memory reporting,
    # not actual memory shortage. Disabling it allows GPU operations to proceed.
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


# Initialize MPS workarounds at module import time (before any GPU operations)
_initialize_mps_workarounds()


@lru_cache(maxsize=1)
def get_device() -> str:
    """Get the best available PyTorch device.

    Priority:
    1. DEVICE env var (explicit override)
    2. FORCE_CPU env var (forces CPU)
    3. CUDA (if available)
    4. MPS (if available on Apple Silicon)
    5. CPU (fallback)

    Returns:
        Device string: "cuda", "cuda:0", "mps", or "cpu"

    Raises:
        ValueError: If DEVICE env var specifies invalid device
    """
    # Priority 1: Explicit device override
    if device := os.getenv("DEVICE"):
        if device.lower() != "auto":
            _validate_device(device)
            return device

    # Priority 2: Force CPU mode
    if os.getenv("FORCE_CPU", "").lower() in ("true", "1", "yes"):
        return "cpu"

    # Priority 3: Auto-detect best available
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def _validate_device(device: str) -> None:
    """Validate device string and raise if invalid.

    Args:
        device: Device string to validate (e.g., "cuda", "cuda:0", "mps", "cpu")

    Raises:
        ValueError: If device string is invalid or device not available
    """
    valid_prefixes = ("cuda", "mps", "cpu")
    if not any(device.startswith(prefix) for prefix in valid_prefixes):
        raise ValueError(f"Invalid DEVICE '{device}'. Must be 'cuda', 'cuda:N', 'mps', or 'cpu'")

    # Validate CUDA device ID if specified
    if device.startswith("cuda:"):
        try:
            device_id = int(device.split(":")[1])
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if device_id >= device_count:
                    raise ValueError(
                        f"CUDA device {device_id} not available. "
                        f"Only {device_count} device(s) found."
                    )
        except (ValueError, IndexError):
            raise ValueError(f"Invalid CUDA device ID in '{device}'")


def clear_device_cache() -> None:
    """Clear the cached device selection. Useful for testing."""
    get_device.cache_clear()
    get_device_info.cache_clear()


@lru_cache(maxsize=1)
def get_device_info() -> dict[str, Any]:
    """Get comprehensive device and platform information.

    Returns:
        Dictionary containing:
        - platform: OS name (Linux, Darwin, Windows)
        - machine: Architecture (x86_64, arm64, etc.)
        - python_version: Python version string
        - pytorch_version: PyTorch version string
        - selected_device: Device selected by get_device()
        - cuda_available: Whether CUDA is available
        - cuda_version: CUDA version (if available)
        - cuda_device_count: Number of CUDA devices (if available)
        - cuda_device_name: Name of first CUDA device (if available)
        - mps_built: Whether PyTorch was built with MPS support
        - mps_available: Whether MPS is available for use
    """
    info: dict[str, Any] = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "selected_device": get_device(),
    }

    # CUDA information
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)

    # MPS information (Apple Silicon)
    if hasattr(torch.backends, "mps"):
        info["mps_built"] = torch.backends.mps.is_built()
        info["mps_available"] = torch.backends.mps.is_available()
    else:
        info["mps_built"] = False
        info["mps_available"] = False

    return info


def get_onnx_providers() -> list[str]:
    """Get ONNX Runtime execution providers in priority order.

    Returns:
        List of ONNX Runtime providers in priority order:
        - CUDAExecutionProvider (NVIDIA GPUs)
        - CoreMLExecutionProvider (Apple Silicon)
        - CPUExecutionProvider (fallback)

    Note:
        Returns ["CPUExecutionProvider"] if onnxruntime is not installed.
    """
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
    except ImportError:
        return ["CPUExecutionProvider"]

    # Priority order: CUDA > CoreML > CPU
    priority = [
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]

    return [p for p in priority if p in available]


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon (M1/M2/M3/M4).

    Returns:
        True if running on macOS with ARM64 architecture, False otherwise
    """
    return platform.system() == "Darwin" and platform.machine() == "arm64"
