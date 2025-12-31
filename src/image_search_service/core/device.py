"""Centralized device management for ML inference.

Device Selection Priority:
1. DEVICE environment variable (explicit override)
2. FORCE_CPU=true (force CPU mode)
3. RQ worker subprocess context (force CPU if MPS would be selected)
4. Auto-detect: CUDA > MPS > CPU

This module provides device abstraction for PyTorch and ONNX Runtime,
supporting CUDA (NVIDIA), MPS (Apple Silicon), and CPU fallback.

Note: MPS is disabled in RQ worker subprocesses on macOS because Apple's
Metal Performance Shaders do not work reliably in forked subprocess contexts.
"""

import multiprocessing
import os
import platform
from functools import lru_cache
from typing import Any

import torch

from image_search_service.core.logging import get_logger

logger = get_logger(__name__)


def is_rq_worker() -> bool:
    """Detect if running in RQ worker subprocess context.

    RQ workers use fork() on Unix systems, which creates a subprocess
    with a different process name. This is unreliable for MPS on macOS.

    Returns:
        True if running in RQ worker subprocess, False otherwise
    """
    try:
        current_proc = multiprocessing.current_process()
        # RQ worker subprocesses have different process names
        # Main process is typically "MainProcess"
        # Worker subprocesses are named "Process-1", "Process-2", etc.
        # or "ForkProcess-1", etc.
        is_subprocess = (
            current_proc.name != "MainProcess"
            and (
                current_proc.name.startswith("Process-")
                or current_proc.name.startswith("ForkProcess-")
                or current_proc.name.startswith("SpawnProcess-")
            )
        )
        return is_subprocess
    except Exception:
        # If we can't determine, assume not in worker
        return False


@lru_cache(maxsize=1)
def get_device() -> str:
    """Get the best available PyTorch device.

    Priority:
    1. DEVICE env var (explicit override)
    2. FORCE_CPU env var (forces CPU)
    3. RQ worker subprocess check (disable MPS if in worker on Apple Silicon)
    4. CUDA (if available)
    5. MPS (if available on Apple Silicon and not in RQ worker)
    6. CPU (fallback)

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

    # Priority 3: Check if in RQ worker subprocess on Apple Silicon
    # MPS doesn't work reliably in forked subprocesses on macOS
    if is_rq_worker() and is_apple_silicon():
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if mps_available:
            logger.info(
                "Detected RQ worker subprocess on Apple Silicon. "
                "Disabling MPS (not reliable in forked processes). Using CPU instead."
            )
            return "cpu"

    # Priority 4: Auto-detect best available
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
