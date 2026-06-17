"""GPU Device Manager — abstracts CUDA, MPS, and CPU backends.

All GPU code in Astraios MUST go through this module.
Never call torch.cuda.* directly elsewhere in the codebase.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


class Backend(Enum):
    CUDA = auto()
    MPS = auto()
    CPU = auto()


@dataclass(frozen=True)
class DeviceInfo:
    backend: Backend
    device: torch.device
    name: str
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    compute_capability: tuple[int, int] | None = None


@dataclass
class DeviceManager:
    """Singleton-style manager for GPU device selection and tensor allocation."""

    _device: torch.device = field(init=False)
    _backend: Backend = field(init=False)
    _info: DeviceInfo = field(init=False)

    def __post_init__(self) -> None:
        self._detect_best_device()

    def _detect_best_device(self) -> None:
        if torch.cuda.is_available():
            self._backend = Backend.CUDA
            self._device = torch.device("cuda", 0)
            props = torch.cuda.get_device_properties(0)
            total_mem = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
            total_mb = total_mem // (1024 * 1024)
            free_mb = total_mb  # approximate at init
            try:
                free_mb = torch.cuda.mem_get_info(0)[0] // (1024 * 1024)
            except Exception:
                log.debug("CUDA mem_get_info unavailable")
            cc = (props.major, props.minor)
            self._info = DeviceInfo(
                backend=Backend.CUDA,
                device=self._device,
                name=props.name,
                vram_total_mb=total_mb,
                vram_free_mb=free_mb,
                compute_capability=cc,
            )
            log.info("CUDA device: %s (%d MB VRAM, CC %d.%d)", props.name, total_mb, *cc)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._backend = Backend.MPS
            self._device = torch.device("mps")
            self._info = DeviceInfo(
                backend=Backend.MPS,
                device=self._device,
                name="Apple Silicon (MPS)",
            )
            log.info("MPS device detected (Apple Silicon)")
        else:
            self._backend = Backend.CPU
            self._device = torch.device("cpu")
            self._info = DeviceInfo(
                backend=Backend.CPU,
                device=self._device,
                name="CPU",
            )
            log.info("No GPU detected, using CPU fallback")

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def info(self) -> DeviceInfo:
        return self._info

    @property
    def is_gpu(self) -> bool:
        return self._backend != Backend.CPU

    def tensor(self, *args, dtype: torch.dtype = torch.float32, **kwargs) -> torch.Tensor:
        """Create a tensor on the managed device."""
        return torch.tensor(*args, dtype=dtype, device=self._device, **kwargs)

    def zeros(self, *shape, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros(*shape, dtype=dtype, device=self._device)

    def ones(self, *shape, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.ones(*shape, dtype=dtype, device=self._device)

    def empty(self, *shape, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.empty(*shape, dtype=dtype, device=self._device)

    def to_device(self, t: torch.Tensor) -> torch.Tensor:
        """Move a tensor to the managed device."""
        return t.to(self._device)

    def to_cpu(self, t: torch.Tensor) -> torch.Tensor:
        """Move a tensor to CPU (for numpy interop, saving, etc.)."""
        return t.cpu()

    def from_numpy(self, arr) -> torch.Tensor:
        """Convert a numpy array to a device tensor."""
        return torch.from_numpy(arr).to(self._device)

    def synchronize(self) -> None:
        """Wait for all operations on the device to complete."""
        if self._backend == Backend.CUDA:
            torch.cuda.synchronize(self._device)
        elif self._backend == Backend.MPS:
            torch.mps.synchronize()

    def empty_cache(self) -> None:
        """Free cached GPU memory."""
        if self._backend == Backend.CUDA:
            torch.cuda.empty_cache()
        elif self._backend == Backend.MPS:
            torch.mps.empty_cache()

    def memory_stats(self) -> dict:
        """Return memory usage statistics."""
        if self._backend == Backend.CUDA:
            free, total = torch.cuda.mem_get_info(self._device)
            allocated = torch.cuda.memory_allocated(self._device)
            return {
                "total_mb": total / (1024 * 1024),
                "free_mb": free / (1024 * 1024),
                "allocated_mb": allocated / (1024 * 1024),
            }
        return {"total_mb": 0, "free_mb": 0, "allocated_mb": 0}

    def optimal_batch_size(self, frame_bytes: int, overhead_factor: float = 2.5) -> int:
        """Estimate how many frames of a given size fit in GPU memory."""
        if not self.is_gpu:
            return 8  # conservative CPU default
        stats = self.memory_stats()
        available = stats.get("free_mb", 2048) * 1024 * 1024
        per_frame = frame_bytes * overhead_factor
        if per_frame <= 0:
            return 1
        return max(1, int(available / per_frame))


# Module-level singleton
_instance: DeviceManager | None = None
_lock: threading.Lock | None = None


def get_device_manager() -> DeviceManager:
    """Return the global DeviceManager singleton."""
    global _instance, _lock
    if _lock is None:
        _lock = threading.Lock()
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = DeviceManager()
    return _instance
