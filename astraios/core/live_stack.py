"""Live stacking — real-time frame accumulation and alignment."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from queue import Queue

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)


@dataclass
class LiveStacker:
    """Simple live stacking engine.

    Accumulates aligned frames, applies rejection, and provides
    a live preview of the current stack.
    """

    reference: NDArray | None = field(default=None, repr=False)
    stack_sum: NDArray | None = field(default=None, repr=False)
    stack_count: NDArray = field(default=None, repr=False)
    n_frames: int = 0
    alignment_mode: str = "fft"
    max_frames: int = 0

    def __post_init__(self):
        self._lock = threading.Lock()
        self._frame_queue: Queue = Queue()
        self._running = False
        self._thread: threading.Thread | None = None

    def set_reference(self, frame: NDArray):
        with self._lock:
            self._set_reference_locked(frame)

    def _set_reference_locked(self, frame: NDArray):
        """Set the reference. Caller must hold self._lock.

        add_frame used to call the public set_reference() while already
        holding the (non-reentrant) lock — the very first add_frame() on a
        fresh stacker deadlocked forever, which made Live Stacking hang on
        frame one since the dialog never seeds the reference itself.
        """
        self.reference = frame.astype(np.float32)
        self.stack_sum = np.zeros_like(self.reference)
        self.stack_count = np.zeros(self.reference.shape[-2:], dtype=np.int32)
        self.n_frames = 0

    def add_frame(self, frame: NDArray):
        with self._lock:
            first = self.reference is None
            if first:
                self._set_reference_locked(frame)

            aligned = frame.astype(np.float32)
            # The first frame IS the reference: no alignment needed, but its
            # signal belongs in the stack (it used to be silently dropped).
            if (not first and self.alignment_mode == "fft"
                    and frame.shape == self.reference.shape):
                from skimage.registration import phase_cross_correlation
                try:
                    shift, _, _ = phase_cross_correlation(
                        self.reference if self.reference.ndim == 2 else self.reference[0],
                        aligned if aligned.ndim == 2 else aligned[0],
                        upsample_factor=10,
                    )
                    from astraios.core.channel_match import _apply_shift
                    aligned = _apply_shift(aligned, shift[0], shift[1])
                except Exception:
                    pass

            aligned = np.clip(aligned, 0, None)
            self.stack_sum += aligned
            self.stack_count += 1
            self.n_frames += 1

    def get_live_preview(self) -> NDArray:
        with self._lock:
            if self.stack_sum is None or self.n_frames == 0:
                return np.zeros((100, 100), dtype=np.float32)
            result = self.stack_sum / max(self.n_frames, 1)

        from astraios.core.stretch import ArcsinhStretchParams, arcsinh_stretch
        return arcsinh_stretch(
            np.clip(result, 0, 1),
            ArcsinhStretchParams(stretch_factor=10.0, black_point=0.001),
        )

    def get_result(self) -> NDArray | None:
        with self._lock:
            if self.stack_sum is None or self.n_frames == 0:
                return None
            return (self.stack_sum / max(self.n_frames, 1)).astype(np.float32)

    def reset(self):
        with self._lock:
            self.reference = None
            self.stack_sum = None
            self.stack_count = None
            self.n_frames = 0
