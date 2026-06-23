"""Frequency Separation — split an image into low- and high-frequency layers.

A staple of high-end retouching (and Seti Astro Suite): the low-frequency (LF)
layer holds large-scale structure, colour, and gradients; the high-frequency
(HF) layer holds fine detail, edges, and star cores. Working on them
independently lets you, e.g., smooth colour mottle / gradients in the LF without
touching detail, or boost detail in the HF without amplifying colour noise.

Two split models:
- ``subtract`` (linear): ``HF = image - LF``;   recombine = ``LF + HF``
- ``divide`` (ratio):    ``HF = image / LF``;   recombine = ``LF * HF``

The divide model is brightness-invariant (detail contrast is preserved equally
in shadows and highlights) — often preferable for stretched astro data.

All blurring runs on the GPU via :func:`astraios.core.filters._gaussian_blur_gpu`
with an automatic CPU fallback.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from astraios.core.device_manager import get_device_manager
from astraios.core.filters import _gaussian_blur_gpu
from astraios.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]

# Floor for the divide model so we never divide by ~0 in the background.
_DIVIDE_EPS = 1e-4


def _noop_progress(fraction: float, message: str) -> None:
    pass


class SeparationMethod(Enum):
    SUBTRACT = auto()  # linear: HF = image - LF
    DIVIDE = auto()    # ratio:  HF = image / LF


@dataclass
class FrequencySeparationParams:
    """Parameters for frequency-separation processing.

    Attributes:
        sigma: Gaussian blur radius (pixels) defining the LF/HF boundary.
            Larger = more detail pushed into the HF layer.
        method: ``SUBTRACT`` (linear) or ``DIVIDE`` (ratio).
        hf_boost: Multiplier applied to the HF layer before recombining.
            >1 sharpens/enhances detail, <1 softens. 1.0 = no change.
        lf_smooth: Extra Gaussian smoothing (pixels) applied to the LF layer
            before recombining — knocks down colour mottle / gradients.
            0 = leave LF untouched.
    """

    sigma: float = 5.0
    method: SeparationMethod = SeparationMethod.SUBTRACT
    hf_boost: float = 1.0
    lf_smooth: float = 0.0


def _blur(channel: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur a single 2-D channel (GPU with CPU fallback)."""
    if sigma <= 0:
        return channel.astype(np.float32, copy=True)
    dm = get_device_manager()
    if dm.device.type != "cpu":
        return _gaussian_blur_gpu(np.ascontiguousarray(channel), sigma, dm)
    import cv2

    ksize = int(np.ceil(sigma * 3)) * 2 + 1
    return cv2.GaussianBlur(channel, (ksize, ksize), sigma).astype(np.float32)


def _iter_channels(image: np.ndarray):
    """Yield (index, 2-D channel) for mono (H,W) or colour (C,H,W) images."""
    if image.ndim == 2:
        yield None, image
    else:
        for c in range(image.shape[0]):
            yield c, image[c]


def separate(
    image: np.ndarray,
    sigma: float = 5.0,
    method: SeparationMethod = SeparationMethod.SUBTRACT,
) -> tuple[np.ndarray, np.ndarray]:
    """Split an image into (low_frequency, high_frequency) layers.

    Args:
        image: ``(H, W)`` or ``(C, H, W)`` float32 in ``[0, 1]``.
        sigma: Blur radius defining the split.
        method: Separation model.

    Returns:
        ``(low, high)`` arrays, same shape/dtype as ``image``. For ``SUBTRACT``
        the HF layer is centred on 0; for ``DIVIDE`` it is centred on 1.
    """
    low = np.empty_like(image, dtype=np.float32)
    for idx, ch in _iter_channels(image):
        blurred = _blur(ch, sigma)
        if idx is None:
            low = blurred
        else:
            low[idx] = blurred

    if method == SeparationMethod.DIVIDE:
        high = image.astype(np.float32) / np.maximum(low, _DIVIDE_EPS)
    else:
        high = image.astype(np.float32) - low
    return low.astype(np.float32), high.astype(np.float32)


def recombine(
    low: np.ndarray,
    high: np.ndarray,
    method: SeparationMethod = SeparationMethod.SUBTRACT,
) -> np.ndarray:
    """Recombine LF and HF layers back into an image.

    Inverse of :func:`separate` for the same ``method``. Result is clipped to
    ``[0, 1]``.
    """
    out = low * high if method == SeparationMethod.DIVIDE else low + high
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def frequency_separation(
    image: np.ndarray,
    params: FrequencySeparationParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    """Single-pass frequency-separation enhancement.

    Splits the image, applies ``hf_boost`` to the detail layer and optional
    ``lf_smooth`` to the structure layer, then recombines. With defaults
    (``hf_boost=1``, ``lf_smooth=0``) this is a no-op round-trip, so any visible
    change comes only from the user's settings.

    Args:
        image: ``(H, W)`` or ``(C, H, W)`` float32 in ``[0, 1]``.
        params: Processing parameters.
        mask: Optional protection mask (applied at the end).
        progress: Optional progress callback.

    Returns:
        Processed image, same shape, clipped to ``[0, 1]``.
    """
    if params is None:
        params = FrequencySeparationParams()

    # no copy: op never mutates the input; apply_mask reads image directly
    progress(0.1, "Separating frequencies…")
    low, high = separate(image, params.sigma, params.method)

    if params.lf_smooth > 0:
        progress(0.4, "Smoothing low-frequency layer…")
        for idx, ch in _iter_channels(low):
            blurred = _blur(ch, params.lf_smooth)
            if idx is None:
                low = blurred
            else:
                low[idx] = blurred

    if params.hf_boost != 1.0:
        progress(0.7, "Boosting high-frequency detail…")
        if params.method == SeparationMethod.DIVIDE:
            # HF is centred on 1.0; scale its deviation from 1.
            high = 1.0 + (high - 1.0) * params.hf_boost
        else:
            high = high * params.hf_boost

    progress(0.9, "Recombining…")
    result = recombine(low, high, params.method)
    progress(1.0, "Frequency separation complete")
    return apply_mask(image, result, mask)
