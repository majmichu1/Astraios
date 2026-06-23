"""Histogram Transformation — interactive black/mid/white point adjustment.

GPU-accelerated via the device manager.
Uses the existing midtone transfer function from stretch.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from astraios.core.device_manager import get_device_manager
from astraios.core.masks import Mask, apply_mask
from astraios.core.stretch import midtone_transfer_function

log = logging.getLogger(__name__)


@dataclass
class HistogramTransformParams:
    """Parameters for histogram transformation."""

    black_point: float = 0.0
    midtone: float = 0.5
    white_point: float = 1.0
    linked: bool = True


def histogram_transform(
    image: np.ndarray,
    params: HistogramTransformParams | None = None,
    mask: Mask | None = None,
) -> np.ndarray:
    """Apply histogram transformation — GPU-accelerated."""
    if params is None:
        params = HistogramTransformParams()

    dm = get_device_manager()
    if dm.is_gpu:
        return _ht_gpu(image, params, mask, dm)
    else:
        return _ht_cpu(image, params, mask)


@torch.no_grad()
def _ht_gpu(
    image: np.ndarray,
    params: HistogramTransformParams,
    mask: Mask | None,
    dm,
) -> np.ndarray:
    """GPU-accelerated histogram transform — stays on GPU throughout."""
    # no copy: op never mutates the input; apply_mask reads image directly
    t = dm.from_numpy(image)  # CPU → GPU (once)
    bp = params.black_point
    wp = params.white_point
    mt = params.midtone

    range_width = max(wp - bp, 1e-10)
    rescaled = torch.clamp((t - bp) / range_width, 0.0, 1.0)

    if abs(mt - 0.5) > 1e-6:
        # MTF entirely on GPU — no round-trip to CPU
        nonzero = rescaled > 0
        denom = (2.0 * mt - 1.0) * rescaled - mt
        safe = nonzero & (torch.abs(denom) > 1e-10)
        result_t = torch.zeros_like(rescaled)
        result_t[safe] = (mt - 1.0) * rescaled[safe] / denom[safe]
        result_t = torch.clamp(result_t, 0.0, 1.0)
    else:
        result_t = rescaled

    result = result_t.cpu().numpy().astype(np.float32)  # GPU → CPU (once, at end)
    return apply_mask(image, result, mask)


def _ht_cpu(
    image: np.ndarray,
    params: HistogramTransformParams,
    mask: Mask | None,
) -> np.ndarray:
    """CPU fallback for histogram transform."""
    # no copy: op never mutates the input; apply_mask reads image directly
    bp = params.black_point
    wp = params.white_point
    mt = params.midtone

    if image.ndim == 2:
        result = _transform_channel(image, bp, wp, mt)
    else:
        result = np.empty_like(image)
        for ch in range(image.shape[0]):
            result[ch] = _transform_channel(image[ch], bp, wp, mt)

    return apply_mask(image, result, mask)


def _transform_channel(
    channel: np.ndarray,
    black_point: float,
    white_point: float,
    midtone: float,
) -> np.ndarray:
    """Apply histogram transform to a single channel."""
    range_width = max(white_point - black_point, 1e-10)
    rescaled = (channel - black_point) / range_width
    rescaled = np.clip(rescaled, 0, 1)

    if abs(midtone - 0.5) > 1e-6:
        return midtone_transfer_function(rescaled, midtone)
    return rescaled
