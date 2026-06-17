"""Cosmetic Correction — hot pixel, cold pixel, and dead pixel removal.

Detects and repairs sensor defects using local neighborhood statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from astraios.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


@dataclass
class CosmeticParams:
    """Parameters for cosmetic correction."""

    hot_sigma: float = 5.0  # sigma threshold for hot pixel detection
    cold_sigma: float = 5.0  # sigma threshold for cold pixel detection
    detect_dead: bool = True  # detect pixels with exactly zero value
    kernel_size: int = 5  # neighborhood kernel size for median comparison


@dataclass
class CosmeticResult:
    """Result of cosmetic correction."""

    data: np.ndarray
    hot_pixels: int = 0
    cold_pixels: int = 0
    dead_pixels: int = 0

    @property
    def total_corrected(self) -> int:
        return self.hot_pixels + self.cold_pixels + self.dead_pixels


def cosmetic_correction(
    image: np.ndarray,
    params: CosmeticParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback = _noop_progress,
) -> CosmeticResult:
    """Detect and repair cosmetic defects in an image.

    Parameters
    ----------
    image : ndarray
        Image data, shape (H, W) or (C, H, W), values in [0, 1].
    params : CosmeticParams, optional
        Detection parameters.
    mask : Mask, optional
        If provided, only correct pixels within the mask.
    progress : callable
        Progress callback.

    Returns
    -------
    CosmeticResult
        Corrected image and defect counts.
    """
    if params is None:
        params = CosmeticParams()

    original = image.copy()
    result = image.copy()
    total_hot = 0
    total_cold = 0
    total_dead = 0

    # Process each channel
    channels = _get_channels(result)
    n_ch = len(channels)

    for ch_idx, (ch_data, setter) in enumerate(channels):
        progress(ch_idx / n_ch * 0.8, f"Correcting channel {ch_idx + 1}/{n_ch}...")

        # Compute local median
        ksize = params.kernel_size
        local_median = cv2.medianBlur(ch_data, ksize)

        # Deviation from local median
        diff = ch_data - local_median

        # Statistics of the deviation
        med_diff = np.median(diff)
        mad_diff = np.median(np.abs(diff - med_diff))
        sigma = max(mad_diff * 1.4826, 1e-10)

        # Hot pixels: significantly brighter than neighbors
        hot_mask = diff > params.hot_sigma * sigma
        total_hot += int(hot_mask.sum())

        # Cold pixels: significantly darker than neighbors
        cold_mask = diff < -params.cold_sigma * sigma
        total_cold += int(cold_mask.sum())

        # Dead pixels: exactly zero
        dead_mask = np.zeros_like(hot_mask)
        if params.detect_dead:
            dead_mask = ch_data == 0.0
            total_dead += int(dead_mask.sum())

        # Replace defective pixels with local median
        defect_mask = hot_mask | cold_mask | dead_mask
        corrected = ch_data.copy()
        corrected[defect_mask] = local_median[defect_mask]
        setter(corrected)

    progress(0.9, "Applying mask...")
    result = apply_mask(original, result, mask)

    progress(1.0, f"Cosmetic correction: {total_hot} hot, {total_cold} cold, {total_dead} dead")
    log.info(
        "Cosmetic correction: %d hot, %d cold, %d dead pixels corrected",
        total_hot, total_cold, total_dead,
    )

    return CosmeticResult(
        data=result,
        hot_pixels=total_hot,
        cold_pixels=total_cold,
        dead_pixels=total_dead,
    )


def _get_channels(image: np.ndarray):
    """Return list of (channel_data, setter_function) for each channel."""
    if image.ndim == 2:
        def setter(data):
            image[:] = data
        return [(image, setter)]
    else:
        result = []
        for ch in range(image.shape[0]):
            def make_setter(c):
                def setter(data):
                    image[c] = data
                return setter
            result.append((image[ch], make_setter(ch)))
        return result
