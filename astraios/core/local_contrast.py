"""Local Contrast Enhancement — CLAHE and local histogram equalization.

Uses OpenCV CLAHE (Apache 2.0) applied to the luminance channel only,
preserving color information.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from astraios.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(f: float, m: str) -> None:
    pass


@dataclass
class LocalContrastParams:
    """Parameters for local contrast enhancement."""

    clip_limit: float = 2.0  # CLAHE contrast limit (1.0-10.0)
    tile_size: int = 8  # tile grid size (4-32)
    amount: float = 1.0  # blend amount (0-1)


def local_contrast_enhance(
    data: np.ndarray,
    params: LocalContrastParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    """Enhance local contrast using CLAHE on the luminance channel.

    For color images, applies CLAHE only to luminance (L in Lab color space),
    preserving chrominance. For mono, applies directly.

    Parameters
    ----------
    data : ndarray
        Image data, shape (H, W) or (C, H, W), float32 in [0, 1].
    params : LocalContrastParams, optional
        Enhancement parameters.
    mask : Mask, optional
        Processing mask.

    Returns
    -------
    ndarray
        Enhanced image.
    """
    if params is None:
        params = LocalContrastParams()

    # no copy: op never mutates the input; apply_mask reads data directly
    progress(0.1, "Building CLAHE…")

    clahe = cv2.createCLAHE(
        clipLimit=params.clip_limit,
        tileGridSize=(params.tile_size, params.tile_size),
    )

    progress(0.3, "Applying local contrast…")
    if data.ndim == 2:
        result = _apply_clahe_mono(data, clahe, params.amount)
    else:
        result = _apply_clahe_color(data, clahe, params.amount)

    progress(0.9, "Blending…")
    result = apply_mask(data, result, mask)
    progress(1.0, "Local contrast complete")
    return result


def _apply_clahe_mono(
    data: np.ndarray,
    clahe: cv2.CLAHE,
    amount: float,
) -> np.ndarray:
    """Apply CLAHE to a mono image."""
    # CLAHE works on uint8 or uint16
    u16 = (data * 65535).clip(0, 65535).astype(np.uint16)
    enhanced = clahe.apply(u16)
    enhanced_f = enhanced.astype(np.float32) / 65535.0

    if amount < 1.0:
        enhanced_f = data * (1 - amount) + enhanced_f * amount

    return np.clip(enhanced_f, 0, 1).astype(np.float32)


def _apply_clahe_color(
    data: np.ndarray,
    clahe: cv2.CLAHE,
    amount: float,
) -> np.ndarray:
    """Apply CLAHE to luminance channel of a color image, preserving chrominance."""
    # Float Lab keeps the pipeline at full depth — the old path quantized
    # color images to 8 bits (the mono path already used uint16), so deep-sky
    # gradients banded after enhancement.
    bgr = np.transpose(data, (1, 2, 0))[:, :, ::-1].astype(np.float32)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)  # L in 0..100

    # Apply CLAHE to L channel only (uint16, like the mono path)
    l_u16 = (lab[:, :, 0] / 100.0 * 65535).clip(0, 65535).astype(np.uint16)
    lab[:, :, 0] = clahe.apply(l_u16).astype(np.float32) / 65535.0 * 100.0

    # Convert back to BGR
    result_bgr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    result_f = np.clip(result_bgr, 0, 1)

    # BGR -> RGB -> (C, H, W)
    result = np.transpose(result_f[:, :, ::-1], (2, 0, 1)).copy()

    if amount < 1.0:
        result = data * (1 - amount) + result * amount

    return np.clip(result, 0, 1).astype(np.float32)
