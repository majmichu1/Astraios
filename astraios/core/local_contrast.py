"""Local Contrast Enhancement — CLAHE and local histogram equalization.

Uses OpenCV CLAHE (Apache 2.0) applied to the luminance channel only,
preserving color information.
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

    original = data.copy()
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
    result = apply_mask(original, result, mask)
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
    # Convert (C, H, W) RGB -> (H, W, C) BGR for OpenCV
    bgr = np.transpose(data, (1, 2, 0))[:, :, ::-1].copy()
    bgr_u8 = (bgr * 255).clip(0, 255).astype(np.uint8)

    # Convert to Lab color space
    lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB)

    # Apply CLAHE to L channel only
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # Convert back to BGR
    result_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    result_f = result_bgr.astype(np.float32) / 255.0

    # BGR -> RGB -> (C, H, W)
    result = np.transpose(result_f[:, :, ::-1], (2, 0, 1)).copy()

    if amount < 1.0:
        result = data * (1 - amount) + result * amount

    return np.clip(result, 0, 1).astype(np.float32)
