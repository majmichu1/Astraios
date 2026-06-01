"""HDR Composition — merge multiple exposures using Mertens fusion or weighted average.

Uses OpenCV's MergeMertens (Apache 2.0) for exposure fusion,
which works without knowing exposure values (no HDR calibration needed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import cv2
import numpy as np

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(f: float, m: str) -> None:
    pass


class HDRMethod(Enum):
    MERTENS = auto()  # Mertens exposure fusion (no exposure data needed)
    WEIGHTED_AVERAGE = auto()  # Custom weighted average


@dataclass
class HDRParams:
    """Parameters for HDR composition."""

    method: HDRMethod = HDRMethod.MERTENS
    # Mertens weights
    contrast_weight: float = 1.0
    saturation_weight: float = 1.0
    exposure_weight: float = 1.0
    # Weighted average
    sigma: float = 0.2  # Gaussian weight width for well-exposedness


def hdr_compose(
    images: list[np.ndarray],
    params: HDRParams | None = None,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    """Merge multiple exposures into a single HDR image.

    Parameters
    ----------
    images : list[ndarray]
        List of exposures, each shape (H, W) or (C, H, W), float32 in [0, 1].
    params : HDRParams, optional
        Composition parameters.

    Returns
    -------
    ndarray
        Merged HDR image, float32 in [0, 1].

    Raises
    ------
    ValueError
        If fewer than 2 images are provided.
    """
    if len(images) < 2:
        raise ValueError("HDR composition requires at least 2 images")

    if params is None:
        params = HDRParams()

    progress(0.0, f"HDR: merging {len(images)} exposures…")
    if params.method == HDRMethod.MERTENS:
        result = _mertens_fusion(images, params)
    else:
        result = _weighted_average(images, params)
    progress(1.0, "HDR complete")
    return result


def _mertens_fusion(images: list[np.ndarray], params: HDRParams) -> np.ndarray:
    """Merge using OpenCV's Mertens exposure fusion."""
    # Convert to OpenCV format: list of (H, W, C) uint8 images
    cv_images = []
    is_color = images[0].ndim == 3

    for img in images:
        if is_color:
            # (C, H, W) -> (H, W, C) BGR
            bgr = np.transpose(img, (1, 2, 0))[:, :, ::-1]
        else:
            bgr = np.stack([img, img, img], axis=-1)
        cv_images.append((bgr * 255).clip(0, 255).astype(np.uint8))

    merge = cv2.createMergeMertens(
        contrast_weight=params.contrast_weight,
        saturation_weight=params.saturation_weight,
        exposure_weight=params.exposure_weight,
    )
    fused = merge.process(cv_images)
    fused = np.clip(fused, 0, 1).astype(np.float32)

    if is_color:
        # (H, W, C) BGR -> (C, H, W) RGB
        return np.transpose(fused[:, :, ::-1], (2, 0, 1)).copy()
    else:
        return np.mean(fused, axis=-1)


def _weighted_average(images: list[np.ndarray], params: HDRParams) -> np.ndarray:
    """Merge using Gaussian-weighted average based on well-exposedness."""
    result = np.zeros_like(images[0], dtype=np.float64)
    weight_sum = np.zeros(images[0].shape[-2:], dtype=np.float64)

    for img in images:
        if img.ndim == 3:
            lum = np.mean(img, axis=0)
        else:
            lum = img

        # Well-exposedness weight: Gaussian centered at 0.5
        w = np.exp(-0.5 * ((lum - 0.5) / params.sigma) ** 2)
        weight_sum += w

        if img.ndim == 3:
            for ch in range(img.shape[0]):
                result[ch] += img[ch].astype(np.float64) * w
        else:
            result += img.astype(np.float64) * w

    # Normalize
    weight_sum = np.maximum(weight_sum, 1e-10)
    if result.ndim == 3:
        for ch in range(result.shape[0]):
            result[ch] /= weight_sum
    else:
        result /= weight_sum

    return np.clip(result, 0, 1).astype(np.float32)
