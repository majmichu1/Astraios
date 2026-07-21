"""Star Reduction — reduce star sizes using morphological erosion within star mask.

Uses OpenCV morphological operations (Apache 2.0 license).
Requires a star mask to operate — either auto-generated or user-provided.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from astraios.core.masks import Mask, MaskType, apply_mask
from astraios.core.morphology import StructuringElement
from astraios.core.star_detection import detect_stars

ProgressCallback = Callable[[float, str], None]


def _noop_progress(f: float, m: str) -> None:
    pass

log = logging.getLogger(__name__)


@dataclass
class StarReductionParams:
    """Parameters for star reduction."""

    amount: float = 0.5  # 0-1, how much to reduce stars
    iterations: int = 2  # morphological erosion iterations
    protect_core: bool = True  # protect brightest star cores
    kernel_size: int = 3  # erosion kernel size
    kernel_type: StructuringElement | None = None


def create_star_mask(
    image: np.ndarray,
    sensitivity: float = 5.0,
    max_stars: int = 500,
    softness: float = 5.0,
    scale: float = 1.5,
) -> Mask:
    """Generate a star mask from an image.

    Detects stars and creates a soft mask with Gaussian blobs
    at each star position.

    Parameters
    ----------
    image : ndarray
        Image data, shape (H, W) or (C, H, W), values in [0, 1].
    sensitivity : float
        Detection sigma threshold (lower = more stars detected).
    max_stars : int
        Maximum number of stars to include.
    softness : float
        Gaussian blur radius for feathering.
    scale : float
        Scale multiplier for star blob size relative to detected FWHM.

    Returns
    -------
    Mask
        Star mask where stars are 1.0 and background is 0.0.
    """
    sf = detect_stars(image, max_stars=max_stars, sigma_threshold=sensitivity)

    if image.ndim == 3:
        h, w = image.shape[1], image.shape[2]
    else:
        h, w = image.shape

    mask_data = np.zeros((h, w), dtype=np.float32)

    for star in sf.stars:
        radius = max(star.fwhm * scale, 2.0)
        sigma = radius / 2.0
        # The Gaussian blob is negligible beyond ~6 sigma (exp(-18) ~ 1e-8, below
        # the mask's float32 precision and softened/clipped anyway), so build it
        # only over the star's bounding box instead of the whole frame — turns an
        # O(N_stars * H * W) loop into O(N_stars * radius^2).
        rad = int(np.ceil(radius * 3.0))
        y0, y1 = max(0, int(star.y) - rad), min(h, int(star.y) + rad + 1)
        x0, x1 = max(0, int(star.x) - rad), min(w, int(star.x) + rad + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist_sq = (xx - star.x) ** 2 + (yy - star.y) ** 2
        blob = np.exp(-dist_sq / (2 * sigma**2)).astype(np.float32)
        region = mask_data[y0:y1, x0:x1]
        np.maximum(region, blob, out=region)

    # Soften edges
    if softness > 0:
        ksize = int(np.ceil(softness * 3)) * 2 + 1
        mask_data = cv2.GaussianBlur(mask_data, (ksize, ksize), softness)

    mask_data = np.clip(mask_data, 0, 1)
    return Mask(data=mask_data, name="Star Mask", mask_type=MaskType.STAR)


def reduce_stars(
    image: np.ndarray,
    star_mask: Mask | None = None,
    params: StarReductionParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    """Reduce star sizes by morphological erosion within the star mask.

    Parameters
    ----------
    image : ndarray
        Image data, shape (H, W) or (C, H, W), values in [0, 1].
    star_mask : Mask, optional
        Mask of star regions. If None, auto-generated.
    params : StarReductionParams, optional
        Reduction parameters.
    mask : Mask, optional
        Additional processing mask.

    Returns
    -------
    ndarray
        Image with reduced stars.
    """
    if params is None:
        params = StarReductionParams()

    if star_mask is None:
        progress(0.0, "Detecting stars for reduction…")
        star_mask = create_star_mask(image)

    progress(0.4, "Applying morphological reduction…")
    # no copy: op never mutates the input; apply_mask reads image directly
    _MORPH_MAP = {
        StructuringElement.CIRCLE: cv2.MORPH_ELLIPSE,
        StructuringElement.SQUARE: cv2.MORPH_RECT,
        StructuringElement.DIAMOND: cv2.MORPH_CROSS,
    }
    morph_type = _MORPH_MAP.get(params.kernel_type or StructuringElement.CIRCLE, cv2.MORPH_ELLIPSE)
    kernel = cv2.getStructuringElement(
        morph_type, (params.kernel_size, params.kernel_size)
    )

    sm = star_mask.data
    if params.protect_core:
        sm = sm * (1.0 - _core_weight(image, sm))

    if image.ndim == 2:
        eroded = _erode_channel(image, kernel, params.iterations)
        # Blend: within star mask, use eroded version proportional to amount
        result = image * (1 - sm * params.amount) + eroded * (sm * params.amount)
    else:
        result = np.empty_like(image)
        for ch in range(image.shape[0]):
            eroded = _erode_channel(image[ch], kernel, params.iterations)
            result[ch] = image[ch] * (1 - sm * params.amount) + eroded * (sm * params.amount)

    result = np.clip(result, 0, 1)
    progress(1.0, "Star reduction complete")
    return apply_mask(image, result, mask)


def _erode_channel(channel: np.ndarray, kernel: np.ndarray, iterations: int) -> np.ndarray:
    """Erode a single channel. OpenCV morphology operates on uint8/float."""
    return cv2.erode(channel, kernel, iterations=iterations)


def _core_weight(image: np.ndarray, star_mask: np.ndarray) -> np.ndarray:
    """Per-pixel weight in [0, 1] that is 1 at bright star cores, 0 in halos.

    Erosion pulls a star's peak down along with its wings, which hollows out
    bright stars into a doughnut. `protect_core` holds the peak back: the
    erosion blend is scaled by ``1 - core_weight``, so the brightest centre
    keeps its original value while the surrounding halo is still reduced.

    The bright/faint split is taken from the distribution of the masked star
    pixels themselves (not a fixed level), so it adapts to linear as well as
    stretched data.
    """
    lum = image if image.ndim == 2 else image.max(axis=0)
    inside = star_mask > 0.05
    if not np.any(inside):
        return np.zeros(lum.shape, dtype=np.float32)

    values = lum[inside]
    # Halo reference vs core reference. The 99.5th percentile rather than the
    # max keeps a single hot pixel from defining the whole scale.
    lo = float(np.percentile(values, 80.0))
    hi = float(np.percentile(values, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(lum.shape, dtype=np.float32)

    weight = np.clip((lum - lo) / (hi - lo), 0.0, 1.0)
    # Square it so protection concentrates on the true peak and fades quickly
    # through the wings, instead of shielding the whole star.
    return (weight * weight).astype(np.float32)
