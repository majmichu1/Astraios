"""Built-in star removal — zero-setup morphological approach.

Uses median-background subtraction + star detection + inpainting.
No external model download needed, works immediately on any image.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)


def remove_stars_builtin(
    image: NDArray,
    threshold: float = 0.5,
) -> NDArray:
    """Remove stars using morphological detection + median inpainting.

    Args:
        image: (H, W) or (C, H, W) float32/float64 in [0, 1].
        threshold: 0-1 slider value (lower = remove more / more aggressive).

    Returns:
        Starless image, same shape and dtype.
    """
    return _remove_stars_morph(image, threshold)


def _remove_stars_morph(
    image: NDArray,
    threshold: float,
) -> NDArray:
    """Morphological star removal via median background + mask."""
    img = image.astype(np.float64)
    is_color = img.ndim == 3

    if is_color:
        lum = 0.2126 * img[0] + 0.7152 * img[1] + 0.0722 * img[2]
    else:
        lum = img.copy()

    orig_h, orig_w = lum.shape

    # ── 1. Median-filtered background ────────────────────────────────
    # Kernel size adapts to image dimensions (stars are ~1-5% of width)
    ksize = max(15, min(orig_h, orig_w) // 30)
    if ksize % 2 == 0:
        ksize += 1
    if ksize > 199:
        ksize = 199  # OpenCV medianBlur limit

    # OpenCV medianBlur only supports 8-bit or float32
    lum_u8 = (np.clip(lum, 0, 1) * 255).astype(np.uint8)
    bg_u8 = cv2.medianBlur(lum_u8, ksize)
    bg = bg_u8.astype(np.float64) / 255.0

    # ── 2. Residuals (star signal) ──────────────────────────────────
    diff = np.clip(lum - bg, 0, None)

    # Noise estimate from MAD of residuals
    mad = np.median(np.abs(diff - np.median(diff)))
    noise_est = max(mad * 1.4826, 1e-6)

    # threshold maps 0..1 → sigma 12..1 (lower slider = more aggressive)
    sigma = 1.0 + (1.0 - threshold) * 12.0
    binary = (diff > sigma * noise_est).astype(np.uint8) * 255

    # ── 3. Dilate mask to cover halos ───────────────────────────────
    radius = max(3, min(orig_h, orig_w) // 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    binary = cv2.dilate(binary, kernel, iterations=2)

    mask = binary > 0

    # ── 4. Composite result ─────────────────────────────────────────
    if is_color:
        # Scale each channel proportionally to the luminance change
        scale = np.clip(bg / (lum + 1e-10), 0.1, 1.0)
        result = img.copy()
        for c in range(3):
            result[c][mask] = img[c][mask] * scale[mask]
    else:
        result = img.copy()
        result[mask] = bg[mask]

    # ── 5. Feather edges of the mask ────────────────────────────────
    # Light Gaussian blend at star boundaries to avoid hard cutoffs
    if np.any(mask):
        blurred = cv2.GaussianBlur(binary.astype(np.float32), (0, 0), sigmaX=radius * 0.8)
        blend = blurred / 255.0
        blend = np.clip(blend, 0, 1)
        blend = cv2.erode(blend.astype(np.float32), kernel, iterations=1)
        if is_color:
            for c in range(3):
                result[c] = img[c] * (1.0 - blend) + result[c] * blend
        else:
            result = img * (1.0 - blend) + result * blend

    return np.clip(result, 0, 1).astype(image.dtype)
