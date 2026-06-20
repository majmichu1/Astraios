"""Background-masked grain reduction — clean post-stretch sky noise.

After a strong stretch the dark sky still carries luminance grain (brightness
mottle) that the black point cannot remove: clipping the black point only crushes
the shadows, not the mid-tone speckle in the background. This module smooths that
grain gently and only in the background (low-signal, non-subject) region, so the
sky reads clean without softening the object or stars.

A bilateral filter is used rather than a wavelet/threshold denoiser: being a
normalized, edge-aware weighted average it preserves the local mean (no
darkening) and the per-channel level (no colour shift), and removes only the
high-frequency grain. The effect is confined to the background by a soft
luminance mask, so the subject and stars are untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from astraios.core.masks import (
    Mask,
    MaskType,
    _get_luminance,
    apply_mask,
    create_luminance_mask,
)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


@dataclass
class LumaDenoiseParams:
    """Parameters for background grain reduction.

    Attributes:
        strength: Grain-reduction strength (0-1); maps to the bilateral colour
            sigma (how much intensity variation is smoothed).
        detail_preservation: Higher keeps more fine structure (0-1).
        bg_threshold: Luminance cutoff that defines "background". If None it is
            estimated from the image as ``median + threshold_sigma * sigma``.
        threshold_sigma: Robust sigma multiplier for the auto threshold.
    """

    strength: float = 0.5
    detail_preservation: float = 0.6
    bg_threshold: float | None = None
    threshold_sigma: float = 3.0


def _auto_bg_threshold(luminance: np.ndarray, sigma: float) -> float:
    """Estimate the background/subject luminance split robustly.

    On a sky-dominated frame the median luminance *is* the sky, so everything up
    to a few robust sigma above it is background. MAD->sigma keeps stars and the
    object (the bright tail) from inflating the estimate.
    """
    med = float(np.median(luminance))
    mad = float(np.median(np.abs(luminance - med)))
    thr = med + sigma * 1.4826 * mad
    return float(np.clip(thr, 0.02, 0.7))


def _bilateral_smooth(
    image: np.ndarray, strength: float, detail_preservation: float
) -> np.ndarray:
    """Edge-preserving smooth of ``image`` (mono ``(H,W)`` or colour ``(C,H,W)``)."""
    # sigma_color sets how much intensity variation is treated as grain; more
    # detail_preservation lowers it so real edges survive.
    sigma_color = (0.03 + 0.10 * float(strength)) * (1.0 - 0.4 * float(detail_preservation))
    sigma_space = 3.0
    diameter = 5

    def _bf(plane: np.ndarray) -> np.ndarray:
        return cv2.bilateralFilter(
            np.ascontiguousarray(plane, dtype=np.float32), diameter, sigma_color, sigma_space
        )

    if image.ndim == 2:
        return _bf(image)
    if image.shape[0] in (1, 3):
        hwc = np.ascontiguousarray(np.transpose(image, (1, 2, 0)))
        out = cv2.bilateralFilter(hwc, diameter, sigma_color, sigma_space)
        return np.ascontiguousarray(np.transpose(out, (2, 0, 1))).astype(np.float32)
    # Uncommon channel count: smooth each plane independently.
    return np.stack([_bf(image[c]) for c in range(image.shape[0])]).astype(np.float32)


def denoise_background_luma(
    image: np.ndarray,
    params: LumaDenoiseParams | None = None,
    object_mask: np.ndarray | None = None,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    """Reduce grain in the background only, preserving the subject and stars.

    Args:
        image: float32 in [0, 1]; mono ``(H, W)`` or colour ``(C, H, W)``.
        params: Reduction parameters (defaults if None).
        object_mask: Optional ``(H, W)`` array, 1.0 on the subject, used to keep
            the smoothing off the object even where it is faint.
        progress: Progress callback ``(fraction, message)``.

    Returns:
        The image with background grain reduced, same shape/dtype.
    """
    if params is None:
        params = LumaDenoiseParams()
    if params.strength <= 0.0:
        return image

    progress(0.1, "Measuring background...")
    luminance = _get_luminance(image)
    thr = (
        params.bg_threshold
        if params.bg_threshold is not None
        else _auto_bg_threshold(luminance, params.threshold_sigma)
    )

    # Background = dark sky: luminance below the threshold, soft falloff above so
    # the object edge is not smoothed abruptly.
    bg_mask = create_luminance_mask(image, low=0.0, high=thr, name="Background")

    # Keep the smoothing off the subject even where it is faint.
    if object_mask is not None and object_mask.shape == bg_mask.data.shape:
        protected = np.clip(object_mask.astype(np.float32), 0.0, 1.0)
        bg_mask = Mask(
            data=bg_mask.data * (1.0 - protected),
            name="Background",
            mask_type=MaskType.LUMINANCE,
        )

    if not bg_mask.data.any():
        return image  # nothing classified as background

    progress(0.4, "Smoothing background grain...")
    processed = _bilateral_smooth(image, params.strength, params.detail_preservation)

    progress(0.9, "Blending into background...")
    result = apply_mask(image, processed, bg_mask)
    progress(1.0, "Background grain reduction complete")
    return result.astype(np.float32)
