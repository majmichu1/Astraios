"""SuperBias — statistically optimized master bias frame.

Reduces read noise by exploiting spatial redundancy in bias structure.
Equivalent to stacking hundreds of bias frames from just ~20 inputs.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)


def create_superbias(
    bias_frames: NDArray,
    rejection_sigma: float = 3.0,
    column_smooth: float = 0.5,
) -> NDArray:
    """Create a SuperBias from a stack of bias frames.

    Algorithm:
    1. Stack bias frames with sigma rejection (pixel-wise).
    2. Detect column-dependent fixed-pattern noise by averaging along columns.
    3. Smooth the column pattern (optional).
    4. Combine pixel-wise and column-wise estimates.

    Args:
        bias_frames: (N, H, W) or (N, H, W, C) float32 bias frames.
        rejection_sigma: Sigma threshold for pixel rejection during stacking.
        column_smooth: Gaussian sigma for column pattern smoothing (0 = no smoothing).

    Returns:
        (H, W) or (H, W, C) float32 SuperBias.
    """
    if bias_frames.ndim < 3:
        msg = f"Expected at least 3D array (N, H, W), got shape {bias_frames.shape}"
        raise ValueError(msg)

    n = bias_frames.shape[0]
    if n < 3:
        log.warning("SuperBias needs at least 3 frames, got %d", n)
        return bias_frames.mean(axis=0).astype(np.float32)

    # Per-pixel sigma rejection
    median = np.median(bias_frames, axis=0)
    mad = np.median(np.abs(bias_frames - median), axis=0)
    sigma = mad * 1.4826 + 1e-10

    mask = np.abs(bias_frames - median) < rejection_sigma * sigma
    mask_count = mask.sum(axis=0)
    masked_sum = (bias_frames * mask).sum(axis=0)
    pixel_bias = np.where(mask_count > 0, masked_sum / mask_count, median)

    # Column-dependent FPN suppression
    if bias_frames.ndim == 3:
        # (N, H, W) grayscale
        column_pattern = np.median(pixel_bias, axis=0, keepdims=True)
        column_pattern = column_pattern - column_pattern.mean()
        if column_smooth > 0:
            column_pattern = _gaussian_smooth_1d(column_pattern[0], column_smooth)
            column_pattern = column_pattern[np.newaxis, :]
        result = pixel_bias - column_pattern
    else:
        # (N, H, W, C) multichannel — vectorized column pattern
        col_pat = np.median(pixel_bias, axis=0, keepdims=True)  # (1, W, C)
        col_pat = col_pat - col_pat.mean(axis=1, keepdims=True)
        if column_smooth > 0:
            for c in range(pixel_bias.shape[2]):
                col_pat[0, :, c] = _gaussian_smooth_1d(col_pat[0, :, c], column_smooth)
        result = pixel_bias - col_pat

    return result.astype(np.float32)


def _gaussian_smooth_1d(data: NDArray, sigma: float) -> NDArray:
    """Simple 1D Gaussian smoothing."""
    if sigma <= 0 or len(data) < 3:
        return data.copy()

    radius = int(sigma * 3)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    padded = np.pad(data, radius, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[radius:-radius]
