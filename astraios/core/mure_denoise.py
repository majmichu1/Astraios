"""MureDenoise — noise estimation from single images (Muresan & Parks method)."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)


def estimate_noise(
    img: NDArray,
    patch_size: int = 8,
    percentile: int = 25,
) -> float | tuple[float, ...]:
    """Estimate Gaussian noise standard deviation from a single image.

    Uses local patch std.dev. histogram approach — robust to structure/texture.

    Args:
        img: (H, W) or (H, W, C) float32 image in [0, 1] range.
        patch_size: Size of square patches (default 8).
        percentile: Histogram percentile for robust estimate (default 25).

    Returns:
        Grayscale: float noise_std.
        RGB: tuple (noise_r, noise_g, noise_b).

    Reference:
        Muresan & Parks, "Adaptive principal components and image denoising", 2003.
    """
    is_gray = img.ndim == 2

    if img.shape[0] < patch_size or img.shape[1] < patch_size:
        log.warning(
            "Image (%d, %d) smaller than patch_size %d, falling back to global std",
            img.shape[0], img.shape[1], patch_size,
        )
        if is_gray:
            return float(img.std())
        return tuple(float(img[..., c].std()) for c in range(img.shape[2]))

    std_map = _patch_std_map(img, patch_size)

    if is_gray:
        return float(np.percentile(std_map, percentile))

    n_ch = std_map.shape[-1]
    estimates: list[float] = []
    for c in range(n_ch):
        estimates.append(float(np.percentile(std_map[..., c], percentile)))
    return tuple(estimates)


def estimate_noise_from_dark(
    dark_frames: NDArray,
) -> float:
    """Estimate read noise from a stack of dark frames.

    Args:
        dark_frames: (N, H, W) or (N, H, W, C) float32.

    Returns:
        Noise std averaged across pixels.
    """
    pixel_std = dark_frames.std(axis=0)
    return float(pixel_std.mean())


def _patch_std_map(img: NDArray, patch_size: int) -> NDArray:
    """Compute per-patch standard deviation map.

    Uses sliding_window_view for efficient non-overlapping patches.
    """
    if img.ndim == 2:
        windows_g = np.lib.stride_tricks.sliding_window_view(
            img, (patch_size, patch_size)
        )
        windows_g = windows_g[::patch_size, ::patch_size]
        patches = windows_g.reshape(-1, patch_size * patch_size)
        stds: NDArray = patches.std(axis=1)
        return stds.reshape(windows_g.shape[0], windows_g.shape[1])

    windows_mc = np.lib.stride_tricks.sliding_window_view(
        img, (patch_size, patch_size), axis=(0, 1)
    )
    windows_mc = windows_mc[::patch_size, ::patch_size]
    patches_mc = windows_mc.reshape(-1, patch_size * patch_size, img.shape[2])
    stds_mc: NDArray = patches_mc.std(axis=1)
    return stds_mc.reshape(windows_mc.shape[0], windows_mc.shape[1], img.shape[2])


def snr_estimate(img: NDArray, noise_std: float) -> float:
    """Estimate SNR from image and its noise level.

    SNR = mean(img) / noise_std (simplified, for linear data).
    """
    mean_val = float(img.mean())
    if noise_std < 1e-10:
        return float("inf")
    return mean_val / noise_std
