"""Noise generation utilities — synthetic noise for testing and calibration."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)


def add_gaussian_noise(
    img: NDArray,
    mean: float = 0.0,
    std: float = 0.01,
    seed: int | None = None,
) -> NDArray:
    """Add Gaussian noise to an image.

    Args:
        img: (H, W) or (H, W, C) float32.
        mean: Noise mean.
        std: Noise standard deviation.
        seed: Random seed for reproducibility.

    Returns:
        Noisy copy of img.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(mean, std, size=img.shape).astype(np.float32)
    result = img.astype(np.float32) + noise
    return result.astype(img.dtype)


def add_poisson_noise(
    img: NDArray,
    scale: float = 1.0,
    seed: int | None = None,
) -> NDArray:
    """Add Poisson (shot) noise to an image.

    Args:
        img: (H, W) or (H, W, C) float32 in [0, 1].
        scale: Scale factor (larger = more noise).
        seed: Random seed.

    Returns:
        Noisy copy.
    """
    rng = np.random.default_rng(seed)
    scaled = np.clip(img, 0, 1) * scale
    noisy = rng.poisson(scaled).astype(np.float32) / max(scale, 1e-8)
    return noisy.astype(img.dtype)


def add_uniform_noise(
    img: NDArray,
    low: float = -0.01,
    high: float = 0.01,
    seed: int | None = None,
) -> NDArray:
    """Add uniform random noise to an image.

    Args:
        img: (H, W) or (H, W, C) float32.
        low: Lower bound of noise.
        high: Upper bound of noise.
        seed: Random seed.

    Returns:
        Noisy copy.
    """
    rng = np.random.default_rng(seed)
    noise = rng.uniform(low, high, size=img.shape).astype(np.float32)
    result = img.astype(np.float32) + noise
    return result.astype(img.dtype)


def add_salt_pepper_noise(
    img: NDArray,
    amount: float = 0.01,
    salt_vs_pepper: float = 0.5,
    seed: int | None = None,
) -> NDArray:
    """Add salt & pepper noise to an image.

    Args:
        img: (H, W) or (H, W, C) float32 in [0, 1].
        amount: Fraction of pixels to corrupt.
        salt_vs_pepper: Ratio of salt (1) to pepper (0).
        seed: Random seed.

    Returns:
        Noisy copy.
    """
    rng = np.random.default_rng(seed)
    result = img.astype(np.float32).copy()
    mask = rng.random(img.shape[:2]) < amount
    salt = rng.random(img.shape[:2]) < salt_vs_pepper
    if img.ndim == 3:
        for c in range(img.shape[2]):
            result[mask & salt, c] = 1.0
            result[mask & ~salt, c] = 0.0
    else:
        result[mask & salt] = 1.0
        result[mask & ~salt] = 0.0
    return result.astype(img.dtype)


def add_bias_frame(
    shape: tuple[int, ...],
    offset: float = 0.01,
    read_noise_std: float = 0.001,
    seed: int | None = None,
) -> NDArray:
    """Generate a synthetic bias frame.

    Args:
        shape: (H, W) or (H, W, C).
        offset: Bias offset value.
        read_noise_std: Read noise std dev.
        seed: Random seed.

    Returns:
        Bias frame as float32.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, read_noise_std, size=shape).astype(np.float32)
    return (offset + noise).astype(np.float32)
