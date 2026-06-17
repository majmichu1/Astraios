"""Channel Operations — split, combine, extract, and replace channels.

Pure numpy operations — no external libraries needed.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def split_channels(image: np.ndarray) -> list[np.ndarray]:
    """Split a color image into individual channel arrays.

    Parameters
    ----------
    image : ndarray
        Color image of shape (C, H, W).

    Returns
    -------
    list[ndarray]
        List of C arrays, each of shape (H, W).
    """
    if image.ndim == 2:
        return [image.copy()]
    return [image[ch].copy() for ch in range(image.shape[0])]


def combine_channels(channels: list[np.ndarray]) -> np.ndarray:
    """Combine individual channel arrays into a color image.

    Parameters
    ----------
    channels : list[ndarray]
        List of 2D arrays (H, W), all same shape.

    Returns
    -------
    ndarray
        Color image of shape (C, H, W).
    """
    if len(channels) == 1:
        return channels[0].copy()
    return np.stack(channels, axis=0).astype(np.float32)


def extract_luminance(image: np.ndarray) -> np.ndarray:
    """Extract luminance channel from a color image.

    Parameters
    ----------
    image : ndarray
        Color image of shape (C, H, W) with C >= 3.

    Returns
    -------
    ndarray
        Luminance array of shape (H, W).
    """
    if image.ndim == 2:
        return image.copy()
    if image.shape[0] >= 3:
        return (0.2126 * image[0] + 0.7152 * image[1] + 0.0722 * image[2]).astype(np.float32)
    return image[0].copy()


def replace_channel(
    image: np.ndarray,
    channel_idx: int,
    new_channel: np.ndarray,
) -> np.ndarray:
    """Replace a single channel in a color image.

    Parameters
    ----------
    image : ndarray
        Color image of shape (C, H, W).
    channel_idx : int
        Index of channel to replace (0=R, 1=G, 2=B).
    new_channel : ndarray
        New channel data of shape (H, W).

    Returns
    -------
    ndarray
        Image with replaced channel.
    """
    result = image.copy()
    if result.ndim == 3 and 0 <= channel_idx < result.shape[0]:
        result[channel_idx] = new_channel
    return result


def replace_luminance(image: np.ndarray, new_luminance: np.ndarray) -> np.ndarray:
    """Replace the luminance of a color image while preserving chrominance.

    Uses the ratio method: scale each channel by new_L / old_L.

    Parameters
    ----------
    image : ndarray
        Color image of shape (C, H, W) with C >= 3.
    new_luminance : ndarray
        New luminance of shape (H, W).

    Returns
    -------
    ndarray
        Image with replaced luminance.
    """
    if image.ndim == 2:
        return new_luminance.copy()

    old_lum = extract_luminance(image)
    ratio = np.where(old_lum > 1e-10, new_luminance / old_lum, 1.0)

    result = image.copy()
    for ch in range(min(image.shape[0], 3)):
        result[ch] = np.clip(image[ch] * ratio, 0, 1)

    return result.astype(np.float32)


def rgb_to_hsl(image: np.ndarray) -> np.ndarray:
    """Convert (3, H, W) RGB to (3, H, W) HSL.

    H in [0, 1], S in [0, 1], L in [0, 1].
    """
    r, g, b = image[0], image[1], image[2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Lightness
    l = (cmax + cmin) / 2.0

    # Saturation
    s = np.zeros_like(l)
    mask = delta > 1e-10
    s[mask] = delta[mask] / (1.0 - np.abs(2 * l[mask] - 1) + 1e-10)

    # Hue
    h = np.zeros_like(l)
    mask_r = mask & (cmax == r)
    mask_g = mask & (cmax == g) & ~mask_r
    mask_b = mask & ~mask_r & ~mask_g

    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
    h[mask_g] = (b[mask_g] - r[mask_g]) / delta[mask_g] + 2
    h[mask_b] = (r[mask_b] - g[mask_b]) / delta[mask_b] + 4
    h = h / 6.0  # normalize to [0, 1]
    h = h % 1.0

    return np.stack([h, s, l], axis=0).astype(np.float32)


def hsl_to_rgb(hsl: np.ndarray) -> np.ndarray:
    """Convert (3, H, W) HSL to (3, H, W) RGB.

    H in [0, 1], S in [0, 1], L in [0, 1].
    """
    h, s, l = hsl[0], hsl[1], hsl[2]

    c = (1 - np.abs(2 * l - 1)) * s
    h6 = h * 6.0
    x = c * (1 - np.abs(h6 % 2 - 1))
    m = l - c / 2

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    for lo, hi, rv, gv, bv in [
        (0, 1, c, x, 0),
        (1, 2, x, c, 0),
        (2, 3, 0, c, x),
        (3, 4, 0, x, c),
        (4, 5, x, 0, c),
        (5, 6, c, 0, x),
    ]:
        mask = (h6 >= lo) & (h6 < hi)
        r[mask] = (rv if isinstance(rv, (int, float)) else rv[mask]) + m[mask]
        g[mask] = (gv if isinstance(gv, (int, float)) else gv[mask]) + m[mask]
        b[mask] = (bv if isinstance(bv, (int, float)) else bv[mask]) + m[mask]

    return np.clip(np.stack([r, g, b], axis=0), 0, 1).astype(np.float32)
