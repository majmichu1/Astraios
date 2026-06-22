"""Color Tools — SCNR, saturation, hue rotation, and vibrance.

GPU-accelerated via the device manager.
All operations expect images in (C, H, W) format with values in [0, 1].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import torch

from astraios.core.device_manager import get_device_manager
from astraios.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)


# ---------- SCNR (Subtractive Chromatic Noise Reduction) ----------


class SCNRMethod(Enum):
    AVERAGE_NEUTRAL = auto()
    MAXIMUM_NEUTRAL = auto()


@dataclass
class SCNRParams:
    """Parameters for SCNR (green noise removal)."""

    method: SCNRMethod = SCNRMethod.AVERAGE_NEUTRAL
    amount: float = 1.0  # 0-1
    preserve_luminance: bool = True


def scnr(
    image: np.ndarray,
    params: SCNRParams | None = None,
    mask: Mask | None = None,
) -> np.ndarray:
    """Apply Subtractive Chromatic Noise Reduction (SCNR) — GPU-accelerated."""
    if params is None:
        params = SCNRParams()

    if image.ndim != 3 or image.shape[0] < 3:
        raise ValueError("SCNR requires a color image with >= 3 channels; got mono or single-channel input")

    dm = get_device_manager()
    if dm.is_gpu:
        return _scnr_gpu(image, params, mask, dm)
    else:
        return _scnr_cpu(image, params, mask)


@torch.no_grad()
def _scnr_gpu(
    image: np.ndarray,
    params: SCNRParams,
    mask: Mask | None,
    dm,
) -> np.ndarray:
    """GPU-accelerated SCNR."""
    t = dm.from_numpy(image)
    r, g, b = t[0], t[1], t[2]

    if params.method == SCNRMethod.AVERAGE_NEUTRAL:
        neutral = (r + b) / 2.0
    else:
        neutral = torch.maximum(r, b)

    excess = g > neutral
    corrected_g = g.clone()
    corrected_g[excess] = g[excess] * (1 - params.amount) + neutral[excess] * params.amount

    if params.preserve_luminance:
        lum_before = 0.2126 * r + 0.7152 * g + 0.0722 * b
        lum_after = 0.2126 * r + 0.7152 * corrected_g + 0.0722 * b
        ratio = torch.where(lum_after > 1e-10, lum_before / lum_after, torch.tensor(1.0, device=t.device))
        t = torch.stack([r * ratio, corrected_g * ratio, b * ratio], dim=0)
        if image.shape[0] > 3:
            t = torch.cat([t, dm.from_numpy(image[3:])], dim=0)
    else:
        t = torch.stack([r, corrected_g, b], dim=0)
        if image.shape[0] > 3:
            t = torch.cat([t, dm.from_numpy(image[3:])], dim=0)

    result = torch.clamp(t, 0.0, 1.0).cpu().numpy().astype(np.float32)
    return apply_mask(image, result, mask)


def _scnr_cpu(
    image: np.ndarray,
    params: SCNRParams,
    mask: Mask | None,
) -> np.ndarray:
    """CPU fallback for SCNR."""
    original = image.copy()
    result = image.copy()
    r, g, b = result[0], result[1], result[2]

    if params.method == SCNRMethod.AVERAGE_NEUTRAL:
        neutral = (r + b) / 2.0
    else:
        neutral = np.maximum(r, b)

    excess = g > neutral
    if excess.any():
        corrected_g = g.copy()
        corrected_g[excess] = g[excess] * (1 - params.amount) + neutral[excess] * params.amount
        result[1] = corrected_g

    if params.preserve_luminance:
        lum_before = 0.2126 * original[0] + 0.7152 * original[1] + 0.0722 * original[2]
        lum_after = 0.2126 * result[0] + 0.7152 * result[1] + 0.0722 * result[2]
        ratio = np.where(lum_after > 1e-10, lum_before / lum_after, 1.0)
        for ch in range(3):
            result[ch] = np.clip(result[ch] * ratio, 0, 1)

    return apply_mask(original, result, mask)


# ---------- Color Saturation / Hue Tools ----------


@dataclass
class ColorAdjustParams:
    """Parameters for color adjustment."""

    saturation: float = 1.0  # multiplier: <1 desaturate, >1 saturate
    hue_shift: float = 0.0  # degrees (-180 to 180)
    vibrance: float = 0.0  # 0-1, boost only desaturated colors


def _rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    """Convert (3, H, W) RGB to (3, H, W) HSV.

    H in [0, 360], S in [0, 1], V in [0, 1].
    Vectorized for performance.
    """
    r, g, b = image[0], image[1], image[2]
    v = np.maximum(np.maximum(r, g), b)
    c = v - np.minimum(np.minimum(r, g), b)

    h = np.zeros_like(v)
    s = np.zeros_like(v)

    # Saturation
    nonzero_v = v > 1e-10
    s[nonzero_v] = c[nonzero_v] / v[nonzero_v]

    # Hue
    nonzero_c = c > 1e-10
    mask_r = nonzero_c & (v == r)
    mask_g = nonzero_c & (v == g) & ~mask_r
    mask_b = nonzero_c & ~mask_r & ~mask_g

    h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / c[mask_r]) % 6)
    h[mask_g] = 60 * ((b[mask_g] - r[mask_g]) / c[mask_g] + 2)
    h[mask_b] = 60 * ((r[mask_b] - g[mask_b]) / c[mask_b] + 4)

    h = h % 360

    return np.stack([h, s, v], axis=0)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Convert (3, H, W) HSV to (3, H, W) RGB.

    H in [0, 360], S in [0, 1], V in [0, 1].
    """
    h, s, v = hsv[0], hsv[1], hsv[2]

    c = v * s
    h_prime = (h / 60) % 6
    x = c * (1 - np.abs(h_prime % 2 - 1))
    m = v - c

    r = np.zeros_like(v)
    g = np.zeros_like(v)
    b = np.zeros_like(v)

    for lo, hi, rv, gv, bv in [
        (0, 1, c, x, 0),
        (1, 2, x, c, 0),
        (2, 3, 0, c, x),
        (3, 4, 0, x, c),
        (4, 5, x, 0, c),
        (5, 6, c, 0, x),
    ]:
        mask = (h_prime >= lo) & (h_prime < hi)
        r[mask] = (rv if isinstance(rv, (int, float)) else rv[mask]) + m[mask]
        g[mask] = (gv if isinstance(gv, (int, float)) else gv[mask]) + m[mask]
        b[mask] = (bv if isinstance(bv, (int, float)) else bv[mask]) + m[mask]

    return np.clip(np.stack([r, g, b], axis=0), 0, 1).astype(np.float32)


@torch.no_grad()
def _rgb_to_hsv_gpu(image: np.ndarray, dm) -> torch.Tensor:
    """GPU-accelerated RGB to HSV conversion."""
    t = dm.from_numpy(image)
    r, g, b = t[0], t[1], t[2]

    v = torch.maximum(torch.maximum(r, g), b)
    c = v - torch.minimum(torch.minimum(r, g), b)

    h = torch.zeros_like(v)
    s = torch.zeros_like(v)

    nonzero_v = v > 1e-10
    s[nonzero_v] = c[nonzero_v] / v[nonzero_v]

    nonzero_c = c > 1e-10
    mask_r = nonzero_c & (v == r)
    mask_g = nonzero_c & (v == g) & ~mask_r
    mask_b = nonzero_c & ~mask_r & ~mask_g

    h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / c[mask_r]) % 6)
    h[mask_g] = 60 * ((b[mask_g] - r[mask_g]) / c[mask_g] + 2)
    h[mask_b] = 60 * ((r[mask_b] - g[mask_b]) / c[mask_b] + 4)

    h = h % 360
    return torch.stack([h, s, v], dim=0)


@torch.no_grad()
def _hsv_to_rgb_gpu(hsv: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated HSV to RGB conversion."""
    h, s, v = hsv[0], hsv[1], hsv[2]

    c = v * s
    h_prime = (h / 60) % 6
    x = c * (1 - torch.abs(h_prime % 2 - 1))
    m = v - c

    r = torch.zeros_like(v)
    g = torch.zeros_like(v)
    b = torch.zeros_like(v)

    for lo, hi, rv, gv, bv in [
        (0, 1, c, x, torch.zeros_like(v)),
        (1, 2, x, c, torch.zeros_like(v)),
        (2, 3, torch.zeros_like(v), c, x),
        (3, 4, torch.zeros_like(v), x, c),
        (4, 5, x, torch.zeros_like(v), c),
        (5, 6, c, torch.zeros_like(v), x),
    ]:
        mask = (h_prime >= lo) & (h_prime < hi)
        r[mask] = rv[mask] + m[mask]
        g[mask] = gv[mask] + m[mask]
        b[mask] = bv[mask] + m[mask]

    return torch.clamp(torch.stack([r, g, b], dim=0), 0.0, 1.0)


def color_adjust(
    image: np.ndarray,
    params: ColorAdjustParams | None = None,
    mask: Mask | None = None,
) -> np.ndarray:
    """Adjust color saturation, hue, and vibrance — GPU-accelerated."""
    if params is None:
        params = ColorAdjustParams()

    if image.ndim != 3 or image.shape[0] < 3:
        log.warning("Color adjustment requires a color image with >= 3 channels")
        return image

    dm = get_device_manager()
    if dm.is_gpu:
        return _color_adjust_gpu(image, params, mask, dm)
    else:
        return _color_adjust_cpu(image, params, mask)


@torch.no_grad()
def _color_adjust_gpu(
    image: np.ndarray,
    params: ColorAdjustParams,
    mask: Mask | None,
    dm,
) -> np.ndarray:
    """GPU-accelerated color adjustment."""
    hsv = _rgb_to_hsv_gpu(image[:3], dm)

    # Hue shift
    if abs(params.hue_shift) > 0.01:
        hsv[0] = (hsv[0] + params.hue_shift) % 360

    # Saturation
    if abs(params.saturation - 1.0) > 1e-6:
        hsv[1] = torch.clamp(hsv[1] * params.saturation, 0.0, 1.0)

    # Vibrance
    if params.vibrance > 0:
        boost = params.vibrance * (1.0 - hsv[1])
        hsv[1] = torch.clamp(hsv[1] + boost, 0.0, 1.0)

    result = _hsv_to_rgb_gpu(hsv)
    result_np = result.cpu().numpy().astype(np.float32)

    if image.shape[0] > 3:
        result_np = np.concatenate([result_np, image[3:]], axis=0)

    # apply_mask only reads `image` (never mutates it) and nothing above touches
    # it, so no defensive copy is needed — saving a full-image allocation.
    return apply_mask(image, result_np, mask)


def _color_adjust_cpu(
    image: np.ndarray,
    params: ColorAdjustParams,
    mask: Mask | None,
) -> np.ndarray:
    """CPU fallback for color adjustment."""
    hsv = _rgb_to_hsv(image[:3])

    if abs(params.hue_shift) > 0.01:
        hsv[0] = (hsv[0] + params.hue_shift) % 360

    if abs(params.saturation - 1.0) > 1e-6:
        hsv[1] = np.clip(hsv[1] * params.saturation, 0, 1)

    if params.vibrance > 0:
        boost = params.vibrance * (1.0 - hsv[1])
        hsv[1] = np.clip(hsv[1] + boost, 0, 1)

    result = _hsv_to_rgb(hsv)

    if image.shape[0] > 3:
        result = np.concatenate([result, image[3:]], axis=0)

    return apply_mask(image, result, mask)
