"""LRGB Combine — merge a luminance (L) channel with an RGB color image.

Standard astrophotography workflow:
  1. Stack and process L (luminance, high SNR) separately
  2. Stack and process RGB (color, lower SNR) separately
  3. Combine: luminance drives brightness/detail, RGB drives hue/saturation

Algorithm:
  - Convert RGB to Lab (perceptual) or HSL
  - Replace L channel with the processed luminance image
  - Optionally blend (saturation boost compensates for L override)
  - Convert back to RGB
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


@dataclass
class LRGBParams:
    """Parameters for LRGB combination."""

    luminance_weight: float = 1.0    # 0–1, blend from RGB-L to pure L
    saturation_boost: float = 1.2    # compensate color saturation after L replace
    chrominance_noise: float = 0.0   # 0–1 wavelet denoise on chroma channels


def _to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert (H,W,3) float32 [0,1] RGB to CIE Lab via sRGB→XYZ→Lab."""
    # sRGB linearise
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

    # XYZ (D65)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz = lin @ M.T

    # Normalise by D65 white
    xyz /= np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)

    def f(t):
        eps = 0.008856
        kappa = 903.3
        return np.where(t > eps, np.cbrt(t), (kappa * t + 16) / 116)

    fx, fy, fz = f(xyz[..., 0]), f(xyz[..., 1]), f(xyz[..., 2])
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def _from_lab(lab: np.ndarray) -> np.ndarray:
    """Convert (H,W,3) CIE Lab → sRGB float32 [0,1]."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    def finv(t):
        eps = 0.206897
        return np.where(t > eps, t ** 3, (116 * t - 16) / 903.3)

    xyz = np.stack([finv(fx), finv(fy), finv(fz)], axis=-1)
    xyz *= np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)

    M_inv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ], dtype=np.float32)
    lin = xyz @ M_inv.T

    # sRGB gamma
    rgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(np.clip(lin, 0, None), 1 / 2.4) - 0.055)
    return np.clip(rgb, 0, 1).astype(np.float32)


def lrgb_combine(
    luminance: np.ndarray,
    rgb: np.ndarray,
    params: LRGBParams | None = None,
    progress: ProgressCallback | None = None,
) -> np.ndarray:
    """Combine a luminance channel with an RGB image.

    Parameters
    ----------
    luminance : ndarray
        Mono luminance image, shape (H, W) or (1, H, W), float32 [0, 1].
    rgb : ndarray
        Color image, shape (3, H, W) or (H, W, 3), float32 [0, 1].
    params : LRGBParams, optional
    progress : callable, optional

    Returns
    -------
    ndarray
        Combined LRGB image, shape (3, H, W), float32 [0, 1].
    """
    if params is None:
        params = LRGBParams()
    if progress is None:
        progress = lambda f, m: None

    # Normalise shapes to HWC
    lum = luminance.squeeze() if luminance.ndim == 3 else luminance
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb_hwc = np.transpose(rgb, (1, 2, 0))
    else:
        rgb_hwc = rgb

    # Resize luminance to match RGB if needed
    if lum.shape != rgb_hwc.shape[:2]:
        from PIL import Image
        lum_pil = Image.fromarray((lum * 65535).astype(np.uint16))
        lum_pil = lum_pil.resize((rgb_hwc.shape[1], rgb_hwc.shape[0]), Image.LANCZOS)
        lum = np.array(lum_pil, dtype=np.float32) / 65535.0
        log.info("Luminance resized to match RGB: %s", lum.shape)

    progress(0.2, "Converting to Lab…")
    lab = _to_lab(rgb_hwc)

    progress(0.4, "Replacing luminance…")
    # L channel in Lab is [0, 100]; scale our [0,1] lum to that range
    lab_L_original = lab[..., 0].copy()
    lab_L_new = lum * 100.0

    # Blend with luminance_weight
    lab[..., 0] = (
        lab_L_original * (1.0 - params.luminance_weight)
        + lab_L_new * params.luminance_weight
    )

    # Saturation boost on a/b channels
    if params.saturation_boost != 1.0:
        lab[..., 1] *= params.saturation_boost
        lab[..., 2] *= params.saturation_boost

    # Optional chroma noise reduction (simple Gaussian blur on a/b)
    if params.chrominance_noise > 0:
        from scipy.ndimage import gaussian_filter
        sigma = params.chrominance_noise * 3.0
        lab[..., 1] = gaussian_filter(lab[..., 1], sigma=sigma)
        lab[..., 2] = gaussian_filter(lab[..., 2], sigma=sigma)

    progress(0.8, "Converting back to RGB…")
    result_hwc = _from_lab(lab)

    progress(1.0, "Done")
    # Return as (3, H, W)
    return np.transpose(result_hwc, (2, 0, 1)).astype(np.float32)
