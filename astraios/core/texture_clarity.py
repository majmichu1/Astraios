"""Texture & Clarity — local-contrast texture enhancement.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro),
Copyright Franklin Marek, GPL-3.0-or-later.
https://github.com/setiastro/setiastrosuitepro

Two independent effects, both GPU-accelerated via the device manager:

* **Texture** — a difference-of-Gaussians band-pass sharpen. Boosts detail
  at a specific spatial scale controlled by `texture_radius`.
* **Clarity** — a bilateral-filter-based local-contrast boost, restricted to
  midtones by default via a parabolic luminance mask (classic "clarity"
  behavior — shadows and highlights are left alone unless `mask_strength`
  is reduced).

Color images are processed on luminance only and the per-pixel gain is
re-applied to all three channels (color ratio preserved). Images are
float32 in [0, 1]; mono is (H, W), color is (C, H, W) channels-first
(SASpro's original works on (H, W, 3) — the math is unchanged, only the
channel axis moved).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
import torch

from astraios.core.device_manager import get_device_manager
from astraios.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


@dataclass
class TextureClarityParams:
    """Parameters for the Texture & Clarity local-contrast tool.

    Attributes
    ----------
    texture_amount : float
        Strength of the difference-of-Gaussians texture boost, roughly
        [-1, 1]. Positive values add texture at `texture_radius`'s scale;
        negative values smooth it away.
    texture_radius : float
        Band-pass radius in pixels (0.1-10.0 in the original UI). Controls
        which detail scale ("texture" vs coarse structure) is boosted.
    clarity_amount : float
        Strength of the bilateral local-contrast ("clarity") effect,
        roughly [-1, 1].
    clarity_radius : float
        Bilateral filter radius in pixels (0.1-10.0 in the original UI);
        scales the filter's spatial sigma.
    mask_strength : float
        Blend between midtone-only masking (1.0 — classic "clarity"
        behavior that leaves shadows/highlights untouched) and uniform
        application everywhere (0.0).
    """

    texture_amount: float = 0.0
    texture_radius: float = 1.0
    clarity_amount: float = 0.0
    clarity_radius: float = 3.0
    mask_strength: float = 1.0


# ---------------------------------------------------------------------
# Gaussian blur — separable GPU convolution, cv2 CPU fallback
# ---------------------------------------------------------------------


def _make_gaussian_kernel_1d(sigma: float, device: torch.device) -> torch.Tensor:
    sigma = max(sigma, 0.5)
    ksize = int(np.ceil(sigma * 3)) * 2 + 1
    ksize = max(ksize, 3)
    x = torch.arange(ksize, dtype=torch.float32, device=device) - ksize // 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


@torch.no_grad()
def _gaussian_blur_gpu_tensor(channel: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur on a (H, W) GPU tensor, reflect-padded."""
    device = channel.device
    k1d = _make_gaussian_kernel_1d(sigma, device)
    pad = k1d.shape[0] // 2
    t = channel.unsqueeze(0).unsqueeze(0)
    t_padded = torch.nn.functional.pad(t, (pad, pad, pad, pad), mode="reflect")
    kh = k1d.reshape(1, 1, 1, -1)
    blurred = torch.nn.functional.conv2d(t_padded, kh, padding=0)
    kv = k1d.reshape(1, 1, -1, 1)
    blurred = torch.nn.functional.conv2d(blurred, kv, padding=0)
    return blurred.squeeze(0).squeeze(0)


def _gaussian_blur_cpu(channel: np.ndarray, sigma: float) -> np.ndarray:
    """CPU Gaussian blur matching SASpro's odd-kernel-size convention."""
    k = int(2 * round(3 * sigma) + 1) | 1
    return cv2.GaussianBlur(channel, (k, k), sigma).astype(np.float32)


# ---------------------------------------------------------------------
# Bilateral filter — tiled GPU implementation, cv2 CPU fallback
#
# SASpro always filters with a fixed 9x9 window (d=9); only sigma_space
# changes with `clarity_radius`. That bounds a GPU unfold-based bilateral
# filter to a small, VRAM-safe per-tile working set even on 4k+ images.
# ---------------------------------------------------------------------

_BILATERAL_D = 9
_BILATERAL_TILE = 768


@torch.no_grad()
def _bilateral_filter_gpu(
    channel: np.ndarray, sigma_color: float, sigma_space: float, device: torch.device
) -> np.ndarray:
    d = _BILATERAL_D
    r = d // 2
    tile = _BILATERAL_TILE

    t = torch.from_numpy(np.ascontiguousarray(channel, dtype=np.float32))
    t = t.unsqueeze(0).unsqueeze(0).to(device)
    padded = torch.nn.functional.pad(t, (r, r, r, r), mode="reflect")[0, 0]
    h, w = channel.shape

    ax = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(ax, ax, indexing="ij")
    spatial = torch.exp(-(gx**2 + gy**2) / (2.0 * max(sigma_space, 1e-6) ** 2))
    spatial = spatial.reshape(1, d * d, 1, 1)

    out = torch.empty((h, w), device=device, dtype=torch.float32)
    for y0 in range(0, h, tile):
        y1 = min(y0 + tile, h)
        for x0 in range(0, w, tile):
            x1 = min(x0 + tile, w)
            block = padded[y0 : y1 + 2 * r, x0 : x1 + 2 * r].unsqueeze(0).unsqueeze(0)
            bh, bw = y1 - y0, x1 - x0
            patches = torch.nn.functional.unfold(block, kernel_size=d)
            patches = patches.view(1, d * d, bh, bw)
            center = padded[y0 + r : y1 + r, x0 + r : x1 + r].unsqueeze(0).unsqueeze(0)
            diff = patches - center
            range_w = torch.exp(-(diff**2) / (2.0 * max(sigma_color, 1e-6) ** 2))
            weight = range_w * spatial
            wsum = weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
            filt = (weight * patches).sum(dim=1, keepdim=True) / wsum
            out[y0:y1, x0:x1] = filt[0, 0]

    return out.cpu().numpy().astype(np.float32)


def _bilateral_filter_cpu(
    channel: np.ndarray, sigma_color: float, sigma_space: float
) -> np.ndarray:
    return cv2.bilateralFilter(
        np.ascontiguousarray(channel, dtype=np.float32),
        d=_BILATERAL_D,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    ).astype(np.float32)


# ---------------------------------------------------------------------
# processing core (mirrors SASpro's _apply_texture / _apply_clarity)
# ---------------------------------------------------------------------


def _midtone_mask(image: np.ndarray, strength: float) -> np.ndarray:
    """strength=1 -> classic midtone-only mask; strength=0 -> apply everywhere."""
    full = np.clip(1.0 - 4.0 * (image - 0.5) ** 2, 0.0, 1.0)
    return float(strength) * full + (1.0 - float(strength)) * np.ones_like(full)


def _apply_texture(image: np.ndarray, amount: float, radius: float, dm) -> np.ndarray:
    """Difference-of-Gaussians band-pass sharpening."""
    if abs(amount) < 0.001:
        return image
    img = np.ascontiguousarray(np.nan_to_num(image), dtype=np.float32)
    s1, s2 = radius, radius * 2.0
    if dm.is_gpu:
        t = torch.from_numpy(img).to(dm.device)
        b1 = _gaussian_blur_gpu_tensor(t, s1)
        b2 = _gaussian_blur_gpu_tensor(t, s2)
        out = t + (b1 - b2) * 2.0 * amount
        return torch.clamp(out, 0.0, 1.0).cpu().numpy().astype(np.float32)
    b1 = _gaussian_blur_cpu(img, s1)
    b2 = _gaussian_blur_cpu(img, s2)
    return np.clip(img + (b1 - b2) * 2.0 * amount, 0.0, 1.0).astype(np.float32)


def _apply_clarity(
    image: np.ndarray, amount: float, radius: float, mask_strength: float, dm
) -> np.ndarray:
    """Bilateral-based local contrast with midtone mask."""
    if abs(amount) < 0.001:
        return image
    img = np.ascontiguousarray(np.nan_to_num(image), dtype=np.float32)
    sigma_space = radius * 10.0
    sigma_color = 0.1

    if sigma_space > 10.0:
        scale = max(0.1, min(5.0 / sigma_space, 1.0))
        h, w = img.shape[:2]
        small = cv2.resize(
            img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
        )
        if dm.is_gpu:
            small_filt = _bilateral_filter_gpu(small, sigma_color, sigma_space * scale, dm.device)
        else:
            small_filt = _bilateral_filter_cpu(small, sigma_color, sigma_space * scale)
        base = cv2.resize(small_filt, (w, h), interpolation=cv2.INTER_LINEAR)
    elif dm.is_gpu:
        base = _bilateral_filter_gpu(img, sigma_color, sigma_space, dm.device)
    else:
        base = _bilateral_filter_cpu(img, sigma_color, sigma_space)

    mask = _midtone_mask(img, mask_strength)
    return np.clip(img + amount * (img - base) * mask, 0.0, 1.0).astype(np.float32)


def _compute(image: np.ndarray, params: TextureClarityParams, dm) -> np.ndarray:
    out = _apply_texture(image, params.texture_amount, params.texture_radius, dm)
    out = _apply_clarity(
        out, params.clarity_amount, params.clarity_radius, params.mask_strength, dm
    )
    return out


# ---------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------


def apply_texture_clarity(
    data: np.ndarray,
    params: TextureClarityParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    """Apply texture + clarity local-contrast enhancement.

    Parameters
    ----------
    data : ndarray
        Image data, float32 in [0, 1]. Mono (H, W) or color (C, H, W).
    params : TextureClarityParams, optional
        Effect parameters. Defaults to a no-op (both amounts 0).
    mask : Mask, optional
        Restrict the effect to a region; see `astraios.core.masks`.
    progress : callable, optional
        `progress(fraction, message)` callback.

    Returns
    -------
    ndarray
        Processed image, same shape/dtype as `data`.
    """
    if params is None:
        params = TextureClarityParams()

    dm = get_device_manager()
    progress(0.05, "Preparing…")

    if data.ndim == 3 and data.shape[0] >= 3:
        r, g, b = data[0], data[1], data[2]
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        progress(0.3, "Applying texture…")
        luminance_new = _compute(luminance, params, dm)
        progress(0.8, "Recombining color…")
        ratio = luminance_new / (luminance + 1e-7)
        out = np.clip(data[:3] * ratio[np.newaxis, :, :], 0.0, 1.0).astype(np.float32)
        if data.shape[0] > 3:
            out = np.concatenate([out, data[3:]], axis=0)
    else:
        mono = data.squeeze() if data.ndim == 3 else data
        progress(0.3, "Applying texture…")
        out = _compute(mono, params, dm)
        if data.ndim == 3:
            out = out[np.newaxis, :, :]
        out = out.astype(np.float32, copy=False)

    progress(0.95, "Blending…")
    result = apply_mask(data, out, mask)
    progress(1.0, "Texture & Clarity complete")
    return result
