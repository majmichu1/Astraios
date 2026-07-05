"""Halo-B-Gon — bright-star halo reduction.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro), Copyright
Franklin Marek, GPL-3.0-or-later. https://github.com/setiastro/setiastrosuitepro
Darkens the smooth glow ("halo") around bright stars while preserving their
cores, by building an unsharp-mask-derived suppression mask from the image's
lightness channel and finishing with a per-level gamma darkening curve.

This is an exact port of SASv2's ``HaloProcessingThread.applyHaloReduction``:
the lightness mask is deliberately built in an 8-bit-like scale (the source
divides the already-[0,1] grayscale by 255, a quirk preserved here for
numerical fidelity with the original tool), followed by an unsharp mask and a
per-reduction-level gamma LUT (1.2 / 1.5 / 1.8 / 2.2). The halo effect mostly
comes from the final gamma curve pulling down faint halo pixels far more than
the bright star core.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from astraios.core.device_manager import get_device_manager
from astraios.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(f: float, m: str) -> None:
    pass


class HaloReductionLevel(IntEnum):
    """Halo suppression strength. Higher levels darken more aggressively."""

    EXTRA_LOW = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


# UI-facing display names, indexed by HaloReductionLevel value.
HALO_REDUCTION_LEVEL_NAMES = ["Extra Low", "Low", "Medium", "High"]

# Per-level darkening gamma (exponent applied to the masked image, > 1 darkens
# faint pixels — i.e. the halo — proportionally more than the bright core).
_GAMMAS = [1.2, 1.5, 1.8, 2.2]

_UNSHARP_SIGMA = 2.0


@dataclass
class HaloReductionParams:
    """Parameters for Halo-B-Gon halo reduction.

    Attributes
    ----------
    reduction_level : HaloReductionLevel
        Suppression strength (0=Extra Low .. 3=High). Higher levels use a
        steeper gamma-darkening curve and subtract more of the smoothed
        unsharp mask, more aggressively dimming halos (and, at high levels,
        faint background structure too).
    is_linear : bool
        Set True for un-stretched (linear) data. Applies a temporary
        ``x ** (1/5)`` gamma boost before computing the halo mask so the
        algorithm can "see" contrast in very dark linear data. Matches the
        original tool: the output remains in this boosted domain (it is not
        converted back to linear afterward).
    """

    reduction_level: HaloReductionLevel = HaloReductionLevel.EXTRA_LOW
    is_linear: bool = False


def _split_rgb_and_extra(t):
    """Split a (C, H, W) array/tensor into an (<=3, H, W) RGB part and any extra channels."""
    c = t.shape[0]
    rgb = t[:3] if c >= 3 else t
    extra = t[3:] if c > 3 else None
    return rgb, extra


# ---------- CPU path (numpy / OpenCV) ----------


def _reduce_halos_cpu(image: np.ndarray, level: int, is_linear: bool) -> np.ndarray:
    work = image
    if is_linear:
        with np.errstate(invalid="ignore"):
            work = np.power(np.clip(work, 0.0, 1.0), 1.0 / 5.0).astype(np.float32)

    extra = None
    if work.ndim == 2:
        light = work.astype(np.float32)
    else:
        rgb, extra = _split_rgb_and_extra(work)
        if rgb.shape[0] == 3:
            light = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]).astype(np.float32)
        else:
            light = rgb[0].astype(np.float32)
        work = rgb

    # SASv2 divides the already-[0,1] lightness by 255 here — preserved for fidelity.
    light = light / 255.0
    blurred = cv2.GaussianBlur(light, (0, 0), sigmaX=_UNSHARP_SIGMA)
    unsharp = 1.66 * light - 0.66 * blurred
    inv = 1.0 - unsharp
    dup = cv2.GaussianBlur(unsharp, (0, 0), sigmaX=_UNSHARP_SIGMA)
    scale = level * 0.33
    enhanced_mask = (inv - dup * scale).astype(np.float32)

    # Broadcasts (H, W) mask against (3, H, W) or (H, W) work.
    masked = work * enhanced_mask

    g = _GAMMAS[level]
    lut = np.clip((np.linspace(0, 1, 256, dtype=np.float32) ** g) * 255.0, 0, 255).astype(np.uint8)
    u8 = (np.clip(masked, 0.0, 1.0) * 255.0).astype(np.uint8, copy=False)
    mapped = lut[u8].astype(np.float32) / 255.0
    out = np.clip(mapped, 0.0, 1.0).astype(np.float32)

    if extra is not None:
        out = np.concatenate([out, extra], axis=0)
    return out


# ---------- GPU path (torch) ----------


def _gaussian_blur_gpu(t: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur matching cv2.GaussianBlur(ksize=(0,0), sigmaX=sigma)
    for float32 input (OpenCV auto-derives ksize = round(sigma*4*2+1)|1 for non-8U
    depth); reflect-101 padding matches OpenCV's default border handling.
    """
    ksize = int(round(sigma * 4 * 2 + 1))
    if ksize % 2 == 0:
        ksize += 1
    radius = ksize // 2
    x = torch.arange(-radius, radius + 1, device=t.device, dtype=t.dtype)
    k = torch.exp(-(x**2) / (2.0 * sigma * sigma))
    k = k / k.sum()
    t4 = t.unsqueeze(0).unsqueeze(0)
    kx = k.view(1, 1, 1, -1)
    ky = k.view(1, 1, -1, 1)
    t4 = F.pad(t4, (radius, radius, 0, 0), mode="reflect")
    t4 = F.conv2d(t4, kx)
    t4 = F.pad(t4, (0, 0, radius, radius), mode="reflect")
    t4 = F.conv2d(t4, ky)
    return t4.squeeze(0).squeeze(0)


@torch.no_grad()
def _reduce_halos_gpu(image: np.ndarray, level: int, is_linear: bool, dm) -> np.ndarray:
    t = dm.from_numpy(image)
    if is_linear:
        t = torch.clamp(t, 0.0, 1.0) ** (1.0 / 5.0)

    extra = None
    if t.ndim == 2:
        light = t
        work = t
    else:
        rgb, extra = _split_rgb_and_extra(t)
        if rgb.shape[0] == 3:
            light = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        else:
            light = rgb[0]
        work = rgb

    light = light / 255.0
    blurred = _gaussian_blur_gpu(light, _UNSHARP_SIGMA)
    unsharp = 1.66 * light - 0.66 * blurred
    inv = 1.0 - unsharp
    dup = _gaussian_blur_gpu(unsharp, _UNSHARP_SIGMA)
    scale = level * 0.33
    enhanced_mask = inv - dup * scale

    masked = work * enhanced_mask

    g = _GAMMAS[level]
    lin = torch.linspace(0.0, 1.0, 256, device=dm.device, dtype=torch.float32)
    # Quantize the LUT to integer 0-255 levels, matching the uint8 LUT of the CPU path.
    lut = torch.clamp((lin**g) * 255.0, 0.0, 255.0).to(torch.int64).to(torch.float32)
    u8_idx = (torch.clamp(masked, 0.0, 1.0) * 255.0).to(torch.int64)
    mapped = lut[u8_idx] / 255.0
    out = torch.clamp(mapped, 0.0, 1.0)

    if extra is not None:
        out = torch.cat([out, extra], dim=0)
    return out.cpu().numpy().astype(np.float32)


# ---------- Public API ----------


def reduce_halos(
    data: np.ndarray,
    params: HaloReductionParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    """Reduce bright-star halos.

    Parameters
    ----------
    data : ndarray
        Image data, ``(H, W)`` mono or ``(C, H, W)`` color, float32 in
        ``[0, 1]``. Mono images are processed directly (the original tool
        supports mono, it's not a no-op).
    params : HaloReductionParams, optional
        Reduction settings. Defaults to the mildest (Extra Low) setting.
    mask : Mask, optional
        If given, blends the result with the original per
        ``result = processed * mask + original * (1 - mask)``.
    progress : callable, optional
        ``progress(fraction, message)`` callback.

    Returns
    -------
    ndarray
        Same shape as ``data`` (extra channels beyond the first 3 in a color
        stack, if any, pass through unmodified).
    """
    if params is None:
        params = HaloReductionParams()

    data = np.clip(np.asarray(data, dtype=np.float32), 0.0, 1.0)
    level = max(0, min(3, int(params.reduction_level)))
    is_linear = bool(params.is_linear)

    level_name = HALO_REDUCTION_LEVEL_NAMES[level]
    progress(0.0, f"Halo-B-Gon: computing suppression mask ({level_name})...")
    dm = get_device_manager()
    if dm.is_gpu:
        out = _reduce_halos_gpu(data, level, is_linear, dm)
    else:
        out = _reduce_halos_cpu(data, level, is_linear)

    progress(1.0, "Halo-B-Gon complete")
    return apply_mask(data, out, mask)
