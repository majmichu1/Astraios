"""Saturation / Chroma Tool — hue-selective saturation or chroma adjustment.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro), Copyright
Franklin Marek, GPL-3.0-or-later. https://github.com/setiastro/setiastrosuitepro
Hue-selective saturation (HSV) or chroma (CIE Lab) boost/cut driven by a
PCHIP-interpolated curve over the hue wheel.

The two modes mirror the original tool exactly:

- ``SATURATION_HSV``: converts to HSV and multiplies the S channel by a
  per-hue factor read off the curve. Simple and fast, but large boosts can
  shift the perceived hue slightly because HSV saturation is not perceptually
  uniform.
- ``CHROMA_LAB``: converts to CIE Lab and scales the (a, b) chroma vector by
  the same per-hue factor. Perceptually cleaner (less hue shift) at the cost
  of a slightly heavier conversion.

The hue curve is a list of ``(hue_degrees, multiplier)`` control points
(0-360 degrees, 0-3x multiplier) interpolated with a monotone cubic spline
(PCHIP), exactly as the original tool's draggable curve widget did. The
first and last points are expected to share the same multiplier so the curve
wraps seamlessly at the 0/360 degree seam.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

import cv2
import numpy as np
import torch
from scipy.interpolate import PchipInterpolator

from astraios.core.color_tools import _hsv_to_rgb_gpu, _rgb_to_hsv_gpu
from astraios.core.device_manager import get_device_manager
from astraios.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(f: float, m: str) -> None:
    pass


# Curve value bounds and resolution (matches the original tool exactly).
CURVE_MIN = 0.0
CURVE_MAX = 3.0
CURVE_NEUTRAL = 1.0
LUT_RESOLUTION = 4096


def _default_curve_points() -> list[tuple[float, float]]:
    """Six evenly-spaced neutral control points (red/yellow/green/cyan/blue/magenta)."""
    return [(h, CURVE_NEUTRAL) for h in (0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0)]


class SatChromaMode(Enum):
    """Which color space the hue curve is applied in."""

    SATURATION_HSV = auto()  # multiply HSV saturation per-hue
    CHROMA_LAB = auto()  # multiply CIE Lab chroma (a, b) per-hue


@dataclass
class SatChromaParams:
    """Parameters for the hue-selective Saturation / Chroma tool.

    Attributes
    ----------
    mode : SatChromaMode
        SATURATION_HSV adjusts HSV saturation; CHROMA_LAB adjusts CIE Lab
        chroma (perceptually cleaner, less hue shift).
    curve_points : list[tuple[float, float]]
        (hue_degrees 0-360, multiplier 0-3) control points defining the
        per-hue adjustment curve, interpolated with PCHIP. First and last
        points should carry the same multiplier to wrap cleanly at 0/360.
    strength : float
        Global multiplier (0-3) applied on top of the curve's own multiplier;
        1.0 leaves the curve as authored, 0 disables all adjustment.
    """

    mode: SatChromaMode = SatChromaMode.SATURATION_HSV
    curve_points: list[tuple[float, float]] = field(default_factory=_default_curve_points)
    strength: float = 1.0


def _pchip_lut(points: list[tuple[float, float]], n: int = LUT_RESOLUTION) -> np.ndarray:
    """Build an n-sample LUT over normalized hue [0, 1] from curve control points."""
    pts = sorted(points, key=lambda p: p[0])
    xs = np.array([p[0] / 360.0 for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    # De-duplicate identical x values (PchipInterpolator requires strictly increasing x).
    if len(xs) > 1:
        keep = np.concatenate([[True], np.diff(xs) > 1e-9])
        xs, ys = xs[keep], ys[keep]
    xi = np.linspace(0.0, 1.0, n, dtype=np.float64)
    if len(xs) < 2:
        lut = np.full(n, ys[0] if len(ys) else CURVE_NEUTRAL, dtype=np.float64)
    else:
        lut = PchipInterpolator(xs, ys, extrapolate=True)(xi)
    return np.clip(lut, CURVE_MIN, CURVE_MAX).astype(np.float32)


def _build_lut(params: SatChromaParams) -> np.ndarray:
    lut = _pchip_lut(params.curve_points)
    return np.clip(lut * float(params.strength), CURVE_MIN, CURVE_MAX).astype(np.float32)


# ---------- CPU path (numpy / OpenCV / scikit-image) ----------


def _apply_saturation_hsv_cpu(img_chw: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """HSV saturation adjustment. img_chw: (3, H, W) float32 in [0, 1]."""
    img_hwc = np.ascontiguousarray(np.transpose(img_chw, (1, 2, 0)))
    hsv = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue_norm = hsv[:, :, 0] / 360.0
    idx = np.clip((hue_norm * (len(lut) - 1)).astype(np.int32), 0, len(lut) - 1)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * lut[idx], 0.0, 1.0)
    out_hwc = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    out_hwc = np.clip(out_hwc, 0.0, 1.0).astype(np.float32)
    return np.ascontiguousarray(np.transpose(out_hwc, (2, 0, 1)))


def _rgb2lab_manual(rgb_hwc: np.ndarray) -> np.ndarray:
    lin = np.where(rgb_hwc <= 0.04045, rgb_hwc / 12.92, ((rgb_hwc + 0.055) / 1.055) ** 2.4)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = lin @ m.T
    xyz /= np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    f = np.where(xyz > 0.008856, xyz ** (1 / 3), 7.787 * xyz + 16 / 116)
    lightness = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([lightness, a, b], axis=-1)


def _lab2rgb_manual(lab: np.ndarray) -> np.ndarray:
    lightness, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (lightness + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    x = np.where(fx**3 > 0.008856, fx**3, (fx - 16 / 116) / 7.787) * 0.95047
    y = np.where(fy**3 > 0.008856, fy**3, (fy - 16 / 116) / 7.787) * 1.00000
    z = np.where(fz**3 > 0.008856, fz**3, (fz - 16 / 116) / 7.787) * 1.08883
    xyz = np.stack([x, y, z], axis=-1)
    m_inv = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=np.float32,
    )
    lin = xyz @ m_inv.T
    rgb = np.where(
        lin <= 0.0031308, 12.92 * lin, 1.055 * np.clip(lin, 0, None) ** (1 / 2.4) - 0.055
    )
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _apply_chroma_lab_cpu(img_chw: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """CIE Lab chroma adjustment. img_chw: (3, H, W) float32 in [0, 1]."""
    img_hwc = np.ascontiguousarray(np.transpose(img_chw, (1, 2, 0))).astype(np.float32)
    try:
        from skimage import color as sk

        lab = sk.rgb2lab(img_hwc)
    except ImportError:
        lab = _rgb2lab_manual(img_hwc)

    a_ch, b_ch = lab[..., 1], lab[..., 2]
    hue_norm = (np.arctan2(b_ch, a_ch) / (2 * math.pi)) % 1.0
    idx = np.clip((hue_norm * (len(lut) - 1)).astype(np.int32), 0, len(lut) - 1)
    mult = lut[idx]
    lab[..., 1] = a_ch * mult
    lab[..., 2] = b_ch * mult

    try:
        from skimage import color as sk

        out_hwc = sk.lab2rgb(lab).astype(np.float32)
    except ImportError:
        out_hwc = _lab2rgb_manual(lab)

    out_hwc = np.clip(out_hwc, 0.0, 1.0).astype(np.float32)
    return np.ascontiguousarray(np.transpose(out_hwc, (2, 0, 1)))


# ---------- GPU path (torch) ----------


@torch.no_grad()
def _rgb_to_lab_gpu(t: torch.Tensor) -> torch.Tensor:
    """t: (3, H, W) float in [0, 1] -> Lab (3, H, W), L in [0,100], a/b roughly [-128,127]."""
    lin = torch.where(t <= 0.04045, t / 12.92, ((t + 0.055) / 1.055) ** 2.4)
    r, g, b = lin[0], lin[1], lin[2]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    x = x / 0.95047
    z = z / 1.08883

    def f(v: torch.Tensor) -> torch.Tensor:
        safe_pow = torch.pow(torch.clamp(v, min=1e-12), 1.0 / 3.0)
        return torch.where(v > 0.008856, safe_pow, 7.787 * v + 16.0 / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    lightness = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    bb = 200.0 * (fy - fz)
    return torch.stack([lightness, a, bb], dim=0)


@torch.no_grad()
def _lab_to_rgb_gpu(lab: torch.Tensor) -> torch.Tensor:
    lightness, a, b = lab[0], lab[1], lab[2]
    fy = (lightness + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    def finv(v: torch.Tensor) -> torch.Tensor:
        cube = v**3
        safe_lin = (v - 16.0 / 116.0) / 7.787
        return torch.where(cube > 0.008856, cube, safe_lin)

    x = finv(fx) * 0.95047
    y = finv(fy) * 1.00000
    z = finv(fz) * 1.08883

    r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    bch = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    lin = torch.stack([r, g, bch], dim=0)
    rgb = torch.where(
        lin <= 0.0031308, 12.92 * lin, 1.055 * torch.clamp(lin, min=0.0) ** (1.0 / 2.4) - 0.055
    )
    return torch.clamp(rgb, 0.0, 1.0)


@torch.no_grad()
def _apply_saturation_hsv_gpu(img_chw: np.ndarray, lut: np.ndarray, dm) -> np.ndarray:
    hsv = _rgb_to_hsv_gpu(img_chw, dm)  # (3,H,W): H in [0,360), S,V in [0,1]
    lut_t = torch.as_tensor(lut, device=dm.device, dtype=torch.float32)
    hue_norm = hsv[0] / 360.0
    idx = torch.clamp((hue_norm * (len(lut) - 1)).long(), 0, len(lut) - 1)
    hsv[1] = torch.clamp(hsv[1] * lut_t[idx], 0.0, 1.0)
    rgb = _hsv_to_rgb_gpu(hsv)
    return rgb.cpu().numpy().astype(np.float32)


@torch.no_grad()
def _apply_chroma_lab_gpu(img_chw: np.ndarray, lut: np.ndarray, dm) -> np.ndarray:
    t = dm.from_numpy(img_chw)
    lab = _rgb_to_lab_gpu(t)
    lut_t = torch.as_tensor(lut, device=dm.device, dtype=torch.float32)
    a_ch, b_ch = lab[1], lab[2]
    hue_norm = torch.remainder(torch.atan2(b_ch, a_ch) / (2 * math.pi), 1.0)
    idx = torch.clamp((hue_norm * (len(lut) - 1)).long(), 0, len(lut) - 1)
    mult = lut_t[idx]
    lab = torch.stack([lab[0], a_ch * mult, b_ch * mult], dim=0)
    rgb = _lab_to_rgb_gpu(lab)
    return rgb.cpu().numpy().astype(np.float32)


# ---------- Public API ----------


def apply_sat_chroma(
    data: np.ndarray,
    params: SatChromaParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    """Apply hue-selective saturation or chroma adjustment.

    Parameters
    ----------
    data : ndarray
        Image data, ``(H, W)`` mono or ``(C, H, W)`` color (``C >= 3``),
        float32 in ``[0, 1]``.
    params : SatChromaParams, optional
        Adjustment settings. Defaults to a flat (no-op) curve.
    mask : Mask, optional
        If given, blends the result with the original per
        ``result = processed * mask + original * (1 - mask)``.
    progress : callable, optional
        ``progress(fraction, message)`` callback.

    Returns
    -------
    ndarray
        Same shape/dtype as ``data``. Mono input is returned unchanged
        (saturation/chroma has no meaning without color).
    """
    if params is None:
        params = SatChromaParams()

    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 3 or data.shape[0] < 3:
        log.warning("SatChroma requires a color image with >= 3 channels; returning input as-is")
        return data

    progress(0.0, "Building hue curve...")
    lut = _build_lut(params)

    rgb = data[:3]
    dm = get_device_manager()
    if dm.is_gpu:
        progress(0.2, "Adjusting color (GPU)...")
        if params.mode == SatChromaMode.SATURATION_HSV:
            out_rgb = _apply_saturation_hsv_gpu(rgb, lut, dm)
        else:
            out_rgb = _apply_chroma_lab_gpu(rgb, lut, dm)
    else:
        progress(0.2, "Adjusting color (CPU)...")
        if params.mode == SatChromaMode.SATURATION_HSV:
            out_rgb = _apply_saturation_hsv_cpu(rgb, lut)
        else:
            out_rgb = _apply_chroma_lab_cpu(rgb, lut)

    if data.shape[0] > 3:
        result = np.concatenate([out_rgb, data[3:]], axis=0)
    else:
        result = out_rgb

    progress(1.0, "SatChroma complete")
    return apply_mask(data, result, mask)
