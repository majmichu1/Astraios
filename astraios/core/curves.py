"""Curves Transformation — spline-based tonal adjustment.

Uses monotone cubic interpolation (PCHIP) through user control points
to create a lookup table for fast per-pixel remapping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator

from astraios.core.device_manager import get_device_manager
from astraios.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)

# LUT resolution
LUT_SIZE = 65536


@dataclass
class CurvePoints:
    """Control points for a single curve channel.

    Points are (x, y) pairs in [0, 1].
    Always starts with (0, 0) and ends with (1, 1) by default.
    """

    points: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.0), (1.0, 1.0)]
    )

    def add_point(self, x: float, y: float) -> None:
        """Add a control point, maintaining sorted order by x."""
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        # Remove any existing point at this x (within tolerance)
        self.points = [(px, py) for px, py in self.points if abs(px - x) > 0.005]
        self.points.append((x, y))
        self.points.sort(key=lambda p: p[0])

    def remove_point(self, index: int) -> None:
        """Remove a control point by index. Cannot remove first or last."""
        if 0 < index < len(self.points) - 1:
            del self.points[index]

    def move_point(self, index: int, x: float, y: float) -> None:
        """Move a control point to a new position."""
        if index == 0:
            x = 0.0
        elif index == len(self.points) - 1:
            x = 1.0
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        self.points[index] = (x, y)
        self.points.sort(key=lambda p: p[0])

    def build_lut(self) -> np.ndarray:
        """Build a lookup table from the control points.

        Returns
        -------
        ndarray
            Float32 array of shape (LUT_SIZE,) mapping input [0, 1] to output [0, 1].
        """
        xs = np.array([p[0] for p in self.points])
        ys = np.array([p[1] for p in self.points])

        if len(xs) < 2:
            return np.linspace(0, 1, LUT_SIZE, dtype=np.float32)

        # Deduplicate x-values with tolerance to prevent PchipInterpolator crash
        if len(xs) > 2:
            keep = [True]
            for i in range(1, len(xs)):
                keep.append(abs(xs[i] - xs[i - 1]) > 1e-8)
            xs = xs[keep]
            ys = ys[keep]
            if len(xs) < 2:
                return np.linspace(0, 1, LUT_SIZE, dtype=np.float32)

        # Monotone cubic interpolation (PCHIP)
        interp = PchipInterpolator(xs, ys, extrapolate=False)
        lut_x = np.linspace(0, 1, LUT_SIZE)
        lut = interp(lut_x)

        # Handle extrapolation at edges
        lut = np.nan_to_num(lut, nan=0.0)
        return np.clip(lut, 0, 1).astype(np.float32)


@dataclass
class CurvesParams:
    """Parameters for curves transformation.

    Supports per-channel curves (RGB) plus a master curve (L).
    """

    master: CurvePoints = field(default_factory=CurvePoints)
    red: CurvePoints = field(default_factory=CurvePoints)
    green: CurvePoints = field(default_factory=CurvePoints)
    blue: CurvePoints = field(default_factory=CurvePoints)


def apply_curve_lut(channel: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply a LUT to a single channel using interpolation.

    Parameters
    ----------
    channel : ndarray
        Float32 array with values in [0, 1].
    lut : ndarray
        Float32 LUT of shape (LUT_SIZE,).

    Returns
    -------
    ndarray
        Remapped channel values.
    """
    # Scale to LUT indices
    indices = np.clip(channel * (LUT_SIZE - 1), 0, LUT_SIZE - 1)
    return np.interp(indices, np.arange(LUT_SIZE), lut).astype(np.float32)


def curves_transform(
    image: np.ndarray,
    params: CurvesParams | None = None,
    mask: Mask | None = None,
) -> np.ndarray:
    """Apply curves transformation to an image — GPU-accelerated."""
    if params is None:
        params = CurvesParams()

    dm = get_device_manager()
    if dm.is_gpu:
        return _curves_gpu(image, params, mask, dm)
    else:
        return _curves_cpu(image, params, mask)


@torch.no_grad()
def _apply_curve_lut_gpu(t: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Apply LUT to tensor using manual GPU interp (torch has no built-in interp)."""
    # LUT lookup via torch indexing: scale to LUT indices
    indices = torch.clamp((t * (LUT_SIZE - 1)).long(), 0, LUT_SIZE - 1)
    return lut[indices]


@torch.no_grad()
def _curves_gpu(
    image: np.ndarray,
    params: CurvesParams,
    mask: Mask | None,
    dm,
) -> np.ndarray:
    """GPU-accelerated curves transform."""
    original = image.copy()
    t = dm.from_numpy(image)
    master_lut = dm.from_numpy(params.master.build_lut())

    if image.ndim == 2:
        t = _apply_curve_lut_gpu(t, master_lut)
    else:
        channel_luts = [
            dm.from_numpy(params.red.build_lut()),
            dm.from_numpy(params.green.build_lut()),
            dm.from_numpy(params.blue.build_lut()),
        ]
        for ch in range(min(image.shape[0], 3)):
            t[ch] = _apply_curve_lut_gpu(t[ch], channel_luts[ch])
        for ch in range(image.shape[0]):
            t[ch] = _apply_curve_lut_gpu(t[ch], master_lut)

    result = torch.clamp(t, 0.0, 1.0).cpu().numpy().astype(np.float32)
    return apply_mask(original, result, mask)


def _curves_cpu(
    image: np.ndarray,
    params: CurvesParams,
    mask: Mask | None,
) -> np.ndarray:
    """CPU fallback for curves transform."""
    original = image.copy()
    result = image.copy()
    master_lut = params.master.build_lut()

    if image.ndim == 2:
        result = apply_curve_lut(result, master_lut)
    else:
        channel_luts = [
            params.red.build_lut(),
            params.green.build_lut(),
            params.blue.build_lut(),
        ]
        for ch in range(min(image.shape[0], 3)):
            result[ch] = apply_curve_lut(result[ch], channel_luts[ch])
        for ch in range(image.shape[0]):
            result[ch] = apply_curve_lut(result[ch], master_lut)

    result = np.clip(result, 0, 1)
    return apply_mask(original, result, mask)
