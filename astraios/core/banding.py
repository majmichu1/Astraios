"""Banding Reduction — remove horizontal and vertical banding artifacts.

Banding is common in CMOS sensors, appearing as faint stripes in the
background. This module detects and subtracts per-row/column offsets
using sigma-clipped median analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.ndimage import median_filter

from astraios.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


@dataclass
class BandingParams:
    """Parameters for banding reduction."""

    horizontal: bool = True  # remove horizontal banding (row offsets)
    vertical: bool = False  # remove vertical banding (column offsets)
    amount: float = 1.0  # correction strength (0-1)
    protection_sigma: float = 3.0  # sigma clip to reject bright objects


def _compute_line_offsets(
    data_2d: np.ndarray,
    axis: int,
    protection_sigma: float,
) -> np.ndarray:
    """Compute per-line (row or column) offset from neighbors.

    Uses batched median + MAD sigma clipping to reject bright objects, then
    computes the offset of each line relative to a scipy median-filtered version.

    Parameters
    ----------
    data_2d : ndarray
        2D channel data, shape (H, W).
    axis : int
        0 = compute row offsets (horizontal banding),
        1 = compute column offsets (vertical banding).
    protection_sigma : float
        Sigma clipping threshold for object rejection.

    Returns
    -------
    ndarray
        Per-line offsets to subtract.
    """
    # Orient so lines run along axis=1 for vectorised ops
    work = data_2d if axis == 0 else data_2d.T  # (n_lines, n_pixels)

    # Batched iterative sigma clip — 3 passes, all lines at once
    valid = work.copy()
    mask = np.ones_like(valid, dtype=bool)
    for _ in range(3):
        meds = np.nanmedian(np.where(mask, valid, np.nan), axis=1)        # (n_lines,)
        abs_dev = np.abs(valid - meds[:, np.newaxis])
        mads = np.nanmedian(np.where(mask, abs_dev, np.nan), axis=1)      # (n_lines,)
        sigma = np.maximum(mads * 1.4826, 1e-10)                       # (n_lines,)
        new_mask = abs_dev < protection_sigma * sigma[:, np.newaxis]
        if new_mask.sum() < 3 * work.shape[0]:
            break
        mask = new_mask

    # Final median per line using only surviving pixels
    line_medians = np.array(
        [np.median(row[m]) if m.any() else np.median(row) for row, m in zip(work, mask)],
        dtype=np.float32,
    )

    # Smooth reference via scipy median_filter (replaces per-element Python loop)
    n_lines = len(line_medians)
    window = max(5, n_lines // 20)
    if window % 2 == 0:
        window += 1
    smooth = median_filter(line_medians, size=window, mode="nearest").astype(np.float32)

    return line_medians - smooth


def banding_reduction(
    image: np.ndarray,
    params: BandingParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    """Remove horizontal and/or vertical banding from an image.

    Parameters
    ----------
    image : ndarray
        Image data, shape (H, W) or (C, H, W), values in [0, 1].
    params : BandingParams, optional
        Banding reduction parameters.
    mask : Mask, optional
        Selective processing mask.
    progress : callable
        Progress callback.

    Returns
    -------
    ndarray
        Corrected image.
    """
    if params is None:
        params = BandingParams()

    if not params.horizontal and not params.vertical:
        return image

    original = image.copy()
    result = image.copy()

    if result.ndim == 2:
        channels = [(result, lambda d: None)]
    else:
        channels = [(result[ch], ch) for ch in range(result.shape[0])]

    n_ch = len(channels)
    step = 0
    total_steps = n_ch * (int(params.horizontal) + int(params.vertical))

    for ch_idx, (ch_data, _) in enumerate(channels):
        if params.horizontal:
            progress(step / total_steps, f"Horizontal banding ch{ch_idx + 1}...")
            offsets = _compute_line_offsets(ch_data, axis=0, protection_sigma=params.protection_sigma)
            correction = offsets[:, np.newaxis] * params.amount
            ch_data -= correction
            step += 1

        if params.vertical:
            progress(step / total_steps, f"Vertical banding ch{ch_idx + 1}...")
            offsets = _compute_line_offsets(ch_data, axis=1, protection_sigma=params.protection_sigma)
            correction = offsets[np.newaxis, :] * params.amount
            ch_data -= correction
            step += 1

        # Write back
        if result.ndim == 2:
            result[:] = ch_data
        else:
            result[ch_idx] = ch_data

    result = np.clip(result, 0, 1)
    progress(1.0, "Banding reduction complete")
    return apply_mask(original, result, mask)
