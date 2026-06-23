"""Chromatic Aberration Correction — detect and correct lateral CA.

Lateral chromatic aberration causes red and blue channels to be slightly
shifted relative to green.  This module can either auto-detect the shift
by matching star centroids across channels, or apply user-supplied manual
shifts.  Sub-pixel translation is performed with ``cv2.warpAffine`` to
preserve image quality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
from scipy import ndimage

from astraios.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(f: float, m: str) -> None:
    pass


@dataclass
class CAParams:
    """Parameters for chromatic aberration correction.

    Attributes
    ----------
    auto_detect : bool
        If *True*, automatically detect the R/B shift relative to G by
        matching star centroids across channels.
    red_shift_x : float
        Manual red channel shift in pixels along the x-axis.
    red_shift_y : float
        Manual red channel shift in pixels along the y-axis.
    blue_shift_x : float
        Manual blue channel shift in pixels along the x-axis.
    blue_shift_y : float
        Manual blue channel shift in pixels along the y-axis.
    max_correction : float
        Maximum allowed shift in pixels (caps both auto and manual shifts).
    """

    auto_detect: bool = True
    red_shift_x: float = 0.0
    red_shift_y: float = 0.0
    blue_shift_x: float = 0.0
    blue_shift_y: float = 0.0
    max_correction: float = 3.0


def _detect_peaks(
    channel: np.ndarray,
    threshold_sigma: float = 5.0,
    max_peaks: int = 200,
) -> np.ndarray:
    """Detect bright peaks in a single 2D channel using labelling.

    Parameters
    ----------
    channel : np.ndarray
        2D float32 array, values in [0, 1].
    threshold_sigma : float
        Detection threshold in MAD-estimated sigma above the median.
    max_peaks : int
        Maximum number of peaks to return (brightest first).

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) containing (x, y) centroids, sorted by
        descending peak brightness.
    """
    med = np.median(channel)
    mad = np.median(np.abs(channel - med))
    noise = max(mad * 1.4826, 1e-8)
    threshold = med + threshold_sigma * noise
    threshold = min(threshold, 0.95)

    binary = channel > threshold
    labelled, n_labels = ndimage.label(binary)

    if n_labels == 0:
        return np.empty((0, 2), dtype=np.float64)

    # Compute center of mass for each labelled region
    centroids = ndimage.center_of_mass(channel, labelled, range(1, n_labels + 1))

    # Get peak brightness for sorting
    peaks = ndimage.maximum(channel, labelled, range(1, n_labels + 1))

    # Convert to (x, y) — center_of_mass returns (row, col) = (y, x)
    results = []
    for (cy, cx), peak in zip(centroids, peaks):
        results.append((cx, cy, peak))

    # Sort by brightness descending
    results.sort(key=lambda t: -t[2])
    results = results[:max_peaks]

    return np.array([(r[0], r[1]) for r in results], dtype=np.float64)


def _match_centroids(
    ref_pts: np.ndarray,
    target_pts: np.ndarray,
    max_dist: float = 10.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Match nearest-neighbor centroids between two point sets.

    Parameters
    ----------
    ref_pts : np.ndarray
        Reference centroids, shape (N, 2) as (x, y).
    target_pts : np.ndarray
        Target centroids, shape (M, 2) as (x, y).
    max_dist : float
        Maximum matching distance in pixels.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of (ref_point, target_point) matched pairs.
    """
    if len(ref_pts) == 0 or len(target_pts) == 0:
        return []

    from scipy.spatial.distance import cdist

    dists = cdist(ref_pts, target_pts)
    matches = []
    used_target: set[int] = set()

    for i in range(len(ref_pts)):
        best_j = -1
        best_d = float("inf")
        for j in range(len(target_pts)):
            if j not in used_target and dists[i, j] < best_d:
                best_d = dists[i, j]
                best_j = j
        if best_j >= 0 and best_d < max_dist:
            matches.append((ref_pts[i], target_pts[best_j]))
            used_target.add(best_j)

    return matches


def _compute_median_offset(
    matches: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    """Compute the median centroid offset from matched pairs.

    Parameters
    ----------
    matches : list[tuple[np.ndarray, np.ndarray]]
        Matched (ref, target) centroid pairs.

    Returns
    -------
    tuple[float, float]
        Median (dx, dy) offset: target - ref.
    """
    if not matches:
        return 0.0, 0.0

    dx_list = [t[0] - r[0] for r, t in matches]
    dy_list = [t[1] - r[1] for r, t in matches]

    return float(np.median(dx_list)), float(np.median(dy_list))


def _clamp(value: float, limit: float) -> float:
    """Clamp a value to [-limit, +limit]."""
    return max(-limit, min(limit, value))


def _shift_channel(
    channel: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Apply sub-pixel translation to a 2D channel using warpAffine.

    Parameters
    ----------
    channel : np.ndarray
        2D float32 array.
    dx : float
        Shift along x (positive = move right).
    dy : float
        Shift along y (positive = move down).

    Returns
    -------
    np.ndarray
        Shifted channel, same shape.
    """
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return channel.copy()

    h, w = channel.shape
    # Translation matrix: [[1, 0, tx], [0, 1, ty]]
    # We negate the shifts because we want to *undo* the aberration:
    # the channel was shifted by (dx, dy), so we shift it back by (-dx, -dy).
    matrix = np.array(
        [[1.0, 0.0, -dx], [0.0, 1.0, -dy]],
        dtype=np.float64,
    )

    return cv2.warpAffine(
        channel,
        matrix,
        (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def correct_chromatic_aberration(
    image: np.ndarray,
    params: CAParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    """Detect and correct lateral chromatic aberration.

    Only operates on colour images with shape (C, H, W) where C >= 3.
    The green channel is used as the spatial reference; red and blue
    channels are shifted to align with it.

    When ``params.auto_detect`` is *True*, bright stars are detected
    independently in R, G, and B, matched by proximity, and the median
    centroid offset between G and R/B is computed.  The offset is capped
    at ``params.max_correction`` pixels.

    When ``params.auto_detect`` is *False*, the manual shift values in
    the params are used directly (still capped at ``max_correction``).

    Parameters
    ----------
    image : np.ndarray
        Colour image, shape (C, H, W) with C >= 3, float32 in [0, 1].
    params : CAParams, optional
        Correction parameters.  If *None*, defaults (auto-detect) are used.
    mask : Mask, optional
        Selective processing mask.

    Returns
    -------
    np.ndarray
        Corrected image, same shape, clipped to [0, 1].

    Raises
    ------
    ValueError
        If the image is not colour with at least 3 channels.
    """
    if params is None:
        params = CAParams()

    if image.ndim != 3 or image.shape[0] < 3:
        raise ValueError(
            "Chromatic aberration correction requires a colour image with "
            f"shape (C, H, W) and C >= 3, got shape {image.shape}"
        )

    # no copy: op never mutates the input; apply_mask reads image directly
    progress(0.0, "Detecting chromatic aberration…" if params.auto_detect else "Preparing CA correction…")

    red = image[0]
    green = image[1]
    blue = image[2]

    max_px = params.max_correction

    if params.auto_detect:
        log.info("Auto-detecting chromatic aberration from star centroids...")

        # Detect peaks in each channel
        green_peaks = _detect_peaks(green)
        red_peaks = _detect_peaks(red)
        blue_peaks = _detect_peaks(blue)

        log.debug(
            "Peaks detected: R=%d, G=%d, B=%d",
            len(red_peaks), len(green_peaks), len(blue_peaks),
        )

        # Match R and B centroids against G (reference)
        red_matches = _match_centroids(green_peaks, red_peaks)
        blue_matches = _match_centroids(green_peaks, blue_peaks)

        # Compute median offsets (target - ref = channel - green)
        red_dx, red_dy = _compute_median_offset(red_matches)
        blue_dx, blue_dy = _compute_median_offset(blue_matches)

        log.info(
            "Detected CA offsets — R: (%.3f, %.3f) px (%d matches), "
            "B: (%.3f, %.3f) px (%d matches)",
            red_dx, red_dy, len(red_matches),
            blue_dx, blue_dy, len(blue_matches),
        )

        # Cap at max_correction
        red_dx = _clamp(red_dx, max_px)
        red_dy = _clamp(red_dy, max_px)
        blue_dx = _clamp(blue_dx, max_px)
        blue_dy = _clamp(blue_dy, max_px)
    else:
        red_dx = _clamp(params.red_shift_x, max_px)
        red_dy = _clamp(params.red_shift_y, max_px)
        blue_dx = _clamp(params.blue_shift_x, max_px)
        blue_dy = _clamp(params.blue_shift_y, max_px)

        log.info(
            "Manual CA correction — R: (%.3f, %.3f) px, B: (%.3f, %.3f) px",
            red_dx, red_dy, blue_dx, blue_dy,
        )

    # Check if correction is negligible
    if (abs(red_dx) < 0.01 and abs(red_dy) < 0.01
            and abs(blue_dx) < 0.01 and abs(blue_dy) < 0.01):
        log.info("CA offsets are negligible; no correction applied")
        return image.copy()

    # Apply sub-pixel shifts to R and B channels
    progress(0.6, "Shifting R channel…")
    result = image.copy()
    result[0] = _shift_channel(red, red_dx, red_dy)
    progress(0.8, "Shifting B channel…")
    result[2] = _shift_channel(blue, blue_dx, blue_dy)

    result = np.clip(result, 0.0, 1.0).astype(np.float32)
    result = apply_mask(image, result, mask)
    progress(1.0, "CA correction complete")
    return result
