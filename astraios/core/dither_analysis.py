"""Dither Analysis — quantify how well a set of registered frames dithered.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro), Copyright
Franklin Marek, GPL-3.0-or-later. https://github.com/setiastro/setiastrosuitepro

SASpro's ``pro/dither_analysis.py`` (``DitherAnalysisWindow``) runs its own
star-triangle registration thread to get a per-frame affine transform, maps
the image center through each transform, and reports the scatter of those
mapped centers as the "dither pattern" -- plus derived stats: RMS/mean/max
offset, step distribution, a convex-hull coverage area, a preferred
direction (circular mean), meridian-flip detection, cluster detection (are
several consecutive frames sitting at the same pointing?), and "walking
noise" diagnostics (PCA linearity, temporal drift correlation, directional
consistency) that flag systematic drift rather than healthy random jitter.

This module ports that statistics core (``_compute_stats`` /
``_cluster_stats`` and their thresholds) verbatim as
:class:`DitherAnalysisParams` settings, but swaps out SASpro's star-triangle
affine registration for direct translation measurement, since we are not
porting the full ``StarRegistrationThread``/UI pipeline here:

* :attr:`DitherOffsetMethod.PHASE_CORRELATION` -- per CLAUDE.md guidance,
  ``skimage.registration.phase_cross_correlation`` on frame luminance,
  matching :mod:`astraios.core.stacking`'s own use of the same routine for
  frame alignment.
* :attr:`DitherOffsetMethod.HEADER_WCS` -- if frames already carry a plate
  solution (``CRVAL1/2`` at minimum), offsets are read straight from the
  WCS instead of touching pixels at all ("from headers", as SASpro's
  ``.sasd``/plate-solved path effectively does via each frame's own
  transform).

Because we measure pure translation (no rotation), SASpro's meridian-flip
detection (which needs the affine's rotation component) is not applicable
and is dropped -- see the module docstring in SASpro's original file for
that logic if a future affine-based path is added.

GPU/CPU decision: offset measurement is FFT-based cross-correlation over at
most a handful of full-resolution frames (already how
:mod:`astraios.core.stacking` does registration), and the statistics that
follow are O(n) to O(n log n) over per-frame scalars (n = frame count, not
pixel count). Nothing here benefits from `device_manager`/GPU dispatch.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


class DitherOffsetMethod(str, Enum):
    """How per-frame dither offsets are measured."""

    PHASE_CORRELATION = "phase_correlation"  # skimage FFT cross-correlation on pixels
    HEADER_WCS = "header_wcs"  # read CRVAL1/2 from each frame's FITS header


@dataclass
class DitherAnalysisParams:
    """Settings for :func:`analyze_dither`.

    The threshold fields default to the exact constants SASpro's
    ``_compute_stats``/``_cluster_stats`` hard-code; they were not exposed
    as UI settings in SASpro either, but are surfaced here as tunable
    ``Params`` fields per project convention.

    Attributes:
        offset_method: How to measure each frame's offset from the reference.
        reference_index: Index of the reference frame within the input list.
        upsample_factor: Subpixel precision for
            ``skimage.registration.phase_cross_correlation`` (only used by
            the PHASE_CORRELATION method); e.g. 10 -> 1/10 px precision.
        linearity_threshold: PCA first/second singular-value ratio above
            which the dither pattern is flagged as linear drift.
        temporal_drift_threshold: |Pearson r| between frame index and radial
            offset above which growth-with-time (walking) is flagged.
        dir_consistency_threshold: Fraction of steps within 45 deg of the
            circular mean step direction above which direction is
            suspiciously consistent (walking) rather than random.
        min_problem_cluster: Minimum run length of near-identical pointings
            to count frames as "clustered" (stuck) rather than dithered.
        cluster_mean_size_threshold: Mean cluster size above which the set
            may be considered clustered.
        cluster_max_size_threshold: Worst-case (max) cluster size above
            which the set may be considered clustered.
        cluster_frac_threshold: Fraction of frames living in problem-sized
            clusters above which the set is flagged clustered.
    """

    offset_method: DitherOffsetMethod = DitherOffsetMethod.PHASE_CORRELATION
    reference_index: int = 0
    upsample_factor: int = 10
    linearity_threshold: float = 4.0
    temporal_drift_threshold: float = 0.85
    dir_consistency_threshold: float = 0.65
    min_problem_cluster: int = 3
    cluster_mean_size_threshold: float = 3.0
    cluster_max_size_threshold: int = 4
    cluster_frac_threshold: float = 0.3


@dataclass
class DitherResult:
    """Result of :func:`analyze_dither`.

    ``dx``/``dy`` are per-frame pixel offsets relative to the reference
    frame (positive = moved right / down in the reference's pixel grid).
    """

    n_frames: int
    dx: np.ndarray
    dy: np.ndarray
    radii: np.ndarray  # per-frame radial offset from reference, px
    steps: np.ndarray  # consecutive-frame step distances, px (length n-1)

    mean_radius: float
    median_radius: float
    max_radius: float
    rms_offset: float

    mean_step: float
    max_step: float
    std_dx: float
    std_dy: float
    span_x: float
    span_y: float

    coverage_px: float  # convex-hull area of the dither scatter, px^2
    preferred_direction_deg: float  # circular mean of step directions

    nearest_neighbor_min_px: float
    nearest_neighbor_mean_px: float

    n_clusters: int
    mean_cluster_size: float
    max_cluster_size: int
    clustered_fraction: float
    is_clustered: bool

    linearity_ratio: float
    temporal_drift_corr: float
    dir_consistency: float
    is_walking: bool

    quality_summary: str
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# offset measurement
# ---------------------------------------------------------------------------
def _to_luma(img: np.ndarray) -> np.ndarray:
    """Collapse an (H, W) or (C, H, W) image to a 2-D luminance array."""
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr.mean(axis=0)
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def _load_luma_frames(
    items: list[Any], progress: ProgressCallback
) -> list[np.ndarray]:
    from astraios.core.image_io import load_image

    n = len(items)
    out = []
    for i, item in enumerate(items):
        progress(0.5 * i / n, f"Loading frame {i + 1}/{n}…")
        data = load_image(str(item)).data if isinstance(item, (str, Path)) else np.asarray(item)
        out.append(_to_luma(data))
    return out


def _headers_from_items(items: list[Any]) -> list[Any]:
    from astraios.core.image_io import load_image

    headers = []
    for item in items:
        if not isinstance(item, (str, Path)):
            raise ValueError(
                "offset_method=HEADER_WCS requires file paths (to read FITS headers) "
                "or an explicit `headers=` argument when passing raw arrays."
            )
        headers.append(load_image(str(item)).header)
    return headers


def _offsets_from_correlation(
    luma_frames: list[np.ndarray], params: DitherAnalysisParams, progress: ProgressCallback
) -> tuple[np.ndarray, np.ndarray]:
    from skimage.registration import phase_cross_correlation

    n = len(luma_frames)
    ref = luma_frames[params.reference_index]
    dx = np.zeros(n, dtype=np.float64)
    dy = np.zeros(n, dtype=np.float64)
    for i, frame in enumerate(luma_frames):
        progress(0.5 + 0.5 * i / n, f"Measuring offset {i + 1}/{n}…")
        if i == params.reference_index:
            continue
        shift_yx, _error, _diffphase = phase_cross_correlation(
            ref, frame, upsample_factor=params.upsample_factor
        )
        # phase_cross_correlation returns the shift that would register
        # `frame` back onto `ref`; negate to get frame's actual displacement.
        dy[i] = -float(shift_yx[0])
        dx[i] = -float(shift_yx[1])
    progress(1.0, "Offsets measured")
    return dx, dy


def _offsets_from_headers(
    headers: list[Any], params: DitherAnalysisParams
) -> tuple[np.ndarray, np.ndarray]:
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS

    n = len(headers)
    ref_hdr = headers[params.reference_index]
    ref_wcs = WCS(ref_hdr)
    origin = SkyCoord(
        ra=float(ref_hdr["CRVAL1"]) * u.deg, dec=float(ref_hdr["CRVAL2"]) * u.deg
    )
    ox, oy = ref_wcs.world_to_pixel(origin)

    dx = np.zeros(n, dtype=np.float64)
    dy = np.zeros(n, dtype=np.float64)
    for i, hdr in enumerate(headers):
        sky = SkyCoord(ra=float(hdr["CRVAL1"]) * u.deg, dec=float(hdr["CRVAL2"]) * u.deg)
        px, py = ref_wcs.world_to_pixel(sky)
        dx[i] = float(px) - float(ox)
        dy[i] = float(py) - float(oy)
    return dx, dy


# ---------------------------------------------------------------------------
# statistics (ported from SASpro's _compute_stats / _cluster_stats)
# ---------------------------------------------------------------------------
def _nearest_neighbor_stats(dx: np.ndarray, dy: np.ndarray) -> tuple[float, float]:
    n = len(dx)
    if n < 2:
        return 0.0, 0.0
    from scipy.spatial import cKDTree

    pts = np.stack([dx, dy], axis=1)
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)  # k=1 is the point itself (distance 0)
    nn = dists[:, 1]
    return float(nn.min()), float(nn.mean())


def _coverage_area(dx: np.ndarray, dy: np.ndarray) -> float:
    if len(dx) < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(np.stack([dx, dy], axis=1))
        return float(hull.volume)  # "volume" of a 2-D hull is its area
    except Exception:
        return 0.0


def _cluster_stats(
    dx: np.ndarray, dy: np.ndarray, steps: np.ndarray, params: DitherAnalysisParams
) -> dict:
    """Detect runs of consecutive frames sharing (near enough) one pointing."""
    if len(steps) == 0:
        return dict(
            n_clusters=1, mean_cluster_size=1.0, max_cluster_size=1,
            clustered_fraction=0.0, is_clustered=False,
        )

    n_frames = len(dx)
    sorted_steps = np.sort(steps)

    if len(sorted_steps) > 2:
        gaps = np.diff(sorted_steps)
        upper_idx = int(len(sorted_steps) * 0.85)
        gaps_clipped = gaps[:upper_idx]
        if len(gaps_clipped) > 0 and gaps_clipped.max() > 0:
            gap_idx = int(np.argmax(gaps_clipped))
            step_threshold = float((sorted_steps[gap_idx] + sorted_steps[gap_idx + 1]) / 2.0)
        else:
            step_threshold = float(sorted_steps.max() * 0.2)
    else:
        step_threshold = float(sorted_steps[0] * 1.5) if len(sorted_steps) else 1.0

    step_threshold = min(max(0.5, min(step_threshold, float(steps.max()) * 0.5)), 5.0)

    cluster_sizes = []
    current_run = 1
    for s in steps:
        if s <= step_threshold:
            current_run += 1
        else:
            cluster_sizes.append(current_run)
            current_run = 1
    cluster_sizes.append(current_run)
    cluster_sizes_arr = np.array(cluster_sizes, dtype=int)

    n_clusters = len(cluster_sizes_arr)
    mean_cluster_size = float(cluster_sizes_arr.mean())
    max_cluster_size = int(cluster_sizes_arr.max())

    problem_frames = int(np.sum(cluster_sizes_arr[cluster_sizes_arr >= params.min_problem_cluster]))
    clustered_fraction = problem_frames / max(1, n_frames)

    is_clustered = (
        mean_cluster_size > params.cluster_mean_size_threshold
        and max_cluster_size > params.cluster_max_size_threshold
        and clustered_fraction > params.cluster_frac_threshold
    )

    return dict(
        n_clusters=n_clusters,
        mean_cluster_size=mean_cluster_size,
        max_cluster_size=max_cluster_size,
        clustered_fraction=clustered_fraction,
        is_clustered=is_clustered,
    )


def _walking_noise_metrics(
    dx: np.ndarray, dy: np.ndarray, params: DitherAnalysisParams
) -> dict:
    """PCA linearity + temporal drift + directional consistency (SASpro's
    walking-noise heuristics)."""
    n = len(dx)

    if n >= 3:
        pts_c = np.stack([dx - dx.mean(), dy - dy.mean()], axis=1)
        sv = np.linalg.svd(pts_c, full_matrices=False, compute_uv=False)
        linearity_ratio = float(sv[0] / (sv[1] + 1e-9))
    else:
        linearity_ratio = 1.0

    radii = np.hypot(dx, dy)
    if n > 3 and radii.std() > 1e-12:
        frame_idx = np.arange(n, dtype=float)
        temporal_drift_corr = float(np.corrcoef(frame_idx, radii)[0, 1])
    else:
        temporal_drift_corr = 0.0

    if n > 1:
        step_rad = np.arctan2(np.diff(dy), np.diff(dx))
        circ_mean = math.degrees(
            math.atan2(np.sin(step_rad).mean(), np.cos(step_rad).mean())
        ) % 360
        step_angles = np.degrees(step_rad) % 360
    else:
        step_angles = np.array([0.0])
        circ_mean = 0.0

    if len(step_angles) > 3:
        angular_diffs = np.abs(((step_angles - circ_mean + 180.0) % 360.0) - 180.0)
        dir_consistency = float(np.mean(angular_diffs < 45.0))
    else:
        dir_consistency = 0.0

    lin_fired = linearity_ratio > params.linearity_threshold
    tdc_fired = abs(temporal_drift_corr) > params.temporal_drift_threshold
    dir_fired = dir_consistency > params.dir_consistency_threshold
    is_walking = bool(lin_fired or (tdc_fired and (lin_fired or dir_fired)))

    return dict(
        linearity_ratio=linearity_ratio,
        temporal_drift_corr=temporal_drift_corr,
        dir_consistency=dir_consistency,
        is_walking=is_walking,
        preferred_direction_deg=circ_mean,
    )


def _quality_summary(
    *, is_walking: bool, is_clustered: bool, n_clusters: int, n_frames: int
) -> str:
    if is_walking:
        return "Poor: dither pattern shows systematic drift (walking) rather than random jitter."
    if is_clustered:
        return (
            f"Fair: {n_clusters} distinct pointing cluster(s) detected across {n_frames} "
            "frames; dither steps may be too small relative to noise/hot-pixel scale."
        )
    return "Good: dither pattern looks like healthy random jitter."


def _build_result(dx: np.ndarray, dy: np.ndarray, params: DitherAnalysisParams) -> DitherResult:
    n = len(dx)
    radii = np.hypot(dx, dy)
    steps = np.hypot(np.diff(dx), np.diff(dy)) if n > 1 else np.array([0.0])

    nn_min, nn_mean = _nearest_neighbor_stats(dx, dy)
    coverage = _coverage_area(dx, dy)
    cluster = _cluster_stats(dx, dy, steps, params)
    walk = _walking_noise_metrics(dx, dy, params)
    quality = _quality_summary(
        is_walking=walk["is_walking"],
        is_clustered=cluster["is_clustered"],
        n_clusters=cluster["n_clusters"],
        n_frames=n,
    )

    return DitherResult(
        n_frames=n,
        dx=dx,
        dy=dy,
        radii=radii,
        steps=steps,
        mean_radius=float(radii.mean()),
        median_radius=float(np.median(radii)),
        max_radius=float(radii.max()),
        rms_offset=float(np.sqrt((dx**2 + dy**2).mean())),
        mean_step=float(steps.mean()),
        max_step=float(steps.max()),
        std_dx=float(dx.std()),
        std_dy=float(dy.std()),
        span_x=float(dx.max() - dx.min()),
        span_y=float(dy.max() - dy.min()),
        coverage_px=coverage,
        preferred_direction_deg=walk["preferred_direction_deg"],
        nearest_neighbor_min_px=nn_min,
        nearest_neighbor_mean_px=nn_mean,
        n_clusters=cluster["n_clusters"],
        mean_cluster_size=cluster["mean_cluster_size"],
        max_cluster_size=cluster["max_cluster_size"],
        clustered_fraction=cluster["clustered_fraction"],
        is_clustered=cluster["is_clustered"],
        linearity_ratio=walk["linearity_ratio"],
        temporal_drift_corr=walk["temporal_drift_corr"],
        dir_consistency=walk["dir_consistency"],
        is_walking=walk["is_walking"],
        quality_summary=quality,
    )


def analyze_dither(
    frames_or_paths: list[Any],
    params: DitherAnalysisParams | None = None,
    progress: ProgressCallback = _noop_progress,
    headers: list[Any] | None = None,
) -> DitherResult:
    """Quantify dither quality across a set of registered frames.

    Args:
        frames_or_paths: List of ``(H, W)``/``(C, H, W)`` float arrays, or a
            list of file paths (loaded via
            :func:`astraios.core.image_io.load_image`). Must contain at
            least 2 frames.
        params: Analysis settings. Defaults to phase-correlation offsets
            against frame 0.
        progress: Optional ``(fraction, message)`` progress callback.
        headers: Optional list of FITS-header-like mappings, one per frame,
            used when ``params.offset_method`` is
            :attr:`DitherOffsetMethod.HEADER_WCS`. If omitted and
            ``frames_or_paths`` are file paths, headers are read from those
            files automatically.

    Returns:
        A :class:`DitherResult` with per-frame offsets and derived
        spread/coverage/nearest-neighbor/clustering/drift statistics.

    Raises:
        ValueError: Fewer than 2 frames, an out-of-range
            ``reference_index``, or ``HEADER_WCS`` requested without headers
            available.
    """
    if params is None:
        params = DitherAnalysisParams()

    items = list(frames_or_paths)
    n = len(items)
    if n < 2:
        raise ValueError("analyze_dither needs at least two frames.")
    if not (0 <= params.reference_index < n):
        raise ValueError(f"reference_index {params.reference_index} out of range for {n} frames.")

    if params.offset_method == DitherOffsetMethod.HEADER_WCS:
        hdrs = headers if headers is not None else _headers_from_items(items)
        if len(hdrs) != n:
            raise ValueError("headers must have the same length as frames_or_paths.")
        progress(0.5, "Reading WCS headers…")
        dx, dy = _offsets_from_headers(hdrs, params)
        progress(1.0, "Offsets measured")
    else:
        luma_frames = _load_luma_frames(items, progress)
        dx, dy = _offsets_from_correlation(luma_frames, params, progress)

    return _build_result(dx, dy, params)
