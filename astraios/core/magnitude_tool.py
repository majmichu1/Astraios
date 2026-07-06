"""Magnitude Tool — instrumental and calibrated star magnitudes.

Wraps :mod:`astraios.core.analysis.aperture_photometry` (source detection +
aperture photometry) and adds what SASpro's magnitude tool does beyond it:
photometric zero-point calibration against a reference-star catalog (or a
user-supplied zero point), a per-star magnitude table, and a limiting
magnitude estimate.

This runs entirely on CPU: aperture photometry already reuses
``run_photometry`` (small, per-star pixel sampling — not GPU-shaped), and the
zero-point fit / sigma-clip / limiting-magnitude estimate are O(n_stars)
scalar operations, far below the threshold where a GPU dispatch would pay
for itself.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro),
Copyright Franklin Marek, GPL-3.0-or-later.
https://github.com/setiastro/setiastrosuitepro
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from astropy.stats import SigmaClip, sigma_clipped_stats
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from astraios.core.analysis.aperture_photometry import PhotometryParams, run_photometry

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


@dataclass
class MagnitudeParams:
    """Settings for :func:`measure_magnitudes`."""

    aperture_radius: float = 10.0  # pixels
    annulus_inner: float = 15.0  # pixels
    annulus_outer: float = 20.0  # pixels
    detection_threshold: float = 5.0  # sigma, source detection threshold
    max_sources: int = 500  # cap on detected sources

    zero_point: float | None = None  # fixed zero point; overrides reference_stars if set
    match_radius_px: float = 5.0  # max pixel distance to match a source to a reference star
    clip_sigma: float = 2.5  # sigma-clip threshold when averaging per-star zero points
    clip_iters: int = 3  # sigma-clip max iterations

    limiting_mag_sigma: float = 5.0  # N-sigma flux threshold defining the limiting magnitude


@dataclass
class ReferenceStar:
    """A reference star with a known catalog magnitude, located by pixel position."""

    x: float
    y: float
    catalog_mag: float


@dataclass
class MagnitudeResult:
    """Per-star photometry/magnitude table plus the fitted calibration."""

    x: NDArray
    y: NDArray
    flux: NDArray
    instrumental_mag: NDArray  # -2.5 * log10(flux)
    calibrated_mag: NDArray | None  # instrumental_mag + zero_point, or None if uncalibrated
    zero_point: float | None
    zero_point_std: float | None
    zero_point_n: int  # number of reference-star matches used for the zero point
    limiting_mag: float | None  # calibrated if zero_point is known, else instrumental
    n_stars: int


def _get_field(entry: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(entry, dict):
            if name in entry and entry[name] is not None:
                return entry[name]
        elif hasattr(entry, name) and getattr(entry, name) is not None:
            return getattr(entry, name)
    return default


def _fit_zero_point(
    xs: NDArray,
    ys: NDArray,
    instrumental_mag: NDArray,
    reference_stars: list[Any],
    params: MagnitudeParams,
) -> tuple[float | None, float | None, int]:
    """Mutual-nearest-neighbor match detected sources to reference stars, then
    sigma-clip the per-star zero-point candidates ``zp_i = catalog_mag_i - m_inst_i``.
    """
    if xs.size == 0 or not reference_stars:
        return None, None, 0

    ref_x = np.array([_get_field(r, "x") for r in reference_stars], dtype=np.float64)
    ref_y = np.array([_get_field(r, "y") for r in reference_stars], dtype=np.float64)
    ref_mag = np.array(
        [_get_field(r, "catalog_mag", "mag") for r in reference_stars], dtype=np.float64
    )
    valid = np.isfinite(ref_x) & np.isfinite(ref_y) & np.isfinite(ref_mag)
    ref_x, ref_y, ref_mag = ref_x[valid], ref_y[valid], ref_mag[valid]
    if ref_x.size == 0:
        return None, None, 0

    dist = cdist(np.column_stack([xs, ys]), np.column_stack([ref_x, ref_y]))
    src_best = np.argmin(dist, axis=1)
    ref_best = np.argmin(dist, axis=0)

    zp_values = []
    for i in range(xs.size):
        j = src_best[i]
        if ref_best[j] != i:
            continue
        if dist[i, j] > params.match_radius_px:
            continue
        if not (np.isfinite(instrumental_mag[i]) and np.isfinite(ref_mag[j])):
            continue
        zp_values.append(float(ref_mag[j]) - float(instrumental_mag[i]))

    if not zp_values:
        return None, None, 0

    zp_arr = np.asarray(zp_values, dtype=np.float64)
    sigclip = SigmaClip(sigma=params.clip_sigma, maxiters=params.clip_iters)
    good = sigclip(zp_arr, masked=True).compressed()
    if good.size == 0:
        good = zp_arr

    zp = float(np.median(good))
    zp_std = float(np.std(good)) if good.size > 1 else 0.0
    return zp, zp_std, int(good.size)


def _estimate_limiting_magnitude(
    image: NDArray, params: MagnitudeParams, zero_point: float | None
) -> float | None:
    """N-sigma point-source detection limit from the global sky background noise."""
    gray = np.mean(image, axis=0) if image.ndim == 3 else image
    gray = np.asarray(gray, dtype=np.float32)

    try:
        _, _, sky_std = sigma_clipped_stats(gray, sigma=3.0, maxiters=5)
    except Exception:
        log.debug("Limiting magnitude: sigma_clipped_stats failed", exc_info=True)
        return None

    if not np.isfinite(sky_std) or sky_std <= 0:
        return None

    aperture_area = math.pi * params.aperture_radius ** 2
    limiting_flux = params.limiting_mag_sigma * float(sky_std) * math.sqrt(aperture_area)
    if not (limiting_flux > 0):
        return None

    instrumental_limiting_mag = -2.5 * math.log10(limiting_flux)
    if zero_point is not None:
        return float(instrumental_limiting_mag + zero_point)
    return float(instrumental_limiting_mag)


def measure_magnitudes(
    image: NDArray,
    params: MagnitudeParams | None = None,
    reference_stars: list[Any] | None = None,
    progress: ProgressCallback | None = None,
) -> MagnitudeResult:
    """Detect stars, run aperture photometry, and compute magnitudes.

    Parameters
    ----------
    image : ndarray
        (H, W) mono or (C, H, W) color float32 image in [0, 1].
    params : MagnitudeParams, optional
        Aperture/annulus/detection/calibration settings.
    reference_stars : list, optional
        ``ReferenceStar`` instances or dict/objects with ``x``, ``y`` (pixel
        position) and ``catalog_mag``/``mag`` (known magnitude). Used to fit
        a zero point when ``params.zero_point`` is not set directly.
    progress : callable(fraction, message), optional

    Returns
    -------
    MagnitudeResult
    """
    params = params or MagnitudeParams()
    progress = progress or _noop_progress

    progress(0.0, "Detecting sources")
    photometry_params = PhotometryParams(
        aperture_radius=params.aperture_radius,
        annulus_inner=params.annulus_inner,
        annulus_outer=params.annulus_outer,
        detection_threshold=params.detection_threshold,
        max_sources=params.max_sources,
    )
    phot = run_photometry(image, photometry_params)

    if phot.n_sources == 0:
        progress(1.0, "Done (no sources detected)")
        return MagnitudeResult(
            x=phot.x, y=phot.y, flux=phot.flux,
            instrumental_mag=phot.mag, calibrated_mag=None,
            zero_point=params.zero_point, zero_point_std=None, zero_point_n=0,
            limiting_mag=None, n_stars=0,
        )

    instrumental_mag = phot.mag

    progress(0.5, "Calibrating zero point")
    zero_point = params.zero_point
    zero_point_std: float | None = None
    zero_point_n = 0

    if zero_point is None and reference_stars:
        zero_point, zero_point_std, zero_point_n = _fit_zero_point(
            phot.x, phot.y, instrumental_mag, reference_stars, params
        )

    calibrated_mag = instrumental_mag + zero_point if zero_point is not None else None

    progress(0.8, "Estimating limiting magnitude")
    limiting_mag = _estimate_limiting_magnitude(image, params, zero_point)

    progress(1.0, "Done")

    return MagnitudeResult(
        x=phot.x, y=phot.y, flux=phot.flux,
        instrumental_mag=instrumental_mag, calibrated_mag=calibrated_mag,
        zero_point=zero_point, zero_point_std=zero_point_std, zero_point_n=zero_point_n,
        limiting_mag=limiting_mag, n_stars=phot.n_sources,
    )
