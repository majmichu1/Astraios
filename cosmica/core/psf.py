"""PSF Measurement — measure the Point Spread Function from stars in an image.

Detects bright unsaturated stars, fits 2D Gaussians to each,
and reports the median FWHM and ellipticity. This gives the actual
PSF of the image (seeing + tracking + optics combined), which is
much more accurate than theoretical calculations from equipment specs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from cosmica.core.star_detection import Star, detect_stars

log = logging.getLogger(__name__)


@dataclass
class PSFMeasurement:
    """Result of PSF measurement from an image."""

    fwhm_x: float  # FWHM along x-axis in pixels
    fwhm_y: float  # FWHM along y-axis in pixels
    fwhm: float  # geometric mean FWHM in pixels
    ellipticity: float  # 1 - minor/major, 0 = circular
    theta: float  # rotation angle in degrees
    n_stars_used: int  # number of stars used for fitting
    fwhm_std: float  # standard deviation of per-star FWHM values
    per_star_fwhm: list[float]  # FWHM per fitted star


def _gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    """2D Gaussian function for fitting."""
    x, y = coords
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    a = cos_t**2 / (2 * sigma_x**2) + sin_t**2 / (2 * sigma_y**2)
    b = -np.sin(2 * theta) / (4 * sigma_x**2) + np.sin(2 * theta) / (4 * sigma_y**2)
    c = sin_t**2 / (2 * sigma_x**2) + cos_t**2 / (2 * sigma_y**2)
    dx = x - x0
    dy = y - y0
    return offset + amplitude * np.exp(-(a * dx**2 + 2 * b * dx * dy + c * dy**2))


def _radial_fwhm(cutout: np.ndarray, r: int) -> float | None:
    """Measure FWHM via radial profile — robust fallback for clipped/stretched stars."""
    cy, cx = r, r
    bg = float(np.percentile(cutout, 10))
    peak = float(cutout[cy, cx])
    if peak - bg < 0.02:
        return None
    half_max = bg + (peak - bg) * 0.5

    # Sample in 16 directions, find half-maximum crossing
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    half_radii: list[float] = []
    for theta in angles:
        for rad in np.linspace(0.5, r - 0.5, 80):
            iy = cy + rad * np.sin(theta)
            ix = cx + rad * np.cos(theta)
            yi, xi = int(round(iy)), int(round(ix))
            if 0 <= yi < cutout.shape[0] and 0 <= xi < cutout.shape[1]:
                if cutout[yi, xi] <= half_max:
                    half_radii.append(rad)
                    break

    if len(half_radii) < 6:
        return None
    fwhm = float(np.median(half_radii)) * 2.0
    if fwhm < 1.0 or fwhm > r:
        return None
    return fwhm


def _fit_star_psf(
    image: np.ndarray,
    star: Star,
    cutout_radius: int = 12,
) -> tuple[float, float, float] | None:
    """Fit a 2D Gaussian to a single star; fall back to radial profile for clipped stars.

    Returns (fwhm_x, fwhm_y, theta) or None if all methods fail.
    """
    h, w = image.shape
    ix, iy = int(round(star.x)), int(round(star.y))

    # Bounds check with margin
    r = cutout_radius
    if ix - r < 0 or ix + r >= w or iy - r < 0 or iy + r >= h:
        return None

    # Extract cutout
    cutout = image[iy - r : iy + r + 1, ix - r : ix + r + 1].copy()
    size = 2 * r + 1

    # Create coordinate grids
    y_grid, x_grid = np.mgrid[0:size, 0:size]
    coords = (x_grid.ravel(), y_grid.ravel())
    data = cutout.ravel()

    # Initial estimates
    bg = float(np.percentile(cutout, 10))
    peak = float(cutout.max())
    amp = peak - bg
    offset = bg
    cx = float(r)
    cy = float(r)
    sigma_init = star.fwhm / 2.355 if star.fwhm > 0 else 2.0

    # Amplitude bound: allow up to max+20% — clipped stars have amp underestimated
    amp_max = max(amp * 2.0, 0.1)
    p0 = [amp, cx, cy, sigma_init, sigma_init, 0.0, offset]
    bounds = (
        [0, cx - 4, cy - 4, 0.5, 0.5, -np.pi, -0.1],
        [amp_max, cx + 4, cy + 4, r, r, np.pi, 1.1],
    )

    try:
        popt, _ = curve_fit(
            _gaussian_2d,
            coords,
            data,
            p0=p0,
            bounds=bounds,
            maxfev=2000,
            method="trf",
        )
        _, _, _, sigma_x, sigma_y, theta, _ = popt

        fwhm_x = abs(sigma_x) * 2.355
        fwhm_y = abs(sigma_y) * 2.355

        if fwhm_x < 1.0 or fwhm_y < 1.0 or fwhm_x > cutout_radius or fwhm_y > cutout_radius:
            raise ValueError("unreasonable fit")

        return (fwhm_x, fwhm_y, float(theta))

    except (RuntimeError, ValueError):
        # Gaussian fitting failed (common for clipped/stretched stars).
        # Fall back to radial half-maximum profile measurement.
        fwhm = _radial_fwhm(cutout, r)
        if fwhm is not None:
            return (fwhm, fwhm, 0.0)
        return None


def measure_psf(
    image: np.ndarray,
    max_stars: int = 50,
    min_flux: float = 0.3,
    max_flux: float = 0.95,
    cutout_radius: int = 12,
    force_cpu: bool = False,
) -> PSFMeasurement:
    """Measure the PSF from stars in an image.

    Detects stars, filters to bright but unsaturated ones,
    fits 2D Gaussians, and reports median FWHM and ellipticity.

    Parameters
    ----------
    image : ndarray
        Image data, shape (H, W) or (C, H, W), float32 in [0, 1].
    max_stars : int
        Maximum number of stars to fit.
    min_flux : float
        Minimum peak flux for star selection (rejects faint stars).
    max_flux : float
        Maximum peak flux for star selection (rejects saturated stars).
    cutout_radius : int
        Half-size of the cutout around each star for fitting.
    force_cpu : bool
        Skip GPU star detection entirely.  Pass True when calling from
        parallel worker threads — multiple threads dispatching CUDA kernels
        via Python all hold the GIL during dispatch, serializing the pool.
        CPU detection (pure numpy) releases the GIL, enabling true
        multi-core parallelism.

    Returns
    -------
    PSFMeasurement
        Measured PSF properties.
    """
    # Get grayscale
    if image.ndim == 3:
        gray = np.mean(image, axis=0).astype(np.float32)
    else:
        gray = image.astype(np.float32)

    # Star detection always uses CPU (OpenCV contour analysis). GPU star detection
    # (max-pool local maxima) does not compute roundness/ellipticity, making
    # candidate selection inconsistent between CPU and GPU paths and causing
    # different FWHM measurements.  CPU detection is deterministic and provides
    # roundness for reliable star/artifact discrimination.
    # 2D Gaussian fitting happens on CPU regardless of detection method.
    # Use 3-sigma threshold so PSF measurement works on both linear and
    # stretched images (5-sigma misses stars when background is lifted by stretch).
    sf = detect_stars(gray, max_stars=max_stars * 3, sigma_threshold=3.0)
    stars_list = sf.stars

    # Filter to bright but unsaturated round stars
    candidates = [
        s for s in stars_list
        if min_flux <= s.flux <= max_flux and s.roundness < 0.4
    ]

    if not candidates:
        # Relax constraints
        candidates = [
            s for s in stars_list
            if s.flux >= min_flux * 0.5 and s.flux <= max_flux + 0.03
        ]

    candidates = candidates[:max_stars]

    if not candidates:
        log.warning("No suitable stars found for PSF measurement")
        return PSFMeasurement(
            fwhm_x=3.0, fwhm_y=3.0, fwhm=3.0, ellipticity=0.0,
            theta=0.0, n_stars_used=0, fwhm_std=0.0, per_star_fwhm=[],
        )

    # Fit each star
    fwhm_x_list = []
    fwhm_y_list = []
    theta_list = []
    per_star = []

    for star in candidates:
        result = _fit_star_psf(gray, star, cutout_radius=cutout_radius)
        if result is not None:
            fx, fy, th = result
            fwhm_x_list.append(fx)
            fwhm_y_list.append(fy)
            theta_list.append(th)
            per_star.append(np.sqrt(fx * fy))  # geometric mean

    if not per_star:
        log.warning("PSF fitting failed for all candidate stars")
        return PSFMeasurement(
            fwhm_x=3.0, fwhm_y=3.0, fwhm=3.0, ellipticity=0.0,
            theta=0.0, n_stars_used=0, fwhm_std=0.0, per_star_fwhm=[],
        )

    # Reject outliers using sigma clipping (2.5 sigma)
    median_fwhm = np.median(per_star)
    mad = np.median(np.abs(np.array(per_star) - median_fwhm))
    sigma_est = mad * 1.4826 if mad > 0 else 0.5
    mask = np.abs(np.array(per_star) - median_fwhm) < 2.5 * sigma_est

    if np.sum(mask) < 3:
        mask = np.ones(len(per_star), dtype=bool)

    fx_arr = np.array(fwhm_x_list)[mask]
    fy_arr = np.array(fwhm_y_list)[mask]
    th_arr = np.array(theta_list)[mask]
    ps_arr = np.array(per_star)[mask]

    med_fx = float(np.median(fx_arr))
    med_fy = float(np.median(fy_arr))
    med_fwhm = float(np.median(ps_arr))
    fwhm_std = float(np.std(ps_arr))
    med_theta = float(np.median(th_arr))

    # Ellipticity: 1 - minor/major
    minor = min(med_fx, med_fy)
    major = max(med_fx, med_fy)
    ellipticity = 1.0 - minor / major if major > 0 else 0.0

    log.info(
        "PSF measured: FWHM=%.2f px (%.2f x %.2f), ellipticity=%.3f, %d stars",
        med_fwhm, med_fx, med_fy, ellipticity, int(np.sum(mask)),
    )

    return PSFMeasurement(
        fwhm_x=med_fx,
        fwhm_y=med_fy,
        fwhm=med_fwhm,
        ellipticity=ellipticity,
        theta=np.degrees(med_theta),
        n_stars_used=int(np.sum(mask)),
        fwhm_std=fwhm_std,
        per_star_fwhm=ps_arr.tolist(),
    )


def fwhm_to_sigma(fwhm: float) -> float:
    """Convert FWHM to Gaussian sigma."""
    return fwhm / 2.355


def sigma_to_fwhm(sigma: float) -> float:
    """Convert Gaussian sigma to FWHM."""
    return sigma * 2.355


def fwhm_pixels_to_arcsec(fwhm_px: float, plate_scale_arcsec_per_px: float) -> float:
    """Convert FWHM from pixels to arcseconds."""
    return fwhm_px * plate_scale_arcsec_per_px
