"""FWHM map — measures PSF quality across the image to detect tilt/aberrations."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

log = logging.getLogger(__name__)


@dataclass
class FWHMMapParams:
    grid_rows: int = 8
    grid_cols: int = 8
    min_stars_per_zone: int = 3
    star_detection_threshold: float = 5.0  # sigma above background


@dataclass
class FWHMMapResult:
    fwhm_map: NDArray  # (rows, cols) mean FWHM per zone in pixels
    star_counts: NDArray  # (rows, cols) number of stars detected per zone
    mean_fwhm: float
    std_fwhm: float
    max_fwhm: float
    min_fwhm: float
    tilt_detected: bool  # True if max zone FWHM > 1.5x min zone FWHM
    tilt_angle: float  # approximate tilt direction in degrees (0=horizontal)


def _gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    """2D Gaussian function for fitting star PSF."""
    x, y = coords
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    a = cos_t ** 2 / (2 * sigma_x ** 2) + sin_t ** 2 / (2 * sigma_y ** 2)
    b = -np.sin(2 * theta) / (4 * sigma_x ** 2) + np.sin(2 * theta) / (4 * sigma_y ** 2)
    c = sin_t ** 2 / (2 * sigma_x ** 2) + cos_t ** 2 / (2 * sigma_y ** 2)
    dx = x - x0
    dy = y - y0
    return offset + amplitude * np.exp(-(a * dx ** 2 + 2 * b * dx * dy + c * dy ** 2))


def _gaussian_fit_2d(
    data: NDArray,
    peak_y: int,
    peak_x: int,
    fit_radius: int = 5,
) -> float | None:
    """Fit a 2D Gaussian to estimate FWHM at a peak location.

    Parameters
    ----------
    data : ndarray
        2D image patch containing the star.
    peak_y : int
        Row index of the star peak within *data*.
    peak_x : int
        Column index of the star peak within *data*.
    fit_radius : int
        Half-size of the fitting region around the peak.

    Returns
    -------
    float or None
        FWHM in pixels (geometric mean of sigma_x and sigma_y), or
        *None* if the fit fails or produces unreasonable values.
    """
    h, w = data.shape
    r = fit_radius

    y0 = max(peak_y - r, 0)
    y1 = min(peak_y + r + 1, h)
    x0 = max(peak_x - r, 0)
    x1 = min(peak_x + r + 1, w)
    cutout = data[y0:y1, x0:x1].copy()

    if cutout.size < 9:
        return None

    cy, cx = cutout.shape
    size_y, size_x = cutout.shape
    y_grid, x_grid = np.mgrid[0:size_y, 0:size_x]
    coords = (x_grid.ravel(), y_grid.ravel())
    flat = cutout.ravel()

    bg = float(np.percentile(cutout, 10))
    peak = float(cutout.max())
    amp = peak - bg
    if amp < 0.01:
        return None

    p_cy = float(peak_y - y0)
    p_cx = float(peak_x - x0)
    sigma_init = 1.5
    p0 = [amp, p_cx, p_cy, sigma_init, sigma_init, 0.0, bg]
    bounds = (
        [0, p_cx - 3, p_cy - 3, 0.3, 0.3, -np.pi, -0.1],
        [amp * 2, p_cx + 3, p_cy + 3, r, r, np.pi, max(bg * 2, 0.5)],
    )

    try:
        popt, _ = curve_fit(
            _gaussian_2d,
            coords,
            flat,
            p0=p0,
            bounds=bounds,
            maxfev=2000,
            method="trf",
        )
        _, _, _, sigma_x, sigma_y, _, _ = popt
        fwhm_x = abs(sigma_x) * 2.355
        fwhm_y = abs(sigma_y) * 2.355

        if fwhm_x < 0.5 or fwhm_y < 0.5 or fwhm_x > r * 2 or fwhm_y > r * 2:
            return None

        return float(np.sqrt(fwhm_x * fwhm_y))

    except (RuntimeError, ValueError):
        return None


def _detect_stars_zone(
    zone: NDArray,
    threshold_sigma: float,
) -> list[tuple[float, float, float]]:
    """Detect stars in a zone and measure their FWHM.

    Uses simple peak detection + Gaussian fit.

    Parameters
    ----------
    zone : ndarray
        2D image patch (zone of the image).
    threshold_sigma : float
        Detection threshold in sigma above background.

    Returns
    -------
    list of (float, float, float)
        List of (y, x, fwhm) for each detected star.
    """
    med = float(np.median(zone))
    mad = float(np.median(np.abs(zone - med)))
    noise = max(mad * 1.4826, 1e-6)
    threshold = med + threshold_sigma * noise

    stars: list[tuple[float, float, float]] = []
    h, w = zone.shape

    # Simple local-maximum detection within 3x3 neighbourhood
    from scipy.ndimage import maximum_filter

    local_max = maximum_filter(zone, size=3) == zone
    peaks = np.argwhere(local_max & (zone > threshold))

    for py, px in peaks:
        if not (0 < py < h - 1 and 0 < px < w - 1):
            continue
        fwhm = _gaussian_fit_2d(zone, int(py), int(px), fit_radius=5)
        if fwhm is not None:
            stars.append((float(py), float(px), fwhm))

    return stars


def compute_fwhm_map(
    image: NDArray,
    params: FWHMMapParams | None = None,
) -> FWHMMapResult:
    """Compute FWHM map from detected stars across the image grid.

    Parameters
    ----------
    image : ndarray
        (H, W) or (C, H, W) float32 in [0, 1]. If RGB, uses luminance.
    params : FWHMMapParams, optional
        Grid and detection parameters. Uses defaults if *None*.

    Returns
    -------
    FWHMMapResult
        Per-zone statistics and tilt diagnosis.
    """
    if params is None:
        params = FWHMMapParams()

    if image.ndim == 3:
        gray = np.mean(image, axis=0).astype(np.float64)
    else:
        gray = image.astype(np.float64)

    h, w = gray.shape
    rows, cols = params.grid_rows, params.grid_cols
    zone_h = max(h // rows, 1)
    zone_w = max(w // cols, 1)

    fwhm_map = np.zeros((rows, cols), dtype=np.float64)
    star_counts = np.zeros((rows, cols), dtype=np.int32)

    for r in range(rows):
        for c in range(cols):
            y0 = r * zone_h
            y1 = min((r + 1) * zone_h, h)
            x0 = c * zone_w
            x1 = min((c + 1) * zone_w, w)
            zone = gray[y0:y1, x0:x1]

            detected = _detect_stars_zone(zone, params.star_detection_threshold)
            if len(detected) >= params.min_stars_per_zone:
                fwhms = [d[2] for d in detected]
                fwhm_map[r, c] = float(np.mean(fwhms))
            else:
                fwhm_map[r, c] = 0.0
            star_counts[r, c] = len(detected)

    valid = fwhm_map[fwhm_map > 0]
    if len(valid) == 0:
        return FWHMMapResult(
            fwhm_map=fwhm_map,
            star_counts=star_counts,
            mean_fwhm=0.0,
            std_fwhm=0.0,
            max_fwhm=0.0,
            min_fwhm=0.0,
            tilt_detected=False,
            tilt_angle=0.0,
        )

    mean_fwhm = float(np.mean(valid))
    std_fwhm = float(np.std(valid))
    max_fwhm = float(np.max(valid))
    min_fwhm = float(np.min(valid))

    tilt_detected = max_fwhm > 1.5 * max(min_fwhm, 0.5)

    # Estimate tilt direction from gradient of FWHM map
    if tilt_detected and rows > 1 and cols > 1:
        grad_y, grad_x = np.gradient(fwhm_map)
        gx = float(np.mean(grad_x[valid.reshape(rows, cols)]))
        gy = float(np.mean(grad_y[valid.reshape(rows, cols)]))
        tilt_angle = float(np.degrees(np.arctan2(gy, gx)))
    else:
        tilt_angle = 0.0

    log.info(
        "FWHM map: %.2f ± %.2f px, %.0f%% zones sampled, tilt=%s (angle=%.1f°)",
        mean_fwhm, std_fwhm,
        100.0 * float(np.sum(star_counts >= params.min_stars_per_zone)) / (rows * cols),
        tilt_detected, tilt_angle,
    )

    return FWHMMapResult(
        fwhm_map=fwhm_map,
        star_counts=star_counts,
        mean_fwhm=mean_fwhm,
        std_fwhm=std_fwhm,
        max_fwhm=max_fwhm,
        min_fwhm=min_fwhm,
        tilt_detected=tilt_detected,
        tilt_angle=tilt_angle,
    )
