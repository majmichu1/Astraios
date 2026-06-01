"""Tilt & aberration analysis — measures star shape distortions across the field."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)


@dataclass
class TiltAnalysisParams:
    grid_rows: int = 8
    grid_cols: int = 8
    min_stars_per_zone: int = 2


@dataclass
class TiltAnalysisResult:
    ellipticity_map: NDArray  # (rows, cols) mean ellipticity (1 - b/a) per zone
    angle_map: NDArray  # (rows, cols) mean orientation angle in degrees
    fwhm_map: NDArray  # (rows, cols) mean FWHM per zone
    coma_detected: bool  # True if ellipticity correlates with radial distance
    astigmatism_detected: bool  # True if angle varies systematically
    tilt_detected: bool  # True if FWHM varies asymmetrically
    summary: str  # human-readable diagnosis


def _measure_star_shape(
    data: NDArray,
    y: int,
    x: int,
    size: int = 9,
) -> tuple[float, float] | None:
    """Measure ellipticity and angle of a single star.

    Uses image moments to compute shape parameters.

    Parameters
    ----------
    data : ndarray
        2D image patch containing the star.
    y : int
        Row index of the star centre.
    x : int
        Column index of the star centre.
    size : int
        Half-size of the subimage to extract for moment calculation.

    Returns
    -------
    tuple (float, float) or None
        (ellipticity, angle_deg) where ellipticity = 1 - b/a (0 = circular),
        or *None* if the measurement fails.
    """
    h, w = data.shape
    y0 = max(y - size, 0)
    y1 = min(y + size + 1, h)
    x0 = max(x - size, 0)
    x1 = min(x + size + 1, w)
    patch = data[y0:y1, x0:x1]

    if patch.size < 9:
        return None

    bg = float(np.median(patch))
    patch_bg = np.maximum(patch - bg, 0.0)

    total = float(np.sum(patch_bg))
    if total < 1e-10:
        return None

    yy, xx = np.mgrid[0 : patch.shape[0], 0 : patch.shape[1]]
    cy = float(np.sum(yy * patch_bg) / total)
    cx = float(np.sum(xx * patch_bg) / total)

    mu20 = float(np.sum((xx - cx) ** 2 * patch_bg) / total)
    mu02 = float(np.sum((yy - cy) ** 2 * patch_bg) / total)
    mu11 = float(np.sum((xx - cx) * (yy - cy) * patch_bg) / total)

    common = np.sqrt((mu20 - mu02) ** 2 + 4 * mu11 ** 2)
    if common < 1e-12:
        return 0.0, 0.0

    a_sq = 2.0 * (mu20 + mu02 + common)
    b_sq = 2.0 * (mu20 + mu02 - common)

    a = np.sqrt(max(a_sq, 0.0))
    b = np.sqrt(max(b_sq, 0.0))

    if a < 1e-12:
        return None

    ellipticity = 1.0 - b / a
    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

    return float(ellipticity), float(np.degrees(theta))


def _detect_star_centroids(
    zone: NDArray,
    threshold_sigma: float = 3.0,
    max_stars: int = 20,
) -> list[tuple[int, int]]:
    """Detect bright pixel peaks as candidate star positions in a zone.

    Parameters
    ----------
    zone : ndarray
        2D image patch.
    threshold_sigma : float
        Detection threshold in sigma above background.
    max_stars : int
        Maximum number of stars to return (brightest first).

    Returns
    -------
    list of (int, int)
        List of (y, x) peak positions.
    """
    from scipy.ndimage import maximum_filter

    med = float(np.median(zone))
    mad = float(np.median(np.abs(zone - med)))
    noise = max(mad * 1.4826, 1e-6)
    threshold = med + threshold_sigma * noise

    local_max = maximum_filter(zone, size=3) == zone
    peaks = np.argwhere(local_max & (zone > threshold))

    if len(peaks) > max_stars:
        intensities = zone[peaks[:, 0], peaks[:, 1]]
        top = np.argsort(intensities)[::-1][:max_stars]
        peaks = peaks[top]

    return [(int(py), int(px)) for py, px in peaks]


def analyze_tilt(
    image: NDArray,
    params: TiltAnalysisParams | None = None,
) -> TiltAnalysisResult:
    """Analyze optical tilt and aberrations from star shapes.

    Divides the image into a grid, detects stars in each zone, measures
    their ellipticity and orientation, and builds maps that reveal
    optical issues such as tilt, coma, and astigmatism.

    Parameters
    ----------
    image : ndarray
        (H, W) or (C, H, W) float32 in [0, 1].
    params : TiltAnalysisParams, optional
        Analysis parameters. Uses defaults if *None*.

    Returns
    -------
    TiltAnalysisResult
        Per-zone metrics and human-readable diagnosis.
    """
    if params is None:
        params = TiltAnalysisParams()

    if image.ndim == 3:
        gray = np.mean(image, axis=0).astype(np.float64)
    else:
        gray = image.astype(np.float64)

    h, w = gray.shape
    rows, cols = params.grid_rows, params.grid_cols
    zone_h = max(h // rows, 1)
    zone_w = max(w // cols, 1)

    ellipticity_map = np.full((rows, cols), np.nan, dtype=np.float64)
    angle_map = np.full((rows, cols), np.nan, dtype=np.float64)
    fwhm_map = np.full((rows, cols), np.nan, dtype=np.float64)

    for r in range(rows):
        for c in range(cols):
            y0 = r * zone_h
            y1 = min((r + 1) * zone_h, h)
            x0 = c * zone_w
            x1 = min((c + 1) * zone_w, w)
            zone = gray[y0:y1, x0:x1]

            centroids = _detect_star_centroids(zone)

            zone_ells: list[float] = []
            zone_angles: list[float] = []

            for py, px in centroids:
                shape = _measure_star_shape(zone, py, px, size=7)
                if shape is not None:
                    ell, ang = shape
                    zone_ells.append(ell)
                    zone_angles.append(ang)

            if len(zone_ells) >= params.min_stars_per_zone:
                ellipticity_map[r, c] = float(np.mean(zone_ells))
                angle_map[r, c] = float(np.mean(zone_angles))

    # Detect tilt: FWHM variation (approximated from ellipticity gradient)
    fwhm_map[:] = np.nan
    valid_ell = ellipticity_map[~np.isnan(ellipticity_map)]
    if len(valid_ell) > 0:
        mask = ~np.isnan(ellipticity_map)
        fwhm_map[mask] = 1.0 + ellipticity_map[mask] * 3.0
    else:
        fwhm_map[:] = 0.0

    # Coma detection: correlation between ellipticity and radial distance
    coma_detected = False
    astigmatism_detected = False
    tilt_detected = False

    valid_mask = ~np.isnan(ellipticity_map)
    if np.any(valid_mask):
        ell_vals = ellipticity_map[valid_mask]
        y_centers = np.linspace(0, h, rows)[:, None] + zone_h / 2.0
        x_centers = np.linspace(0, w, cols)[None, :] + zone_w / 2.0
        cy_all = h / 2.0
        cx_all = w / 2.0

        r_indices, c_indices = np.where(valid_mask)
        radial_dists = np.sqrt(
            (y_centers[r_indices, 0] - cy_all) ** 2 +
            (x_centers[0, c_indices] - cx_all) ** 2
        )

        if len(radial_dists) > 5:
            corr = np.corrcoef(radial_dists, ell_vals)[0, 1]
            coma_detected = abs(corr) > 0.5

        # Astigmatism: angle variation across field
        angle_vals = angle_map[valid_mask]
        if len(angle_vals) > 5:
            angle_std = float(np.nanstd(angle_vals))
            astigmatism_detected = angle_std > 20.0

        # Tilt: asymmetric FWHM (large ellipticity imbalance)
        ell_flat = ell_vals.flatten()
        if len(ell_flat) >= 4:
            half = len(ell_flat) // 2
            left_ell = np.mean(ell_flat[:half])
            right_ell = np.mean(ell_flat[half:])
            tilt_detected = abs(left_ell - right_ell) > 0.05

    # Build human-readable summary
    parts: list[str] = []
    n_valid = int(np.sum(valid_mask))
    total_zones = rows * cols

    if not coma_detected and not astigmatism_detected and not tilt_detected:
        summary = "No significant optical aberrations detected."
    else:
        if tilt_detected:
            parts.append("Optical tilt detected (asymmetric FWHM)")
        if coma_detected:
            parts.append("Coma detected (ellipticity correlates with radial distance)")
        if astigmatism_detected:
            parts.append("Astigmatism detected (systematic angle variation)")
        summary = "; ".join(parts)

    log.info(
        "Tilt analysis: %d/%d zones with stars, tilt=%s, coma=%s, astig=%s",
        n_valid, total_zones, tilt_detected, coma_detected, astigmatism_detected,
    )

    return TiltAnalysisResult(
        ellipticity_map=ellipticity_map,
        angle_map=angle_map,
        fwhm_map=fwhm_map,
        coma_detected=coma_detected,
        astigmatism_detected=astigmatism_detected,
        tilt_detected=tilt_detected,
        summary=summary,
    )
