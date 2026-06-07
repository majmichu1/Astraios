"""Photometric Color Calibration — calibrate color balance using star photometry.

Uses detected star fluxes and optional catalog data to compute a
color correction matrix. Falls back to statistical white balance
when no catalog is available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from cosmica.core.masks import Mask, apply_mask
from cosmica.core.star_detection import detect_stars
from cosmica.core.wcs import normalise_wcs_dict

ProgressCallback = Callable[[float, str], None]


def _noop_progress(f: float, m: str) -> None:
    pass

log = logging.getLogger(__name__)


@dataclass
class ColorCalibrationParams:
    """Parameters for photometric color calibration."""

    white_reference: str = "average"  # "average", "G2V" (solar type), "custom"
    custom_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    neutralize_background: bool = True
    background_percentile: float = 10.0  # use darkest N% for background


@dataclass
class ColorCalibrationResult:
    """Result of color calibration."""

    data: np.ndarray
    correction_factors: tuple[float, float, float]  # R, G, B multipliers
    background_offset: tuple[float, float, float] | None = None


def color_calibrate(
    image: np.ndarray,
    params: ColorCalibrationParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback = _noop_progress,
) -> ColorCalibrationResult:
    """Calibrate color balance using star photometry.

    For images without WCS/catalog data, uses statistical methods:
    1. Background neutralization (make background neutral gray)
    2. White balance using average star color or specified reference

    Parameters
    ----------
    image : ndarray
        Color image, shape (C, H, W) with C >= 3, values in [0, 1].
    params : ColorCalibrationParams, optional
        Calibration parameters.
    mask : Mask, optional
        Selective processing mask.

    Returns
    -------
    ColorCalibrationResult
        Calibrated image and correction factors.
    """
    if params is None:
        params = ColorCalibrationParams()

    if image.ndim != 3 or image.shape[0] < 3:
        log.warning("Color calibration requires a color image with >= 3 channels")
        return ColorCalibrationResult(data=image, correction_factors=(1.0, 1.0, 1.0))

    original = image.copy()
    result = image.copy()

    bg_offset = None

    # Step 1: Background neutralization
    if params.neutralize_background:
        progress(0.1, "Neutralizing background…")
        bg_offset = _neutralize_background(result, params.background_percentile)

    # Step 2: White balance using star photometry
    progress(0.4, "Computing white balance…")
    correction = _compute_white_balance(result, params)
    progress(0.8, "Applying correction…")
    for ch in range(3):
        result[ch] = np.clip(result[ch] * correction[ch], 0, 1)

    result = apply_mask(original, result, mask)
    progress(1.0, "Color calibration complete")

    return ColorCalibrationResult(
        data=result,
        correction_factors=correction,
        background_offset=bg_offset,
    )


def _neutralize_background(
    image: np.ndarray,
    percentile: float,
) -> tuple[float, float, float]:
    """Neutralize the background by subtracting per-channel offsets.

    Uses the darkest pixels as the background reference.
    Modifies image in-place.

    Returns the offsets that were subtracted.
    """
    offsets = []
    for ch in range(3):
        # Use percentile of non-zero pixels as background
        channel = image[ch]
        valid = channel[channel > 0]
        if len(valid) == 0:
            offsets.append(0.0)
            continue
        bg_val = float(np.percentile(valid, percentile))
        offsets.append(bg_val)

    # Find the minimum offset (darkest channel background)
    min_offset = min(offsets)

    # Subtract so all channels have the same background level
    for ch in range(3):
        excess = offsets[ch] - min_offset
        image[ch] = np.clip(image[ch] - excess, 0, 1)

    return (offsets[0], offsets[1], offsets[2])


def _compute_white_balance(
    image: np.ndarray,
    params: ColorCalibrationParams,
) -> tuple[float, float, float]:
    """Compute per-channel correction factors for white balance.

    Uses star photometry: detects stars, measures their average color,
    and computes correction to make them match the reference.
    """
    if params.white_reference == "custom":
        return params.custom_rgb

    # Detect stars and measure their average RGB
    sf = detect_stars(image, max_stars=100, sigma_threshold=5.0)

    if len(sf) < 5:
        log.info("Too few stars for photometric calibration, using global statistics")
        return _global_white_balance(image)

    # Measure star fluxes in each channel
    star_colors = []
    for star in sf.stars:
        ix, iy = int(round(star.x)), int(round(star.y))
        # Aperture photometry: sum within small radius
        r = max(int(star.fwhm), 2)
        y_lo = max(0, iy - r)
        y_hi = min(image.shape[1], iy + r + 1)
        x_lo = max(0, ix - r)
        x_hi = min(image.shape[2], ix + r + 1)

        if y_hi - y_lo < 2 or x_hi - x_lo < 2:
            continue

        rgb = [float(image[ch, y_lo:y_hi, x_lo:x_hi].sum()) for ch in range(3)]
        if all(v > 0 for v in rgb):
            star_colors.append(rgb)

    if len(star_colors) < 3:
        return _global_white_balance(image)

    colors = np.array(star_colors)
    avg_rgb = np.median(colors, axis=0)

    if params.white_reference == "G2V":
        # G2V stars (solar type, ~5770K) have a slightly blue spectral slope.
        # Using Gaia BP/RP-based G2V synthetic colors: redder channels need
        # slightly more gain to neutralize the blue bias.
        target = np.array([1.035, 1.0, 0.965])
    else:  # "average"
        # Make the average star color neutral
        target = np.array([1.0, 1.0, 1.0])

    # Correction factors to make avg_rgb match target
    correction = target / np.maximum(avg_rgb, 1e-10)
    # Normalize so the brightest channel has factor 1.0
    correction = correction / correction.max()

    log.info("PCC correction factors: R=%.3f G=%.3f B=%.3f", *correction)
    return (float(correction[0]), float(correction[1]), float(correction[2]))


def _global_white_balance(image: np.ndarray) -> tuple[float, float, float]:
    """Fallback: use global median for white balance."""
    medians = [float(np.median(image[ch])) for ch in range(3)]
    max_med = max(medians)
    if max_med < 1e-10:
        return (1.0, 1.0, 1.0)
    correction = [max_med / max(m, 1e-10) for m in medians]
    # Normalize
    max_corr = max(correction)
    correction = [c / max_corr for c in correction]
    return (correction[0], correction[1], correction[2])


def photometric_color_calibrate(
    image: np.ndarray,
    catalog_stars: list | None = None,
    wcs: dict | None = None,
    neutralize_bg: bool = True,
    background_percentile: float = 10.0,
) -> ColorCalibrationResult:
    """Photometric color calibration using star catalog data.

    If catalog stars are provided with WCS, performs true PCC using
    Gaia DR3 color information. Otherwise falls back to statistical
    white balance.

    Parameters
    ----------
    image : ndarray
        Color image, shape (C, H, W) with C >= 3, values in [0, 1].
    catalog_stars : list, optional
        List of StarCatalogEntry from Gaia query.
    wcs : dict, optional
        WCS info with 'ra', 'dec' keys for pixel-to-sky conversion.
    neutralize_bg : bool
        Whether to neutralize background.
    background_percentile : float
        Percentile for background estimation.

    Returns
    -------
    ColorCalibrationResult
        Calibrated image and correction factors.
    """
    if catalog_stars is None or wcs is None:
        params = ColorCalibrationParams(
            neutralize_background=neutralize_bg,
            background_percentile=background_percentile,
        )
        return color_calibrate(image, params)

    if image.ndim != 3 or image.shape[0] < 3:
        log.warning("PCC requires a color image with >= 3 channels")
        return ColorCalibrationResult(data=image, correction_factors=(1.0, 1.0, 1.0))

    result = image.copy()
    bg_offset = None

    if neutralize_bg:
        bg_offset = _neutralize_background(result, background_percentile)

    det_stars = detect_stars(result, max_stars=100, sigma_threshold=5.0)
    if len(det_stars.stars) < 5:
        log.warning("Too few detected stars for PCC, using statistical calibration")
        return color_calibrate(result, ColorCalibrationParams(neutralize_background=False))

    matched_stars = _match_stars_to_catalog(
        result, det_stars, catalog_stars, wcs, result.shape[1], result.shape[2]
    )

    if len(matched_stars) < 5:
        log.warning("Not enough matched stars for PCC, using statistical calibration")
        return color_calibrate(result, ColorCalibrationParams(neutralize_background=False))

    correction = _compute_ccm_from_matches(matched_stars)

    for ch in range(3):
        result[ch] = np.clip(result[ch] * correction[ch], 0, 1)

    log.info("PCC with catalog: correction factors R=%.3f G=%.3f B=%.3f", *correction)
    return ColorCalibrationResult(
        data=result,
        correction_factors=correction,
        background_offset=bg_offset,
    )


def _match_stars_to_catalog(image, detected, catalog_stars, wcs, height, width):
    """Match detected stars to catalog stars using WCS pixel→sky conversion.

    Uses astropy WCS when a full WCS header is available, falls back to
    simple linear projection otherwise.
    """
    if wcs is not None:
        wcs = normalise_wcs_dict(wcs)
    if not wcs or "ra" not in wcs or wcs["ra"] is None:
        return []

    # Try astropy WCS for accurate pixel→sky conversion
    sky_fn = _make_pixel_to_sky(wcs, width, height)

    scale_deg = (wcs.get("scale") or 1.0) / 3600.0  # arcsec/px → deg/px
    match_radius_deg = max(scale_deg * 5, 0.002)   # 5 px or at least 7"

    matches = []
    for star in detected.stars:
        px, py = float(star.x), float(star.y)
        star_ra, star_dec = sky_fn(px, py)

        best_match = None
        best_sep = float("inf")
        for cat_star in catalog_stars:
            # Angular separation (small angle — Euclidean OK here)
            dra = (cat_star.ra_deg - star_ra) * np.cos(np.radians(star_dec))
            ddec = cat_star.dec_deg - star_dec
            sep = np.hypot(dra, ddec)
            if sep < best_sep and sep < match_radius_deg:
                best_sep = sep
                best_match = cat_star

        if best_match is None:
            continue

        # Aperture photometry on the star
        r = max(int(round(star.fwhm * 1.5)), 3)
        y_lo = max(0, int(py) - r)
        y_hi = min(height, int(py) + r + 1)
        x_lo = max(0, int(px) - r)
        x_hi = min(width, int(px) + r + 1)

        if y_hi - y_lo < 2 or x_hi - x_lo < 2:
            continue

        fluxes = [float(image[ch, y_lo:y_hi, x_lo:x_hi].sum()) for ch in range(3)]

        if all(f > 0 for f in fluxes):
            matches.append({"instrumental": fluxes, "catalog": best_match})

    log.info("PCC matched %d/%d stars to catalog", len(matches), len(detected.stars))
    return matches


def _make_pixel_to_sky(wcs: dict, width: int, height: int):
    """Return a (px, py) → (ra_deg, dec_deg) callable.

    Uses astropy WCS when a full header is present, otherwise falls back
    to a simple tangent-plane linear approximation.
    """
    header = wcs.get("wcs_header")
    if header:
        try:
            from astropy.io import fits as afits
            from astropy.wcs import WCS as AstroWCS
            h = afits.Header(header)
            awcs = AstroWCS(h)

            def sky_astropy(px, py):
                sky = awcs.pixel_to_world(px, py)
                return float(sky.ra.deg), float(sky.dec.deg)

            return sky_astropy
        except Exception:
            log.warning("WCS pixel_to_world failed, using linear fallback")

    # Linear approximation fallback
    ra0 = float(wcs["ra"])
    dec0 = float(wcs["dec"])
    scale_deg = (wcs.get("scale") or 1.0) / 3600.0
    cos_dec = np.cos(np.radians(dec0))

    def sky_linear(px, py):
        ra = ra0 + (px - width / 2) * scale_deg / max(cos_dec, 1e-6)
        dec = dec0 - (py - height / 2) * scale_deg   # y axis flipped
        return ra, dec

    return sky_linear


def _compute_ccm_from_matches(matches: list) -> tuple[float, float, float]:
    """Compute per-channel correction factors from catalog-matched stars.

    Uses Gaia BP-RP color index as a proxy for the true R/G/B ratio.
    The model: observed color ratio should match catalog color temperature.
    """
    if len(matches) < 3:
        return (1.0, 1.0, 1.0)

    # Build least-squares system: find R,G,B scale factors that minimise
    # the difference between instrumental and catalog-predicted colors.
    r_ratios = []
    b_ratios = []

    for m in matches:
        inst = np.array(m["instrumental"], dtype=np.float64)
        cat = m["catalog"]

        if cat.bp_mag is None or cat.rp_mag is None or cat.g_mag is None:
            continue
        if inst[1] <= 0:   # green channel as reference
            continue

        bp_rp = float(cat.bp_mag - cat.rp_mag)

        # Gaia BP-RP → expected (R-G) and (B-G) color offsets
        # Empirical relation from Gaia DR3 passbands:
        #   R/G ≈ 10^( 0.27 * bp_rp)
        #   B/G ≈ 10^(-0.30 * bp_rp)
        expected_r_g = 10 ** (0.27 * bp_rp)
        expected_b_g = 10 ** (-0.30 * bp_rp)

        observed_r_g = inst[0] / inst[1]
        observed_b_g = inst[2] / inst[1]

        if observed_r_g > 0:
            r_ratios.append(expected_r_g / observed_r_g)
        if observed_b_g > 0:
            b_ratios.append(expected_b_g / observed_b_g)

    if not r_ratios or not b_ratios:
        return (1.0, 1.0, 1.0)

    # Robust median to reject outliers
    r_factor = float(np.median(r_ratios))
    b_factor = float(np.median(b_ratios))
    g_factor = 1.0

    # Normalize so green=1 (or normalize to max=1)
    max_f = max(r_factor, g_factor, b_factor)
    r_factor /= max_f
    g_factor /= max_f
    b_factor /= max_f

    # Clamp to reasonable range
    r_factor = float(np.clip(r_factor, 0.3, 3.0))
    g_factor = float(np.clip(g_factor, 0.3, 3.0))
    b_factor = float(np.clip(b_factor, 0.3, 3.0))

    log.info("PCC CCM from %d stars: R=%.3f G=%.3f B=%.3f", len(r_ratios), r_factor, g_factor, b_factor)
    return (r_factor, g_factor, b_factor)
