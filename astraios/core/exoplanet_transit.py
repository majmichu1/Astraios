"""Exoplanet transit detector — differential aperture photometry over an image sequence.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro), Copyright Franklin
Marek, GPL-3.0-or-later. https://github.com/setiastro/setiastrosuitepro

Given a chronologically-ordered sequence of frames (paths or in-memory arrays), a
target star position, and a set of comparison-star positions (or an auto-selected
set), this module measures each star's flux per frame with the shared aperture
photometry helpers, builds a differential ("ensemble") light curve for the target,
detrends it, and flags a transit dip using the moving-average threshold + temporal
SNR check that SASpro's "Dip threshold" / "Temporal SNR min" controls implement.

Design notes
------------
* This is deliberately CPU/numpy, not GPU: transit photometry samples small
  (~tens-of-pixel) apertures at a handful of fixed positions per frame — there is no
  large tensor to vectorise on a GPU, and the per-frame cost is dominated by disk I/O
  when streaming from paths. `device_manager` is not used here.
* Frames given as paths are loaded and discarded one at a time (peak RAM: ~2 frames:
  the first frame cached for star detection/reference, plus the frame under
  measurement) so long time-series runs don't need to hold every sub in memory.
* SASpro auto-sorts loaded subs by DATE-OBS before measuring. This port does not
  re-sort: callers are expected to pass ``frames_or_paths`` already in chronological
  order, since output arrays are index-aligned with the input sequence.
* Deferred vs. SASpro: the Lomb-Scargle + Box-Least-Squares period search (multi-
  transit period/duration folding in SASpro's "Analyze Star..." dialog), the full
  per-frame SEP-based star_count/FWHM/eccentricity frame-quality metrics, and
  TESScut/AAVSO/SIMBAD export/lookup are all UI-adjacent or period-search features
  out of scope for this single-dip core port. `quality` here is a lightweight
  median/MAD proxy instead of SASpro's SEP-derived per-frame stats.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from astraios.core.analysis.aperture_photometry import _annulus_stats, _aperture_sum
from astraios.core.star_detection import detect_stars

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


class DetrendMethod(Enum):
    """Baseline-detrending polynomial degree (mirrors SASpro's Detrend combo box)."""

    NONE = "none"
    LINEAR = "linear"
    QUADRATIC = "quadratic"


_DETREND_DEGREE = {DetrendMethod.NONE: None, DetrendMethod.LINEAR: 1, DetrendMethod.QUADRATIC: 2}


class TimeSource(Enum):
    """Where per-frame timestamps come from."""

    AUTO = "auto"  # FITS DATE-OBS/DATE-END/MJD-OBS header if available, else frame index
    INDEX = "index"  # always use the frame index (0, 1, 2, ...)


class ComparisonCombine(Enum):
    """How multiple comparison-star fluxes are combined into one reference flux."""

    MEAN = "mean"
    MEDIAN = "median"
    SUM = "sum"


@dataclass
class ExoplanetTransitParams:
    """Settings for :func:`analyze_transit`."""

    aperture_radius: float = 10.0  # pixels
    annulus_inner: float = 15.0
    annulus_outer: float = 20.0

    # Comparison-star selection
    n_comparison_stars: int = 5  # how many auto-selected comparisons to keep
    comparison_combine: ComparisonCombine = ComparisonCombine.MEAN
    star_detection_sigma: float = 5.0  # sigma threshold for auto candidate detection
    max_auto_comparison_candidates: int = 200  # cap for the detect_stars() pass
    border_fraction: float = 0.10  # ignore candidates within this fraction of the edge
    target_exclusion_radius: float | None = None  # px; default 2x annulus_outer

    # Detrend / detection (mirrors SASpro's Detrend combo, dip-threshold slider,
    # and temporal-SNR spin box)
    detrend_method: DetrendMethod = DetrendMethod.QUADRATIC
    ma_window: int = 5  # moving-average window (frames)
    detection_threshold_ppt: float = 20.0  # min dip depth, parts-per-thousand
    temporal_snr_threshold: float = 2.0  # min (peak dip) / (MAD of MA residuals); 0 disables
    min_good_frames: int = 5  # minimum finite, positive-flux frames required to analyze

    # Time source
    time_source: TimeSource = TimeSource.AUTO


@dataclass
class TransitStar:
    """A star position used in the differential photometry."""

    x: float
    y: float
    role: str = "comparison"  # "target" | "comparison"


@dataclass
class TransitResult:
    """Output of :func:`analyze_transit`."""

    times: NDArray  # (n_frames,) JD if time_is_jd else frame index
    time_is_jd: bool
    target_flux: NDArray  # (n_frames,) sky-subtracted target aperture flux
    comparison_flux: NDArray  # (n_frames,) combined comparison-star flux
    relative_flux: NDArray  # (n_frames,) the light curve: target/comparison, unit-median
    normalized_baseline: float  # out-of-transit median of relative_flux (~1.0)
    transit_depth: float  # fractional depth of the deepest detected dip (0..1)
    transit_detected: bool
    mid_transit_time: float | None  # time of deepest dip, or None if no dip found
    target_star: TransitStar
    comparison_stars: list[TransitStar]
    quality: NDArray | None = None  # (n_frames,) lightweight per-frame SNR proxy
    airmass: NDArray | None = None  # (n_frames,) AIRMASS or altitude estimate, NaN if unknown
    n_good_frames: int = 0


# --------------------------------------------------------------------------- #
# Frame loading
# --------------------------------------------------------------------------- #


def _load_plane_and_header(item: Any) -> tuple[NDArray, dict[str, Any] | None]:
    """Load one frame as a 2D luminance plane + optional header dict.

    Paths are loaded (and released) one at a time via the shared image loader;
    in-memory arrays are used directly. Color frames (C, H, W) are photometered
    on luminance (mean over channels), matching the rest of the codebase's
    star-detection convention.
    """
    if isinstance(item, (str, Path)):
        from astraios.core.image_io import load_image

        img = load_image(item)
        data = img.data
        header = img.header
    else:
        data = np.asarray(item, dtype=np.float32)
        header = None

    plane = np.mean(data, axis=0) if data.ndim == 3 else data
    return np.ascontiguousarray(plane.astype(np.float32, copy=False)), header


def _parse_frame_time(header: dict[str, Any]):
    """Parse an observation timestamp from a FITS-style header dict.

    Preference order: DATE-OBS, DATE-END, MJD-OBS.
    """
    from astropy.time import Time

    for key in ("DATE-OBS", "DATE-END"):
        v = header.get(key)
        if isinstance(v, str) and v.strip():
            try:
                return Time(v, scale="utc")
            except Exception:
                continue

    v = header.get("MJD-OBS")
    if v is not None:
        try:
            return Time(float(v), format="mjd", scale="utc")
        except Exception:
            pass

    return None


def _estimate_airmass_from_altitude(alt_deg: float) -> float:
    """Simple plane-parallel airmass estimate: 1 / sin(altitude)."""
    alt_rad = np.deg2rad(np.clip(alt_deg, 0.1, 90.0))
    return float(1.0 / np.sin(alt_rad))


def _extract_airmass(header: dict[str, Any]) -> float:
    if "AIRMASS" in header:
        try:
            return float(header["AIRMASS"])
        except (TypeError, ValueError):
            pass
    for key in ("OBJCTALT", "ALT", "ALTITUDE", "EL"):
        if key in header:
            try:
                return _estimate_airmass_from_altitude(float(header[key]))
            except (TypeError, ValueError):
                continue
    return float("nan")


# --------------------------------------------------------------------------- #
# Photometry
# --------------------------------------------------------------------------- #


def _measure_flux(
    plane: NDArray, x: float, y: float, params: ExoplanetTransitParams
) -> float:
    """Sky-subtracted aperture flux at (x, y), reusing the shared photometry helpers.

    Same background model as ``aperture_photometry._photutils_photometry``:
    a circular aperture sum minus (annulus median * aperture area).
    """
    ap_sum = _aperture_sum(plane, x, y, params.aperture_radius)
    bg_median, _bg_std = _annulus_stats(plane, x, y, params.annulus_inner, params.annulus_outer)
    ap_area = np.pi * params.aperture_radius**2
    return float(ap_sum - bg_median * ap_area)


def _auto_candidate_positions(
    plane: NDArray, target_xy: tuple[float, float], params: ExoplanetTransitParams
) -> list[tuple[float, float]]:
    """Detect candidate comparison stars on the reference (first) frame.

    Excludes stars near the target and near the frame border, keeping the
    brightest-first ordering from :func:`detect_stars`.
    """
    field = detect_stars(
        plane,
        max_stars=params.max_auto_comparison_candidates + 5,
        sigma_threshold=params.star_detection_sigma,
    )
    h, w = plane.shape
    bf = params.border_fraction
    excl_r = (
        params.target_exclusion_radius
        if params.target_exclusion_radius is not None
        else 2.0 * params.annulus_outer
    )
    tx, ty = target_xy

    candidates: list[tuple[float, float]] = []
    for star in field.stars:
        if not (w * bf <= star.x <= w * (1 - bf) and h * bf <= star.y <= h * (1 - bf)):
            continue
        if (star.x - tx) ** 2 + (star.y - ty) ** 2 <= excl_r**2:
            continue
        candidates.append((float(star.x), float(star.y)))
        if len(candidates) >= params.max_auto_comparison_candidates:
            break
    return candidates


def _select_best_comparisons(flux_candidates: NDArray, n_select: int) -> NDArray:
    """Rank candidate comparison stars by brightness and frame-to-frame stability.

    Score = median flux / coefficient-of-variation, so brighter and more stable
    (lower relative scatter) stars are preferred — analogous to SASpro's ensemble
    neighbor selection, but scored explicitly rather than picked purely by nearest
    median flux.
    """
    medians = np.nanmedian(flux_candidates, axis=1)
    stds = np.nanstd(flux_candidates, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = stds / np.maximum(medians, 1e-12)

    valid = np.isfinite(medians) & (medians > 0) & np.isfinite(cv)
    score = np.where(valid, medians / (cv + 1e-6), -np.inf)
    order = np.argsort(score)[::-1]
    n_select = max(1, min(n_select, int(np.sum(valid))))
    return order[:n_select]


def _combine_comparison(flux_rows: NDArray, method: ComparisonCombine) -> NDArray:
    with np.errstate(invalid="ignore"):
        if method == ComparisonCombine.MEAN:
            return np.nanmean(flux_rows, axis=0)
        if method == ComparisonCombine.MEDIAN:
            return np.nanmedian(flux_rows, axis=0)
        return np.nansum(flux_rows, axis=0)


# --------------------------------------------------------------------------- #
# Detrend + dip detection
# --------------------------------------------------------------------------- #


def _detrend_curve(curve: NDArray, degree: int, mask: NDArray | None = None) -> NDArray:
    """Normalize a curve to its own polynomial baseline (ported from SASpro).

    Fits a degree-``degree`` polynomial to the good (finite, positive) points and
    divides the curve by that fitted trend, so slow systematic drifts (airmass,
    focus, cloud gradients) are removed while a short dip is preserved.
    """
    x = np.arange(curve.size)
    if mask is None:
        mask = np.isfinite(curve) & (curve > 0)
    n_good = int(mask.sum())
    if n_good < 2:
        return curve
    fit_deg = min(degree, n_good - 1)
    if fit_deg < 1:
        return curve
    try:
        coeffs = np.polyfit(x[mask], curve[mask], fit_deg)
    except Exception:
        return curve
    trend = np.polyval(coeffs, x)
    trend[trend == 0] = 1.0
    return curve / trend


def _moving_average(curve: NDArray, window: int) -> NDArray:
    """Edge-padded moving average (ported from SASpro's ``moving_average``).

    The window is forced odd so the edge-padded convolution always returns an
    array the same length as the input (SASpro always calls this with a fixed
    odd window of 5; this generalizes safely for a user-configurable window).
    """
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    ext = np.pad(curve, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(ext, kernel, mode="valid")


def _detect_dip(
    times_good: NDArray, flux_good: NDArray, params: ExoplanetTransitParams
) -> tuple[bool, float, float | None, float]:
    """Moving-average dip detection (ported from SASpro's ``apply_threshold``).

    Finds the deepest below-baseline excursion of the moving average, in parts
    per thousand, and flags a detection when it clears both the depth threshold
    and the temporal-SNR check (peak depth vs. MAD scatter of MA residuals).

    Returns
    -------
    detected, depth_fraction, mid_time, baseline
    """
    n = flux_good.size
    if n < params.min_good_frames:
        baseline = float(np.nanmedian(flux_good)) if n > 0 else float("nan")
        return False, 0.0, None, baseline

    med = float(np.nanmedian(flux_good))
    f_norm = flux_good / med if med > 0 else flux_good.copy()

    window = max(1, min(params.ma_window, f_norm.size))
    ma = _moving_average(f_norm, window)

    dip_ppt = (1.0 - ma) * 1000.0
    peak_idx = int(np.nanargmax(dip_ppt))
    peak_ppt = float(dip_ppt[peak_idx])

    residuals = f_norm - ma
    med_res = float(np.nanmedian(residuals))
    mad_res_ppt = float(1.4826 * np.nanmedian(np.abs(residuals - med_res)) * 1000.0)

    detected = False
    if peak_ppt > 0:
        meets_depth = peak_ppt >= params.detection_threshold_ppt
        meets_snr = True
        if params.temporal_snr_threshold > 0 and mad_res_ppt > 0:
            meets_snr = (peak_ppt / mad_res_ppt) >= params.temporal_snr_threshold
        detected = meets_depth and meets_snr

    depth_fraction = max(0.0, peak_ppt / 1000.0)
    mid_time = float(times_good[peak_idx]) if peak_ppt > 0 else None
    return detected, depth_fraction, mid_time, med


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def analyze_transit(
    frames_or_paths: Sequence[Any],
    target_xy: tuple[float, float],
    comparison_xys: Sequence[tuple[float, float]] | None = None,
    params: ExoplanetTransitParams | None = None,
    progress: ProgressCallback | None = None,
) -> TransitResult:
    """Measure a differential light curve and detect a transit dip.

    Parameters
    ----------
    frames_or_paths : sequence
        Chronologically-ordered frames: either (H, W) / (C, H, W) float32 [0, 1]
        arrays, or paths to image files (loaded and released one at a time).
    target_xy : (float, float)
        Pixel (x, y) of the target star, in the coordinate system of the first frame.
    comparison_xys : sequence of (float, float), optional
        Pixel positions of comparison stars. If *None*, comparison stars are
        auto-detected on the first frame and ranked by brightness/stability
        (see ``params.n_comparison_stars``).
    params : ExoplanetTransitParams, optional
        Photometry, comparison-selection, detrend, and detection settings.
    progress : callable, optional
        ``progress(fraction, message)`` called after each frame is measured.

    Returns
    -------
    TransitResult
    """
    if params is None:
        params = ExoplanetTransitParams()
    if progress is None:
        progress = _noop_progress

    frames = list(frames_or_paths)
    n_frames = len(frames)
    if n_frames < 3:
        raise ValueError("analyze_transit requires at least 3 frames")

    target_xy = (float(target_xy[0]), float(target_xy[1]))

    progress(0.0, "Loading reference frame…")
    first_plane, first_header = _load_plane_and_header(frames[0])

    auto_select = comparison_xys is None
    if comparison_xys is None:
        candidate_positions = _auto_candidate_positions(first_plane, target_xy, params)
        if not candidate_positions:
            raise ValueError(
                "No comparison-star candidates found; supply comparison_xys explicitly"
            )
    else:
        candidate_positions = [(float(x), float(y)) for x, y in comparison_xys]
        if not candidate_positions:
            raise ValueError("comparison_xys was provided but is empty")

    positions = [target_xy, *candidate_positions]
    n_positions = len(positions)

    flux = np.full((n_positions, n_frames), np.nan, dtype=np.float64)
    times_raw: list[Any] = []
    airmass = np.full(n_frames, np.nan, dtype=np.float64)
    quality = np.full(n_frames, np.nan, dtype=np.float64)

    for i, item in enumerate(frames):
        plane, header = (first_plane, first_header) if i == 0 else _load_plane_and_header(item)

        for p_idx, (x, y) in enumerate(positions):
            flux[p_idx, i] = _measure_flux(plane, x, y, params)

        med = float(np.median(plane))
        mad = float(np.median(np.abs(plane - med)))
        quality[i] = med / max(mad, 1e-8)

        if header is not None:
            airmass[i] = _extract_airmass(header)
            times_raw.append(
                _parse_frame_time(header) if params.time_source != TimeSource.INDEX else None
            )
        else:
            times_raw.append(None)

        progress((i + 1) / n_frames, f"Measured frame {i + 1}/{n_frames}")

    # --- Comparison-star selection & combination ---
    candidate_flux = flux[1:, :]
    if auto_select:
        selected = _select_best_comparisons(candidate_flux, params.n_comparison_stars)
    else:
        selected = np.arange(candidate_flux.shape[0])

    comparison_positions = [candidate_positions[i] for i in selected]
    comparison_flux = _combine_comparison(candidate_flux[selected, :], params.comparison_combine)
    target_flux = flux[0, :]

    # --- Differential light curve ---
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_flux = target_flux / comparison_flux
    valid = (
        np.isfinite(relative_flux)
        & (relative_flux > 0)
        & np.isfinite(target_flux)
        & (target_flux > 0)
        & np.isfinite(comparison_flux)
        & (comparison_flux > 0)
    )
    relative_flux = np.where(valid, relative_flux, np.nan)

    unit_med = float(np.nanmedian(relative_flux[valid])) if np.any(valid) else np.nan
    if np.isfinite(unit_med) and unit_med > 0:
        relative_flux = relative_flux / unit_med

    # --- Detrend against out-of-transit baseline ---
    degree = _DETREND_DEGREE[params.detrend_method]
    if degree is not None:
        good_mask = np.isfinite(relative_flux) & (relative_flux > 0)
        relative_flux = _detrend_curve(relative_flux, degree, mask=good_mask)
        good_mask = np.isfinite(relative_flux) & (relative_flux > 0)
        if np.any(good_mask):
            re_med = float(np.nanmedian(relative_flux[good_mask]))
            if re_med > 0:
                relative_flux = relative_flux / re_med

    # --- Time axis ---
    if params.time_source == TimeSource.INDEX:
        times_arr = np.arange(n_frames, dtype=np.float64)
        time_is_jd = False
    else:
        valid_times = [t for t in times_raw if t is not None]
        if len(valid_times) == n_frames:
            from astropy.time import Time

            times_arr = Time(times_raw).utc.jd.astype(np.float64)
            time_is_jd = True
        else:
            if 0 < len(valid_times) < n_frames:
                log.warning(
                    "Only %d/%d frames had a parseable timestamp; falling back to "
                    "frame-index time axis for all frames",
                    len(valid_times), n_frames,
                )
            times_arr = np.arange(n_frames, dtype=np.float64)
            time_is_jd = False

    # --- Dip detection ---
    good_mask = np.isfinite(relative_flux) & (relative_flux > 0)
    n_good = int(np.sum(good_mask))
    detected, depth, mid_time, baseline = _detect_dip(
        times_arr[good_mask], relative_flux[good_mask], params
    )

    target_star = TransitStar(x=target_xy[0], y=target_xy[1], role="target")
    comparison_stars = [TransitStar(x=x, y=y, role="comparison") for x, y in comparison_positions]

    return TransitResult(
        times=times_arr,
        time_is_jd=time_is_jd,
        target_flux=target_flux,
        comparison_flux=comparison_flux,
        relative_flux=relative_flux,
        normalized_baseline=baseline,
        transit_depth=depth,
        transit_detected=detected,
        mid_transit_time=mid_time,
        target_star=target_star,
        comparison_stars=comparison_stars,
        quality=quality,
        airmass=airmass,
        n_good_frames=n_good,
    )
