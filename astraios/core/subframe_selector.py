"""Subframe Selector — score and rank light frames for quality-based rejection.

Loads each frame, measures its PSF (FWHM, eccentricity), estimates SNR and
background level, counts detected stars, and produces a weighted quality
score.  Frames that fall outside a configurable sigma threshold from the
mean quality score are rejected.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

import numpy as np

from astraios.core.image_io import load_image
from astraios.core.psf import measure_psf
from astraios.core.stretch import compute_channel_stats

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]

# ProcessPoolExecutor context: forkserver on Linux gives fork-speed startup without
# inheriting the parent process's CUDA handles (which would be UB in child processes).
# Falls back to spawn on Windows/macOS where forkserver is unavailable.
try:
    _MP_CTX: mp.context.BaseContext = mp.get_context("forkserver")
except ValueError:
    _MP_CTX = mp.get_context("spawn")


def _noop_progress(fraction: float, message: str) -> None:
    """No-op progress callback."""


@dataclass
class SubframeScore:
    """Quality score for a single light frame.

    Attributes
    ----------
    file_path : str
        Path to the source image file.
    fwhm : float
        Measured full width at half maximum in pixels (lower is better).
    eccentricity : float
        Star eccentricity / ellipticity (0 = round, lower is better).
    snr : float
        Estimated signal-to-noise ratio (higher is better).
    background : float
        Median background level in [0, 1].
    n_stars : int
        Number of stars used for PSF measurement.
    quality_score : float
        Weighted composite quality score (higher is better).
    accepted : bool
        *True* if the frame passed sigma-clipping rejection.
    """

    file_path: str
    fwhm: float
    eccentricity: float
    snr: float
    background: float
    n_stars: int
    quality_score: float
    accepted: bool


@dataclass
class SubframeSelectorParams:
    """Parameters controlling subframe scoring and rejection.

    Attributes
    ----------
    fwhm_weight : float
        Weight for FWHM in the quality score (lower FWHM = better).
    eccentricity_weight : float
        Weight for eccentricity (lower eccentricity = better).
    snr_weight : float
        Weight for SNR (higher SNR = better).
    stars_weight : float
        Weight for star count (more stars = better).
    rejection_sigma : float
        Sigma-clipping threshold.  Frames with quality score more than
        this many standard deviations below the mean are rejected.
    """

    fwhm_weight: float = 0.3
    eccentricity_weight: float = 0.2
    snr_weight: float = 0.3
    stars_weight: float = 0.2
    rejection_sigma: float = 2.0


def _normalize_metric(
    values: np.ndarray,
    invert: bool = False,
) -> np.ndarray:
    """Min-max normalise a 1D array of metric values to [0, 1].

    Parameters
    ----------
    values : np.ndarray
        Raw metric values.
    invert : bool
        If *True*, invert the normalisation so that lower raw values
        produce higher normalised scores (e.g. FWHM, eccentricity).

    Returns
    -------
    np.ndarray
        Normalised values in [0, 1].
    """
    v_min = values.min()
    v_max = values.max()
    spread = v_max - v_min

    if spread < 1e-12:
        # All values identical — return uniform 1.0
        return np.ones_like(values, dtype=np.float64)

    normalised = (values - v_min) / spread

    if invert:
        normalised = 1.0 - normalised

    return normalised


def _measure_frame(file_path: str) -> dict:
    """Load a single frame and measure its quality metrics.

    Parameters
    ----------
    file_path : str
        Path to the image file.

    Returns
    -------
    dict
        Dictionary with keys: fwhm, eccentricity, snr, background, n_stars.
        On failure, returns defaults with fwhm=999, snr=0, n_stars=0.
    """
    try:
        image_data = load_image(file_path)
        data = image_data.data

        # Measure PSF (FWHM + eccentricity).  force_cpu=True so parallel threads
        # don't contend on CUDA dispatch (Python GIL is held during kernel launch,
        # serializing the pool).  Pure numpy/scipy releases the GIL → true multi-core.
        psf = measure_psf(data, force_cpu=True)
        fwhm = psf.fwhm
        eccentricity = psf.ellipticity
        n_stars = psf.n_stars_used

        # Compute channel stats for SNR and background
        if data.ndim == 3:
            # Use luminance-like average for overall stats
            gray = np.mean(data, axis=0).astype(np.float32)
        else:
            gray = data

        stats = compute_channel_stats(gray)
        background = stats["median"]

        # SNR estimate: median / MAD-estimated noise
        mad = stats["mad"]
        noise_est = max(mad * 1.4826, 1e-10)
        snr = background / noise_est

        return {
            "fwhm": fwhm,
            "eccentricity": eccentricity,
            "snr": snr,
            "background": background,
            "n_stars": n_stars,
        }

    except Exception:
        log.exception("Failed to measure frame: %s", file_path)
        return {
            "fwhm": 999.0,
            "eccentricity": 1.0,
            "snr": 0.0,
            "background": 0.0,
            "n_stars": 0,
        }


def score_subframes(
    frame_paths: list[str],
    params: SubframeSelectorParams | None = None,
    progress: ProgressCallback | None = None,
    frame_callback: Callable[[int, dict], None] | None = None,
) -> list[SubframeScore]:
    """Score and rank a list of light frames for quality-based rejection.

    For each frame the function loads the image, measures the PSF
    (FWHM, eccentricity, star count), estimates SNR, and records the
    background level.  A composite quality score is then computed as a
    weighted sum of min-max normalised metrics.  Finally, frames with a
    quality score more than ``params.rejection_sigma`` standard
    deviations below the mean are marked as rejected.

    Parameters
    ----------
    frame_paths : list[str]
        Paths to the light frame image files.
    params : SubframeSelectorParams, optional
        Scoring and rejection parameters.  Defaults are used if *None*.
    progress : callable, optional
        Progress callback with signature ``(fraction: float, message: str)``.
    frame_callback : callable, optional
        Called as each frame finishes measuring with ``(idx, metrics_dict)``
        so callers can stream per-frame raw metrics before global normalisation.

    Returns
    -------
    list[SubframeScore]
        One ``SubframeScore`` per input frame, in the same order as
        *frame_paths*.
    """
    if params is None:
        params = SubframeSelectorParams()

    if progress is None:
        progress = _noop_progress

    n_frames = len(frame_paths)
    if n_frames == 0:
        log.warning("No frames provided to score_subframes")
        return []

    log.info("Scoring %d subframes...", n_frames)

    # -- Phase 1: Measure each frame (parallel, multi-process) ----------------
    # ProcessPoolExecutor gives each worker its own Python interpreter and its
    # own GIL.  This is essential here because scipy.optimize.curve_fit calls
    # the Python callback _gaussian_2d on every MINPACK function evaluation —
    # that callback holds the GIL, so ThreadPoolExecutor workers would all
    # serialize on it.  Separate processes bypass this entirely.
    # _measure_frame is a module-level function with str input / dict output,
    # so it's trivially picklable for cross-process IPC.
    n_workers = min(os.cpu_count() or 4, n_frames)
    measurements: list[dict] = [{}] * n_frames
    completed = 0

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=_MP_CTX) as pool:
        futures = {pool.submit(_measure_frame, path): idx for idx, path in enumerate(frame_paths)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                measurements[idx] = future.result()
            except Exception:
                log.exception("Failed to measure frame at index %d", idx)
                measurements[idx] = {
                    "fwhm": 999.0, "eccentricity": 1.0,
                    "snr": 0.0, "background": 0.0, "n_stars": 0,
                }
            completed += 1
            if frame_callback is not None:
                frame_callback(idx, measurements[idx])
            progress(completed / n_frames * 0.8, f"Measured {completed}/{n_frames} frames…")

    progress(0.8, "Computing quality scores...")

    # -- Phase 2: Normalise and score -----------------------------------------
    fwhm_arr = np.array([m["fwhm"] for m in measurements], dtype=np.float64)
    ecc_arr = np.array([m["eccentricity"] for m in measurements], dtype=np.float64)
    snr_arr = np.array([m["snr"] for m in measurements], dtype=np.float64)
    stars_arr = np.array([m["n_stars"] for m in measurements], dtype=np.float64)

    # Normalise: FWHM and eccentricity are "lower is better" (inverted)
    norm_fwhm = _normalize_metric(fwhm_arr, invert=True)
    norm_ecc = _normalize_metric(ecc_arr, invert=True)
    norm_snr = _normalize_metric(snr_arr, invert=False)
    norm_stars = _normalize_metric(stars_arr, invert=False)

    quality_scores = (
        params.fwhm_weight * norm_fwhm
        + params.eccentricity_weight * norm_ecc
        + params.snr_weight * norm_snr
        + params.stars_weight * norm_stars
    )

    # -- Phase 3: Sigma-clip rejection ----------------------------------------
    mean_q = float(np.mean(quality_scores))
    std_q = float(np.std(quality_scores))

    if std_q > 1e-12:
        threshold = mean_q - params.rejection_sigma * std_q
    else:
        # All scores are identical — accept everything
        threshold = mean_q - 1.0

    progress(0.9, "Building results...")

    # -- Phase 4: Build results -----------------------------------------------
    results: list[SubframeScore] = []
    n_rejected = 0

    for idx in range(n_frames):
        m = measurements[idx]
        q = float(quality_scores[idx])
        accepted = q >= threshold

        if not accepted:
            n_rejected += 1

        results.append(
            SubframeScore(
                file_path=frame_paths[idx],
                fwhm=m["fwhm"],
                eccentricity=m["eccentricity"],
                snr=m["snr"],
                background=m["background"],
                n_stars=m["n_stars"],
                quality_score=q,
                accepted=accepted,
            )
        )

    log.info(
        "Subframe scoring complete: %d/%d accepted (mean=%.3f, sigma=%.3f, threshold=%.3f)",
        n_frames - n_rejected,
        n_frames,
        mean_q,
        std_q,
        threshold,
    )

    progress(1.0, f"Done — {n_frames - n_rejected}/{n_frames} frames accepted")

    return results


def _get_fwhm(s: SubframeScore) -> float:
    return s.fwhm


def _get_snr(s: SubframeScore) -> float:
    return s.snr


def _get_quality_score(s: SubframeScore) -> float:
    return s.quality_score


def filter_by_metric(
    scores: list[SubframeScore],
    metric: str = "fwhm",
    mode: str = "top_n",
    top_n: int | None = None,
    top_percent: float = 80.0,
    sigma: float = 2.0,
) -> list[SubframeScore]:
    """Filter frames by quality metric.

    Parameters
    ----------
    scores : list[SubframeScore]
        List of scored frames from `score_subframes`.
    metric : str
        Quality metric to use: "fwhm", "snr", or "quality_score".
    mode : str
        Filtering mode: "top_n", "top_percent", or "sigma".
    top_n : int | None
        Number of best frames to keep (for mode="top_n").
    top_percent : float
        Percentage of best frames to keep (for mode="top_percent").
    sigma : float
        Sigma rejection threshold (for mode="sigma").

    Returns
    -------
    list[SubframeScore]
        Frames with `accepted` field updated based on filtering.
    """
    if not scores:
        return scores

    result = [SubframeScore(**vars(s)) for s in scores]

    if metric == "fwhm":
        sort_key = _get_fwhm
        ascending = True
    elif metric == "snr":
        sort_key = _get_snr
        ascending = False
    else:
        sort_key = _get_quality_score
        ascending = False

    sorted_scores = sorted(result, key=sort_key, reverse=not ascending)

    if mode == "top_n":
        n_keep = top_n if top_n is not None else len(result)
        for i, s in enumerate(sorted_scores):
            s.accepted = i < n_keep
    elif mode == "top_percent":
        n_keep = max(1, int(len(result) * top_percent / 100.0))
        for i, s in enumerate(sorted_scores):
            s.accepted = i < n_keep
    elif mode == "sigma":
        if metric == "fwhm":
            values = [s.fwhm for s in result]
        elif metric == "snr":
            values = [s.snr for s in result]
        else:
            values = [s.quality_score for s in result]
        mean_v = float(np.mean(values))
        std_v = float(np.std(values))
        if metric == "fwhm":
            threshold = mean_v + sigma * std_v
            for s in result:
                s.accepted = s.fwhm <= threshold
        else:
            threshold = mean_v - sigma * std_v
            for s in result:
                if metric == "snr":
                    s.accepted = s.snr >= threshold
                else:
                    s.accepted = s.quality_score >= threshold
    else:
        for s in result:
            s.accepted = True

    return result
