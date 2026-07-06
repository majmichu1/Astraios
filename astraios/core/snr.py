"""Signal-to-noise ratio (SNR) measurement tool.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro), Copyright
Franklin Marek, GPL-3.0-or-later. https://github.com/setiastro/setiastrosuitepro

SASpro's ``SNRToolDialog._calculate`` measures, per channel, the median of a
user-picked object mask against the median/stddev of an auto-selected
background rectangle::

    So      = max(signal_median - background_median, 0.0)
    noise   = max(background_std, _EPS)
    SNR     = So / noise
    SNR_dB  = 10 * log10(max(SNR, _EPS))

This port keeps that formula exactly (including the ``10 * log10`` -- not
``20 * log10`` -- convention SASpro uses) but replaces its interactive
mask/rectangle picker with plain ``(x, y, w, h)`` bounding boxes supplied via
:class:`SNRParams`, since Astraios's read-only measurement dialogs use
spin-box regions rather than a freehand region-picker overlay. When a region
is omitted, this falls back to a robust whole-frame estimate (see
:func:`measure_snr`).

CPU vs. GPU
-----------
This is a one-shot statistical reduction over a (typically small) region or,
at most, the full frame -- not a per-frame hot loop. The robust background
estimate (used when no background rectangle is given) uses astropy's
``SigmaClip`` per the project rule that sigma-clipping must use astropy/numpy
and never a hand-rolled PyTorch version; astropy's sigma-clipping has no GPU
implementation. Porting the rest (plain mean/median/std reductions) through
`device_manager`/PyTorch would only pay off for large, repeated, per-frame
reductions -- this tool runs once per button click on a bounded region, so a
GPU path was not built. No benchmark was fabricated to justify this; it is a
reasoning call based on operation shape (single small reduction, astropy
dependency) rather than measured performance.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

_EPS = 1e-12  # matches SASpro's snr_tool._EPS exactly

ProgressFn = Callable[[float, str], None]

BBox = tuple[int, int, int, int]  # (x, y, w, h) in pixels


@dataclass
class SNRParams:
    """Parameters for :func:`measure_snr`.

    Attributes
    ----------
    background_bbox : tuple[int, int, int, int] | None
        ``(x, y, w, h)`` pixel rectangle to sample as background. If *None*,
        the background is estimated robustly from the whole frame using
        astropy sigma-clipped statistics (see module docstring).
    signal_bbox : tuple[int, int, int, int] | None
        ``(x, y, w, h)`` pixel rectangle to sample as the signal/target
        region. If *None*, the whole frame's median stands in for the signal
        level (a quick global check rather than a per-object measurement).
    sigma : float
        Sigma-clip threshold, used only when ``background_bbox`` is *None*.
    maxiters : int
        Sigma-clip maximum iterations, used only when ``background_bbox`` is
        *None*.
    per_channel : bool
        Also compute per-channel SNR for colour images (in addition to the
        always-present pooled "Overall" entry).
    """

    background_bbox: BBox | None = None
    signal_bbox: BBox | None = None
    sigma: float = 3.0
    maxiters: int = 5
    per_channel: bool = True


@dataclass
class ChannelSNR:
    """SNR measurement for a single channel (or the pooled "Overall" entry).

    Attributes
    ----------
    name : str
        Channel name ("Mono", "R"/"G"/"B", or "Overall").
    signal_mean, signal_median : float
        Mean and median of the signal-region sample.
    background_mean, background_median, background_std, background_mad : float
        Mean, median, standard deviation, and median absolute deviation of
        the background-region sample. ``background_std`` is the noise term
        actually used in the SNR calculation (see :func:`measure_snr` for how
        it differs between the explicit-rectangle and auto-estimate paths).
    net_signal : float
        Background-subtracted signal level: ``max(signal_median -
        background_median, 0.0)`` (SASpro's "So").
    snr : float
        Linear signal-to-noise ratio: ``net_signal / max(background_std, eps)``.
    snr_db : float
        ``10 * log10(max(snr, eps))``, matching SASpro's dB convention.
    n_signal_px, n_background_px : int
        Sample sizes used for the signal and background regions.
    """

    name: str
    signal_mean: float
    signal_median: float
    background_mean: float
    background_median: float
    background_std: float
    background_mad: float
    net_signal: float
    snr: float
    snr_db: float
    n_signal_px: int
    n_background_px: int


@dataclass
class SNRResult:
    """Full SNR measurement result: per-channel entries plus a pooled overall."""

    channels: list[ChannelSNR]
    overall: ChannelSNR
    background_bbox: BBox | None
    signal_bbox: BBox | None
    background_auto: bool
    signal_auto: bool
    n_channels: int
    width: int
    height: int


def _channel_names(n_channels: int) -> list[str]:
    """Return human-readable channel names based on the channel count."""
    if n_channels == 1:
        return ["Mono"]
    if n_channels == 3:
        return ["R", "G", "B"]
    if n_channels == 4:
        return ["L", "R", "G", "B"]
    return [f"Ch{i}" for i in range(n_channels)]


def _clip_bbox(bbox: BBox, width: int, height: int) -> tuple[slice, slice]:
    """Clip an (x, y, w, h) box to the image bounds and return (row, col) slices."""
    x, y, w, h = bbox
    x0 = max(0, min(width, int(x)))
    y0 = max(0, min(height, int(y)))
    x1 = max(0, min(width, int(x) + int(w)))
    y1 = max(0, min(height, int(y) + int(h)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Region {bbox} is empty after clipping to {width}x{height}.")
    return slice(y0, y1), slice(x0, x1)


def _background_stats(
    bg_vals: NDArray, auto: bool, sigma: float, maxiters: int
) -> tuple[float, float, float, float]:
    """Return (mean, median, std_used_as_noise, mad) for a background sample.

    When ``auto`` is False (an explicit background rectangle was given), this
    mirrors SASpro's ``_stats_on_mask`` exactly: a plain mean/median and a
    sample standard deviation (``ddof=1``) used directly as the noise term.

    When ``auto`` is True (no rectangle given -- the whole frame stands in for
    "background"), stray bright pixels (stars, the target itself) would
    otherwise bias a plain mean/std upward, so an astropy ``SigmaClip`` first
    rejects outliers; noise is then the MAD-based robust estimate
    (``mad * 1.4826``) already used elsewhere in this codebase (see
    ``star_detection.py``'s noise floor), with the same ``max(..., 1e-6)``
    floor to avoid a degenerate zero-noise SNR on a perfectly flat sample.
    """
    if bg_vals.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    if not auto:
        mean_val = float(np.mean(bg_vals))
        median_val = float(np.median(bg_vals))
        std_val = float(np.std(bg_vals, ddof=1)) if bg_vals.size > 1 else 0.0
        mad_val = float(np.median(np.abs(bg_vals - median_val)))
        return mean_val, median_val, std_val, mad_val

    from astropy.stats import SigmaClip

    clip = SigmaClip(sigma=sigma, maxiters=maxiters)
    clipped = clip(bg_vals, masked=True)
    clean = clipped.compressed()
    if clean.size == 0:
        clean = bg_vals

    mean_val = float(np.mean(clean))
    median_val = float(np.median(clean))
    mad_val = float(np.median(np.abs(clean - median_val)))
    std_val = max(mad_val * 1.4826, 1e-6)
    return mean_val, median_val, std_val, mad_val


def _channel_snr(
    name: str,
    sig_vals: NDArray,
    bg_vals: NDArray,
    background_auto: bool,
    sigma: float,
    maxiters: int,
) -> ChannelSNR:
    signal_mean = float(np.mean(sig_vals)) if sig_vals.size else 0.0
    signal_median = float(np.median(sig_vals)) if sig_vals.size else 0.0

    bg_mean, bg_median, bg_std, bg_mad = _background_stats(
        bg_vals, background_auto, sigma, maxiters
    )

    net_signal = max(signal_median - bg_median, 0.0)
    noise = max(bg_std, _EPS)
    snr = net_signal / noise
    snr_db = 10.0 * math.log10(max(snr, _EPS))  # matches SASpro exactly

    return ChannelSNR(
        name=name,
        signal_mean=signal_mean,
        signal_median=signal_median,
        background_mean=bg_mean,
        background_median=bg_median,
        background_std=bg_std,
        background_mad=bg_mad,
        net_signal=net_signal,
        snr=snr,
        snr_db=snr_db,
        n_signal_px=int(sig_vals.size),
        n_background_px=int(bg_vals.size),
    )


def measure_snr(
    image: NDArray,
    params: SNRParams | None = None,
    progress: ProgressFn | None = None,
) -> SNRResult:
    """Measure signal-to-noise ratio for an image.

    Parameters
    ----------
    image : ndarray
        float32 image in [0, 1]. Mono: (H, W). Colour: (C, H, W).
    params : SNRParams, optional
        Measurement parameters. Defaults estimate everything automatically:
        whole-frame median as the signal level vs. a sigma-clipped
        whole-frame background.
    progress : callable, optional
        ``progress(fraction, message)`` callback.

    Returns
    -------
    SNRResult
        Per-channel (optional) and pooled ("Overall") SNR measurements.
    """
    if params is None:
        params = SNRParams()

    if image.ndim == 2:
        n_channels = 1
        height, width = image.shape
        planes = [image]
    elif image.ndim == 3:
        n_channels, height, width = image.shape
        planes = [image[c] for c in range(n_channels)]
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    names = _channel_names(n_channels)

    background_auto = params.background_bbox is None
    signal_auto = params.signal_bbox is None

    bg_slices = (
        _clip_bbox(params.background_bbox, width, height)
        if params.background_bbox is not None
        else None
    )
    sig_slices = (
        _clip_bbox(params.signal_bbox, width, height)
        if params.signal_bbox is not None
        else None
    )

    channels: list[ChannelSNR] = []
    pooled_sig: list[NDArray] = []
    pooled_bg: list[NDArray] = []

    for idx, plane in enumerate(planes):
        if progress is not None:
            progress(idx / max(n_channels, 1), f"Measuring {names[idx]}...")

        bg_vals = plane[bg_slices].ravel() if bg_slices is not None else plane.ravel()
        sig_vals = plane[sig_slices].ravel() if sig_slices is not None else plane.ravel()

        pooled_bg.append(bg_vals)
        pooled_sig.append(sig_vals)

        if params.per_channel:
            channels.append(
                _channel_snr(
                    names[idx], sig_vals, bg_vals, background_auto,
                    params.sigma, params.maxiters,
                )
            )

    overall = _channel_snr(
        "Overall",
        np.concatenate(pooled_sig),
        np.concatenate(pooled_bg),
        background_auto,
        params.sigma,
        params.maxiters,
    )

    if progress is not None:
        progress(1.0, "Done.")

    log.info(
        "SNR measured: overall=%.4g (%.2f dB), background_auto=%s, signal_auto=%s",
        overall.snr, overall.snr_db, background_auto, signal_auto,
    )

    return SNRResult(
        channels=channels,
        overall=overall,
        background_bbox=params.background_bbox,
        signal_bbox=params.signal_bbox,
        background_auto=background_auto,
        signal_auto=signal_auto,
        n_channels=n_channels,
        width=width,
        height=height,
    )
