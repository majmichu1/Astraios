"""Image Peeker — quick multi-file inspector for culling a night's subs.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro), Copyright
Franklin Marek, GPL-3.0-or-later. https://github.com/setiastro/setiastrosuitepro

SASpro's "Image Peeker" opens a single already-loaded image and tiles 100%
crops from its corners/edges/center into a mosaic, with an optional dispatch
to tilt / focal-plane / astrometric-distortion analysis dialogs. Astraios
already has that per-image field-quality workflow covered by
``astraios.core.analysis.tilt_analysis`` and ``astraios.core.analysis.fwhm_map``
(zone-based ellipticity/FWHM maps across a single frame). What Astraios was
missing — and what this module provides under the same tool name and the same
"peek fast, don't fully load" spirit — is a *multi-file* inspector: point it
at a folder of subs and get an auto-stretched thumbnail plus quick per-frame
stats for each one, so a night's subs can be culled without opening every
frame individually. This reuses ``astraios.core.psf.measure_psf`` (itself
built on ``astraios.core.star_detection.detect_stars``) for the FWHM /
eccentricity / star-count columns exactly as ``subframe_selector.py`` does,
``astraios.core.statistics.compute_image_statistics`` for the min/median/
mean/max/std columns, and ``astraios.core.stretch.auto_stretch`` (GPU via
``device_manager``) for the thumbnail preview — no quality scoring or
rejection logic is duplicated here; see ``subframe_selector.py`` for that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from astraios.core.image_io import load_image
from astraios.core.psf import measure_psf
from astraios.core.statistics import ImageStatistics, compute_image_statistics
from astraios.core.stretch import StretchParams, auto_stretch

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    """No-op progress callback."""


@dataclass
class ImagePeekParams:
    """Parameters controlling per-frame thumbnail generation and measurement.

    Attributes
    ----------
    thumbnail_size : int
        Longest-edge size in pixels for the generated preview thumbnail.
    measure_stars : bool
        If *True*, measure FWHM / eccentricity / star count per frame via
        ``measure_psf``. Disable for a faster pass over very large batches
        when only the brightness stats and thumbnail are needed.
    max_stars : int
        Maximum number of stars used for the PSF fit (passed to
        ``measure_psf``).
    """

    thumbnail_size: int = 220
    measure_stars: bool = True
    max_stars: int = 30


@dataclass
class FramePeek:
    """Quick-look summary of a single light frame.

    Attributes
    ----------
    path : str
        Path to the source image file.
    thumbnail : ndarray
        Auto-stretched uint8 RGB preview, shape ``(h, w, 3)`` with its
        longest edge downscaled to ``params.thumbnail_size``.
    width, height : int
        Full-resolution image dimensions (not the thumbnail's).
    is_color : bool
        *True* for multi-channel (color) frames.
    n_channels : int
        Number of image channels (1 for mono, 3 for RGB).
    stats : ImageStatistics
        Full per-channel statistics from ``compute_image_statistics``.
    fwhm : float | None
        Measured FWHM in pixels (geometric mean), or *None* if star
        measurement was skipped or failed.
    eccentricity : float | None
        Measured star eccentricity (0 = round), or *None*.
    n_stars : int | None
        Number of stars used for the PSF fit, or *None*.
    exposure : float | None
        Exposure time in seconds, from the FITS/XISF header.
    filter_name : str | None
        Filter name, from the header.
    temperature : float | None
        CCD/sensor temperature, from the header.
    date_obs : str | None
        Observation timestamp, from the header.
    error : str | None
        Set if this frame partially failed (e.g. star measurement raised)
        while the frame itself still loaded successfully.
    """

    path: str
    thumbnail: NDArray
    width: int
    height: int
    is_color: bool
    n_channels: int
    stats: ImageStatistics
    fwhm: float | None = None
    eccentricity: float | None = None
    n_stars: int | None = None
    exposure: float | None = None
    filter_name: str | None = None
    temperature: float | None = None
    date_obs: str | None = None
    error: str | None = None

    @property
    def median(self) -> float:
        """Representative brightness: mean of per-channel medians."""
        return float(np.mean([c.median for c in self.stats.channels]))

    @property
    def mean(self) -> float:
        """Representative brightness: mean of per-channel means."""
        return float(np.mean([c.mean for c in self.stats.channels]))

    @property
    def min_val(self) -> float:
        """Darkest pixel value across all channels."""
        return float(min(c.min_val for c in self.stats.channels))

    @property
    def max_val(self) -> float:
        """Brightest pixel value across all channels."""
        return float(max(c.max_val for c in self.stats.channels))

    @property
    def std(self) -> float:
        """Representative noise: mean of per-channel standard deviations."""
        return float(np.mean([c.std for c in self.stats.channels]))

    @property
    def snr_estimate(self) -> float:
        """Representative SNR: mean of per-channel SNR estimates."""
        return float(np.mean([c.snr_estimate for c in self.stats.channels]))


def _make_thumbnail(data: NDArray, max_size: int) -> NDArray:
    """Downscale then auto-stretch to build a small uint8 RGB preview.

    Downscaling *before* stretching keeps this fast even for large
    full-resolution frames — mirrors the thumbnail helper in
    ``subframe_dialog.py``.
    """
    h, w = data.shape[-2], data.shape[-1]
    if max(h, w) > max_size:
        # Ceiling-divide so the strided result never exceeds max_size (floor
        # division under-strides whenever h/max_size falls just above an
        # integer, e.g. h=90, max_size=48 -> h//new_h=1 gives no downscale).
        new_h, new_w = max(1, int(h * max_size / max(h, w))), max(1, int(w * max_size / max(h, w)))
        ry = max(1, -(-h // new_h))
        rx = max(1, -(-w // new_w))
        small = data[..., ::ry, ::rx]
    else:
        small = data

    stretched = auto_stretch(small, StretchParams())

    if stretched.ndim == 2:
        rgb = np.stack([stretched] * 3, axis=-1)
    else:
        rgb = np.transpose(stretched, (1, 2, 0))  # (C,H,W) -> (H,W,C)
        if rgb.shape[2] == 1:
            rgb = np.repeat(rgb, 3, axis=2)
        elif rgb.shape[2] > 3:
            rgb = rgb[:, :, :3]

    rgb8 = np.clip(rgb, 0.0, 1.0) * 255.0
    return np.ascontiguousarray(rgb8).astype(np.uint8)


def peek_frames(
    paths: list[str],
    params: ImagePeekParams | None = None,
    progress: ProgressCallback | None = None,
) -> list[FramePeek]:
    """Load, thumbnail, and measure quick stats for a batch of light frames.

    For each path this loads the frame, builds a small auto-stretched
    preview thumbnail, computes min/median/mean/max/std statistics, and
    (optionally) measures FWHM / eccentricity / star count. Files that fail
    to load are logged and skipped — they simply do not appear in the
    returned list, so one corrupt file never aborts the batch.

    Parameters
    ----------
    paths : list[str]
        Paths to the light frame image files.
    params : ImagePeekParams, optional
        Thumbnail size and measurement options. Defaults are used if *None*.
    progress : callable, optional
        Progress callback with signature ``(fraction: float, message: str)``.

    Returns
    -------
    list[FramePeek]
        One ``FramePeek`` per *readable* input path, in input order.
    """
    if params is None:
        params = ImagePeekParams()
    if progress is None:
        progress = _noop_progress

    n_paths = len(paths)
    if n_paths == 0:
        log.warning("No frames provided to peek_frames")
        return []

    log.info("Peeking %d frames...", n_paths)
    results: list[FramePeek] = []

    for idx, path in enumerate(paths):
        name = Path(path).name
        try:
            image = load_image(path)
            data = image.data
        except Exception:
            log.warning("Skipping unreadable frame: %s", path, exc_info=True)
            progress((idx + 1) / n_paths, f"Skipped (unreadable): {name}")
            continue

        stats = compute_image_statistics(data)
        thumbnail = _make_thumbnail(data, params.thumbnail_size)

        fwhm = eccentricity = n_stars = None
        error = None
        if params.measure_stars:
            try:
                psf = measure_psf(data, max_stars=params.max_stars, force_cpu=True)
                fwhm = psf.fwhm
                eccentricity = psf.ellipticity
                n_stars = psf.n_stars_used
            except Exception as exc:
                log.warning("Star measurement failed for %s: %s", path, exc)
                error = f"Star measurement failed: {exc}"

        results.append(
            FramePeek(
                path=str(path),
                thumbnail=thumbnail,
                width=image.width,
                height=image.height,
                is_color=image.is_color,
                n_channels=image.channels,
                stats=stats,
                fwhm=fwhm,
                eccentricity=eccentricity,
                n_stars=n_stars,
                exposure=image.exposure,
                filter_name=image.filter_name,
                temperature=image.temperature,
                date_obs=image.header.get("DATE-OBS"),
                error=error,
            )
        )

        progress((idx + 1) / n_paths, f"Peeked {idx + 1}/{n_paths}: {name}")

    n_ok = len(results)
    log.info("Frame peek complete: %d/%d frames readable", n_ok, n_paths)
    progress(1.0, f"Done — {n_ok}/{n_paths} frames readable")

    return results
