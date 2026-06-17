"""Image Statistics — read-only computation of per-channel image statistics.

Provides comprehensive statistical analysis of astronomical images including
mean, median, standard deviation, MAD, SNR estimate, percentiles, and
clipping information.  No image data is modified.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ChannelStatistics:
    """Statistics for a single image channel.

    Attributes
    ----------
    name : str
        Human-readable channel name (e.g. "R", "G", "B", "Mono").
    mean : float
        Arithmetic mean of all pixel values.
    median : float
        Median pixel value.
    std : float
        Standard deviation of pixel values.
    min_val : float
        Minimum pixel value.
    max_val : float
        Maximum pixel value.
    mad : float
        Median absolute deviation from the median.
    snr_estimate : float
        Estimated signal-to-noise ratio: median / MAD-estimated noise.
        Returns 0.0 when noise estimate is effectively zero.
    percentile_01 : float
        1st percentile value (shadows).
    percentile_99 : float
        99th percentile value (highlights).
    pixel_count : int
        Total number of pixels in the channel.
    clipped_low_pct : float
        Percentage of pixels at exactly 0.0 (clipped shadows).
    clipped_high_pct : float
        Percentage of pixels at exactly 1.0 (clipped highlights / saturated).
    """

    name: str
    mean: float
    median: float
    std: float
    min_val: float
    max_val: float
    mad: float
    snr_estimate: float
    percentile_01: float
    percentile_99: float
    pixel_count: int
    clipped_low_pct: float
    clipped_high_pct: float


@dataclass
class ImageStatistics:
    """Aggregate statistics for an entire image.

    Attributes
    ----------
    channels : list[ChannelStatistics]
        Per-channel statistics, one entry per channel.
    total_pixels : int
        Total number of pixels across all channels.
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    n_channels : int
        Number of channels (1 for mono, 3 for RGB, 4 for LRGB).
    is_linear : bool
        Heuristic linearity estimate.  *True* when the median of the
        first (or only) channel is below 0.1, suggesting the image has
        not been stretched.
    """

    channels: list[ChannelStatistics]
    total_pixels: int
    width: int
    height: int
    n_channels: int
    is_linear: bool


def _channel_names(n_channels: int) -> list[str]:
    """Return human-readable channel names based on the channel count.

    Parameters
    ----------
    n_channels : int
        Number of image channels.

    Returns
    -------
    list[str]
        Channel name list.
    """
    if n_channels == 1:
        return ["Mono"]
    if n_channels == 3:
        return ["R", "G", "B"]
    if n_channels == 4:
        return ["L", "R", "G", "B"]
    # Fallback for unusual channel counts
    return [f"Ch{i}" for i in range(n_channels)]


def _compute_single_channel(name: str, data: np.ndarray) -> ChannelStatistics:
    """Compute all statistics for a single 2D channel.

    Parameters
    ----------
    name : str
        Channel name.
    data : np.ndarray
        2D float32 array with values in [0, 1].

    Returns
    -------
    ChannelStatistics
        Computed statistics for the channel.
    """
    flat = data.ravel()
    pixel_count = flat.size
    if pixel_count == 0:
        return ChannelStatistics(
            mean=0.0, median=0.0, std=0.0, min_val=0.0, max_val=0.0,
            mad=0.0, snr=0.0, entropy=0.0, bayer_weighted=1.0,
        )

    mean_val = float(np.mean(flat))
    median_val = float(np.median(flat))
    std_val = float(np.std(flat))
    min_val = float(np.min(flat))
    max_val = float(np.max(flat))

    # Median absolute deviation
    mad_val = float(np.median(np.abs(flat - median_val)))

    # SNR estimate: median / (MAD * 1.4826)
    noise_est = mad_val * 1.4826
    if noise_est > 1e-10:
        snr_estimate = median_val / noise_est
    else:
        snr_estimate = 0.0

    # Percentiles
    percentile_01 = float(np.percentile(flat, 1))
    percentile_99 = float(np.percentile(flat, 99))

    # Clipping percentages
    clipped_low_pct = float(np.sum(flat == 0.0)) / max(pixel_count, 1) * 100.0
    clipped_high_pct = float(np.sum(flat == 1.0)) / max(pixel_count, 1) * 100.0

    return ChannelStatistics(
        name=name,
        mean=mean_val,
        median=median_val,
        std=std_val,
        min_val=min_val,
        max_val=max_val,
        mad=mad_val,
        snr_estimate=snr_estimate,
        percentile_01=percentile_01,
        percentile_99=percentile_99,
        pixel_count=pixel_count,
        clipped_low_pct=clipped_low_pct,
        clipped_high_pct=clipped_high_pct,
    )


def compute_image_statistics(image: np.ndarray) -> ImageStatistics:
    """Compute comprehensive statistics for an image.

    Analyses each channel independently and produces an aggregate
    ``ImageStatistics`` result.  The ``is_linear`` flag is estimated
    heuristically: if the median of the first channel is below 0.1, the
    image is assumed to still be in a linear (unstretched) state.

    Parameters
    ----------
    image : np.ndarray
        Image data, float32 in [0, 1].
        Mono: shape (H, W).  Colour: shape (C, H, W).

    Returns
    -------
    ImageStatistics
        Full per-channel and aggregate statistics.
    """
    if image.ndim == 2:
        h, w = image.shape
        n_channels = 1
        channel_data = [image]
    elif image.ndim == 3:
        n_channels, h, w = image.shape
        channel_data = [image[ch] for ch in range(n_channels)]
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    names = _channel_names(n_channels)

    channel_stats: list[ChannelStatistics] = []
    for idx, data_2d in enumerate(channel_data):
        name = names[idx] if idx < len(names) else f"Ch{idx}"
        stats = _compute_single_channel(name, data_2d)
        channel_stats.append(stats)

    total_pixels = h * w * n_channels

    # Linearity heuristic: median of first channel < 0.1
    is_linear = channel_stats[0].median < 0.1

    log.debug(
        "Image statistics: %dx%d, %d ch, linear=%s, median=%.4f",
        w, h, n_channels, is_linear, channel_stats[0].median,
    )

    return ImageStatistics(
        channels=channel_stats,
        total_pixels=total_pixels,
        width=w,
        height=h,
        n_channels=n_channels,
        is_linear=is_linear,
    )
