"""Lucky-imaging planetary/lunar/solar stacking from SER video.

Pipeline: quality-rank every frame, keep the best N%, align the kept frames
to a quality-picked reference (translation-only: phase correlation or
disk-centroid), and integrate (average/median/sigma-clip) into a single
stacked image.

Memory is bounded: the analysis pass streams frames one at a time from disk
(:func:`astraios.core.ser_reader.iter_ser_frames`) and only the kept subset
(``keep_percent`` of the total) is ever held for alignment/integration.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro)
``ser_stacker.py`` / ``ser_stack_config.py`` / ``ser_tracking.py``,
Copyright Franklin Marek, GPL-3.0-or-later.
https://github.com/setiastro/setiastrosuitepro

Deferred relative to SASpro (not ported — out of scope for this pass):
  - Multi-point aperture (AP) grid dense-field surface alignment/derotation.
  - Planetary pole/axis field-rotation correction and synthetic derotation.
  - True drizzle (pixfrac footprint splatting with square/circle/gaussian
    kernels) — ``drizzle_scale`` here only does a bicubic canvas upsample.
  - Per-channel atmospheric-dispersion centroid correction.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from astropy.stats import SigmaClip
from skimage.registration import phase_cross_correlation

from astraios.core.device_manager import get_device_manager
from astraios.core.gpu_stars import warp_image_gpu
from astraios.core.ser_reader import SERFrameReader, SERHeader, read_ser_header
from astraios.core.stacking import _apply_shift_cpu, _gpu_fft_shift

log = logging.getLogger(__name__)

__all__ = [
    "SERStackParams",
    "stack_ser",
    "compute_ser_quality_scores",
    "frame_quality_score",
    "read_ser_header",
    "SERHeader",
]

ProgressCB = Callable[[int, int, str], None]

QualityMetric = Literal["laplacian_gradient"]
AlignmentMethod = Literal["phase_correlation", "centroid", "none"]
ReferenceMode = Literal["best_frame", "best_stack"]
IntegrationMethod = Literal["average", "median", "sigma_clip"]
ColorMode = Literal["auto", "mono", "rgb"]

_SIGNAL_FLOOR = 0.05


@dataclass
class SERStackParams:
    """Settings for :func:`stack_ser` (mirrors SASpro's ``SERStackConfig``,
    narrowed to the translation-alignment lucky-imaging pipeline)."""

    #: Percent (0-100] of frames (ranked by quality) to keep for stacking.
    keep_percent: float = 20.0
    #: Per-frame sharpness/quality metric used for lucky-imaging ranking.
    quality_metric: QualityMetric = "laplacian_gradient"
    #: How kept frames are registered: FFT phase correlation, disk centroid, or off.
    alignment_method: AlignmentMethod = "phase_correlation"
    #: Sub-pixel upsampling factor for phase-correlation alignment.
    upsample_factor: int = 4
    #: Single best frame, or the mean of the top ``reference_count`` frames, as
    #: the alignment reference.
    reference_mode: ReferenceMode = "best_frame"
    #: Frames averaged into the reference when reference_mode="best_stack".
    reference_count: int = 5
    #: How the kept, aligned frames are combined into the final stack.
    integration_method: IntegrationMethod = "average"
    #: Sigma-clip lower/upper thresholds and iteration cap (integration_method="sigma_clip").
    sigma_low: float = 3.0
    sigma_high: float = 3.0
    max_iterations: int = 5
    #: Scale each aligned frame's brightness to match the reference before combining.
    normalize_frames: bool = True
    #: Canvas upsample factor applied after alignment ("drizzle-lite"; >1.0 enables it).
    drizzle_scale: float = 1.0
    #: Force mono ("mono"), force color ("rgb"), or leave frames as read ("auto").
    color_mode: ColorMode = "auto"
    #: Override the SER header's declared CFA pattern (RGGB/BGGR/GRBG/GBRG).
    bayer_pattern: str | None = None
    #: Demosaic algorithm passed through to astraios.core.debayer.debayer.
    debayer_method: str = "bilinear"
    #: (x, y, w, h) crop applied to every frame before quality/alignment/stacking.
    roi: tuple[int, int, int, int] | None = None
    #: Optional cap on the number of frames read from the file (quick previews on huge SERs).
    max_frames: int | None = None
    #: Allow GPU-accelerated phase correlation/warp when the kept-frame count is large.
    use_gpu: bool = True
    #: Kept-frame count above which the GPU alignment path is used instead of CPU/skimage.
    gpu_frame_threshold: int = 64

    def __post_init__(self) -> None:
        self.keep_percent = float(np.clip(self.keep_percent, 0.1, 100.0))
        self.upsample_factor = max(1, int(self.upsample_factor))
        self.reference_count = max(1, int(self.reference_count))
        self.max_iterations = max(1, int(self.max_iterations))
        self.drizzle_scale = max(1.0, float(self.drizzle_scale))
        if self.alignment_method not in ("phase_correlation", "centroid", "none"):
            raise ValueError(f"Unknown alignment_method: {self.alignment_method!r}")
        if self.reference_mode not in ("best_frame", "best_stack"):
            raise ValueError(f"Unknown reference_mode: {self.reference_mode!r}")
        if self.integration_method not in ("average", "median", "sigma_clip"):
            raise ValueError(f"Unknown integration_method: {self.integration_method!r}")
        if self.color_mode not in ("auto", "mono", "rgb"):
            raise ValueError(f"Unknown color_mode: {self.color_mode!r}")


# ---------------------------------------------------------------------------
# Small array helpers (channels-first: mono (H, W), color (C, H, W))
# ---------------------------------------------------------------------------


def _to_mono(frame: np.ndarray) -> np.ndarray:
    """Luma-reduce a channels-first frame to mono float32 (H, W)."""
    if frame.ndim == 2:
        return frame.astype(np.float32, copy=False)
    if frame.shape[0] >= 3:
        weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        return (frame[:3] * weights[:, None, None]).sum(axis=0).astype(np.float32)
    return frame.mean(axis=0).astype(np.float32)


def _apply_roi(frame: np.ndarray, roi: tuple[int, int, int, int] | None) -> np.ndarray:
    if roi is None:
        return frame
    x, y, w, h = (int(v) for v in roi)
    height, width = frame.shape[-2], frame.shape[-1]
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    w = max(1, min(width - x, w))
    h = max(1, min(height - y, h))
    if frame.ndim == 2:
        return frame[y : y + h, x : x + w]
    return frame[:, y : y + h, x : x + w]


def _apply_color_mode(frame: np.ndarray, mode: ColorMode) -> np.ndarray:
    if mode == "mono" and frame.ndim == 3:
        return _to_mono(frame)
    if mode == "rgb" and frame.ndim == 2:
        return np.stack([frame, frame, frame], axis=0)
    return frame


def _resize_channels_first(frame: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    chans = [
        cv2.resize(frame[c], (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        for c in range(frame.shape[0])
    ]
    return np.stack(chans, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Quality metric (ported from SASpro's _quality_score)
# ---------------------------------------------------------------------------


def frame_quality_score(gray: np.ndarray) -> float:
    """Lucky-imaging sharpness/feature-richness score for a mono frame.

    ``0.70 * log1p(1e4 * Laplacian-variance) + 0.30 * gradient-richness``,
    both computed only over pixels above ``_SIGNAL_FLOOR`` so black
    background/empty-sky pixels don't dilute the score. Higher is better
    (sharper, more fine structure). Not normalized across a frame set —
    only comparable within one clip.
    """
    m = np.asarray(gray, dtype=np.float32)
    mask = m > _SIGNAL_FLOOR
    n_sig = int(np.count_nonzero(mask))

    if n_sig < 16:
        mean_gx = float(np.abs(m[:, 1:] - m[:, :-1]).mean()) if m.shape[1] > 1 else 0.0
        mean_gy = float(np.abs(m[1:, :] - m[:-1, :]).mean()) if m.shape[0] > 1 else 0.0
        return float(mean_gx + mean_gy)

    lap = cv2.Laplacian(m, cv2.CV_32F, ksize=5)
    sharpness = float(np.var(lap[mask]))

    sobel_x = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)
    gmag_sig = cv2.magnitude(sobel_x, sobel_y)[mask]

    gmax = float(gmag_sig.max()) if gmag_sig.size else 0.0
    richness = float(gmag_sig.mean()) / gmax if gmax > 1e-9 else 0.0

    sharpness_log = float(np.log1p(sharpness * 1e4))
    return float(0.70 * sharpness_log + 0.30 * richness)


def compute_ser_quality_scores(
    ser_path: str,
    params: SERStackParams | None = None,
    *,
    progress: ProgressCB | None = None,
) -> np.ndarray:
    """Stream every frame once and return its quality score (float32 array).

    Bounded memory: only one frame is held at a time.
    """
    params = params or SERStackParams()
    with SERFrameReader(ser_path) as reader:
        n = len(reader)
        n_eff = n if params.max_frames is None else min(n, int(params.max_frames))
        scores = np.empty(n_eff, dtype=np.float32)
        for i in range(n_eff):
            frame = reader.read_frame(
                i,
                debayer=True,
                bayer_pattern=params.bayer_pattern,
                debayer_method=params.debayer_method,
            )
            frame = _apply_roi(frame, params.roi)
            scores[i] = frame_quality_score(_to_mono(frame))
            if progress is not None:
                progress(i + 1, n_eff, "Analyzing")
    return scores


# ---------------------------------------------------------------------------
# Alignment (translation-only: FFT phase correlation or disk centroid)
# ---------------------------------------------------------------------------


def _disk_centroid(gray: np.ndarray, thresh_frac: float = 0.5) -> tuple[float, float]:
    """Simplified planetary-disk centroid (row, col). No smoothing/multi-scale
    refinement — see module docstring for what SASpro's fuller tracker does."""
    g = np.nan_to_num(gray.astype(np.float32, copy=False))
    peak = float(g.max())
    if peak <= 0.0:
        cy, cx = g.shape[0] * 0.5, g.shape[1] * 0.5
        return float(cy), float(cx)
    mask = g >= (peak * thresh_frac)
    ys, xs = np.nonzero(mask)
    if xs.size < 8:
        idx = int(np.argmax(g))
        cy, cx = np.unravel_index(idx, g.shape)
        return float(cy), float(cx)
    return float(ys.mean()), float(xs.mean())


def _centroid_shift(ref_gray: np.ndarray, cur_gray: np.ndarray) -> tuple[float, float]:
    """Return (row_shift, col_shift) to move ``cur``'s disk centroid onto ``ref``'s."""
    ref_cy, ref_cx = _disk_centroid(ref_gray)
    cur_cy, cur_cx = _disk_centroid(cur_gray)
    return ref_cy - cur_cy, ref_cx - cur_cx


def _estimate_shift(
    ref_gray: np.ndarray,
    cur_gray: np.ndarray,
    *,
    method: AlignmentMethod,
    upsample_factor: int,
    use_gpu: bool,
    ref_tensor_gpu=None,
) -> tuple[float, float]:
    """Return (row_shift, col_shift) needed to align ``cur`` onto ``ref``."""
    if method == "none":
        return 0.0, 0.0
    if method == "centroid":
        return _centroid_shift(ref_gray, cur_gray)

    # phase_correlation
    if use_gpu:
        dm = get_device_manager()
        cur_tensor = dm.from_numpy(cur_gray.astype(np.float32))
        row_shift, col_shift = _gpu_fft_shift(ref_tensor_gpu, cur_tensor, upsample_factor)
        del cur_tensor
        return row_shift, col_shift

    shift, _error, _phase_diff = phase_cross_correlation(
        ref_gray, cur_gray, upsample_factor=upsample_factor
    )
    return float(shift[0]), float(shift[1])


def _apply_translation(
    frame: np.ndarray,
    row_shift: float,
    col_shift: float,
    *,
    use_gpu: bool,
) -> np.ndarray:
    if abs(row_shift) < 1e-6 and abs(col_shift) < 1e-6:
        return frame
    if use_gpu:
        dm = get_device_manager()
        tensor = dm.from_numpy(frame.astype(np.float32))
        matrix = np.array([[1.0, 0.0, -col_shift], [0.0, 1.0, -row_shift]], dtype=np.float32)
        warped = warp_image_gpu(tensor, matrix, mode="bicubic")
        out = dm.to_cpu(warped).numpy().astype(np.float32)
        if out.ndim == 3 and out.shape[0] == 1 and frame.ndim == 2:
            out = out[0]
        del tensor, warped
        return out
    return _apply_shift_cpu(frame, row_shift, col_shift).astype(np.float32)


def _normalize_to_reference(frame: np.ndarray, ref_mean: float) -> np.ndarray:
    """Scale ``frame``'s masked-mean brightness to match the reference's."""
    if ref_mean <= 1e-6:
        return frame
    gray = _to_mono(frame)
    mask = gray > _SIGNAL_FLOOR
    if int(np.count_nonzero(mask)) < 16:
        return frame
    cur_mean = float(gray[mask].mean())
    if cur_mean <= 1e-6:
        return frame
    scale = float(np.clip(ref_mean / cur_mean, 0.5, 2.0))
    if abs(scale - 1.0) < 1e-3:
        return frame
    return (frame * scale).astype(np.float32)


def _sigma_clip_combine(stack: np.ndarray, params: SERStackParams) -> np.ndarray:
    """Reject outlier pixels across the frame axis (astropy SigmaClip) and average."""
    sc = SigmaClip(
        sigma_lower=params.sigma_low, sigma_upper=params.sigma_high, maxiters=params.max_iterations
    )
    masked = sc(stack, axis=0, masked=True)
    combined = np.ma.mean(masked, axis=0)
    result = np.ma.filled(combined.astype(np.float32), np.nan)
    nan_mask = np.isnan(result)
    if nan_mask.any():
        # Pixels rejected in every frame (e.g. a hot column) fall back to the plain mean.
        fallback = np.mean(stack, axis=0)
        result = np.where(nan_mask, fallback, result)
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _read_and_prepare(reader: SERFrameReader, index: int, params: SERStackParams) -> np.ndarray:
    frame = reader.read_frame(
        index,
        debayer=True,
        bayer_pattern=params.bayer_pattern,
        debayer_method=params.debayer_method,
    )
    frame = _apply_roi(frame, params.roi)
    frame = _apply_color_mode(frame, params.color_mode)
    return frame.astype(np.float32, copy=False)


def stack_ser(
    ser_path: str,
    params: SERStackParams | None = None,
    *,
    progress: ProgressCB | None = None,
) -> np.ndarray:
    """Lucky-imaging stack of a SER planetary/lunar/solar capture.

    Ranks every frame by :func:`frame_quality_score`, keeps the best
    ``params.keep_percent`` percent, aligns them (translation-only) to a
    quality-picked reference, and integrates them into one float32 [0, 1]
    image (mono ``(H, W)`` or color ``(3, H, W)``).

    Streams frames from disk; only the kept subset is ever held in memory
    at once (never the full clip).
    """
    params = params or SERStackParams()
    header = read_ser_header(ser_path)
    if header.frame_count <= 0:
        raise ValueError(f"SER file has no frames: {ser_path}")

    n_eff = header.frame_count if params.max_frames is None else min(
        header.frame_count, int(params.max_frames)
    )

    # ---- Pass 1: quality ranking (bounded memory: one frame at a time) ----
    quality = compute_ser_quality_scores(ser_path, params, progress=progress)

    order = np.argsort(-quality, kind="stable")
    k = max(1, int(round(n_eff * params.keep_percent / 100.0)))
    keep_by_quality = order[:k]
    # Ascending frame-index order for sequential disk access in pass 2.
    keep_idx = np.sort(keep_by_quality)

    dm = get_device_manager()
    use_gpu = bool(params.use_gpu and dm.is_gpu and len(keep_idx) >= params.gpu_frame_threshold
                   and params.alignment_method == "phase_correlation")

    with SERFrameReader(ser_path) as reader:
        # ---- Build alignment reference from the best-quality kept frames ----
        ranked_keep = keep_by_quality  # already sorted by descending quality
        ref_frame = _read_and_prepare(reader, int(ranked_keep[0]), params)
        if params.reference_mode == "best_stack" and params.reference_count > 1:
            acc = ref_frame.copy()
            count = 1
            for idx in ranked_keep[1 : params.reference_count]:
                acc += _read_and_prepare(reader, int(idx), params)
                count += 1
            ref_frame = np.clip(acc / count, 0.0, 1.0).astype(np.float32)

        ref_gray = _to_mono(ref_frame)
        ref_signal = ref_gray > _SIGNAL_FLOOR
        ref_mean = (
            float(ref_gray[ref_signal].mean())
            if int(np.count_nonzero(ref_signal)) >= 16
            else float(ref_gray.mean())
        )

        ref_tensor_gpu = dm.from_numpy(ref_gray.astype(np.float32)) if use_gpu else None

        canvas_h, canvas_w = ref_frame.shape[-2], ref_frame.shape[-1]
        drizzle_on = params.drizzle_scale > 1.0001
        out_h = int(round(canvas_h * params.drizzle_scale)) if drizzle_on else canvas_h
        out_w = int(round(canvas_w * params.drizzle_scale)) if drizzle_on else canvas_w

        needs_all_frames = params.integration_method in ("median", "sigma_clip")
        collected: list[np.ndarray] = [] if needs_all_frames else []
        acc_sum: np.ndarray | None = None
        n_stacked = 0
        total = int(len(keep_idx))

        for idx in keep_idx:
            frame = _read_and_prepare(reader, int(idx), params)
            gray = _to_mono(frame)

            row_shift, col_shift = _estimate_shift(
                ref_gray,
                gray,
                method=params.alignment_method,
                upsample_factor=params.upsample_factor,
                use_gpu=use_gpu,
                ref_tensor_gpu=ref_tensor_gpu,
            )
            aligned = _apply_translation(frame, row_shift, col_shift, use_gpu=use_gpu)

            if params.normalize_frames:
                aligned = _normalize_to_reference(aligned, ref_mean)

            if drizzle_on:
                aligned = _resize_channels_first(aligned, out_h, out_w)

            if needs_all_frames:
                collected.append(aligned)
            else:
                acc_sum = aligned.copy() if acc_sum is None else acc_sum + aligned

            n_stacked += 1
            if progress is not None:
                progress(n_stacked, total, "Stacking")

        if ref_tensor_gpu is not None:
            del ref_tensor_gpu
            dm.empty_cache()

    if needs_all_frames:
        stack_arr = np.stack(collected, axis=0)
        if params.integration_method == "median":
            result = np.median(stack_arr, axis=0)
        else:
            result = _sigma_clip_combine(stack_arr, params)
    else:
        assert acc_sum is not None
        result = acc_sum / float(n_stacked)

    result = np.clip(result, 0.0, 1.0).astype(np.float32)
    return result
