"""Multi-Session / Multi-Setup Image Stacking.

Handles frames from different telescopes, cameras, nights, or filter sets.
Pipeline:
  1. Stack each session independently (alignment + rejection + integration).
  2. Align sub-stacks to reference (star-based or FFT).
  3. Normalize background and signal scale across sessions.
  4. Compute per-session weights (SNR or integration time).
  5. Weighted integration of all sub-stacks.

This mirrors the APP "multi-session integration" workflow and is more
approachable than PixInsight's manual process.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from cosmica.core.image_io import FrameType, ImageData
from cosmica.core.stacking import (
    IntegrationMethod,
    RejectionMethod,
    StackingParams,
    StackResult,
    _reject_sigma_clip,
    align_frames,
    stack_images,
)

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop(f: float, m: str) -> None:
    pass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SessionGroup:
    """A batch of light frames from one observing session / setup."""

    frames: list[ImageData]
    name: str = "Session"
    integration_time: float | None = None
    """Total exposure time in seconds — used for SNR weight if provided."""
    params: StackingParams | None = None
    """Per-session override. Falls back to MultiSessionParams.per_session_params."""


@dataclass
class MultiSessionParams:
    """Configuration for multi-session integration."""

    per_session_params: StackingParams = field(default_factory=StackingParams)
    """Params applied when stacking each individual session."""

    weight_mode: str = "snr"
    """How to weight sessions for final integration.
    'snr'   — weight by estimated SNR of each sub-stack (recommended).
    'time'  — weight by integration_time (requires EXPTIME headers or manual entry).
    'equal' — no weighting; simple sigma-clip integration.
    """

    final_rejection: RejectionMethod = RejectionMethod.SIGMA_CLIP
    final_integration: IntegrationMethod = IntegrationMethod.AVERAGE
    final_kappa: float = 3.0

    align_sub_stacks: bool = True
    """Register sub-stacks to each other before final integration."""

    normalize_background: bool = True
    """Match background level (additive) and gradient (multiplicative) across sessions."""


@dataclass
class MultiSessionResult:
    image: ImageData
    sub_stacks: list[StackResult]
    weights: list[float]
    session_names: list[str]
    n_sessions: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _estimate_sky_noise(data: np.ndarray) -> tuple[float, float]:
    """Return (sky_median, sky_noise_mad) of an image, ignoring bright objects.

    Uses sigma-clipped statistics on the darkest 75% of pixels to avoid
    star/galaxy contamination.
    """
    if data.ndim == 3:
        # Use luminance channel (mean across colour channels)
        lum = data.mean(axis=0)
    else:
        lum = data

    flat = lum.ravel()
    p75 = np.percentile(flat, 75)
    sky_pixels = flat[flat < p75]
    if sky_pixels.size < 100:
        sky_pixels = flat

    median = float(np.median(sky_pixels))
    mad = float(np.median(np.abs(sky_pixels - median))) * 1.4826
    return median, max(mad, 1e-6)


def _snr_weight(stack_result: StackResult) -> float:
    """Estimate integration SNR weight from a sub-stack.

    SNR ∝ signal / noise, where signal is estimated from the upper quartile
    and noise from sky MAD.  We use SNR² as the weight so that a session
    with 2× the SNR contributes 4× the weight (matches Poisson statistics).
    """
    data = stack_result.image.data
    sky_med, noise = _estimate_sky_noise(data)
    if data.ndim == 3:
        signal = float(np.percentile(data, 90)) - sky_med
    else:
        signal = float(np.percentile(data, 90)) - sky_med
    snr = max(signal / noise, 0.0)
    return snr * snr


def _time_weight(session: SessionGroup, stack_result: StackResult) -> float:
    """Weight based on total integration time (falls back to frame count)."""
    if session.integration_time is not None and session.integration_time > 0:
        return float(session.integration_time)
    # Fallback: use number of frames as proxy for integration time
    return float(len(session.frames))


def _normalize_background(stack: np.ndarray, ref_median: float, ref_mad: float) -> np.ndarray:
    """Scale and shift a sub-stack to match a reference background level."""
    sky_med, noise = _estimate_sky_noise(stack)
    if noise < 1e-8:
        return stack
    # Multiplicative scale to match noise level, then additive shift for background
    scale = ref_mad / noise
    shift = ref_median - sky_med * scale
    result = stack * scale + shift
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _pad_to_shape(data: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Centre-pad image to target dimensions with zeros."""
    if data.ndim == 2:
        h, w = data.shape
        pad_top = (target_h - h) // 2
        pad_bot = target_h - h - pad_top
        pad_lft = (target_w - w) // 2
        pad_rgt = target_w - w - pad_lft
        return np.pad(data, ((pad_top, pad_bot), (pad_lft, pad_rgt)), mode="constant")
    else:
        c, h, w = data.shape
        pad_top = (target_h - h) // 2
        pad_bot = target_h - h - pad_top
        pad_lft = (target_w - w) // 2
        pad_rgt = target_w - w - pad_lft
        return np.pad(data, ((0, 0), (pad_top, pad_bot), (pad_lft, pad_rgt)), mode="constant")


def _align_sub_stacks(
    sub_images: list[ImageData],
    params: StackingParams,
    progress: ProgressCallback,
) -> list[ImageData]:
    """Align sub-stack images to each other using the existing star-based alignment."""
    if len(sub_images) < 2:
        return sub_images
    try:
        return align_frames(sub_images, params, progress=progress)
    except Exception as exc:
        log.warning("Sub-stack alignment failed (%s); proceeding without alignment", exc)
        return sub_images


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def stack_multi_session(
    sessions: list[SessionGroup],
    params: MultiSessionParams | None = None,
    progress: ProgressCallback = _noop,
) -> MultiSessionResult:
    """Stack frames from multiple sessions / setups into one combined image.

    Parameters
    ----------
    sessions :
        List of :class:`SessionGroup`.  Each group is stacked independently
        before cross-session integration.
    params :
        Configuration.  Uses sensible defaults if *None*.
    progress :
        ``(fraction: float, message: str)`` callback for UI progress updates.

    Returns
    -------
    MultiSessionResult
        Combined image and per-session diagnostics.
    """
    if params is None:
        params = MultiSessionParams()

    n = len(sessions)
    if n == 0:
        raise ValueError("No sessions provided")
    if n == 1 and len(sessions[0].frames) > 0:
        # Degenerate case: single session — use standard stacking
        sub_params = sessions[0].params or params.per_session_params
        result = stack_images(sessions[0].frames, sub_params, progress=progress)
        return MultiSessionResult(
            image=result.image,
            sub_stacks=[result],
            weights=[1.0],
            session_names=[sessions[0].name],
            n_sessions=1,
        )

    # ── Phase 1: Stack each session (0 → 0.55 of total progress) ────────────
    sub_results: list[StackResult] = []
    per_session_frac = 0.55 / n

    for i, session in enumerate(sessions):
        if not session.frames:
            log.warning("Session '%s' has no frames — skipping", session.name)
            continue
        sub_params = session.params or params.per_session_params
        base = i * per_session_frac

        def _sub_progress(f: float, m: str, _base: float = base) -> None:
            progress(_base + f * per_session_frac, f"[{session.name}] {m}")

        progress(base, f"Stacking session {i + 1}/{n}: {session.name}…")
        try:
            result = stack_images(session.frames, sub_params, progress=_sub_progress)
            sub_results.append(result)
        except Exception as exc:
            log.error("Failed to stack session '%s': %s", session.name, exc)
            raise

    if not sub_results:
        raise ValueError("All sessions failed to stack")

    valid_sessions = [s for s in sessions if s.frames]

    # ── Phase 2: Collect sub-stack arrays, unify sizes ──────────────────────
    progress(0.55, "Matching frame sizes across sessions…")
    sub_data = [r.image.data for r in sub_results]

    # Determine shapes
    if sub_data[0].ndim == 3:
        heights = [d.shape[1] for d in sub_data]
        widths  = [d.shape[2] for d in sub_data]
    else:
        heights = [d.shape[0] for d in sub_data]
        widths  = [d.shape[1] for d in sub_data]

    max_h, max_w = max(heights), max(widths)

    # Pad smaller sub-stacks to common bounding box
    padded = []
    for d in sub_data:
        if d.ndim == 3:
            h, w = d.shape[1], d.shape[2]
        else:
            h, w = d.shape[0], d.shape[1]
        if h != max_h or w != max_w:
            log.info("Padding sub-stack from %dx%d to %dx%d", w, h, max_w, max_h)
            d = _pad_to_shape(d, max_h, max_w)
        padded.append(d)

    # ── Phase 3: Align sub-stacks to reference ───────────────────────────────
    if params.align_sub_stacks and len(padded) > 1:
        progress(0.60, "Aligning sub-stacks across sessions…")
        sub_images_for_align = [
            ImageData(data=d, header=sub_results[i].image.header.copy(), frame_type=FrameType.RESULT)
            for i, d in enumerate(padded)
        ]
        try:
            aligned_images = _align_sub_stacks(
                sub_images_for_align,
                params.per_session_params,
                lambda f, m: progress(0.60 + f * 0.10, m),
            )
            padded = [img.data for img in aligned_images]
        except Exception as exc:
            log.warning("Sub-stack alignment failed: %s", exc)

    # ── Phase 4: Background normalization ────────────────────────────────────
    if params.normalize_background and len(padded) > 1:
        progress(0.70, "Normalising background across sessions…")
        ref_med, ref_mad = _estimate_sky_noise(padded[0])
        normalised = [padded[0]]
        for d in padded[1:]:
            normalised.append(_normalize_background(d, ref_med, ref_mad))
        padded = normalised

    # ── Phase 5: Compute weights ─────────────────────────────────────────────
    progress(0.75, "Computing session weights…")
    if params.weight_mode == "snr":
        raw_weights = [_snr_weight(r) for r in sub_results]
    elif params.weight_mode == "time":
        raw_weights = [_time_weight(s, r) for s, r in zip(valid_sessions, sub_results)]
    else:  # equal
        raw_weights = [1.0] * len(sub_results)

    total = sum(raw_weights)
    weights = [w / total for w in raw_weights] if total > 0 else [1.0 / len(sub_results)] * len(sub_results)

    for i, (sess, w) in enumerate(zip(valid_sessions, weights)):
        log.info("Session '%s': weight=%.3f", sess.name, w)

    # ── Phase 6: Weighted integration ────────────────────────────────────────
    progress(0.80, "Integrating sessions…")

    # Build stack array (N, C, H, W) or (N, H, W)
    stack_array = np.array(padded, dtype=np.float32)
    weight_array = np.array(weights, dtype=np.float32)

    if params.final_rejection != RejectionMethod.NONE and len(padded) >= 3:
        masked_data = _reject_sigma_clip(stack_array, params.final_kappa, params.final_kappa, 5)
        # Build rejection-aware weighted average
        alive_mask = ~masked_data.mask  # shape (N, ...)
        w_broadcast = weight_array.reshape((-1,) + (1,) * (stack_array.ndim - 1))
        weighted_sum = np.sum(masked_data.data * alive_mask * w_broadcast, axis=0)
        weight_sum = np.sum(alive_mask * w_broadcast, axis=0)
        weight_sum = np.where(weight_sum == 0, 1.0, weight_sum)
        result_data = (weighted_sum / weight_sum).astype(np.float32)
    else:
        # Simple weighted average
        w_broadcast = weight_array.reshape((-1,) + (1,) * (stack_array.ndim - 1))
        result_data = np.sum(stack_array * w_broadcast, axis=0).astype(np.float32)

    result_data = np.clip(result_data, 0.0, 1.0).astype(np.float32)

    ref_header = sub_results[0].image.header.copy()
    ref_header["COSMICA_SESSIONS"] = n
    ref_header["COSMICA_TOTAL_FRAMES"] = sum(r.n_frames for r in sub_results)

    result_image = ImageData(
        data=result_data,
        header=ref_header,
        frame_type=FrameType.RESULT,
    )

    progress(1.0, f"Multi-session integration complete: {n} sessions, "
                  f"{sum(r.n_frames for r in sub_results)} total frames")

    return MultiSessionResult(
        image=result_image,
        sub_stacks=sub_results,
        weights=weights,
        session_names=[s.name for s in valid_sessions],
        n_sessions=len(sub_results),
    )


# ---------------------------------------------------------------------------
# Auto-grouping from FITS headers
# ---------------------------------------------------------------------------


def auto_group_sessions(images: list[ImageData]) -> list[SessionGroup]:
    """Group frames into sessions based on FITS header metadata.

    Groups by (INSTRUME, FILTER, DATE-OBS date part, XPIXSZ/FOCALLEN).
    Returns a list of :class:`SessionGroup` with sensible names derived
    from the distinguishing header fields.
    """
    from collections import defaultdict

    groups: dict[tuple, list[ImageData]] = defaultdict(list)

    for img in images:
        h = img.header or {}
        instrument = str(h.get("INSTRUME", h.get("CAMERA", "Unknown"))).strip()
        filt = str(h.get("FILTER", h.get("FILT", ""))).strip()
        # Date: use DATE-OBS truncated to day
        dateobs = str(h.get("DATE-OBS", "")).split("T")[0]
        # Pixel scale proxy: round to 1 decimal to allow slight WCS variation
        xpix = round(float(h.get("XPIXSZ", h.get("PIXELSIZE", 0.0))), 1)
        focal = round(float(h.get("FOCALLEN", h.get("FOCALLEN", 0.0))), -1)  # round to 10mm

        key = (instrument, filt, dateobs, xpix, focal)
        groups[key].append(img)

    result = []
    for key, frames in groups.items():
        instrument, filt, dateobs, _, _ = key
        parts = [instrument]
        if filt:
            parts.append(filt)
        if dateobs:
            parts.append(dateobs)
        name = " / ".join(p for p in parts if p and p != "Unknown") or "Session"

        # Sum integration time from EXPTIME header
        total_time = 0.0
        for img in frames:
            h = img.header or {}
            exp = h.get("EXPTIME", h.get("EXPOSURE", None))
            if exp is not None:
                try:
                    total_time += float(exp)
                except (ValueError, TypeError):
                    pass

        result.append(SessionGroup(
            frames=frames,
            name=name,
            integration_time=total_time if total_time > 0 else None,
        ))

    log.info("auto_group_sessions: %d frames → %d sessions", len(images), len(result))
    return result
