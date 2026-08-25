"""HDR Tonemap Operators — Reinhard, Drago, and Core-blend.

All functions accept and return ``float32`` data in ``[0, 1]``.
Mono: ``(H, W)``, Color: ``(C, H, W)``.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from enum import Enum

import numpy as np

log = logging.getLogger(__name__)


class HDROperator(Enum):
    """Selectable HDR tonemap operators."""

    REINHARD = "reinhard"
    DRAGO = "drago"
    CORE_BLEND = "core_blend"


@dataclasses.dataclass
class ReinhardParams:
    """Reinhard global tonemap parameters."""

    intensity: float = 0.0   # -8 .. 8, 0 = auto
    light_adapt: float = 0.0  # 0 .. 1, 0 = global
    color_adapt: float = 0.0  # 0 .. 1


@dataclasses.dataclass
class DragoParams:
    """Drago logarithmic tonemap parameters."""

    gamma: float = 1.0
    saturation: float = 1.0
    bias: float = 0.85  # 0 .. 1, lower = more compression


@dataclasses.dataclass
class CoreBlendParams:
    """Core-blend: gentle stretch blended with normal stretch via a gaussian mask."""

    core_threshold: float = 0.3   # fraction of p99 to define "core" region
    gentle_midtone: float = 0.25
    gentle_shadow: float = -2.0
    blur_sigma_factor: float = 0.015  # fraction of min(H, W)


def tonemap_reinhard(
    data: np.ndarray,
    params: ReinhardParams | None = None,
) -> np.ndarray:
    """Reinhard photographic tone reproduction operator.

    Converts linear HDR data to LDR. Runs in numpy: the work is a handful of
    whole-array expressions, not a per-pixel loop, so there is nothing here
    for a GPU transfer to win back.

    ``light_adapt`` and ``color_adapt`` used to be declared and then ignored,
    which made the dataclass advertise two knobs that did nothing. They now
    behave as Reinhard defines them, and the descriptions below are measured
    rather than assumed:

    * ``light_adapt`` 0 = one global adaptation level for the whole frame,
      which is what blows out a bright core. Raising it lets each pixel adapt
      to its own level, pulling clipped highlights back down while leaving
      the background where it was. On a test field with a core driven to 3.0,
      that core is 100% clipped at 0.0 and 0% clipped by 0.5, while the
      background mean moves 0.264 -> 0.265. This is the knob for a saturated
      nebula core.
    * ``color_adapt`` 0 = every channel adapts to the shared luminance, which
      preserves colour; 1 = each channel adapts to itself, which desaturates
      hard (measured channel spread 0.198 -> 0.011) but tames a single
      channel that is clipping on its own.

    Both default to 0, and at 0 the adaptation term is exactly 1.0, so the
    result is bit-for-bit what this function produced before they were
    implemented. Existing projects re-run to the same pixels.
    """
    if params is None:
        params = ReinhardParams()
    intensity = max(-8.0, min(8.0, params.intensity))
    light_adapt = float(np.clip(params.light_adapt, 0.0, 1.0))
    color_adapt = float(np.clip(params.color_adapt, 0.0, 1.0))

    is_color = data.ndim == 3
    scale = 2.0 ** (-intensity) if intensity != 0.0 else 1.0
    scaled_all = data * scale if scale != 1.0 else data

    # Shared luminance drives adaptation when color_adapt is 0.
    luminance = scaled_all.mean(axis=0) if is_color else scaled_all
    lum_mean = max(float(luminance.mean()), 1e-10)

    ch_list = list(range(data.shape[0])) if is_color else [None]
    result = np.empty_like(data, dtype=np.float32)

    for ch in ch_list:
        scaled = scaled_all[ch] if is_color else scaled_all
        l_white = max(float(np.max(scaled)), 1e-10)

        # Adaptation, normalised so that light_adapt=0 gives exactly 1.0 and
        # the classic global curve below is untouched.
        local = color_adapt * scaled + (1.0 - color_adapt) * luminance
        global_ = color_adapt * max(float(scaled.mean()), 1e-10) + (1.0 - color_adapt) * lum_mean
        adapt = (light_adapt * local + (1.0 - light_adapt) * global_) / max(global_, 1e-10)

        tone = scaled * (1.0 + scaled / (l_white * l_white)) / (scaled + adapt)
        tone = np.clip(tone, 0.0, 1.0)
        if is_color:
            result[ch] = tone.astype(np.float32)
        else:
            result = tone.astype(np.float32)

    return result


def tonemap_drago(
    data: np.ndarray,
    params: DragoParams | None = None,
) -> np.ndarray:
    """Drago logarithmic tonemap operator.

    Better at compressing extreme dynamic range (like M42 core).
    Bias < 1.0 compresses highlights more aggressively.
    """
    if params is None:
        params = DragoParams()
    bias = np.clip(params.bias, 0.0, 1.0)
    gamma = max(0.1, params.gamma)

    is_color = data.ndim == 3
    ch_list = list(range(data.shape[0])) if is_color else [None]
    result = np.empty_like(data, dtype=np.float32)

    for ch in ch_list:
        d = data[ch] if is_color else data
        l_max = float(np.max(d))
        l_max = max(l_max, 1e-10)
        l_scaled = np.maximum(d, 0.0) / l_max
        # Drago et al. 2003: Ld ∝ [log10(1+L) / log10(1+Lmax)] · ld(L/Lmax),
        # where ld(w) = log10(2 + 8·w^b) is the bias curve with ld(1) = 1,
        # so the bias term needs no extra denominator. Without the log-ratio
        # factor the curve started at log10(2) ≈ 0.30 — blacks were lifted by
        # 30% and the whole image compressed into [0.30, 1].
        log_term = np.log10(1.0 + np.maximum(d, 0.0)) / max(
            float(np.log10(1.0 + l_max)), 1e-10
        )
        bias_term = np.log10(2.0 + 8.0 * l_scaled ** (1.0 / max(bias, 1e-6)))
        tone = np.clip(log_term * bias_term, 0.0, 1.0)
        if gamma != 1.0:
            tone = tone ** gamma
        if is_color:
            result[ch] = tone.astype(np.float32)
        else:
            result = tone.astype(np.float32)

    return result


def tonemap_core_blend(
    data: np.ndarray,
    params: CoreBlendParams | None = None,
) -> np.ndarray:
    """Core-blend: isolate the bright core, stretch it gently, and blend.

    This preserves Trapezium / bright-core detail while still bringing
    out faint outer nebulosity.
    """
    if params is None:
        params = CoreBlendParams()

    p99 = float(np.percentile(data, 99))
    core_linear = (data > p99 * params.core_threshold).astype(np.float32)
    if not np.any(core_linear):
        from astraios.core.stretch import StretchParams, auto_stretch
        return auto_stretch(data, StretchParams())

    from astraios.core.filters import gaussian_blur as _gf
    from astraios.core.stretch import StretchParams, auto_stretch

    sigma = max(8, min(data.shape[-2], data.shape[-1]) * params.blur_sigma_factor)
    # Blur a 2-D mask: blurring the (C, H, W) volume would smear across the
    # channel axis, and the later [np.newaxis, ...] broadcast would then
    # inflate a color image to (C, C, H, W).
    core_2d = core_linear.max(axis=0) if data.ndim == 3 else core_linear
    # gaussian_blur runs this on the GPU when the work is worth the transfer.
    # sigma is 1.5% of the short side, so the kernel grows with the image: 123
    # taps at 1024px, 721 taps on a 6000px frame. That convolution was 55ms of
    # this tool's 226ms at 1024px and would dominate outright at full size;
    # it is about 3.6ms now.
    core_mask = _gf(core_2d, sigma=sigma)
    core_mask = np.clip(core_mask, 0, 1)

    gentle = auto_stretch(
        data,
        StretchParams(midtone=params.gentle_midtone, shadow_clip=params.gentle_shadow),
    )
    normal = auto_stretch(data, StretchParams())

    cm = core_mask[np.newaxis, ...] if data.ndim == 3 else core_mask

    working = normal * (1.0 - cm) + gentle * cm
    return np.clip(working, 0, 1).astype(np.float32)


# Dispatch table
_HDR_DISPATCH: dict[HDROperator, Callable] = {
    HDROperator.REINHARD: tonemap_reinhard,
    HDROperator.DRAGO: tonemap_drago,
    HDROperator.CORE_BLEND: tonemap_core_blend,
}


def apply_hdr(
    data: np.ndarray,
    operator: HDROperator | str,
    params: ReinhardParams | DragoParams | CoreBlendParams | None = None,
) -> np.ndarray:
    """Apply the selected HDR tonemap operator.

    Parameters
    ----------
    data : ndarray
        Linear float32 image in ``[0, 1]``, ``(H, W)`` or ``(C, H, W)``.
    operator : HDROperator | str
        Which operator to apply.
    params : optional
        Operator-specific parameters.  ``None`` = defaults.

    Returns
    -------
    ndarray
        Tonemapped float32 image in ``[0, 1]``.
    """
    if isinstance(operator, str):
        operator = HDROperator(operator)
    func = _HDR_DISPATCH.get(operator)
    if func is None:
        log.warning("Unknown HDR operator %s, falling back to core_blend", operator)
        func = tonemap_core_blend
    return func(data, params)
