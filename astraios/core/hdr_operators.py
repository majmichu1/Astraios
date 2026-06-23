"""HDR Tonemap Operators — Reinhard, Drago, and Core-blend.

All functions accept and return ``float32`` data in ``[0, 1]``.
Mono: ``(H, W)``, Color: ``(C, H, W)``.
"""

from __future__ import annotations

import dataclasses
import logging
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
    """Reinhard global tonemap operator.

    Converts linear HDR data to LDR using the Reinhard photographic
    tone reproduction operator.  Operates per-channel on GPU when
    available.
    """
    if params is None:
        params = ReinhardParams()
    intensity = max(-8.0, min(8.0, params.intensity))

    is_color = data.ndim == 3
    ch_list = list(range(data.shape[0])) if is_color else [None]
    result = np.empty_like(data, dtype=np.float32)

    for ch in ch_list:
        d = data[ch] if is_color else data
        scaled = d * (2.0 ** (-intensity)) if intensity != 0.0 else d
        l_white = float(np.max(scaled))
        l_white = max(l_white, 1e-10)
        tone = scaled * (1.0 + scaled / (l_white * l_white)) / (1.0 + scaled)
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
        l_scaled = d / l_max
        # Drago normalisation = the curve evaluated at the max luminance
        # (l_scaled = 1) → log10(2 + 8) = 1.0. The old code recomputed this
        # per-pixel (it divided l_scaled by l_max a second time), which both
        # mis-scaled the curve and made the next line's max(denom, 1e-10) crash
        # with "truth value of an array is ambiguous".
        denom = max(float(np.log10(10.0)), 1e-10)
        log_base = np.log10(2.0 + 8.0 * l_scaled ** (1.0 / max(bias, 1e-6)))
        tone = np.clip(log_base / denom, 0.0, 1.0)
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

    from scipy.ndimage import gaussian_filter as _gf

    from astraios.core.stretch import StretchParams, auto_stretch

    sigma = max(8, min(data.shape[-2], data.shape[-1]) * params.blur_sigma_factor)
    core_mask = _gf(core_linear, sigma=sigma)
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
