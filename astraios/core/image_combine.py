"""Image Combine — pixel arithmetic between two images.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro), Copyright
Franklin Marek, GPL-3.0-or-later. https://github.com/setiastro/setiastrosuitepro

SASpro's ``pro/image_combine.py`` implements an "Image Combine" dialog whose
blend dispatcher (``_blend_dispatch``) supports nine pixel-arithmetic modes
(Average, Add, Subtract, Blend, Multiply, Divide, Screen, Overlay, Difference)
mixed against source A through an opacity slider. This module ports that
arithmetic core (minus the Qt dialog, view manager, and mask-overlay preview,
which are UI concerns) and adapts the single "opacity" knob into a pair of
explicit ``weight_a`` / ``weight_b`` weights so callers can build weighted
sums/averages (e.g. combining unequally-exposed frames) as well as a
straight opacity cross-fade (set ``weight_a = 1 - alpha, weight_b = alpha``
for BLEND). MIN and MAX modes are not present in SASpro's dialog but are
added here since they are a standard, trivially-cheap part of any
professional pixel-arithmetic toolkit (PixInsight PixelMath, Siril
``pm``) and were explicitly requested for parity.

GPU/CPU decision: these are simple elementwise array ops on at most two
full-resolution frames. A `numpy` implementation is already effectively
bandwidth-bound and avoids a host<->device round trip for what is normally a
one-shot, interactive operation; there is no iterative/tiled workload here
to justify GPU dispatch through :mod:`astraios.core.device_manager`. Kept as
plain NumPy per CLAUDE.md's guidance that light-weight ops don't need torch.

Conventions: float32 images in ``[0, 1]``. Mono is ``(H, W)``, color is
``(C, H, W)`` (channels-first), matching :mod:`astraios.core.image_io`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


class CombineOperation(str, Enum):
    """Pixel-arithmetic combine operation.

    All ops except AVERAGE operate on the *weighted* inputs
    ``wa * A`` and ``wb * B`` (see :class:`ImageCombineParams`).
    """

    AVERAGE = "average"  # weighted mean: (wa*A + wb*B) / (wa + wb)
    ADD = "add"  # wa*A + wb*B  (linear dodge / additive stacking)
    SUBTRACT = "subtract"  # wa*A - wb*B
    BLEND = "blend"  # wa*A + wb*B  (cross-fade; use wa+wb == 1)
    MULTIPLY = "multiply"  # (wa*A) * (wb*B)
    DIVIDE = "divide"  # (wa*A) / (wb*B + eps)
    SCREEN = "screen"  # 1 - (1-wa*A)(1-wb*B)
    OVERLAY = "overlay"  # photoshop-style overlay of wa*A over wb*B
    DIFFERENCE = "difference"  # |wa*A - wb*B|
    MIN = "min"  # elementwise minimum(wa*A, wb*B) -- not in SASpro, added for parity
    MAX = "max"  # elementwise maximum(wa*A, wb*B) -- not in SASpro, added for parity


_EPS = 1e-6


@dataclass
class ImageCombineParams:
    """Settings for :func:`combine_images`.

    Attributes:
        operation: Which pixel-arithmetic operation to apply.
        weight_a: Multiplier applied to image A before the operation.
            SASpro's single "opacity" slider is generalized into this pair of
            weights; the default ``1.0`` reproduces SASpro's un-weighted
            behavior for every op except BLEND (see ``weight_b``).
        weight_b: Multiplier applied to image B before the operation. For a
            SASpro-style opacity cross-fade with BLEND, set
            ``weight_a = 1 - alpha`` and ``weight_b = alpha``.
        clip: Clamp the result to ``[0, 1]`` after the operation (SASpro
            clamps per-op internally; here it is one final, configurable
            clamp so behavior is consistent and testable across all ops).
        rescale: Instead of clamping, min-max rescale the raw result back
            into ``[0, 1]``. Takes priority over ``clip`` when both are set.
            Useful when weights/ops push values outside range and a full
            dynamic-range remap is preferred over hard clipping (mirrors
            PixInsight's "rescale result" option).
    """

    operation: CombineOperation = CombineOperation.AVERAGE
    weight_a: float = 1.0
    weight_b: float = 1.0
    clip: bool = True
    rescale: bool = False


def _to_channels_first(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return ``(C, H, W)`` view of ``img`` and whether it was originally mono."""
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 2:
        return img[None, ...], True
    if img.ndim == 3:
        return img, False
    raise ValueError(f"Unsupported image shape: {img.shape}")


def _broadcast_channels(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast mono (1, H, W) against color (C, H, W) by channel repeat."""
    ca, ha, wa = a.shape
    cb, hb, wb = b.shape
    if (ha, wa) != (hb, wb):
        raise ValueError(
            f"Images must have the same spatial size; got {(ha, wa)} vs {(hb, wb)}."
        )
    if ca == cb:
        return a, b
    if ca == 1:
        a = np.repeat(a, cb, axis=0)
    elif cb == 1:
        b = np.repeat(b, ca, axis=0)
    else:
        raise ValueError(f"Incompatible channel counts: A has {ca}, B has {cb}.")
    return a, b


def _apply_op(a: np.ndarray, b: np.ndarray, op: CombineOperation) -> np.ndarray:
    if op == CombineOperation.AVERAGE:
        return a + b  # caller divides by (wa+wb); this branch receives raw weighted a,b
    if op == CombineOperation.ADD or op == CombineOperation.BLEND:
        return a + b
    if op == CombineOperation.SUBTRACT:
        return a - b
    if op == CombineOperation.MULTIPLY:
        return a * b
    if op == CombineOperation.DIVIDE:
        return a / (b + _EPS)
    if op == CombineOperation.SCREEN:
        return 1.0 - (1.0 - a) * (1.0 - b)
    if op == CombineOperation.OVERLAY:
        return np.where(a <= 0.5, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b))
    if op == CombineOperation.DIFFERENCE:
        return np.abs(a - b)
    if op == CombineOperation.MIN:
        return np.minimum(a, b)
    if op == CombineOperation.MAX:
        return np.maximum(a, b)
    raise ValueError(f"Unknown combine operation: {op}")


def combine_images(
    image_a: np.ndarray,
    image_b: np.ndarray,
    params: ImageCombineParams | None = None,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    """Combine two images with pixel arithmetic.

    Args:
        image_a: ``(H, W)`` mono or ``(C, H, W)`` color, float32-like in ``[0, 1]``.
        image_b: Same spatial size as ``image_a``. Mono/color mismatch is
            broadcast (a mono operand is repeated across the other's channels).
        params: Combine settings. Defaults to a plain AVERAGE.
        progress: Optional ``(fraction, message)`` progress callback.

    Returns:
        Combined image, float32, same shape convention as the wider of the
        two inputs (mono stays mono only if both inputs are mono).

    Raises:
        ValueError: On spatial size mismatch or an unrecognized channel
            layout/operation.
    """
    if params is None:
        params = ImageCombineParams()

    progress(0.0, f"Combining ({params.operation.value})…")

    a3, a_was_mono = _to_channels_first(image_a)
    b3, b_was_mono = _to_channels_first(image_b)
    a3, b3 = _broadcast_channels(a3, b3)

    wa = float(params.weight_a)
    wb = float(params.weight_b)
    aw = a3 * wa
    bw = b3 * wb

    progress(0.4, "Applying operation…")
    raw = _apply_op(aw, bw, params.operation)
    if params.operation == CombineOperation.AVERAGE:
        denom = wa + wb
        raw = raw / denom if denom != 0 else (a3 + b3) * 0.5

    progress(0.8, "Finalizing…")
    if params.rescale:
        lo, hi = float(raw.min()), float(raw.max())
        if hi > lo:
            raw = (raw - lo) / (hi - lo)
        else:
            raw = np.zeros_like(raw)
    elif params.clip:
        raw = np.clip(raw, 0.0, 1.0)

    result = raw.astype(np.float32, copy=False)
    if a_was_mono and b_was_mono:
        result = result[0]

    progress(1.0, "Combine complete")
    return result
