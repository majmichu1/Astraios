"""Add Stars — recombine a stars-only layer back onto a starless image.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro), Copyright
Franklin Marek, GPL-3.0-or-later. https://github.com/setiastro/setiastrosuitepro

SASpro's ``pro/add_stars.py`` implements an "Add Stars to Image" dialog that
recombines a starless view with a stars-only view (the inverse of a star
removal like StarNet). Its blend math (``AddStarsDialog._blend_images``)
offers exactly two modes selectable from a combo box:

    Screen: base = starless + stars - starless * stars   (== 1-(1-a)(1-b))
    Add:    base = starless + stars

and a single "Blend Ratio (Screen/Add Intensity)" slider ``r`` in ``[0, 1]``
(default ``1.0``, i.e. full effect) that cross-fades between the untouched
starless image and the full blend::

    blended = (1 - r) * starless + r * base
    blended = clip(blended, 0, 1)

This module ports that arithmetic core verbatim (minus the Qt dialog / open
docs enumeration, which are UI concerns handled by
:mod:`astraios.ui.dialogs.add_stars_dialog`). SASpro has no separate "clip"
option — the final clip is unconditional, as above — and no per-channel
control; both are preserved as-is (no cutting, no invention).

GPU/CPU decision: Screen and Add are elementwise, bandwidth-bound ops, so
the GPU win is modest but real. Re-benchmarked on a genuinely idle RTX 5060
(2026-07-10; the port-time attempt OOM'd because LM Studio held ~7 of 8GB
VRAM): end-to-end including transfers, the fused device path (blend + amount
+ clamp on GPU, one upload per input and one download) beat numpy at every
size measured — 1.20x at 3x3000x3000 (27M elements), 1.29x at 3x4500x4500,
1.30x at 3x6000x6000 (953 ms vs 1243 ms). Below ~25M elements both paths
finish in ~0.2 s and the difference stops mattering, so GPU dispatch is
gated there. The GPU branch stays wrapped so any CUDA/MPS failure (including
out-of-memory from a contending process) falls back to the numpy path rather
than raising.

Conventions: float32 images in ``[0, 1]``. Mono is ``(H, W)``, color is
``(C, H, W)`` (channels-first), matching :mod:`astraios.core.image_io`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

from astraios.core.device_manager import get_device_manager
from astraios.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


# Elementwise, bandwidth-bound op: GPU wins ~1.2-1.3x end-to-end at every
# size from 27M elements up (idle-GPU benchmark, see module docstring).
# Below this count both paths are near-instant and transfer overhead erodes
# the win, so stay on numpy there.
_GPU_ELEMENT_THRESHOLD = 25_000_000


class AddStarsBlendMode(str, Enum):
    """Blend mode for recombining stars onto a starless image.

    SASpro's ``AddStarsDialog`` offers exactly these two ("Screen", "Add") in
    its blend-type combo box -- no others exist in the source to port.
    """

    SCREEN = "screen"  # 1 - (1-starless)(1-stars) -- never clips highlights
    ADD = "add"  # starless + stars -- linear dodge, can clip/blow out


@dataclass
class AddStarsParams:
    """Settings for :func:`add_stars`.

    Attributes:
        blend_mode: SCREEN or ADD (see :class:`AddStarsBlendMode`).
        amount: SASpro's "Blend Ratio (Screen/Add Intensity)" slider, in
            ``[0, 1]``. ``0.0`` returns the starless image unchanged; ``1.0``
            (the SASpro default) applies the full blend. Values are clamped.
    """

    blend_mode: AddStarsBlendMode = AddStarsBlendMode.SCREEN
    amount: float = 1.0


def _to_channels_first(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return a ``(C, H, W)`` view of ``img`` and whether it was mono."""
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 2:
        return img[None, ...], True
    if img.ndim == 3:
        return img, False
    raise ValueError(f"Unsupported image shape: {img.shape}")


def _broadcast_channels(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast a mono ``(1, H, W)`` operand against a color ``(C, H, W)`` one.

    Matches SASpro's behavior of always widening mono to match the other
    layer's channel count (there, both are forced to RGB via ``_to_rgb01``
    before blending). Raises on spatial size mismatch or incompatible
    (non-mono) channel counts.
    """
    ca, ha, wa = a.shape
    cb, hb, wb = b.shape
    if (ha, wa) != (hb, wb):
        raise ValueError(
            "Starless and stars-only images must have the same pixel "
            f"dimensions; got {(ha, wa)} vs {(hb, wb)}."
        )
    if ca == cb:
        return a, b
    if ca == 1:
        a = np.repeat(a, cb, axis=0)
    elif cb == 1:
        b = np.repeat(b, ca, axis=0)
    else:
        raise ValueError(
            f"Incompatible channel counts: starless has {ca}, stars has {cb}."
        )
    return a, b


def _blend_numpy(a: np.ndarray, b: np.ndarray, mode: AddStarsBlendMode) -> np.ndarray:
    if mode == AddStarsBlendMode.SCREEN:
        return a + b - a * b
    return a + b  # ADD


def _blend_torch(a: torch.Tensor, b: torch.Tensor, mode: AddStarsBlendMode) -> torch.Tensor:
    if mode == AddStarsBlendMode.SCREEN:
        return 1.0 - (1.0 - a) * (1.0 - b)
    return a + b  # ADD


def _run_gpu(a3: np.ndarray, b3: np.ndarray, mode: AddStarsBlendMode, amount: float) -> np.ndarray:
    dm = get_device_manager()
    with torch.no_grad():
        a = dm.from_numpy(np.ascontiguousarray(a3.astype(np.float32)))
        b = dm.from_numpy(np.ascontiguousarray(b3.astype(np.float32)))
        base = _blend_torch(a, b, mode)
        out = (1.0 - amount) * a + amount * base
        out = out.clamp(0.0, 1.0)
        dm.synchronize()
        return out.cpu().numpy().astype(np.float32)


def add_stars(
    starless: np.ndarray,
    stars: np.ndarray,
    params: AddStarsParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    """Recombine a stars-only layer onto a starless image.

    Args:
        starless: ``(H, W)`` mono or ``(C, H, W)`` color, float32-like in
            ``[0, 1]`` -- the destination image (star-removed).
        stars: The stars-only layer to add back in. Same spatial size as
            ``starless``; a mono/color mismatch is broadcast (the mono
            operand is repeated across the other's channels), matching
            SASpro's behavior of normalizing both layers before blending.
        params: Blend settings. Defaults to SCREEN at full (``1.0``) amount.
        mask: Optional protection mask -- ``result = processed * mask +
            original * (1 - mask)``.
        progress: Optional ``(fraction, message)`` progress callback.

    Returns:
        Blended image, float32, clipped to ``[0, 1]``. Stays mono only if
        both inputs were mono; otherwise widened to color.

    Raises:
        ValueError: On spatial size mismatch or an incompatible (non-mono)
            channel-count mismatch.
    """
    if params is None:
        params = AddStarsParams()

    progress(0.0, f"Adding stars ({params.blend_mode.value})…")

    a3, a_was_mono = _to_channels_first(starless)
    b3, b_was_mono = _to_channels_first(stars)
    a3, b3 = _broadcast_channels(a3, b3)

    amount = float(np.clip(params.amount, 0.0, 1.0))

    progress(0.4, "Blending…")
    dm = get_device_manager()
    use_gpu = dm.is_gpu and a3.size >= _GPU_ELEMENT_THRESHOLD
    out: np.ndarray | None = None
    if use_gpu:
        try:
            out = _run_gpu(a3, b3, params.blend_mode, amount)
        except Exception:
            log.warning(
                "Add Stars: GPU blend failed (falling back to CPU)", exc_info=True
            )
            out = None

    if out is None:
        base = _blend_numpy(a3, b3, params.blend_mode)
        out = (1.0 - amount) * a3 + amount * base
        out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    if a_was_mono and b_was_mono:
        out = out[0]

    progress(1.0, "Stars added")
    return apply_mask(starless, out, mask)
