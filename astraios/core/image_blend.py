"""Image Blend — combine two images with photoshop-style blend modes.

The missing half of the starless+stars workflow: after processing a starless
image and stretching the star layer separately (see
:mod:`astraios.core.star_stretch`), ``SCREEN`` blends the stars back in without
clipping. Also useful for compositing, lighten/darken stacking, and difference
inspection.

All modes run on the GPU via the device manager. The blend layer is resized to
the base's H×W when needed, and a mono layer broadcasts across a colour base
(and vice-versa) so mixed-channel inputs "just work".
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import torch

from astraios.core.device_manager import get_device_manager
from astraios.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


class BlendMode(Enum):
    NORMAL = auto()      # opacity mix only
    SCREEN = auto()      # 1-(1-a)(1-b) — add stars/light without clipping
    ADD = auto()         # a + b (linear dodge)
    SUBTRACT = auto()    # a - b
    MULTIPLY = auto()    # a * b
    LIGHTEN = auto()     # max(a, b)
    DARKEN = auto()      # min(a, b)
    DIFFERENCE = auto()  # |a - b|
    AVERAGE = auto()     # (a + b) / 2
    OVERLAY = auto()     # contrast-boosting combine


@dataclass
class BlendParams:
    """Parameters for image blending.

    Attributes:
        mode: Blend mode.
        opacity: Strength of the blend layer in ``[0, 1]``. The final result is
            ``base*(1-opacity) + blended*opacity``.
    """

    mode: BlendMode = BlendMode.SCREEN
    opacity: float = 1.0


def _match_to_base(base: torch.Tensor, layer: torch.Tensor) -> torch.Tensor:
    """Resize/broadcast ``layer`` to ``base``'s (C, H, W) shape on-device."""
    # Normalise both to (C, H, W) for the math; callers pass 3-D already.
    bc, bh, bw = base.shape
    lc, lh, lw = layer.shape

    if (lh, lw) != (bh, bw):
        layer = torch.nn.functional.interpolate(
            layer.unsqueeze(0), size=(bh, bw), mode="bilinear", align_corners=False
        ).squeeze(0)

    if lc != bc:
        if lc == 1:
            layer = layer.expand(bc, -1, -1)
        elif bc == 1:
            layer = layer.mean(dim=0, keepdim=True)
        else:
            # e.g. RGBA vs RGB — use the overlapping leading channels.
            n = min(lc, bc)
            layer = layer[:n]
            if n < bc:
                layer = torch.cat([layer, layer[-1:].expand(bc - n, -1, -1)], dim=0)
    return layer


def _apply_mode(a: torch.Tensor, b: torch.Tensor, mode: BlendMode) -> torch.Tensor:
    if mode == BlendMode.NORMAL:
        return b
    if mode == BlendMode.SCREEN:
        return 1.0 - (1.0 - a) * (1.0 - b)
    if mode == BlendMode.ADD:
        return a + b
    if mode == BlendMode.SUBTRACT:
        return a - b
    if mode == BlendMode.MULTIPLY:
        return a * b
    if mode == BlendMode.LIGHTEN:
        return torch.maximum(a, b)
    if mode == BlendMode.DARKEN:
        return torch.minimum(a, b)
    if mode == BlendMode.DIFFERENCE:
        return torch.abs(a - b)
    if mode == BlendMode.AVERAGE:
        return (a + b) * 0.5
    if mode == BlendMode.OVERLAY:
        return torch.where(a <= 0.5, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b))
    raise ValueError(f"Unknown blend mode: {mode}")


def blend_images(
    base: np.ndarray,
    blend: np.ndarray,
    params: BlendParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    """Blend ``blend`` onto ``base`` with the given mode and opacity.

    Args:
        base: ``(H, W)`` or ``(C, H, W)`` float32 in ``[0, 1]``.
        blend: The layer to combine, any size/channel count (matched to base).
        params: Blend parameters.
        mask: Optional protection mask (applied at the end).
        progress: Optional progress callback.

    Returns:
        Result with ``base``'s shape, clipped to ``[0, 1]``.
    """
    if params is None:
        params = BlendParams()

    # no copy: op never mutates the input; apply_mask reads base directly
    base_was_2d = base.ndim == 2
    base3 = base[None] if base_was_2d else base
    blend3 = blend[None] if blend.ndim == 2 else blend

    dm = get_device_manager()
    progress(0.2, f"Blending ({params.mode.name.lower()})…")
    with torch.no_grad():
        a = dm.from_numpy(np.ascontiguousarray(base3.astype(np.float32)))
        b = dm.from_numpy(np.ascontiguousarray(blend3.astype(np.float32)))
        b = _match_to_base(a, b)

        blended = _apply_mode(a, b, params.mode)
        op = float(np.clip(params.opacity, 0.0, 1.0))
        out = a * (1.0 - op) + blended * op
        out = out.clamp(0.0, 1.0).cpu().numpy().astype(np.float32)

    if base_was_2d:
        out = out[0]
    progress(1.0, "Blend complete")
    return apply_mask(base, out, mask)
