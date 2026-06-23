"""Star Stretch — colour-preserving stretch tuned for star layers.

A counterpart to Seti Astro Suite's *Star Stretch*: stars are tiny but very
bright, so a normal stretch either clips them white or, if gentle, leaves the
field flat. This applies a colour-preserving arcsinh stretch (the brightness is
stretched while each pixel's RGB *ratios* are kept, so stars keep their hue) and
then optionally boosts saturation to bring out the blue/gold star colours that
stretching tends to wash out.

Typical workflow: remove stars (StarNet), process the starless image, then run
Star Stretch on the extracted star image and screen it back in.

The stretch reuses :func:`astraios.core.stretch.arcsinh_stretch` (GPU); the
saturation boost runs on the GPU via the device manager.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from astraios.core.device_manager import get_device_manager
from astraios.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


@dataclass
class StarStretchParams:
    """Parameters for the star stretch.

    Attributes:
        amount: Stretch strength in ``[0, 1]``. Mapped to the arcsinh factor
            ``beta = 1 + amount * 49`` — higher pulls up faint stars more.
        color_boost: Saturation multiplier applied after stretching. ``1.0``
            leaves colour unchanged; ``>1`` enriches star colour, ``<1``
            desaturates. Ignored for mono images.
    """

    amount: float = 0.2
    color_boost: float = 1.0


def _boost_saturation(image: np.ndarray, boost: float) -> np.ndarray:
    """Scale colour saturation about the per-pixel luminance (GPU).

    ``out = lum + (rgb - lum) * boost`` — a hue-preserving saturation control.
    """
    dm = get_device_manager()
    with torch.no_grad():
        t = dm.from_numpy(np.ascontiguousarray(image))  # (C, H, W)
        lum = t.mean(dim=0, keepdim=True)
        out = (lum + (t - lum) * boost).clamp(0.0, 1.0)
        return out.cpu().numpy().astype(np.float32)


def star_stretch(
    image: np.ndarray,
    params: StarStretchParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    """Apply a colour-preserving star stretch.

    Args:
        image: ``(H, W)`` or ``(C, H, W)`` float32 in ``[0, 1]``.
        params: Stretch parameters.
        mask: Optional protection mask (applied at the end).
        progress: Optional progress callback.

    Returns:
        Stretched image, same shape, clipped to ``[0, 1]``.
    """
    from astraios.core.stretch import ArcsinhStretchParams, arcsinh_stretch

    if params is None:
        params = StarStretchParams()

    # no copy: op never mutates the input; apply_mask reads image directly
    beta = 1.0 + float(np.clip(params.amount, 0.0, 1.0)) * 49.0

    progress(0.2, "Stretching stars (colour-preserving)…")
    result = arcsinh_stretch(
        image, ArcsinhStretchParams(stretch_factor=beta, linked=True)
    )

    if image.ndim == 3 and image.shape[0] >= 3 and abs(params.color_boost - 1.0) > 1e-6:
        progress(0.7, "Boosting star colour…")
        result = _boost_saturation(result, params.color_boost)

    progress(1.0, "Star stretch complete")
    return apply_mask(image, result, mask)
