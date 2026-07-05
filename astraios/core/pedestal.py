"""Pedestal — uniform offset management (add / remove a black-level pedestal).

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro)
`pedestal.py`, Copyright Franklin Marek, GPL-3.0-or-later.
https://github.com/setiastro/setiastrosuitepro

SASpro's original only implements automatic pedestal *removal*
(``out[c] = image[c] - min(image[c])``, always per-channel). This port keeps
that exact math for the default "remove" case, and extends it — per the
Astraios feature request — with an explicit "add" mode and a "global"
(single statistic across all channels, rather than per-channel) option, so
the tool can both add and remove a uniform offset.

GPU note: pedestal management is pure scalar arithmetic (a per-channel or
whole-image min/subtract/add). There is nothing to vectorize beyond what
numpy already does in a single pass, and moving a handful of scalar
reductions to the GPU would cost more in host<->device transfer than it
could ever save. This module is therefore CPU-only (numpy), by design —
not an oversight of the "always use device_manager" rule.

Images are float32 in [0, 1]; mono is (H, W), color is (C, H, W)
channels-first (SASpro's original works on (H, W, C); the math is
axis-order-independent so only the reduction axis changed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from astraios.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


@dataclass
class PedestalParams:
    """Parameters for pedestal (uniform black-level offset) management.

    Attributes
    ----------
    mode : str
        ``"add"`` to add a constant offset, or ``"remove"`` to subtract a
        pedestal. Defaults to ``"add"`` with `amount` = 0, i.e. identity.
    per_channel : bool
        When True (and the image is color), operate independently on each
        channel. When False, a single scalar statistic/offset is applied
        uniformly across every channel ("global"). Ignored for mono images.
    amount : float
        Constant value to add, when `mode` == "add". For per-channel add
        without explicit `channel_amounts`, this same amount is applied
        uniformly to every channel.
    channel_amounts : list[float] or None
        Optional explicit per-channel offsets (one float per channel), for
        `mode` == "add" with `per_channel` == True. Overrides `amount` on a
        per-channel basis when given. JSON-serializable list.
    remove_amount : float, list[float], or None
        Optional explicit pedestal value(s) to subtract for `mode` ==
        "remove". If None (default), the pedestal is computed automatically
        as the per-channel (or global) minimum of the image — this matches
        Seti Astro Suite Pro's automatic "Remove Pedestal" behavior exactly.
        If provided (a scalar, or a per-channel list when `per_channel` is
        True), that exact value is subtracted instead of auto-detecting the
        minimum — this enables an exact round-trip with a prior "add".
    clip : bool
        Clip the result back into [0, 1] after the operation.
    """

    mode: str = "add"
    per_channel: bool = True
    amount: float = 0.0
    channel_amounts: list[float] | None = None
    remove_amount: float | list[float] | None = None
    clip: bool = True


def apply_pedestal(
    data: np.ndarray,
    params: PedestalParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    """Add or remove a uniform pedestal (black-level offset).

    Parameters
    ----------
    data : ndarray
        Image data, float32-ish. Mono (H, W) or color (C, H, W).
    params : PedestalParams, optional
        Defaults to a no-op (mode="add", amount=0).
    mask : Mask, optional
        Restrict the effect to a region; see `astraios.core.masks`.
    progress : callable, optional
        `progress(fraction, message)` callback.

    Returns
    -------
    ndarray
        float32 array, same shape as `data`.
    """
    if params is None:
        params = PedestalParams()

    progress(0.1, "Computing pedestal…")
    src = _as_float01(data)

    if params.mode == "add":
        out = _add_pedestal(src, params)
    elif params.mode == "remove":
        out = _remove_pedestal(src, params)
    else:
        raise ValueError(f"Unknown pedestal mode: {params.mode!r} (expected 'add' or 'remove')")

    if params.clip:
        out = np.clip(out, 0.0, 1.0)
    out = out.astype(np.float32, copy=False)

    progress(0.9, "Blending…")
    result = apply_mask(data, out, mask)
    progress(1.0, "Pedestal complete")
    return result


def _as_float01(img: np.ndarray) -> np.ndarray:
    """Return float32 image; compress a stray >1.0 range down to ~[0, 1]."""
    a = np.asarray(img)
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    if a.size:
        mx = float(a.max())
        if mx > 5.0:
            a = a / mx
    return a


def _channel_axis_values(
    data: np.ndarray, values: float | list[float], n_channels: int
) -> np.ndarray:
    """Broadcast a scalar or per-channel list to an (C,) array, validating length."""
    if np.isscalar(values):
        return np.full(n_channels, float(values), dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape != (n_channels,):
        raise ValueError(f"Expected {n_channels} channel values, got {arr.shape}")
    return arr


def _add_pedestal(a: np.ndarray, p: PedestalParams) -> np.ndarray:
    if a.ndim == 2:
        return a + float(p.amount if p.channel_amounts is None else p.channel_amounts[0])

    n = a.shape[0]
    if p.per_channel and p.channel_amounts is not None:
        offsets = _channel_axis_values(a, p.channel_amounts, n)
    elif p.per_channel:
        offsets = np.full(n, float(p.amount), dtype=np.float32)
    else:
        offsets = np.full(n, float(p.amount), dtype=np.float32)

    return a + offsets.reshape(n, *([1] * (a.ndim - 1)))


def _remove_pedestal(a: np.ndarray, p: PedestalParams) -> np.ndarray:
    if a.ndim == 2:
        if p.remove_amount is not None:
            offset = float(p.remove_amount if np.isscalar(p.remove_amount) else p.remove_amount[0])
        else:
            offset = float(a.min()) if a.size else 0.0
        return a - offset

    n = a.shape[0]
    if p.remove_amount is not None:
        if p.per_channel:
            offsets = _channel_axis_values(a, p.remove_amount, n)
        else:
            offset = float(p.remove_amount if np.isscalar(p.remove_amount) else p.remove_amount[0])
            offsets = np.full(n, offset, dtype=np.float32)
    elif p.per_channel:
        # Exactly SASpro's `_remove_pedestal_array`: subtract each channel's own minimum.
        offsets = np.array(
            [float(a[c].min()) if a[c].size else 0.0 for c in range(n)],
            dtype=np.float32,
        )
    else:
        # "Global" extension: subtract one minimum computed across all channels.
        offset = float(a.min()) if a.size else 0.0
        offsets = np.full(n, offset, dtype=np.float32)

    return a - offsets.reshape(n, *([1] * (a.ndim - 1)))
