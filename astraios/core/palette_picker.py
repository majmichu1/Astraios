"""Perfect Palette Picker — narrowband-to-RGB false-color palette blending.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro),
Copyright Franklin Marek, GPL-3.0-or-later.
https://github.com/setiastro/setiastrosuitepro

SASpro's ``perfect_palette_picker.py`` lets a user load up to three
narrowband filter planes (Ha, OIII, SII) and instantly preview all of the
community's standard false-color palette recipes -- the Hubble "SHO", the
bicolor "HOO", the SII-forward "OSS"/"HSS", "Realistic" color-matched blends,
and the dynamic (per-pixel) "Foraxx" blend -- then push the chosen one out as
a finished RGB composite.

This module reproduces every palette recipe from
``PerfectPalettePicker._map_channels_or_special`` exactly, including its
channel-substitution rule (a missing Ha falls back to SII and vice versa, so
every palette can be built from just OIII + one of Ha/SII) and its "Linear
input" pre-stretch (SASpro's ``stretch_mono_image``/``stretch_color_image``
with ``target_median=0.25``, reproduced here via Astraios's own
:func:`astraios.core.stretch.statistical_stretch`). The final-image
normalize-by-max step from SASpro's ``_generate_for_palette`` is reproduced
as well (toggle via :attr:`PalettePickerParams.normalize`).

Two things in this module are Astraios additions, *not* part of SASpro's
Perfect Palette Picker (which has no such inputs at all):

* :attr:`PalettePickerParams.custom_weights` / :attr:`Palette.CUSTOM` -- a
  free-form 3x3 [Ha, OIII, SII] -> [R, G, B] weight matrix, for palette
  recipes the preset list doesn't cover. Mirrors the ``CUSTOM`` matrix
  convention already used by :mod:`astraios.core.narrowband`.
* ``stars`` / :attr:`PalettePickerParams.stars_opacity` -- an optional
  broadband/OSC RGB "stars" layer, screen-blended over the finished palette
  at the end. This is the same idea as
  :mod:`astraios.core.nb_star_color` (natural star color doesn't come from
  narrowband filters) but implemented as a simple post-blend here rather than
  a full recombination, since Perfect Palette Picker's own job is the
  nebula-color mapping, not star-color correction.

Data convention: all channel inputs are float32 ``(H, W)`` mono planes in
``[0, 1]``; ``stars`` (if given) is channels-first ``(3, H, W)`` (or mono,
broadcast to all three). The output is always ``(3, H, W)`` RGB float32 in
``[0, 1]``, matching Astraios's channels-first color convention (SASpro
itself works in ``(H, W, 3)``).

Unlike SASpro's UI, which silently warps mismatched channel sizes after
prompting the user, this module raises ``ValueError`` on a spatial-shape
mismatch -- the same "reject, don't resample" policy Astraios already uses
for ``NBStarColorDialog`` and ``ContinuumSubtractDialog``. Resize the input
yourself (e.g. at the dialog layer) if you need to combine mismatched frames.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

import numpy as np
import torch

from astraios.core.device_manager import get_device_manager

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


class Palette(Enum):
    """Every palette recipe from SASpro's Perfect Palette Picker.

    The nine "basic" members are fixed linear channel permutations/mixes of
    [Ha, OIII, SII]. ``REALISTIC_1``/``REALISTIC_2``/``FORAXX`` are nonlinear
    per-pixel blends. ``CUSTOM`` is an Astraios addition (free-form weight
    matrix; see :attr:`PalettePickerParams.custom_weights`).
    """

    SHO = auto()  # Hubble palette: R=SII, G=Ha, B=OIII
    HOO = auto()  # bicolor: R=Ha, G=OIII, B=OIII
    HSO = auto()  # R=Ha, G=SII, B=OIII
    HOS = auto()  # R=Ha, G=OIII, B=SII
    OSS = auto()  # R=OIII, G=SII, B=SII
    OHH = auto()  # R=OIII, G=Ha, B=Ha
    OSH = auto()  # R=OIII, G=SII, B=Ha
    OHS = auto()  # R=OIII, G=Ha, B=SII
    HSS = auto()  # R=Ha, G=SII, B=SII
    REALISTIC_1 = auto()  # color-matched blend #1
    REALISTIC_2 = auto()  # color-matched blend #2
    FORAXX = auto()  # dynamic per-pixel Ha/OIII green blend
    CUSTOM = auto()  # Astraios addition: free-form [Ha,OIII,SII] -> [R,G,B]


# Display names matching SASpro's PerfectPalettePicker.PALETTES thumbnail
# labels exactly (used by the dialog's combo box).
PALETTE_LABELS: dict[Palette, str] = {
    Palette.SHO: "SHO",
    Palette.HOO: "HOO",
    Palette.HSO: "HSO",
    Palette.HOS: "HOS",
    Palette.OSS: "OSS",
    Palette.OHH: "OHH",
    Palette.OSH: "OSH",
    Palette.OHS: "OHS",
    Palette.HSS: "HSS",
    Palette.REALISTIC_1: "Realistic1",
    Palette.REALISTIC_2: "Realistic2",
    Palette.FORAXX: "Foraxx",
    Palette.CUSTOM: "Custom",
}

# Short, plain-language descriptions for the dialog's help text.
PALETTE_DESCRIPTIONS: dict[Palette, str] = {
    Palette.SHO: "Hubble palette. R=SII, G=Ha, B=OIII -- the classic 'Hubble' look.",
    Palette.HOO: "Bicolor. R=Ha, G=OIII, B=OIII -- for OIII with no/weak SII.",
    Palette.HSO: "R=Ha, G=SII, B=OIII.",
    Palette.HOS: "R=Ha, G=OIII, B=SII -- closer to natural than SHO.",
    Palette.OSS: "R=OIII, G=SII, B=SII -- SII-forward, teal/gold look.",
    Palette.OHH: "R=OIII, G=Ha, B=Ha -- Ha-forward, gold/blue look.",
    Palette.OSH: "R=OIII, G=SII, B=Ha.",
    Palette.OHS: "R=OIII, G=Ha, B=SII.",
    Palette.HSS: "R=Ha, G=SII, B=SII -- doesn't need OIII data at all.",
    Palette.REALISTIC_1: (
        "Color-matched blend: R=(Ha+SII)/2, G=0.3 Ha + 0.7 OIII, B=0.9 OIII + 0.1 Ha."
    ),
    Palette.REALISTIC_2: "Color-matched blend: R=0.7 Ha + 0.3 SII, G=0.3 SII + 0.7 OIII, B=OIII.",
    Palette.FORAXX: (
        "Dynamic per-pixel blend popular in the imaging community: R blends Ha/SII "
        "weighted by OIII strength, G blends Ha/OIII weighted by their product, "
        "B=OIII -- breaks up SHO's flat green/gold cast."
    ),
    Palette.CUSTOM: "Free-form weight matrix: mix any amount of Ha/OIII/SII into R/G/B.",
}

# Linear palette matrices: row i = [Ha_weight, OIII_weight, SII_weight] for
# output channel i in [R, G, B]. Reproduces
# PerfectPalettePicker._map_channels_or_special's `basic` dict exactly.
_LINEAR_MATRICES: dict[Palette, np.ndarray] = {
    Palette.SHO: np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    ),
    Palette.HOO: np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    ),
    Palette.HSO: np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32
    ),
    Palette.HOS: np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    ),
    Palette.OSS: np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32
    ),
    Palette.OHH: np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
    ),
    Palette.OSH: np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32
    ),
    Palette.OHS: np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    ),
    Palette.HSS: np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32
    ),
}

_NONLINEAR_PALETTES = frozenset({Palette.REALISTIC_1, Palette.REALISTIC_2, Palette.FORAXX})

# GPU transfer overhead outweighs this module's few elementwise ops even at
# large frame sizes (benchmarked: CPU faster at 3x4000x4000; see
# scratchpad benchmark run for this port). Kept as a named threshold, gated
# very high like astraios.core.diffraction_spikes.GPU_PIXEL_THRESHOLD, so a
# future benchmark on bigger/rarer hardware can re-enable it by lowering
# this constant rather than changing call sites.
GPU_PIXEL_THRESHOLD = 10_000_000_000


@dataclass
class PalettePickerParams:
    """Parameters for :func:`apply_palette`.

    Attributes:
        palette: Which palette recipe to build. See :class:`Palette`.
        custom_weights: ``(3, 3)`` matrix, row i = weights of
            ``[Ha, OIII, SII]`` for output channel i in ``[R, G, B]``. Only
            used when ``palette is Palette.CUSTOM``. Defaults to the
            identity (R=Ha, G=OIII, B=SII).
        linear_input: Reproduces SASpro's "Linear input (apply statistical
            stretch before build)" checkbox (default checked): each loaded
            channel is stretched independently to `target_median` before
            the palette math runs, since narrowband subs are usually still
            linear when loaded here.
        target_median: Target background median for the `linear_input`
            stretch. SASpro hardcodes ``0.25``; exposed here so it isn't a
            hidden magic number.
        normalize: Reproduces SASpro's ``_generate_for_palette`` step of
            dividing the finished RGB by its own max (then clipping to
            ``[0, 1]``) so the preview is never all-clipped/blown out.
        stars_opacity: Astraios addition (not in SASpro). Screen-blend
            strength for an optional broadband ``stars`` layer over the
            finished palette, ``[0, 1]``. ``0.0`` (default) leaves the
            palette untouched even if `stars` is supplied.
    """

    palette: Palette = Palette.SHO
    custom_weights: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float32)
    )
    linear_input: bool = True
    target_median: float = 0.25
    normalize: bool = True
    stars_opacity: float = 0.0


def _as_mono01(arr: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"{name} must be a mono (H, W) plane; got shape {arr.shape}")
    return np.clip(a, 0.0, 1.0).astype(np.float32)


def _ensure_rgb(arr: np.ndarray) -> np.ndarray:
    """Return a (3, H, W) float32 RGB view; mono is replicated x3."""
    a = np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)
    if a.ndim == 2:
        return np.stack([a, a, a], axis=0)
    if a.ndim == 3:
        if a.shape[0] == 1:
            return np.repeat(a, 3, axis=0)
        if a.shape[0] >= 3:
            return np.ascontiguousarray(a[:3])
    raise ValueError(f"stars must be (H, W) or (C, H, W) with >=1 channel; got shape {arr.shape}")


def _stretch_channel(channel: np.ndarray, target_median: float) -> np.ndarray:
    """SASpro's `stretch_mono_image(img, target_median=0.25)`, via Astraios's
    own statistical (target-median) stretch -- see module docstring."""
    from astraios.core.stretch import StatisticalStretchParams, statistical_stretch

    return np.clip(
        statistical_stretch(channel, StatisticalStretchParams(target_median=target_median)),
        0.0,
        1.0,
    ).astype(np.float32)


def _linear_mix_gpu(stack_t: torch.Tensor, matrix: np.ndarray, dm) -> torch.Tensor:
    matrix_t = torch.from_numpy(np.ascontiguousarray(matrix, dtype=np.float32)).to(dm.device)
    return torch.einsum("ci,ihw->chw", matrix_t, stack_t)


def _linear_mix_np(stack: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.einsum("ci,ihw->chw", matrix.astype(np.float32), stack).astype(np.float32)


@torch.no_grad()
def _realistic_1_gpu(ha: torch.Tensor, oo: torch.Tensor, si: torch.Tensor) -> torch.Tensor:
    r = 0.5 * ha + 0.5 * si
    g = 0.3 * ha + 0.7 * oo
    b = 0.9 * oo + 0.1 * ha
    return torch.stack([r, g, b], dim=0)


def _realistic_1_np(ha: np.ndarray, oo: np.ndarray, si: np.ndarray) -> np.ndarray:
    r = 0.5 * ha + 0.5 * si
    g = 0.3 * ha + 0.7 * oo
    b = 0.9 * oo + 0.1 * ha
    return np.stack([r, g, b], axis=0).astype(np.float32)


@torch.no_grad()
def _realistic_2_gpu(ha: torch.Tensor, oo: torch.Tensor, si: torch.Tensor) -> torch.Tensor:
    r = 0.7 * ha + 0.3 * si
    g = 0.3 * si + 0.7 * oo
    b = oo
    return torch.stack([r, g, b], dim=0)


def _realistic_2_np(ha: np.ndarray, oo: np.ndarray, si: np.ndarray) -> np.ndarray:
    r = 0.7 * ha + 0.3 * si
    g = 0.3 * si + 0.7 * oo
    b = oo
    return np.stack([r, g, b], axis=0).astype(np.float32)


@torch.no_grad()
def _foraxx_gpu(ha: torch.Tensor, oo: torch.Tensor, si: torch.Tensor) -> torch.Tensor:
    # Reproduces PerfectPalettePicker's "ha, oo, si all present" Foraxx branch.
    # (Its "si is None" branch is unreachable here: the caller's Ha/SII
    # substitution above already guarantees si is not None whenever ha is,
    # matching SASpro's own substitution rule -- see module docstring.)
    oo_c = oo.clamp(1e-6, 1.0)
    t = torch.pow(oo_c, 1.0 - oo_c)
    r = t * si + (1.0 - t) * ha
    t2 = ha * oo
    g = torch.pow(t2, 1.0 - t2) * ha + (1.0 - torch.pow(t2, 1.0 - t2)) * oo
    b = oo
    return torch.stack([r, g, b], dim=0)


def _foraxx_np(ha: np.ndarray, oo: np.ndarray, si: np.ndarray) -> np.ndarray:
    oo_c = np.clip(oo, 1e-6, 1.0)
    t = oo_c ** (1.0 - oo_c)
    r = t * si + (1.0 - t) * ha
    t2 = ha * oo
    g = (t2 ** (1.0 - t2)) * ha + (1.0 - (t2 ** (1.0 - t2))) * oo
    b = oo
    return np.stack([r, g, b], axis=0).astype(np.float32)


def apply_palette(
    ha: np.ndarray | None,
    oiii: np.ndarray | None,
    sii: np.ndarray | None = None,
    stars: np.ndarray | None = None,
    params: PalettePickerParams | None = None,
    progress: ProgressCallback | None = None,
) -> np.ndarray:
    """Blend narrowband channels into a false-color RGB palette.

    Args:
        ha: Ha (H-alpha) mono plane, float32 ``[0, 1]``, or ``None`` if not
            available (falls back to `sii`; see below).
        oiii: OIII mono plane. Required for every palette -- reproduces
            SASpro's own gating (``PerfectPalettePicker`` refuses to build
            *any* palette, even Ha/SII-only ones like HSS, without OIII
            loaded).
        sii: SII mono plane, or ``None`` if not available (falls back to
            `ha`). At least one of `ha`/`sii` must be given.
        stars: Optional broadband/OSC RGB frame (Astraios addition, not in
            SASpro), screen-blended over the finished palette at strength
            `PalettePickerParams.stars_opacity`. ``(3, H, W)`` or mono,
            channels-first, same spatial shape as the narrowband channels.
        params: Palette parameters. Defaults to :class:`PalettePickerParams`
            (SHO, linear-input stretch on, normalize on).
        progress: Optional ``progress(fraction, message)`` callback.

    Returns:
        ``(3, H, W)`` float32 RGB image in ``[0, 1]``.

    Raises:
        ValueError: `oiii` is missing, both `ha` and `sii` are missing, or
            the supplied channels/`stars` don't share the same spatial shape.
    """
    if params is None:
        params = PalettePickerParams()
    progress = progress or _noop_progress

    if oiii is None:
        raise ValueError("OIII is required to build any Perfect Palette Picker palette")
    if ha is None and sii is None:
        raise ValueError("At least one of Ha or SII is required (in addition to OIII)")

    oo = _as_mono01(oiii, "OIII")
    ha_in = _as_mono01(ha, "Ha") if ha is not None else None
    si_in = _as_mono01(sii, "SII") if sii is not None else None

    # SASpro's substitution rule: a missing Ha/SII falls back to the other.
    if ha_in is None:
        ha_in = si_in
    if si_in is None:
        si_in = ha_in
    # Guaranteed non-None: the guard above requires at least one of ha/sii.
    assert ha_in is not None and si_in is not None
    ha_arr: np.ndarray = ha_in
    si_arr: np.ndarray = si_in

    if ha_arr.shape != oo.shape or si_arr.shape != oo.shape:
        raise ValueError(
            "Ha/OIII/SII must share the same (H, W) shape; got "
            f"Ha={ha_arr.shape}, OIII={oo.shape}, SII={si_arr.shape}"
        )

    stars_rgb = None
    if stars is not None and params.stars_opacity > 0:
        stars_rgb = _ensure_rgb(stars)
        if stars_rgb.shape[-2:] != oo.shape:
            raise ValueError(
                f"stars must share the narrowband spatial shape {oo.shape}; "
                f"got {stars_rgb.shape[-2:]}"
            )

    progress(0.05, "Preparing channels…")
    if params.linear_input:
        ha_arr = _stretch_channel(ha_arr, params.target_median)
        oo = _stretch_channel(oo, params.target_median)
        si_arr = _stretch_channel(si_arr, params.target_median)

    dm = get_device_manager()
    h, w = oo.shape
    use_gpu = dm.is_gpu and (h * w) >= GPU_PIXEL_THRESHOLD

    progress(0.3, f"Building {PALETTE_LABELS.get(params.palette, params.palette.name)} palette…")
    if params.palette in _NONLINEAR_PALETTES:
        fn_gpu = {
            Palette.REALISTIC_1: _realistic_1_gpu,
            Palette.REALISTIC_2: _realistic_2_gpu,
            Palette.FORAXX: _foraxx_gpu,
        }[params.palette]
        fn_np = {
            Palette.REALISTIC_1: _realistic_1_np,
            Palette.REALISTIC_2: _realistic_2_np,
            Palette.FORAXX: _foraxx_np,
        }[params.palette]
        if use_gpu:
            with torch.no_grad():
                ha_t = dm.from_numpy(np.ascontiguousarray(ha_arr))
                oo_t = dm.from_numpy(np.ascontiguousarray(oo))
                si_t = dm.from_numpy(np.ascontiguousarray(si_arr))
                rgb_t = fn_gpu(ha_t, oo_t, si_t)
                rgb = rgb_t.cpu().numpy().astype(np.float32)
        else:
            rgb = fn_np(ha_arr, oo, si_arr)
    else:
        matrix = (
            params.custom_weights
            if params.palette == Palette.CUSTOM
            else _LINEAR_MATRICES[params.palette]
        )
        stack = np.stack([ha_arr, oo, si_arr], axis=0)  # (3, H, W): [Ha, OIII, SII]
        if use_gpu:
            with torch.no_grad():
                stack_t = dm.from_numpy(np.ascontiguousarray(stack))
                rgb_t = _linear_mix_gpu(stack_t, matrix, dm)
                rgb = rgb_t.cpu().numpy().astype(np.float32)
        else:
            rgb = _linear_mix_np(stack, matrix)

    rgb = np.clip(np.nan_to_num(rgb), 0.0, 1.0).astype(np.float32)

    progress(0.7, "Normalizing…")
    if params.normalize:
        mx = float(rgb.max()) or 1.0
        rgb = np.clip(rgb / mx, 0.0, 1.0).astype(np.float32)

    if stars_rgb is not None:
        progress(0.85, "Blending star color…")
        opacity = float(np.clip(params.stars_opacity, 0.0, 1.0))
        screened = 1.0 - (1.0 - rgb) * (1.0 - stars_rgb)
        rgb = np.clip(rgb + opacity * (screened - rgb), 0.0, 1.0).astype(np.float32)

    progress(1.0, "Palette complete")
    return rgb


__all__ = [
    "Palette",
    "PALETTE_LABELS",
    "PALETTE_DESCRIPTIONS",
    "PalettePickerParams",
    "apply_palette",
]
