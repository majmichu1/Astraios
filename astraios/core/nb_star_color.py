"""Narrowband-to-RGB star color recombination.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro),
Copyright Franklin Marek, GPL-3.0-or-later.
https://github.com/setiastro/setiastrosuitepro

SASpro's ``nbtorgb_stars.py`` ("NB -> RGB Stars") tool solves a common
narrowband-palette problem: mapping Ha/OIII/SII straight onto R/G/B (or
blending them) gives stars odd, bloated colors, because a star's flux in
each narrowband filter has nothing to do with its real optical color. The
fix is to borrow star color from a broadband (OSC) frame -- typically a
stars-only extraction -- and blend it into the narrowband composite:

    r = 0.5 * broadband_r + 0.5 * SII   (SII falls back to broadband_r)
    g = ratio * Ha + (1 - ratio) * broadband_g
    b = OIII                            (falls back to broadband_b)

followed by an optional non-linear "star stretch" boost, green-channel SCNR
(to kill the residual narrowband green cast), and a saturation adjustment.
SASpro implemented SCNR and saturation with hand-rolled Numba kernels; this
port reuses Astraios's own GPU-capable equivalents in
:mod:`astraios.core.color_tools` (``scnr`` / ``color_adjust``) instead of
duplicating that math, configured to reproduce SASpro's exact defaults
(full-strength average-neutral SCNR, no luminance preservation).

Data convention: narrowband channels and the broadband star frame are
channels-first, e.g. ``(3, H, W)`` for a stacked Ha/OIII/SII cube or an RGB
star frame, matching Astraios's ``(C, H, W)`` color convention. The output
is always a ``(3, H, W)`` RGB float32 image in ``[0, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NBStarColorParams:
    """Parameters for :func:`recombine_star_color`.

    Attributes:
        ratio: Ha:broadband-green blend weight for the output green channel,
            in ``[0, 1]``. ``1.0`` uses pure Ha; ``0.0`` uses pure broadband
            green.
        enable_star_stretch: Apply SASpro's non-linear "star stretch" boost
            (``t = 3**stretch_factor; out = t*x / ((t-1)*x + 1)``) after
            combining.
        stretch_factor: Exponent ``k`` in the star-stretch formula above.
            Higher values push faint signal (and star wings) up harder.
        saturation: HSV saturation multiplier applied after combining.
            ``1.0`` leaves color unchanged.
        apply_scnr: Apply green-channel SCNR (average-neutral) after
            combining, to remove the narrowband green cast. SASpro always
            applied this unconditionally; kept toggleable here.
        scnr_amount: SCNR strength in ``[0, 1]``. ``1.0`` reproduces
            SASpro's ``g = min(g, (r+b)/2)`` exactly.
        preserve_luminance: Whether SCNR rescales R/G/B to preserve
            perceived luminance. SASpro's original SCNR kernel did not do
            this; defaults to ``False`` to match.
    """

    ratio: float = 0.30
    enable_star_stretch: bool = True
    stretch_factor: float = 5.0
    saturation: float = 1.0
    apply_scnr: bool = True
    scnr_amount: float = 1.0
    preserve_luminance: bool = False


def _split_nb(nb_image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Split `nb_image` into (Ha, OIII|None, SII|None) planes.

    `nb_image` is either a single mono plane (treated as Ha only) or a
    channels-first stack where channel 0 = Ha, channel 1 = OIII (optional),
    channel 2 = SII (optional) -- SASpro's dialog let the user load any
    subset of Ha/OIII/SII independently.
    """
    nb = np.asarray(nb_image, dtype=np.float32)
    if nb.ndim == 2:
        return nb, None, None
    if nb.ndim == 3:
        n = nb.shape[0]
        if n < 1:
            raise ValueError(f"nb_image must have at least one channel; got shape {nb.shape}")
        ha = nb[0]
        oiii = nb[1] if n >= 2 else None
        sii = nb[2] if n >= 3 else None
        return ha, oiii, sii
    raise ValueError(f"nb_image must be (H, W) or (C, H, W); got shape {nb.shape}")


def _split_rgb_stars(rgb_stars: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split `rgb_stars` into (R, G, B) planes.

    Accepts a ``(3, H, W)`` (or more channels, extra ignored) broadband
    frame, or a ``(H, W)`` / ``(1, H, W)`` mono frame used as R=G=B (SASpro's
    fallback for a mono "OSC" input).
    """
    stars = np.asarray(rgb_stars, dtype=np.float32)
    if stars.ndim == 2:
        return stars, stars, stars
    if stars.ndim == 3:
        if stars.shape[0] == 1:
            m = stars[0]
            return m, m, m
        if stars.shape[0] >= 3:
            return stars[0], stars[1], stars[2]
    raise ValueError(
        f"rgb_stars must be (H, W) or (C, H, W) with >=1 channel; got shape {stars.shape}"
    )


def recombine_star_color(
    nb_image: np.ndarray,
    rgb_stars: np.ndarray,
    params: NBStarColorParams | None = None,
) -> np.ndarray:
    """Recombine a narrowband composite with natural broadband star color.

    Args:
        nb_image: ``(H, W)`` mono Ha plane, or a channels-first stack of
            narrowband planes: channel 0 = Ha, channel 1 = OIII (optional),
            channel 2 = SII (optional).
        rgb_stars: Broadband color frame supplying natural star color --
            typically a stars-only extraction of an OSC/RGB exposure.
            ``(3, H, W)`` (or more channels; extras ignored) or ``(H, W)``
            mono, channels-first, same spatial shape as `nb_image`.
        params: Recombination parameters. Defaults to
            :class:`NBStarColorParams`.

    Returns:
        ``(3, H, W)`` float32 RGB image in ``[0, 1]``.
    """
    if params is None:
        params = NBStarColorParams()

    ha, oiii, sii = _split_nb(nb_image)
    r, g, b = _split_rgb_stars(rgb_stars)

    if ha.shape != r.shape:
        raise ValueError(
            f"nb_image and rgb_stars must share spatial shape; got {ha.shape} vs {r.shape}"
        )

    ratio = float(np.clip(params.ratio, 0.0, 1.0))

    # Missing narrowband channels fall back to the corresponding broadband
    # star-frame channel (matches SASpro's "OSC present" branch, which is
    # the general case here since `rgb_stars` is always supplied).
    sii_eff = sii if sii is not None else r
    oiii_eff = oiii if oiii is not None else b

    r_out = 0.5 * r + 0.5 * sii_eff
    g_out = ratio * ha + (1.0 - ratio) * g
    b_out = oiii_eff

    rgb = np.clip(np.stack([r_out, g_out, b_out], axis=0), 0.0, 1.0).astype(np.float32)

    if params.enable_star_stretch:
        t = 3.0 ** float(params.stretch_factor)
        rgb = (t * rgb) / ((t - 1.0) * rgb + 1.0)
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

    if params.apply_scnr:
        from astraios.core.color_tools import SCNRMethod, SCNRParams, scnr

        rgb = scnr(
            rgb,
            SCNRParams(
                method=SCNRMethod.AVERAGE_NEUTRAL,
                amount=float(np.clip(params.scnr_amount, 0.0, 1.0)),
                preserve_luminance=params.preserve_luminance,
            ),
        )

    if abs(params.saturation - 1.0) > 1e-6:
        from astraios.core.color_tools import ColorAdjustParams, color_adjust

        rgb = color_adjust(rgb, ColorAdjustParams(saturation=float(params.saturation)))

    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


__all__ = ["NBStarColorParams", "recombine_star_color"]
