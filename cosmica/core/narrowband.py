"""Narrowband Processing — palette mapping and continuum subtraction.

Maps narrowband filter data (Ha, OIII, SII) into RGB composites
using standard or custom palette matrices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import torch

from cosmica.core.device_manager import get_device_manager

log = logging.getLogger(__name__)


class NarrowbandPalette(Enum):
    # Linear permutation palettes (channels mapped directly to R/G/B)
    SHO = auto()  # Hubble palette: R=SII, G=Ha, B=OIII
    HOO = auto()  # R=Ha, G=OIII, B=OIII (bicolor)
    HOS = auto()  # R=Ha, G=OIII, B=SII
    HSO = auto()  # R=Ha, G=SII, B=OIII
    OHS = auto()  # R=OIII, G=Ha, B=SII
    OSH = auto()  # R=OIII, G=SII, B=Ha
    # Dynamic (nonlinear) palettes — the green channel is a per-pixel blend of
    # Ha and OIII weighted by OIII strength, instead of a fixed channel map.
    # This breaks up the flat green of a static SHO and is the basis of the
    # popular "Foraxx" look.
    FORAXX = auto()       # Foraxx-style: R=Ha, G=blend(Ha,OIII), B=OIII
    DYNAMIC_SHO = auto()  # SHO with the same dynamic green blend (R=SII)
    CUSTOM = auto()


# Predefined palette matrices: each row is [Ha_weight, OIII_weight, SII_weight]
# for [R, G, B] output channels
PALETTE_MATRICES = {
    NarrowbandPalette.SHO: np.array([
        [0.0, 0.0, 1.0],  # R = SII
        [1.0, 0.0, 0.0],  # G = Ha
        [0.0, 1.0, 0.0],  # B = OIII
    ], dtype=np.float32),
    NarrowbandPalette.HOO: np.array([
        [1.0, 0.0, 0.0],  # R = Ha
        [0.0, 1.0, 0.0],  # G = OIII
        [0.0, 1.0, 0.0],  # B = OIII
    ], dtype=np.float32),
    NarrowbandPalette.HOS: np.array([
        [1.0, 0.0, 0.0],  # R = Ha
        [0.0, 1.0, 0.0],  # G = OIII
        [0.0, 0.0, 1.0],  # B = SII
    ], dtype=np.float32),
    NarrowbandPalette.HSO: np.array([
        [1.0, 0.0, 0.0],  # R = Ha
        [0.0, 0.0, 1.0],  # G = SII
        [0.0, 1.0, 0.0],  # B = OIII
    ], dtype=np.float32),
    NarrowbandPalette.OHS: np.array([
        [0.0, 1.0, 0.0],  # R = OIII
        [1.0, 0.0, 0.0],  # G = Ha
        [0.0, 0.0, 1.0],  # B = SII
    ], dtype=np.float32),
    NarrowbandPalette.OSH: np.array([
        [0.0, 1.0, 0.0],  # R = OIII
        [0.0, 0.0, 1.0],  # G = SII
        [1.0, 0.0, 0.0],  # B = Ha
    ], dtype=np.float32),
}

# Palettes computed with a nonlinear (per-pixel) blend rather than a fixed matrix.
DYNAMIC_PALETTES = frozenset(
    {NarrowbandPalette.FORAXX, NarrowbandPalette.DYNAMIC_SHO}
)


@dataclass
class NarrowbandParams:
    """Parameters for narrowband combination."""

    palette: NarrowbandPalette = NarrowbandPalette.SHO
    custom_matrix: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float32)
    )
    normalize: bool = True  # normalize output to [0, 1]


def combine_narrowband(
    channels: dict[str, np.ndarray],
    params: NarrowbandParams | None = None,
) -> np.ndarray:
    """Combine narrowband channels into an RGB image.

    Parameters
    ----------
    channels : dict
        Mapping of filter names to 2D arrays. Expected keys: "ha", "oiii", "sii".
        At least "ha" and one other must be present.
    params : NarrowbandParams, optional
        Combination parameters.

    Returns
    -------
    ndarray
        RGB image of shape (3, H, W), values in [0, 1].
    """
    if params is None:
        params = NarrowbandParams()

    # Get channels, defaulting missing ones to zeros
    ha = channels.get("ha")
    oiii = channels.get("oiii")
    sii = channels.get("sii")

    if ha is None:
        raise ValueError("Ha channel is required for narrowband combination")

    h, w = ha.shape
    if oiii is None:
        oiii = np.zeros((h, w), dtype=np.float32)
    if sii is None:
        sii = np.zeros((h, w), dtype=np.float32)

    dm = get_device_manager()

    # Dynamic (nonlinear) palettes blend Ha/OIII per pixel.
    if params.palette in DYNAMIC_PALETTES:
        result = _combine_dynamic(ha, oiii, sii, params.palette, dm)
        if params.normalize:
            m = float(result.max())
            if m > 1e-10:
                result = result / m
        return np.clip(result, 0.0, 1.0).astype(np.float32)

    # Stack inputs: (3, H, W) — [Ha, OIII, SII]
    stack = np.stack([ha, oiii, sii], axis=0)  # (3, H, W)

    # Get palette matrix
    if params.palette == NarrowbandPalette.CUSTOM:
        matrix = params.custom_matrix
    else:
        matrix = PALETTE_MATRICES[params.palette]

    # Apply: result[c,H,W] = einsum("ci,ihw->chw", matrix, stack) on GPU
    with torch.no_grad():
        matrix_t = torch.from_numpy(matrix).to(dm.device)  # (3, 3)
        stack_t = torch.from_numpy(stack).to(dm.device)    # (3, H, W)
        result_t = torch.einsum("ci,ihw->chw", matrix_t, stack_t)
        if params.normalize:
            max_val = result_t.max()
            if max_val > 1e-10:
                result_t = result_t / max_val
        result = result_t.clamp(0, 1).cpu().numpy().astype(np.float32)

    return result


def _combine_dynamic(
    ha: np.ndarray,
    oiii: np.ndarray,
    sii: np.ndarray,
    palette: NarrowbandPalette,
    dm,
) -> np.ndarray:
    """Compute a dynamic (nonlinear) narrowband palette on the GPU.

    The green channel is a per-pixel blend of Ha and OIII weighted by
    ``factor = OIII**(1 - OIII)`` (the community "Foraxx" weighting):
    ``green = factor*Ha + (1 - factor)*OIII``. Red/blue depend on the palette:

    - ``FORAXX``:      R=Ha,  B=OIII
    - ``DYNAMIC_SHO``: R=SII, B=OIII

    Returns an unnormalised ``(3, H, W)`` float32 array.
    """
    with torch.no_grad():
        dev = dm.device
        h_t = torch.from_numpy(np.ascontiguousarray(ha)).to(dev)
        o_t = torch.from_numpy(np.ascontiguousarray(oiii)).to(dev)
        factor = torch.pow(o_t.clamp(0, 1), (1.0 - o_t.clamp(0, 1)))
        green = factor * h_t + (1.0 - factor) * o_t

        if palette == NarrowbandPalette.DYNAMIC_SHO:
            red = torch.from_numpy(np.ascontiguousarray(sii)).to(dev)
        else:  # FORAXX
            red = h_t
        blue = o_t

        result_t = torch.stack([red, green, blue], dim=0)
        return result_t.cpu().numpy().astype(np.float32)


def continuum_subtraction(
    narrowband: np.ndarray,
    broadband: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Subtract scaled broadband from narrowband to isolate emission.

    Parameters
    ----------
    narrowband : ndarray
        Narrowband channel, shape (H, W).
    broadband : ndarray
        Broadband channel (e.g., R for Ha), same shape.
    scale : float
        Scale factor for broadband before subtraction.

    Returns
    -------
    ndarray
        Emission-only image, clipped to [0, 1].
    """
    result = narrowband - scale * broadband
    return np.clip(result, 0, 1).astype(np.float32)
