"""Debayering — convert raw OSC Bayer mosaic to color image.

Supports RGGB, BGGR, GRBG, GBRG patterns via OpenCV's optimized demosaicing.
Input: float32 (H, W) in [0, 1] with Bayer pattern
Output: float32 (3, H, W) in [0, 1]  — channels = R, G, B
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# cv2 debayer codes indexed by (pattern, method)
_CV2_BAYER_CODES: dict[str, int] = {}

def _get_cv2_code(pattern: str, method: str) -> int:
    """Return cv2 COLOR_Bayer*2BGR* constant."""
    import cv2

    # cv2 Bayer convention: named from top-left 2x2 quad
    # cv2 calls it by first two channels (e.g. BG = RGGB when read as BGR)
    # Mapping from FITS BAYERPAT → cv2 constant
    _bilinear = {
        "RGGB": cv2.COLOR_BayerBG2RGB,
        "BGGR": cv2.COLOR_BayerRG2RGB,
        "GRBG": cv2.COLOR_BayerGB2RGB,
        "GBRG": cv2.COLOR_BayerGR2RGB,
    }
    _vng = {
        "RGGB": cv2.COLOR_BayerBG2RGB_VNG,
        "BGGR": cv2.COLOR_BayerRG2RGB_VNG,
        "GRBG": cv2.COLOR_BayerGB2RGB_VNG,
        "GBRG": cv2.COLOR_BayerGR2RGB_VNG,
    }
    _ea = {
        "RGGB": cv2.COLOR_BayerBG2RGB_EA,
        "BGGR": cv2.COLOR_BayerRG2RGB_EA,
        "GRBG": cv2.COLOR_BayerGB2RGB_EA,
        "GBRG": cv2.COLOR_BayerGR2RGB_EA,
    }

    codes = {"bilinear": _bilinear, "vng": _vng, "ea": _ea}
    return codes.get(method, _bilinear)[pattern]


def detect_bayer_pattern(header: dict) -> str | None:
    """Extract Bayer pattern string from a FITS header dict.

    Checks BAYERPAT, COLORTYP, CFA-PAT, BAYER keywords.
    Returns e.g. 'RGGB', or None if not a Bayer image.
    """
    for kw in ("BAYERPAT", "COLORTYP", "CFA-PAT", "BAYER", "CFATYPE"):
        val = str(header.get(kw, "")).strip().upper()
        if val in ("RGGB", "BGGR", "GRBG", "GBRG"):
            return val
    return None


def debayer(
    data: np.ndarray,
    pattern: str = "RGGB",
    method: str = "vng",
) -> np.ndarray:
    """Debayer a raw OSC Bayer mosaic into a color image.

    Parameters
    ----------
    data : ndarray
        Input float32 (H, W) in [0, 1].  Must be the raw Bayer mosaic.
        If already color (ndim == 3), returned unchanged.
    pattern : str
        Bayer CFA pattern: 'RGGB' | 'BGGR' | 'GRBG' | 'GBRG'.
    method : str
        Demosaicing algorithm: 'bilinear' (fast), 'vng' (better, default),
        'ea' (edge-aware, best quality but slower).

    Returns
    -------
    ndarray
        float32 (3, H, W) color image in [0, 1].
    """
    import cv2

    if data.ndim == 3:
        log.debug("debayer: input already has %d channels, returning unchanged", data.shape[0])
        return data.astype(np.float32)

    if data.ndim != 2:
        raise ValueError(f"debayer: expected 2D input, got shape {data.shape}")

    pattern = pattern.upper().strip()
    if pattern not in ("RGGB", "BGGR", "GRBG", "GBRG"):
        log.warning("Unknown Bayer pattern '%s', defaulting to RGGB", pattern)
        pattern = "RGGB"

    method = method.lower()
    if method == "superpixel":
        return _debayer_superpixel(data, pattern)
    if method not in ("bilinear", "vng", "ea"):
        method = "vng"

    # OpenCV VNG and EA only support uint8; bilinear supports uint16.
    # For astrophotography, uint16 bilinear is preferred (preserves 16-bit precision).
    # VNG/EA work on uint8 (256 levels — adequate for OSC since stars are overexposed anyway).
    if method == "bilinear":
        data_in = (data.clip(0, 1) * 65535.0).astype(np.uint16)
        code = _get_cv2_code(pattern, "bilinear")
        rgb_out = cv2.cvtColor(data_in, code)  # (H, W, 3) uint16
        rgb = rgb_out.astype(np.float32) / 65535.0
    else:
        # VNG / EA: use uint8 (OpenCV limitation)
        data_in = (data.clip(0, 1) * 255.0).astype(np.uint8)
        code = _get_cv2_code(pattern, method)
        try:
            rgb_out = cv2.cvtColor(data_in, code)  # (H, W, 3) uint8
        except cv2.error:
            # Fallback to uint16 bilinear if VNG/EA not supported on this build
            log.warning("cv2 %s not available, falling back to uint16 bilinear", method)
            data_in16 = (data.clip(0, 1) * 65535.0).astype(np.uint16)
            code_bl = _get_cv2_code(pattern, "bilinear")
            rgb_out = cv2.cvtColor(data_in16, code_bl)
            rgb = rgb_out.astype(np.float32) / 65535.0
            result = np.transpose(rgb, (2, 0, 1))
            return result.astype(np.float32)
        rgb = rgb_out.astype(np.float32) / 255.0

    result = np.transpose(rgb, (2, 0, 1))  # (H,W,3) → (3,H,W)

    log.debug(
        "Debayered %s → shape %s using %s/%s",
        data.shape, result.shape, pattern, method,
    )
    return result.astype(np.float32)


def _debayer_superpixel(data: np.ndarray, pattern: str) -> np.ndarray:
    """Superpixel (2×2 bin) debayer — each 2×2 Bayer quad → one color pixel.

    Output is half the spatial resolution but has no interpolation artefacts
    (each channel uses only its actual sensor pixels).  Useful for severely
    underexposed frames where interpolation noise would dominate.
    """
    h, w = data.shape
    h2, w2 = h // 2, w // 2

    # Map pattern to the (row, col) offsets of each colour in the 2×2 quad
    _offsets = {
        "RGGB": {"R": (0, 0), "G1": (0, 1), "G2": (1, 0), "B": (1, 1)},
        "BGGR": {"B": (0, 0), "G1": (0, 1), "G2": (1, 0), "R": (1, 1)},
        "GRBG": {"G1": (0, 0), "R": (0, 1), "B": (1, 0), "G2": (1, 1)},
        "GBRG": {"G1": (0, 0), "B": (0, 1), "R": (1, 0), "G2": (1, 1)},
    }
    offs = _offsets.get(pattern, _offsets["RGGB"])

    r0, c0 = offs["R"]
    g1r, g1c = offs["G1"]
    g2r, g2c = offs["G2"]
    br, bc = offs["B"]

    out = np.empty((3, h2, w2), dtype=np.float32)
    out[0] = data[r0::2, c0::2][:h2, :w2]  # R
    out[1] = (data[g1r::2, g1c::2][:h2, :w2] + data[g2r::2, g2c::2][:h2, :w2]) * 0.5  # G avg
    out[2] = data[br::2, bc::2][:h2, :w2]  # B

    log.debug("Superpixel debayer %s → shape %s", data.shape, out.shape)
    return out
