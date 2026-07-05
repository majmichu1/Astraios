"""Selective Color / Selective Luminance adjustments.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro) —
`selective_color.py` and `selective_luma.py` — Copyright Franklin Marek,
GPL-3.0-or-later. https://github.com/setiastro/setiastrosuitepro

Two related tools, both GPU-accelerated via the device manager:

* **Selective Color** — select pixels by hue (one or more bands on the hue
  circle, gated by chroma/lightness) and apply CMY/RGB/Luminance/
  Chroma-or-Saturation/Contrast adjustments only within that selection.
* **Selective Luminance** — the same adjustment set, but the selection is a
  luminance band instead of a hue band. Its contrast control uses an
  anchored sigmoid S-curve (anchored at the band edges) instead of a plain
  linear stretch, so the band boundaries never seam.

Images are float32 in [0, 1]; mono is (H, W), color is (C, H, W)
channels-first. SASpro's originals work in (H, W, 3); the mask and
adjustment math below is otherwise unchanged, only the channel axis moved.
Anything SASpro did with an imported/region mask is now the job of the
standard Astraios `mask: Mask | None` parameter, applied once at the end.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np
import torch

from astraios.core.device_manager import get_device_manager
from astraios.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


# Named hue-band presets (degrees on the 0-360 hue circle), matching SASpro.
HUE_PRESETS: dict[str, list[tuple[float, float]]] = {
    "Red": [(340.0, 360.0), (0.0, 15.0)],
    "Orange": [(15.0, 40.0)],
    "Yellow": [(40.0, 70.0)],
    "Green": [(70.0, 170.0)],
    "Cyan": [(170.0, 200.0)],
    "Blue": [(200.0, 270.0)],
    "Magenta": [(270.0, 340.0)],
}

# Named luminance-band presets (0..1), matching SASpro.
LUMINANCE_PRESETS: dict[str, tuple[float, float]] = {
    "Shadows": (0.00, 0.25),
    "Dark Mids": (0.20, 0.45),
    "Midtones": (0.35, 0.65),
    "Bright Mids": (0.55, 0.80),
    "Highlights": (0.75, 1.00),
}


@dataclass
class SelectiveColorParams:
    """Parameters for hue-selective color correction.

    The hue mask selects pixels whose hue falls in `hue_ranges` (degrees on
    the 0-360 hue circle; multiple ranges are unioned), gated by chroma and
    lightness, then CMY/RGB/Luminance/Chroma-or-Saturation/Contrast
    adjustments are applied only within that selection.
    """

    hue_ranges: list[tuple[float, float]] = field(
        default_factory=lambda: [(340.0, 360.0), (0.0, 15.0)]
    )  # union of forward hue arcs (deg, wrap-around) to select; default = "Red"
    invert_range: bool = False  # select the complement of hue_ranges instead
    smooth_deg: float = 10.0  # hue-band edge feather width, in degrees
    min_chroma: float = 0.05  # gate out near-gray pixels below this chroma (S*V)
    min_light: float = 0.0  # gate out pixels darker (V) than this
    max_light: float = 1.0  # gate out pixels brighter (V) than this
    shadows: float = 0.0  # fade the selection out below this luminance
    highlights: float = 1.0  # fade the selection out above this luminance
    shadow_highlight_balance: float = 0.5  # feather width for the shadow/highlight gates
    edge_blur: float = 0.0  # Gaussian blur (px) softening the final selection mask
    intensity: float = 1.0  # overall selection strength multiplier (0-2)
    cyan: float = 0.0  # -1..1: negative adds red, positive removes red
    magenta: float = 0.0  # -1..1: negative adds green, positive removes green
    yellow: float = 0.0  # -1..1: negative adds blue, positive removes blue
    red: float = 0.0  # -1..1 additive red offset
    green: float = 0.0  # -1..1 additive green offset
    blue: float = 0.0  # -1..1 additive blue offset
    luminance: float = 0.0  # -1..1 additive brightness offset
    chroma: float = 0.0  # -1..1 luminance-preserving colorfulness boost/cut
    saturation: float = 0.0  # -1..1 HSV-S multiplicative boost/cut (alt. to chroma)
    contrast: float = 0.0  # -1..1 linear contrast around 0.5
    use_chroma_mode: bool = True  # True: apply `chroma`; False: apply `saturation`


@dataclass
class SelectiveLumaParams:
    """Parameters for luminance-band-selective color correction.

    Pixels whose luminance falls within [lo, hi] (with feathered edges) are
    selected; CMY/RGB/Luminance/Chroma-or-Saturation/Contrast adjustments
    are applied only within that band. Contrast uses an anchored sigmoid
    S-curve anchored at [lo, hi] rather than a simple linear stretch, so the
    band's own edges are always preserved (never seam).
    """

    lo: float = 0.0  # lower luminance band edge, [0, 1]
    hi: float = 0.25  # upper luminance band edge, [0, 1] (default = "Shadows")
    smooth: float = 0.05  # feather width, as a fraction of the [0, 1] range
    invert: bool = False  # select the complement of [lo, hi] instead
    edge_blur: float = 5.0  # Gaussian blur (px) softening the final selection mask
    intensity: float = 1.0  # overall selection strength multiplier (0-2)
    cyan: float = 0.0
    magenta: float = 0.0
    yellow: float = 0.0
    red: float = 0.0
    green: float = 0.0
    blue: float = 0.0
    luminance: float = 0.0
    chroma: float = 0.0
    saturation: float = 0.0
    contrast: float = 0.0  # -1..1; below ~-0.667 the anchored S-curve turns non-monotonic
    use_chroma_mode: bool = True  # True: apply `chroma`; False: apply `saturation`


# ---------------------------------------------------------------------
# small shared helpers (CPU / numpy)
# ---------------------------------------------------------------------


def _ensure_rgb_chw(data: np.ndarray) -> np.ndarray:
    """Return a (3, H, W) float32 RGB view; mono is replicated x3."""
    a = np.clip(data.astype(np.float32, copy=False), 0.0, 1.0)
    if a.ndim == 2:
        return np.stack([a, a, a], axis=0)
    return np.ascontiguousarray(a[:3])


def _luminance_chw(rgb: np.ndarray) -> np.ndarray:
    return (0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]).astype(np.float32)


def _softstep_np(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _gaussian_blur_mask_np(mask: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return mask
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), float(radius))


def _rgb_to_hsv_np(rgb: np.ndarray) -> np.ndarray:
    """(3, H, W) RGB -> (3, H, W) HSV. H in degrees [0, 360), S, V in [0, 1]."""
    r, g, b = rgb[0], rgb[1], rgb[2]
    v = np.maximum(np.maximum(r, g), b)
    c = v - np.minimum(np.minimum(r, g), b)
    h = np.zeros_like(v)
    s = np.zeros_like(v)

    nz_v = v > 1e-10
    s[nz_v] = c[nz_v] / v[nz_v]

    nz_c = c > 1e-10
    is_r = nz_c & (v == r)
    is_g = nz_c & (v == g) & ~is_r
    is_b = nz_c & ~is_r & ~is_g

    h[is_r] = 60.0 * (((g[is_r] - b[is_r]) / c[is_r]) % 6.0)
    h[is_g] = 60.0 * ((b[is_g] - r[is_g]) / c[is_g] + 2.0)
    h[is_b] = 60.0 * ((r[is_b] - g[is_b]) / c[is_b] + 4.0)
    h = h % 360.0

    return np.stack([h, s, v], axis=0).astype(np.float32)


def _hsv_to_rgb_np(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[0], hsv[1], hsv[2]
    c = v * s
    h_prime = (h / 60.0) % 6.0
    x = c * (1 - np.abs(h_prime % 2.0 - 1))
    m = v - c

    r = np.zeros_like(v)
    g = np.zeros_like(v)
    b = np.zeros_like(v)
    for lo, hi, rv, gv, bv in (
        (0, 1, c, x, 0),
        (1, 2, x, c, 0),
        (2, 3, 0, c, x),
        (3, 4, 0, x, c),
        (4, 5, x, 0, c),
        (5, 6, c, 0, x),
    ):
        sel = (h_prime >= lo) & (h_prime < hi)
        r[sel] = (rv if np.isscalar(rv) else rv[sel]) + m[sel]
        g[sel] = (gv if np.isscalar(gv) else gv[sel]) + m[sel]
        b[sel] = (bv if np.isscalar(bv) else bv[sel]) + m[sel]

    return np.clip(np.stack([r, g, b], axis=0), 0.0, 1.0).astype(np.float32)


def _apply_chroma_boost_np(rgb: np.ndarray, m: np.ndarray, chroma: float) -> np.ndarray:
    """L-preserving chroma change: rgb' = Y + (rgb - Y) * (1 + chroma * m)."""
    y = _luminance_chw(rgb)[np.newaxis, :, :]
    d = rgb - y
    k = 1.0 + float(chroma) * m
    return np.clip(y + d * k[np.newaxis, :, :], 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------
# Selective Color: hue-band mask (numpy)
# ---------------------------------------------------------------------


def _hue_band_np(hdeg: np.ndarray, lo: float, hi: float, smooth_deg: float) -> np.ndarray:
    h = hdeg.astype(np.float32)
    lo = float(lo) % 360.0
    hi = float(hi) % 360.0
    length = (hi - lo) % 360.0
    if length <= 1e-6:
        return np.zeros_like(h, dtype=np.float32)

    s = float(max(smooth_deg, 0.0))
    fwd = (h - lo) % 360.0
    bwd = (lo - h) % 360.0

    band = np.zeros_like(h, dtype=np.float32)
    inside = fwd <= length
    band[inside] = 1.0

    if s > 1e-6:
        upper = (fwd > length) & (fwd < length + s)
        band[upper] = np.maximum(band[upper], 1.0 - (fwd[upper] - length) / s)
        lower = (bwd > 0) & (bwd < s)
        band[lower] = np.maximum(band[lower], 1.0 - bwd[lower] / s)

    return np.clip(band, 0.0, 1.0).astype(np.float32)


def _hue_mask_np(rgb: np.ndarray, p: SelectiveColorParams) -> np.ndarray:
    hsv = _rgb_to_hsv_np(rgb)
    hdeg, s_ch, v_ch = hsv[0], hsv[1], hsv[2]

    m = np.zeros_like(hdeg, dtype=np.float32)
    for lo, hi in p.hue_ranges:
        m = np.maximum(m, _hue_band_np(hdeg, lo, hi, p.smooth_deg))

    if p.invert_range:
        m = 1.0 - m

    if p.min_chroma > 0:
        chroma = (s_ch * v_ch).astype(np.float32)
        m = m * _softstep_np(chroma, float(p.min_chroma) * 0.7, float(p.min_chroma))
    if p.min_light > 0:
        m = m * (v_ch >= float(p.min_light)).astype(np.float32)
    if p.max_light < 1:
        m = m * (v_ch <= float(p.max_light)).astype(np.float32)

    return np.clip(m, 0.0, 1.0).astype(np.float32)


def _weight_shadows_highlights_np(
    mask: np.ndarray, rgb: np.ndarray, shadows: float, highlights: float, balance: float
) -> np.ndarray:
    luminance = _luminance_chw(rgb)
    w = np.ones_like(luminance, dtype=np.float32)
    feather = 0.08 + 0.12 * balance

    if shadows > 1e-3:
        s0 = max(0.0, shadows - feather)
        s1 = min(1.0, shadows + 1e-6)
        w = w * _softstep_np(luminance, s0, s1)
    if highlights < 0.999:
        h0 = max(0.0, highlights - 1e-6)
        h1 = min(1.0, highlights + feather)
        w = w * (1.0 - _softstep_np(luminance, h0, h1))

    return np.clip(mask * w, 0.0, 1.0)


# ---------------------------------------------------------------------
# Selective Luminance: luminance-band mask (numpy)
# ---------------------------------------------------------------------


def _lum_band_mask_np(
    luminance: np.ndarray, lo: float, hi: float, smooth: float, invert: bool
) -> np.ndarray:
    lo = float(np.clip(lo, 0.0, 1.0))
    hi = float(np.clip(hi, 0.0, 1.0))
    if lo > hi:
        lo, hi = hi, lo
    s = float(max(smooth, 0.0))

    mask = ((luminance >= lo) & (luminance <= hi)).astype(np.float32)
    if s > 1e-6:
        lower = (luminance >= lo - s) & (luminance < lo)
        mask[lower] = np.maximum(mask[lower], (luminance[lower] - (lo - s)) / s)
        upper = (luminance > hi) & (luminance <= hi + s)
        mask[upper] = np.maximum(mask[upper], 1.0 - (luminance[upper] - hi) / s)

    if invert:
        mask = 1.0 - mask

    return np.clip(mask, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------
# Shared color adjustments (numpy)
# ---------------------------------------------------------------------


def _band_contrast_np(
    out: np.ndarray, m: np.ndarray, con: float, band_lo: float, band_hi: float
) -> np.ndarray:
    """Anchored sigmoid S-curve contrast within [band_lo, band_hi]. con=0 is identity."""
    span = float(band_hi) - float(band_lo)
    if span < 1e-6 or abs(con) < 1e-4:
        return out

    k = abs(float(con)) * 6.0
    s_neg = 1.0 / (1.0 + math.exp(k))
    s_pos = 1.0 / (1.0 + math.exp(-k))
    s_rng = s_pos - s_neg

    out_f = out.astype(np.float32)
    t = (out_f - band_lo) / span
    u = 2.0 * t - 1.0
    raw = 1.0 / (1.0 + np.exp(-k * u))
    e = (raw - s_neg) / s_rng

    anchored = e if con > 0 else (2.0 * t - e)
    result = anchored * span + band_lo

    m3 = m[np.newaxis, :, :] if out_f.ndim == 3 else m
    blended = out_f * (1.0 - m3) + result * m3
    return np.clip(blended, 0.0, 1.0).astype(np.float32)


def _apply_color_adjustments_np(
    rgb: np.ndarray, mask01: np.ndarray, p: SelectiveColorParams
) -> np.ndarray:
    m = np.clip(mask01.astype(np.float32) * float(p.intensity), 0.0, 1.0)
    r, g, b = rgb[0], rgb[1], rgb[2]

    r = np.clip(r + (-p.cyan) * m, 0.0, 1.0)
    g = np.clip(g + (-p.magenta) * m, 0.0, 1.0)
    b = np.clip(b + (-p.yellow) * m, 0.0, 1.0)
    r = np.clip(r + p.red * m, 0.0, 1.0)
    g = np.clip(g + p.green * m, 0.0, 1.0)
    b = np.clip(b + p.blue * m, 0.0, 1.0)
    out = np.stack([r, g, b], axis=0)

    if any(abs(x) > 1e-6 for x in (p.luminance, p.chroma, p.saturation, p.contrast)):
        if abs(p.luminance) > 0:
            out = np.clip(out + p.luminance * m[np.newaxis, :, :], 0.0, 1.0)
        if abs(p.contrast) > 0:
            out = np.clip((out - 0.5) * (1.0 + p.contrast * m[np.newaxis, :, :]) + 0.5, 0.0, 1.0)
        if p.use_chroma_mode:
            if abs(p.chroma) > 0:
                out = _apply_chroma_boost_np(out, m, p.chroma)
        elif abs(p.saturation) > 0:
            hsv = _rgb_to_hsv_np(out)
            hsv[1] = np.clip(hsv[1] * (1.0 + p.saturation * m), 0.0, 1.0)
            out = _hsv_to_rgb_np(hsv)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _apply_luma_adjustments_np(
    rgb: np.ndarray, mask01: np.ndarray, p: SelectiveLumaParams, band_lo: float, band_hi: float
) -> np.ndarray:
    m = np.clip(mask01.astype(np.float32) * float(p.intensity), 0.0, 1.0)
    r, g, b = rgb[0], rgb[1], rgb[2]

    r = np.clip(r + (-p.cyan) * m, 0.0, 1.0)
    g = np.clip(g + (-p.magenta) * m, 0.0, 1.0)
    b = np.clip(b + (-p.yellow) * m, 0.0, 1.0)
    r = np.clip(r + p.red * m, 0.0, 1.0)
    g = np.clip(g + p.green * m, 0.0, 1.0)
    b = np.clip(b + p.blue * m, 0.0, 1.0)
    out = np.stack([r, g, b], axis=0)

    if any(abs(x) > 1e-6 for x in (p.luminance, p.chroma, p.saturation, p.contrast)):
        if abs(p.luminance) > 0:
            out = np.clip(out + p.luminance * m[np.newaxis, :, :], 0.0, 1.0)
        if abs(p.contrast) > 0:
            out = _band_contrast_np(out, m, p.contrast, band_lo, band_hi)
        if p.use_chroma_mode:
            if abs(p.chroma) > 0:
                out = _apply_chroma_boost_np(out, m, p.chroma)
        elif abs(p.saturation) > 0:
            hsv = _rgb_to_hsv_np(out)
            hsv[1] = np.clip(hsv[1] * (1.0 + p.saturation * m), 0.0, 1.0)
            out = _hsv_to_rgb_np(hsv)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------
# GPU mirrors (torch)
# ---------------------------------------------------------------------


def _make_gaussian_kernel_1d(sigma: float, device: torch.device) -> torch.Tensor:
    sigma = max(sigma, 0.5)
    ksize = int(np.ceil(sigma * 3)) * 2 + 1
    ksize = max(ksize, 3)
    x = torch.arange(ksize, dtype=torch.float32, device=device) - ksize // 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


@torch.no_grad()
def _gaussian_blur_gpu(channel: torch.Tensor, sigma: float) -> torch.Tensor:
    device = channel.device
    k1d = _make_gaussian_kernel_1d(sigma, device)
    pad = k1d.shape[0] // 2
    t = channel.unsqueeze(0).unsqueeze(0)
    t_padded = torch.nn.functional.pad(t, (pad, pad, pad, pad), mode="reflect")
    kh = k1d.reshape(1, 1, 1, -1)
    blurred = torch.nn.functional.conv2d(t_padded, kh, padding=0)
    kv = k1d.reshape(1, 1, -1, 1)
    blurred = torch.nn.functional.conv2d(blurred, kv, padding=0)
    return blurred.squeeze(0).squeeze(0)


@torch.no_grad()
def _softstep_gpu(x: torch.Tensor, edge0: float, edge1: float) -> torch.Tensor:
    t = torch.clamp((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3 - 2 * t)


@torch.no_grad()
def _rgb_to_hsv_gpu(rgb: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb[0], rgb[1], rgb[2]
    v = torch.maximum(torch.maximum(r, g), b)
    c = v - torch.minimum(torch.minimum(r, g), b)
    h = torch.zeros_like(v)
    s = torch.zeros_like(v)

    nz_v = v > 1e-10
    s[nz_v] = c[nz_v] / v[nz_v]

    nz_c = c > 1e-10
    is_r = nz_c & (v == r)
    is_g = nz_c & (v == g) & ~is_r
    is_b = nz_c & ~is_r & ~is_g

    h[is_r] = 60.0 * torch.remainder((g[is_r] - b[is_r]) / c[is_r], 6.0)
    h[is_g] = 60.0 * ((b[is_g] - r[is_g]) / c[is_g] + 2.0)
    h[is_b] = 60.0 * ((r[is_b] - g[is_b]) / c[is_b] + 4.0)
    h = torch.remainder(h, 360.0)

    return torch.stack([h, s, v], dim=0)


@torch.no_grad()
def _hsv_to_rgb_gpu(hsv: torch.Tensor) -> torch.Tensor:
    h, s, v = hsv[0], hsv[1], hsv[2]
    c = v * s
    h_prime = torch.remainder(h / 60.0, 6.0)
    x = c * (1 - torch.abs(torch.remainder(h_prime, 2.0) - 1))
    m = v - c
    z = torch.zeros_like(v)

    r = torch.zeros_like(v)
    g = torch.zeros_like(v)
    b = torch.zeros_like(v)
    for lo, hi, rv, gv, bv in (
        (0, 1, c, x, z),
        (1, 2, x, c, z),
        (2, 3, z, c, x),
        (3, 4, z, x, c),
        (4, 5, x, z, c),
        (5, 6, c, z, x),
    ):
        sel = (h_prime >= lo) & (h_prime < hi)
        r[sel] = rv[sel]
        g[sel] = gv[sel]
        b[sel] = bv[sel]

    return torch.clamp(torch.stack([r + m, g + m, b + m], dim=0), 0.0, 1.0)


@torch.no_grad()
def _apply_chroma_boost_gpu(rgb: torch.Tensor, m: torch.Tensor, chroma: float) -> torch.Tensor:
    y = (0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]).unsqueeze(0)
    d = rgb - y
    k = 1.0 + float(chroma) * m
    return torch.clamp(y + d * k.unsqueeze(0), 0.0, 1.0)


@torch.no_grad()
def _hue_band_gpu(hdeg: torch.Tensor, lo: float, hi: float, smooth_deg: float) -> torch.Tensor:
    lo = float(lo) % 360.0
    hi = float(hi) % 360.0
    length = (hi - lo) % 360.0
    if length <= 1e-6:
        return torch.zeros_like(hdeg)

    s = float(max(smooth_deg, 0.0))
    fwd = torch.remainder(hdeg - lo, 360.0)
    bwd = torch.remainder(lo - hdeg, 360.0)

    band = torch.zeros_like(hdeg)
    inside = fwd <= length
    band[inside] = 1.0

    if s > 1e-6:
        upper = (fwd > length) & (fwd < length + s)
        band[upper] = torch.maximum(band[upper], 1.0 - (fwd[upper] - length) / s)
        lower = (bwd > 0) & (bwd < s)
        band[lower] = torch.maximum(band[lower], 1.0 - bwd[lower] / s)

    return torch.clamp(band, 0.0, 1.0)


@torch.no_grad()
def _hue_mask_gpu(rgb: torch.Tensor, p: SelectiveColorParams) -> torch.Tensor:
    hsv = _rgb_to_hsv_gpu(rgb)
    hdeg, s_ch, v_ch = hsv[0], hsv[1], hsv[2]

    m = torch.zeros_like(hdeg)
    for lo, hi in p.hue_ranges:
        m = torch.maximum(m, _hue_band_gpu(hdeg, lo, hi, p.smooth_deg))

    if p.invert_range:
        m = 1.0 - m

    if p.min_chroma > 0:
        chroma = s_ch * v_ch
        m = m * _softstep_gpu(chroma, float(p.min_chroma) * 0.7, float(p.min_chroma))
    if p.min_light > 0:
        m = m * (v_ch >= float(p.min_light)).float()
    if p.max_light < 1:
        m = m * (v_ch <= float(p.max_light)).float()

    return torch.clamp(m, 0.0, 1.0)


@torch.no_grad()
def _weight_shadows_highlights_gpu(
    mask: torch.Tensor, rgb: torch.Tensor, shadows: float, highlights: float, balance: float
) -> torch.Tensor:
    luminance = 0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]
    w = torch.ones_like(luminance)
    feather = 0.08 + 0.12 * balance

    if shadows > 1e-3:
        s0 = max(0.0, shadows - feather)
        s1 = min(1.0, shadows + 1e-6)
        w = w * _softstep_gpu(luminance, s0, s1)
    if highlights < 0.999:
        h0 = max(0.0, highlights - 1e-6)
        h1 = min(1.0, highlights + feather)
        w = w * (1.0 - _softstep_gpu(luminance, h0, h1))

    return torch.clamp(mask * w, 0.0, 1.0)


@torch.no_grad()
def _lum_band_mask_gpu(
    luminance: torch.Tensor, lo: float, hi: float, smooth: float, invert: bool
) -> torch.Tensor:
    lo = float(min(max(lo, 0.0), 1.0))
    hi = float(min(max(hi, 0.0), 1.0))
    if lo > hi:
        lo, hi = hi, lo
    s = float(max(smooth, 0.0))

    mask = ((luminance >= lo) & (luminance <= hi)).float()
    if s > 1e-6:
        lower = (luminance >= lo - s) & (luminance < lo)
        mask[lower] = torch.maximum(mask[lower], (luminance[lower] - (lo - s)) / s)
        upper = (luminance > hi) & (luminance <= hi + s)
        mask[upper] = torch.maximum(mask[upper], 1.0 - (luminance[upper] - hi) / s)

    if invert:
        mask = 1.0 - mask

    return torch.clamp(mask, 0.0, 1.0)


@torch.no_grad()
def _band_contrast_gpu(
    out: torch.Tensor, m: torch.Tensor, con: float, band_lo: float, band_hi: float
) -> torch.Tensor:
    span = float(band_hi) - float(band_lo)
    if span < 1e-6 or abs(con) < 1e-4:
        return out

    k = abs(float(con)) * 6.0
    s_neg = 1.0 / (1.0 + math.exp(k))
    s_pos = 1.0 / (1.0 + math.exp(-k))
    s_rng = s_pos - s_neg

    t = (out - band_lo) / span
    u = 2.0 * t - 1.0
    raw = 1.0 / (1.0 + torch.exp(-k * u))
    e = (raw - s_neg) / s_rng

    anchored = e if con > 0 else (2.0 * t - e)
    result = anchored * span + band_lo

    m3 = m.unsqueeze(0) if out.ndim == 3 else m
    blended = out * (1.0 - m3) + result * m3
    return torch.clamp(blended, 0.0, 1.0)


@torch.no_grad()
def _apply_color_adjustments_gpu(
    rgb: torch.Tensor, mask01: torch.Tensor, p: SelectiveColorParams
) -> torch.Tensor:
    m = torch.clamp(mask01 * float(p.intensity), 0.0, 1.0)
    r, g, b = rgb[0], rgb[1], rgb[2]

    r = torch.clamp(r + (-p.cyan) * m, 0.0, 1.0)
    g = torch.clamp(g + (-p.magenta) * m, 0.0, 1.0)
    b = torch.clamp(b + (-p.yellow) * m, 0.0, 1.0)
    r = torch.clamp(r + p.red * m, 0.0, 1.0)
    g = torch.clamp(g + p.green * m, 0.0, 1.0)
    b = torch.clamp(b + p.blue * m, 0.0, 1.0)
    out = torch.stack([r, g, b], dim=0)

    if any(abs(x) > 1e-6 for x in (p.luminance, p.chroma, p.saturation, p.contrast)):
        if abs(p.luminance) > 0:
            out = torch.clamp(out + p.luminance * m.unsqueeze(0), 0.0, 1.0)
        if abs(p.contrast) > 0:
            out = torch.clamp((out - 0.5) * (1.0 + p.contrast * m.unsqueeze(0)) + 0.5, 0.0, 1.0)
        if p.use_chroma_mode:
            if abs(p.chroma) > 0:
                out = _apply_chroma_boost_gpu(out, m, p.chroma)
        elif abs(p.saturation) > 0:
            hsv = _rgb_to_hsv_gpu(out)
            hsv[1] = torch.clamp(hsv[1] * (1.0 + p.saturation * m), 0.0, 1.0)
            out = _hsv_to_rgb_gpu(hsv)

    return torch.clamp(out, 0.0, 1.0)


@torch.no_grad()
def _apply_luma_adjustments_gpu(
    rgb: torch.Tensor, mask01: torch.Tensor, p: SelectiveLumaParams, band_lo: float, band_hi: float
) -> torch.Tensor:
    m = torch.clamp(mask01 * float(p.intensity), 0.0, 1.0)
    r, g, b = rgb[0], rgb[1], rgb[2]

    r = torch.clamp(r + (-p.cyan) * m, 0.0, 1.0)
    g = torch.clamp(g + (-p.magenta) * m, 0.0, 1.0)
    b = torch.clamp(b + (-p.yellow) * m, 0.0, 1.0)
    r = torch.clamp(r + p.red * m, 0.0, 1.0)
    g = torch.clamp(g + p.green * m, 0.0, 1.0)
    b = torch.clamp(b + p.blue * m, 0.0, 1.0)
    out = torch.stack([r, g, b], dim=0)

    if any(abs(x) > 1e-6 for x in (p.luminance, p.chroma, p.saturation, p.contrast)):
        if abs(p.luminance) > 0:
            out = torch.clamp(out + p.luminance * m.unsqueeze(0), 0.0, 1.0)
        if abs(p.contrast) > 0:
            out = _band_contrast_gpu(out, m, p.contrast, band_lo, band_hi)
        if p.use_chroma_mode:
            if abs(p.chroma) > 0:
                out = _apply_chroma_boost_gpu(out, m, p.chroma)
        elif abs(p.saturation) > 0:
            hsv = _rgb_to_hsv_gpu(out)
            hsv[1] = torch.clamp(hsv[1] * (1.0 + p.saturation * m), 0.0, 1.0)
            out = _hsv_to_rgb_gpu(hsv)

    return torch.clamp(out, 0.0, 1.0)


# ---------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------


def _finish(data: np.ndarray, out_rgb: np.ndarray, is_mono: bool, mask: Mask | None) -> np.ndarray:
    if is_mono:
        # Fold RGB back to gray. When the adjustment left R/G/B perfectly equal
        # (e.g. all-zero/no-op adjustments, or channel-symmetric ones), take a
        # channel directly rather than the luminance formula: Rec.601 weights
        # (0.2989 + 0.5870 + 0.1140 = 0.9999) don't sum to exactly 1.0 in
        # float32, which would otherwise nudge an identity result off by ~1e-4.
        if np.array_equal(out_rgb[0], out_rgb[1]) and np.array_equal(out_rgb[1], out_rgb[2]):
            out = out_rgb[0]
        else:
            out = _luminance_chw(out_rgb)
    else:
        out = out_rgb
        if data.ndim == 3 and data.shape[0] > 3:
            out = np.concatenate([out, data[3:]], axis=0)
    return apply_mask(data, out.astype(np.float32), mask)


def apply_selective_color(
    data: np.ndarray,
    params: SelectiveColorParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    """Apply hue-selective color correction.

    Parameters
    ----------
    data : ndarray
        Image data, float32 in [0, 1]. Mono (H, W) or color (C, H, W).
        Mono input is processed as a replicated-gray RGB image and the
        result is folded back to luminance.
    params : SelectiveColorParams, optional
        Defaults to a no-op (all adjustment amounts 0).
    mask : Mask, optional
        Restrict the effect to a region; see `astraios.core.masks`.
    progress : callable, optional
        `progress(fraction, message)` callback.
    """
    if params is None:
        params = SelectiveColorParams()

    is_mono = data.ndim == 2
    rgb = _ensure_rgb_chw(data)
    dm = get_device_manager()

    progress(0.1, "Building hue selection mask…")
    if dm.is_gpu:
        rgb_t = dm.from_numpy(rgb)
        sel = _hue_mask_gpu(rgb_t, params)
        sel = _weight_shadows_highlights_gpu(
            sel, rgb_t, params.shadows, params.highlights, params.shadow_highlight_balance
        )
        if params.edge_blur > 0:
            sel = _gaussian_blur_gpu(sel, params.edge_blur)
        sel = torch.clamp(sel, 0.0, 1.0)
        progress(0.5, "Applying color adjustments…")
        out_t = _apply_color_adjustments_gpu(rgb_t, sel, params)
        out_rgb = out_t.cpu().numpy().astype(np.float32)
    else:
        sel = _hue_mask_np(rgb, params)
        sel = _weight_shadows_highlights_np(
            sel, rgb, params.shadows, params.highlights, params.shadow_highlight_balance
        )
        if params.edge_blur > 0:
            sel = _gaussian_blur_mask_np(sel, params.edge_blur)
        sel = np.clip(sel, 0.0, 1.0).astype(np.float32)
        progress(0.5, "Applying color adjustments…")
        out_rgb = _apply_color_adjustments_np(rgb, sel, params)

    progress(0.9, "Finalizing…")
    result = _finish(data, out_rgb, is_mono, mask)
    progress(1.0, "Selective color complete")
    return result


def apply_selective_luma(
    data: np.ndarray,
    params: SelectiveLumaParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    """Apply luminance-band-selective color correction.

    Parameters
    ----------
    data : ndarray
        Image data, float32 in [0, 1]. Mono (H, W) or color (C, H, W).
        Mono input is processed as a replicated-gray RGB image and the
        result is folded back to luminance.
    params : SelectiveLumaParams, optional
        Defaults to a no-op (all adjustment amounts 0).
    mask : Mask, optional
        Restrict the effect to a region; see `astraios.core.masks`.
    progress : callable, optional
        `progress(fraction, message)` callback.
    """
    if params is None:
        params = SelectiveLumaParams()

    is_mono = data.ndim == 2
    rgb = _ensure_rgb_chw(data)
    dm = get_device_manager()
    band_lo, band_hi = (params.hi, params.lo) if params.lo > params.hi else (params.lo, params.hi)

    progress(0.1, "Building luminance selection mask…")
    if dm.is_gpu:
        rgb_t = dm.from_numpy(rgb)
        luminance_t = 0.2989 * rgb_t[0] + 0.5870 * rgb_t[1] + 0.1140 * rgb_t[2]
        sel = _lum_band_mask_gpu(luminance_t, params.lo, params.hi, params.smooth, params.invert)
        if params.edge_blur > 0:
            sel = _gaussian_blur_gpu(sel, params.edge_blur)
        sel = torch.clamp(sel, 0.0, 1.0)
        progress(0.5, "Applying color adjustments…")
        out_t = _apply_luma_adjustments_gpu(rgb_t, sel, params, band_lo, band_hi)
        out_rgb = out_t.cpu().numpy().astype(np.float32)
    else:
        luminance = _luminance_chw(rgb)
        sel = _lum_band_mask_np(luminance, params.lo, params.hi, params.smooth, params.invert)
        if params.edge_blur > 0:
            sel = _gaussian_blur_mask_np(sel, params.edge_blur)
        sel = np.clip(sel, 0.0, 1.0).astype(np.float32)
        progress(0.5, "Applying color adjustments…")
        out_rgb = _apply_luma_adjustments_np(rgb, sel, params, band_lo, band_hi)

    progress(0.9, "Finalizing…")
    result = _finish(data, out_rgb, is_mono, mask)
    progress(1.0, "Selective luminance complete")
    return result
