"""Diffraction Spikes — synthetic star diffraction spike generation.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro)'s
``astrospike_python`` procedural spike renderer, Copyright Franklin Marek,
GPL-3.0-or-later. https://github.com/setiastro/setiastrosuitepro

Detects bright stars and draws directional "blade" gradients (plus an
optional secondary spike set, soft flare glow, halo ring, and rainbow
overlay) radiating from each one, screen-blended onto the image — the classic
telescope-diffraction look. Star detection is delegated to Astraios's shared
``astraios.core.star_detection`` (SASpro's own SEP/flood-fill detectors are
not reused); the "Detection" knobs are therefore adapted to that detector's
parameters rather than copied verbatim (documented per-field below). Every
rendering knob from SASpro's ``SpikeConfig`` is preserved.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from astraios.core.device_manager import get_device_manager
from astraios.core.masks import Mask, apply_mask
from astraios.core.star_detection import detect_stars

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(f: float, m: str) -> None:
    pass


# Canvases at or above this pixel count use the GPU for compositing (when
# available).
# TODO(benchmark): first GPU-vs-CPU measurements were taken while the GPU was
# loaded with an LLM and are unreliable — re-measure on an idle GPU before
# deciding the final dispatch (see ROADMAP).
GPU_PIXEL_THRESHOLD = 4_000_000  # roughly 2000x2000


@dataclass
class DiffractionSpikeParams:
    """Settings for synthetic diffraction-spike rendering.

    Mirrors SASpro's ``astrospike_python.SpikeConfig`` field-for-field, except
    the four "Detection" fields, which are adapted to Astraios's shared star
    detector (see field docs).
    """

    # --- Detection (adapted to astraios.core.star_detection) ---
    detect_sigma_threshold: float = 5.0
    """Detection threshold in MAD-sigma units above the background median —
    replaces SASpro's own 1-100 UI threshold now that detection is delegated
    to the shared Astraios detector (higher = fewer, brighter-only stars)."""
    max_stars: int = 500
    """Maximum number of detected stars considered, brightest first."""
    star_amount: float = 100.0
    """Percentage (0-100) of the detected stars (brightest first) that
    actually receive spikes."""
    min_star_size: float = 0.0
    """0-100 percentile-style filter removing the smallest detected stars."""
    max_star_size: float = 100.0
    """0-100 percentile-style filter removing the largest detected stars
    (e.g. saturated blobs)."""

    # --- Main Spikes ---
    quantity: int = 4
    """Number of primary diffraction spikes drawn per star."""
    length: float = 300.0
    """Base length of the primary spikes (scaled by each star's radius)."""
    global_scale: float = 1.0
    """Overall multiplier applied to spike length and thickness."""
    angle: float = 45.0
    """Rotation angle, in degrees, of the primary spike pattern."""
    intensity: float = 1.0
    """Opacity/brightness of the primary spikes."""
    spike_width: float = 1.0
    """Thickness multiplier of the primary spikes."""
    sharpness: float = 0.5
    """Controls the brightness falloff shape along each spike (0-1)."""

    # --- Appearance ---
    color_saturation: float = 1.0
    """Saturation applied to the sampled star colour; values above 1.0
    hyper-saturate and shift the spike toward a pure hue."""
    hue_shift: float = 0.0
    """Hue rotation, in degrees, applied to the spike colour."""

    # --- Secondary Spikes ---
    secondary_intensity: float = 0.5
    """Opacity of the secondary (angularly offset) spike set; 0 disables it."""
    secondary_length: float = 120.0
    """Length of the secondary spikes (relative to ``length``)."""
    secondary_offset: float = 45.0
    """Angular offset, in degrees, of the secondary spikes from the primary."""

    # --- Soft Flare ---
    soft_flare_intensity: float = 3.0
    """Brightness of the soft radial glow drawn under the spikes."""
    soft_flare_size: float = 15.0
    """Size multiplier of the soft flare glow."""

    # --- Halo ---
    enable_halo: bool = False
    """Toggle the diffraction halo ring."""
    halo_intensity: float = 0.5
    """Brightness of the halo ring."""
    halo_scale: float = 5.0
    """Radius multiplier of the halo ring relative to the star's radius."""
    halo_width: float = 1.0
    """Thickness of the halo ring."""
    halo_blur: float = 0.5
    """Softness of the halo ring edge."""
    halo_saturation: float = 1.0
    """Saturation of the halo ring colour."""

    # --- Rainbow ---
    enable_rainbow: bool = False
    """Toggle a chromatic rainbow overlay on the primary spikes."""
    rainbow_spikes: bool = True
    """Apply the rainbow overlay to the primary spike set."""
    rainbow_spike_intensity: float = 0.8
    """Strength of the rainbow overlay."""
    rainbow_spike_frequency: float = 1.0
    """How many hue cycles are packed along the spike length."""
    rainbow_spike_length: float = 0.8
    """Fraction (0-1) of the spike length covered by the rainbow overlay."""


# ─── colour helpers (scalar, called once per star — not a hot path) ───────────


def _hsl_to_rgb(h: float, s: float, l: float) -> tuple[float, float, float]:  # noqa: E741
    if s == 0:
        return l, l, l

    def hue_to_rgb(p: float, q: float, t: float) -> float:
        if t < 0:
            t += 1
        if t > 1:
            t -= 1
        if t < 1 / 6:
            return p + (q - p) * 6 * t
        if t < 1 / 2:
            return q
        if t < 2 / 3:
            return p + (q - p) * (2 / 3 - t) * 6
        return p

    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q
    r = hue_to_rgb(p, q, h + 1 / 3)
    g = hue_to_rgb(p, q, h)
    b = hue_to_rgb(p, q, h - 1 / 3)
    return r, g, b


def _get_star_color(
    color_rgb: tuple[float, float, float],
    x: float,
    y: float,
    hue_shift: float,
    saturation_input: float,
    alpha: float,
) -> tuple[float, float, float, float]:
    """Derive a spike colour from a sampled star colour, hue shift, and saturation."""
    r, g, b = color_rgb
    max_c, min_c = max(r, g, b), min(r, g, b)
    l = (max_c + min_c) / 2.0  # noqa: E741
    h, s = 0.0, 0.0

    if max_c != min_c:
        d = max_c - min_c
        s = d / (2.0 - max_c - min_c) if l > 0.5 else d / (max_c + min_c)
        if max_c == r:
            h = (g - b) / d + (6.0 if g < b else 0.0)
        elif max_c == g:
            h = (b - r) / d + 2.0
        else:
            h = (r - g) / d + 4.0
        h /= 6.0
    else:
        # Colourless star (e.g. mono image): hash a stable hue from position.
        h = (x * 0.618 + y * 0.382) % 1.0

    new_h = (h * 360.0) + hue_shift
    boosted_s = min(1.0, s * 16.0)

    if saturation_input <= 1.0:
        final_s = boosted_s * saturation_input
        final_l = max(l, 0.65)
    else:
        hyper_factor = saturation_input - 1.0
        final_s = boosted_s + (1.0 - boosted_s) * hyper_factor
        base_l = max(l, 0.65)
        final_l = base_l + (0.5 - base_l) * hyper_factor

    final_s = max(0.0, min(1.0, final_s))
    final_l = max(0.4, min(0.95, final_l))
    final_h = (new_h % 360.0) / 360.0

    r_out, g_out, b_out = _hsl_to_rgb(final_h, final_s, final_l)
    return (r_out, g_out, b_out, alpha)


def _sample_star_color(
    img_rgb: np.ndarray, x: float, y: float, radius: float
) -> tuple[float, float, float]:
    """Average the image colour in a ring around a star (halo sampling)."""
    h, w = img_rgb.shape[:2]
    inner_r = max(1.0, radius * 1.5)
    outer_r = max(inner_r + 1.0, radius * 3.0)

    samples = 24
    theta = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    r_mid = (inner_r + outer_r) / 2.0
    xs = np.round(x + np.cos(theta) * r_mid).astype(np.int64)
    ys = np.round(y + np.sin(theta) * r_mid).astype(np.int64)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if not valid.any():
        # Fall back to a small patch directly on the star.
        x0, x1 = max(0, int(x - 3)), min(w, int(x + 4))
        y0, y1 = max(0, int(y - 3)), min(h, int(y + 4))
        if x1 <= x0 or y1 <= y0:
            return (1.0, 1.0, 1.0)
        patch = img_rgb[y0:y1, x0:x1, :]
        mean = patch.reshape(-1, patch.shape[-1]).mean(axis=0)
    else:
        mean = img_rgb[ys[valid], xs[valid], :].mean(axis=0)
    mx = float(np.max(mean))
    if mx < 1e-9:
        return (1.0, 1.0, 1.0)
    return (float(mean[0] / mx), float(mean[1] / mx), float(mean[2] / mx))


# ─── vectorized tile generators (numpy; shared by both GPU and CPU paths) ─────


def _blade_alpha(tile_half: int, angle_rad: float, length: float,
                  half_thick: float, sharpness: float) -> np.ndarray:
    """Directional gradient "blade" alpha map, brightest near the star, fading out."""
    sharpness = float(np.clip(sharpness, 1e-3, 0.999))
    ys, xs = np.mgrid[-tile_half:tile_half + 1, -tile_half:tile_half + 1].astype(np.float32)
    cos_t, sin_t = math.cos(angle_rad), math.sin(angle_rad)
    u = xs * cos_t + ys * sin_t
    v = -xs * sin_t + ys * cos_t
    length = max(length, 1e-6)
    t = np.clip(u / length, 0.0, 1.0)
    fade = np.where(
        t < sharpness,
        1.0 - (t / sharpness) * 0.2,
        0.8 * (1.0 - (t - sharpness) / max(1.0 - sharpness, 1e-6)),
    )
    fade = np.clip(fade, 0.0, 1.0)
    thickness_falloff = np.clip(1.0 - np.abs(v) / (half_thick + 1.0), 0.0, 1.0)
    in_range = (u >= 0.0) & (u <= length) & (np.abs(v) <= half_thick + 1.0)
    alpha = np.where(in_range, fade * thickness_falloff, 0.0)
    return alpha.astype(np.float32)


def _rainbow_tile(tile_half: int, angle_rad: float, length: float, half_thick: float,
                   frequency: float, rainbow_length: float,
                   intensity: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel rainbow colour + alpha map covering a fraction of a spike's length."""
    ys, xs = np.mgrid[-tile_half:tile_half + 1, -tile_half:tile_half + 1].astype(np.float32)
    cos_t, sin_t = math.cos(angle_rad), math.sin(angle_rad)
    u = xs * cos_t + ys * sin_t
    v = -xs * sin_t + ys * cos_t
    length = max(length, 1e-6)
    t = np.clip(u / length, 0.0, 1.0)
    in_range = (u >= 0.0) & (u <= length) & (np.abs(v) <= half_thick + 1.0) & (t <= rainbow_length)

    hue = np.mod(t * frequency, 1.0)
    s_arr = np.full_like(hue, 0.8, dtype=np.float32)
    l_arr = np.full_like(hue, 0.6, dtype=np.float32)
    r, g, b = _hsl_to_rgb_vec(hue, s_arr, l_arr)
    alpha = np.clip(intensity * 2.0 * (1.0 - t), 0.0, 1.0)
    alpha = np.where(in_range, alpha, 0.0).astype(np.float32)
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    return rgb, alpha


def _hsl_to_rgb_vec(
    h: np.ndarray, s: np.ndarray, l: np.ndarray  # noqa: E741
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized HSL -> RGB (matches ``_hsl_to_rgb`` for constant s, l)."""
    q = np.where(l < 0.5, l * (1.0 + s), l + s - l * s)
    p = 2.0 * l - q

    def hue2rgb(p: np.ndarray, q: np.ndarray, t: np.ndarray) -> np.ndarray:
        t = np.mod(t, 1.0)
        return np.where(
            t < 1 / 6, p + (q - p) * 6 * t,
            np.where(t < 1 / 2, q, np.where(t < 2 / 3, p + (q - p) * (2 / 3 - t) * 6, p)),
        )

    r = hue2rgb(p, q, h + 1 / 3)
    g = hue2rgb(p, q, h)
    b = hue2rgb(p, q, h - 1 / 3)
    return r, g, b


def _flare_alpha(tile_half: int) -> np.ndarray:
    """Radial soft-flare glow alpha map (three-zone falloff)."""
    ys, xs = np.mgrid[-tile_half:tile_half + 1, -tile_half:tile_half + 1].astype(np.float32)
    dist = np.sqrt(xs**2 + ys**2) / max(tile_half, 1e-6)
    alpha = np.where(
        dist <= 0.2, 1.0 - (dist / 0.2) * 0.6,
        np.where(dist <= 0.6, 0.4 - ((dist - 0.2) / 0.4) * 0.35,
                 np.maximum(0.0, 0.05 - ((dist - 0.6) / 0.4) * 0.05)),
    )
    alpha = np.where(dist <= 1.0, alpha, 0.0)
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def _halo_alpha(tile_half: int, r_halo: float, ring_width: float, halo_blur: float) -> np.ndarray:
    """Ring-shaped halo alpha map."""
    ys, xs = np.mgrid[-tile_half:tile_half + 1, -tile_half:tile_half + 1].astype(np.float32)
    r = np.sqrt(xs**2 + ys**2)
    dist_from_center = np.abs(r - r_halo) / (ring_width / 2.0 + 0.1)
    falloff = np.clip(1.0 - dist_from_center, 0.0, 1.0)
    falloff *= 1.0 - halo_blur * 0.5
    return falloff.astype(np.float32)


# ─── canvas compositing (numpy CPU / torch GPU, identical math) ───────────────


def _blit_screen_np(canvas: np.ndarray, alpha: np.ndarray, color: tuple[float, float, float],
                     cx: float, cy: float, tile_half: int) -> None:
    h, w = canvas.shape[:2]
    x0, y0 = int(round(cx)) - tile_half, int(round(cy)) - tile_half
    x1, y1 = x0 + alpha.shape[1], y0 + alpha.shape[0]
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(w, x1), min(h, y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return
    a = alpha[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    overlay = np.asarray(color, dtype=np.float32)[None, None, :] * a[..., None]
    patch = canvas[cy0:cy1, cx0:cx1, :]
    canvas[cy0:cy1, cx0:cx1, :] = 1.0 - (1.0 - patch) * (1.0 - overlay)


def _blit_screen_rgb_np(canvas: np.ndarray, alpha: np.ndarray, rgb: np.ndarray,
                         cx: float, cy: float, tile_half: int) -> None:
    h, w = canvas.shape[:2]
    x0, y0 = int(round(cx)) - tile_half, int(round(cy)) - tile_half
    x1, y1 = x0 + alpha.shape[1], y0 + alpha.shape[0]
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(w, x1), min(h, y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return
    a = alpha[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    c = rgb[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0, :]
    overlay = c * a[..., None]
    patch = canvas[cy0:cy1, cx0:cx1, :]
    canvas[cy0:cy1, cx0:cx1, :] = 1.0 - (1.0 - patch) * (1.0 - overlay)


def _blit_screen_torch(canvas: torch.Tensor, alpha: np.ndarray, color: tuple[float, float, float],
                        cx: float, cy: float, tile_half: int) -> None:
    h, w = canvas.shape[:2]
    x0, y0 = int(round(cx)) - tile_half, int(round(cy)) - tile_half
    x1, y1 = x0 + alpha.shape[1], y0 + alpha.shape[0]
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(w, x1), min(h, y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return
    a_np = alpha[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    a = torch.from_numpy(np.ascontiguousarray(a_np)).to(canvas.device, dtype=canvas.dtype)
    color_t = torch.tensor(color, device=canvas.device, dtype=canvas.dtype)
    overlay = color_t.view(1, 1, 3) * a.unsqueeze(-1)
    patch = canvas[cy0:cy1, cx0:cx1, :]
    canvas[cy0:cy1, cx0:cx1, :] = 1.0 - (1.0 - patch) * (1.0 - overlay)


def _blit_screen_rgb_torch(canvas: torch.Tensor, alpha: np.ndarray, rgb: np.ndarray,
                            cx: float, cy: float, tile_half: int) -> None:
    h, w = canvas.shape[:2]
    x0, y0 = int(round(cx)) - tile_half, int(round(cy)) - tile_half
    x1, y1 = x0 + alpha.shape[1], y0 + alpha.shape[0]
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(w, x1), min(h, y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return
    a_np = alpha[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    c_np = rgb[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0, :]
    a = torch.from_numpy(np.ascontiguousarray(a_np)).to(canvas.device, dtype=canvas.dtype)
    c = torch.from_numpy(np.ascontiguousarray(c_np)).to(canvas.device, dtype=canvas.dtype)
    overlay = c * a.unsqueeze(-1)
    patch = canvas[cy0:cy1, cx0:cx1, :]
    canvas[cy0:cy1, cx0:cx1, :] = 1.0 - (1.0 - patch) * (1.0 - overlay)


# ─── star selection ─────────────────────────────────────────────────────────


class _RenderStar:
    __slots__ = ("x", "y", "brightness", "radius", "color")

    def __init__(self, x: float, y: float, brightness: float, radius: float,
                 color: tuple[float, float, float]):
        self.x = x
        self.y = y
        self.brightness = brightness
        self.radius = radius
        self.color = color


def _select_stars(
    img_rgb: np.ndarray, image: np.ndarray, params: DiffractionSpikeParams
) -> list[_RenderStar]:
    field = detect_stars(
        image,
        max_stars=params.max_stars,
        sigma_threshold=params.detect_sigma_threshold,
    )
    stars = list(field.stars)  # brightest first

    # A non-zero percentage always spikes at least the single brightest star:
    # with few stars in frame, a small percentage floored to zero and the
    # tool silently did nothing. star_amount == 0 stays an explicit off.
    if params.star_amount <= 0 or not stars:
        limit = 0
    else:
        limit = max(1, int(len(stars) * (params.star_amount / 100.0)))
    stars = stars[:limit]

    if params.min_star_size > 0:
        internal_min = params.min_star_size * 0.02
        stars = [s for s in stars if max(s.fwhm / 2.0, 0.5) >= internal_min]

    internal_max = 96.0 + params.max_star_size * 0.04
    if internal_max < 100.0 and stars:
        sorted_by_r = sorted(stars, key=lambda s: s.fwhm, reverse=True)
        removal_pct = (100.0 - internal_max) / 100.0
        num_remove = int(len(sorted_by_r) * removal_pct)
        if num_remove > 0:
            remove_ids = {id(s) for s in sorted_by_r[:num_remove]}
            stars = [s for s in stars if id(s) not in remove_ids]

    out: list[_RenderStar] = []
    for s in stars:
        radius = max(1.0, s.fwhm / 2.0)
        brightness = float(np.clip(s.flux, 0.0, 1.0))
        color = _sample_star_color(img_rgb, s.x, s.y, radius)
        out.append(_RenderStar(s.x, s.y, brightness, radius, color))
    return out


# ─── main rendering pass ────────────────────────────────────────────────────


def _render_one_star(canvas, star: _RenderStar, params: DiffractionSpikeParams,
                      blit_solid, blit_rgb) -> None:
    """Render soft flare, spikes, and halo for one star into ``canvas``."""
    # Soft flare (independent of spike length gating, matches SASpro ordering).
    if params.soft_flare_intensity > 0:
        glow_r = star.radius * params.soft_flare_size * 0.4 + star.radius * 2.0
        if glow_r > 2:
            opacity = min(1.0, params.soft_flare_intensity * 0.8 * star.brightness)
            tile_half = max(2, int(round(glow_r)))
            alpha = _flare_alpha(tile_half) * opacity
            flare_color = _get_star_color(
                star.color, star.x, star.y, params.hue_shift, params.color_saturation, 1.0
            )[:3]
            blit_solid(canvas, alpha, flare_color, star.x, star.y, tile_half)

    radius_factor = math.pow(star.radius, 1.2)
    base_length = radius_factor * (params.length / 40.0) * params.global_scale
    thickness = max(0.5, star.radius * params.spike_width * 0.15 * params.global_scale)
    if base_length < 2:
        return

    main_angle_rad = math.radians(params.angle)
    sec_angle_rad = math.radians(params.angle + params.secondary_offset)
    half_thick = thickness / 2.0

    quantity = max(1, int(round(params.quantity)))

    if params.intensity > 0:
        color = _get_star_color(
            star.color, star.x, star.y, params.hue_shift, params.color_saturation, params.intensity
        )
        rainbow_str = (
            params.rainbow_spike_intensity
            if (params.enable_rainbow and params.rainbow_spikes)
            else 0.0
        )
        opacity_mult = 0.4 if rainbow_str > 0 else 1.0
        tile_half = max(2, int(math.ceil(base_length + half_thick + 2)))
        for i in range(quantity):
            theta = main_angle_rad + i * (2.0 * math.pi) / quantity
            alpha = _blade_alpha(tile_half, theta, base_length, half_thick, params.sharpness)
            blit_solid(canvas, alpha * (color[3] * opacity_mult), color[:3],
                       star.x, star.y, tile_half)

            if rainbow_str > 0:
                rgb, r_alpha = _rainbow_tile(
                    tile_half, theta, base_length, half_thick,
                    params.rainbow_spike_frequency, params.rainbow_spike_length, rainbow_str,
                )
                blit_rgb(canvas, r_alpha, rgb, star.x, star.y, tile_half)

    if params.secondary_intensity > 0:
        sec_color = _get_star_color(
            star.color, star.x, star.y, params.hue_shift, params.color_saturation,
            params.secondary_intensity,
        )
        sec_len = base_length * (params.secondary_length / max(params.length, 1e-6))
        sec_half_thick = half_thick * 0.6
        tile_half = max(2, int(math.ceil(sec_len + sec_half_thick + 2)))
        for i in range(quantity):
            theta = sec_angle_rad + i * (2.0 * math.pi) / quantity
            alpha = _blade_alpha(tile_half, theta, sec_len, sec_half_thick, params.sharpness)
            blit_solid(canvas, alpha * sec_color[3], sec_color[:3], star.x, star.y, tile_half)

    if params.enable_halo and params.halo_intensity > 0:
        classification_score = star.radius * star.brightness
        intensity_weight = min(1.0, classification_score / 10.0) ** 2
        if intensity_weight > 0.01:
            final_halo_intensity = params.halo_intensity * intensity_weight
            halo_color = _get_star_color(
                star.color, star.x, star.y, params.hue_shift, params.halo_saturation,
                final_halo_intensity,
            )
            r_halo = star.radius * params.halo_scale
            if r_halo > 0.5:
                ring_width = r_halo * params.halo_width * 0.15
                tile_half = max(2, int(math.ceil(r_halo + ring_width + 2)))
                alpha = _halo_alpha(tile_half, r_halo, ring_width, params.halo_blur)
                blit_solid(canvas, alpha * halo_color[3], halo_color[:3],
                           star.x, star.y, tile_half)


def _render_spikes_cpu(
    img_rgb: np.ndarray, stars: list[_RenderStar], params: DiffractionSpikeParams,
    progress: ProgressCallback,
) -> np.ndarray:
    canvas = img_rgb.copy()
    n = max(1, len(stars))
    for i, star in enumerate(stars):
        _render_one_star(canvas, star, params, _blit_screen_np, _blit_screen_rgb_np)
        if i % 32 == 0:
            progress(0.2 + 0.7 * (i / n), f"Rendering spikes… ({i}/{n})")
    return np.clip(canvas, 0.0, 1.0).astype(np.float32)


@torch.no_grad()
def _render_spikes_gpu(
    img_rgb: np.ndarray, stars: list[_RenderStar], params: DiffractionSpikeParams,
    dm, progress: ProgressCallback,
) -> np.ndarray:
    canvas = dm.from_numpy(np.ascontiguousarray(img_rgb, dtype=np.float32))
    n = max(1, len(stars))
    for i, star in enumerate(stars):
        _render_one_star(canvas, star, params, _blit_screen_torch, _blit_screen_rgb_torch)
        if i % 32 == 0:
            progress(0.2 + 0.7 * (i / n), f"Rendering spikes (GPU)… ({i}/{n})")
    return torch.clamp(canvas, 0.0, 1.0).cpu().numpy().astype(np.float32)


# ─── public entry point ─────────────────────────────────────────────────────


def render_spikes(
    data: np.ndarray,
    params: DiffractionSpikeParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback | None = None,
) -> np.ndarray:
    """Detect bright stars and synthesize diffraction spikes onto an image.

    Parameters
    ----------
    data : ndarray
        Image data, float32 in [0, 1]. Mono (H, W) or color (C, H, W).
    params : DiffractionSpikeParams, optional
        Spike-rendering settings.
    mask : Mask, optional
        Processing mask; ``result = processed * mask + original * (1 - mask)``.
    progress : callable, optional
        ``progress(fraction, message)`` callback.

    Returns
    -------
    ndarray
        Image with spikes composited, same shape and dtype as ``data``.
    """
    if params is None:
        params = DiffractionSpikeParams()
    progress = progress or _noop_progress
    dm = get_device_manager()

    img = np.asarray(data, dtype=np.float32)
    original = img
    is_color = img.ndim == 3

    if is_color:
        work = np.transpose(img, (1, 2, 0))
        img_rgb = work if work.shape[-1] >= 3 else np.repeat(work[..., :1], 3, axis=-1)
    else:
        img_rgb = np.stack([img, img, img], axis=-1)

    h, w = img_rgb.shape[:2]

    progress(0.02, "Detecting stars…")
    stars = _select_stars(img_rgb, img, params)

    if not stars:
        progress(1.0, "No stars detected — nothing to do.")
        return original.copy()

    progress(0.1, f"Rendering spikes for {len(stars)} stars…")
    use_gpu = dm.is_gpu and (h * w) >= GPU_PIXEL_THRESHOLD
    if use_gpu:
        try:
            canvas = _render_spikes_gpu(img_rgb, stars, params, dm, progress)
        except Exception:
            log.exception("GPU spike rendering failed, falling back to CPU")
            canvas = _render_spikes_cpu(img_rgb, stars, params, progress)
    else:
        canvas = _render_spikes_cpu(img_rgb, stars, params, progress)

    if is_color and img.shape[0] >= 3:
        out = np.transpose(canvas[..., :img.shape[0]] if img.shape[0] > 3 else canvas, (2, 0, 1))
        if img.shape[0] > 3:
            # Extra channels (e.g. alpha) pass through unchanged.
            out = np.concatenate([out, img[3:]], axis=0)
    elif is_color:
        # Mono-as-single-channel-3D input: collapse back via luminance.
        lum = 0.2126 * canvas[..., 0] + 0.7152 * canvas[..., 1] + 0.0722 * canvas[..., 2]
        out = lum[np.newaxis, ...]
    else:
        out = 0.2126 * canvas[..., 0] + 0.7152 * canvas[..., 1] + 0.0722 * canvas[..., 2]

    out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    progress(0.95, "Blending mask…")
    result = apply_mask(original, out, mask)
    progress(1.0, "Diffraction spikes complete")
    return result
