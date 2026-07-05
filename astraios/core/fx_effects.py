"""FX Effects — creative post-processing effects for astrophotography.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro),
Copyright Franklin Marek, GPL-3.0-or-later.
https://github.com/setiastro/setiastrosuitepro

Six stylistic effects (Orton glow, soft focus, bloom/star glow, vignette,
film grain, split tone) selected via ``FXParams.effect`` and applied through
the single :func:`apply_fx` entry point, mirroring SASpro's ``fx_module``
effect registry. SASpro operates on channels-last ``(H, W, 3)`` arrays; here
the incoming Astraios ``(C, H, W)`` array is transposed to ``(H, W, C)`` once
on entry, processed with (adapted) SASpro math, then transposed back once on
exit — no per-operation transposing.
"""

from __future__ import annotations

import colorsys
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from astraios.core.device_manager import get_device_manager
from astraios.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(f: float, m: str) -> None:
    pass


# Frames at or above this pixel count use the GPU blur path (when a GPU is available).
GPU_PIXEL_THRESHOLD = 4_000_000  # roughly 2000x2000


class BlendMode(str, Enum):
    """Blend mode used to combine a glow layer with the base image."""

    SCREEN = "screen"
    SOFT_LIGHT = "soft_light"
    LIGHTEN = "lighten"


class FXEffect(str, Enum):
    """Selects which FX effect :func:`apply_fx` renders."""

    ORTON_GLOW = "orton_glow"
    SOFT_FOCUS = "soft_focus"
    BLOOM = "bloom"
    VIGNETTE = "vignette"
    FILM_GRAIN = "film_grain"
    SPLIT_TONE = "split_tone"


@dataclass
class FXParams:
    """Settings for every FX effect. Only the fields relevant to ``effect`` are used.

    Field groups mirror SASpro's ``fx_module.EFFECTS`` registry (one dataclass covers
    all six effects, matching the original's single dialog + effect-selector design).
    """

    effect: FXEffect = FXEffect.ORTON_GLOW

    # --- Orton Glow / shared blur-glow params (Orton Glow, Soft Focus, Bloom) ---
    blur_radius: float = 15.0
    """Size of the soft glow/blur halo, in pixels (Orton Glow, Soft Focus, Bloom)."""
    opacity: float = 0.5
    """Overall blend strength of the effect (Orton Glow, Soft Focus, Bloom)."""

    # --- Orton Glow only ---
    glow_brightness: float = 1.4
    """Brightness boost of the blurred duplicate before blending (Orton Glow)."""
    blend_mode: BlendMode = BlendMode.SCREEN
    """Blend mode combining the glow with the original (Orton Glow): Screen is
    brightest/hazy, Soft Light is gentler and protects shadows, Lighten shifts
    colour the least."""
    highlight_protect: float = 0.5
    """Fades the glow out near already-clipped highlights so they don't blow out
    further (Orton Glow)."""
    luma_recovery: float = 0.7
    """Rescales the blended result back toward the average luma of the two source
    layers, correcting the brightness overshoot inherent to screen-like blends
    (Orton Glow, Bloom)."""

    # --- Bloom / Star Glow only ---
    bloom_threshold: float = 0.7
    """Luminance above which pixels are treated as highlights to bloom (Bloom)."""
    bloom_brightness: float = 1.5
    """Brightness boost applied to isolated highlights before blending (Bloom)."""

    # --- Vignette ---
    vignette_amount: float = 0.5
    """Darkening strength at the frame edges (Vignette)."""
    vignette_radius: float = 1.0
    """Normalized distance from centre where the darkening falloff begins;
    1.0 is approximately the frame edge (Vignette)."""
    vignette_softness: float = 0.4
    """Softness of the vignette falloff transition (Vignette)."""

    # --- Film Grain ---
    grain_intensity: float = 0.3
    """Grain strength (Film Grain)."""
    grain_size: float = 1.0
    """Grain clump size — 0 is fine grain, higher is coarser (Film Grain)."""
    grain_mono: bool = True
    """True for monochrome grain, False for independent per-channel colour grain
    (Film Grain)."""

    # --- Split Tone ---
    shadow_hue: float = 220.0
    """Hue, in degrees, tinting the shadows (Split Tone)."""
    highlight_hue: float = 40.0
    """Hue, in degrees, tinting the highlights (Split Tone)."""
    tone_balance: float = 0.0
    """Shifts the shadow/highlight split point, range -1..1 (Split Tone)."""
    tone_strength: float = 0.3
    """Overall tint strength (Split Tone)."""


# ─── shared blend-mode primitives ──────────────────────────────────────────────


def _blend_screen(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return 1.0 - (1.0 - base) * (1.0 - blend)


def _blend_soft_light(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """W3C compositing-spec soft-light formula (matches Photoshop's Soft Light)."""
    d_lo = base - (1.0 - 2.0 * blend) * base * (1.0 - base)
    d_hi_inner = np.where(
        base <= 0.25,
        ((16.0 * base - 12.0) * base + 4.0) * base,
        np.sqrt(np.clip(base, 0.0, 1.0)),
    )
    d_hi = base + (2.0 * blend - 1.0) * (d_hi_inner - base)
    return np.where(blend <= 0.5, d_lo, d_hi)


def _blend_lighten(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return np.maximum(base, blend)


_BLEND_FUNCS = {
    BlendMode.SCREEN: _blend_screen,
    BlendMode.SOFT_LIGHT: _blend_soft_light,
    BlendMode.LIGHTEN: _blend_lighten,
}


def _luminance(image: np.ndarray) -> np.ndarray:
    """Luminance of an (H, W) or (H, W, C>=3) array (channels-last, internal use)."""
    if image.ndim == 3 and image.shape[2] >= 3:
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    return image.squeeze() if image.ndim == 3 else image


def _hue_to_rgb(hue_deg: float) -> np.ndarray:
    r, g, b = colorsys.hsv_to_rgb((hue_deg % 360.0) / 360.0, 1.0, 1.0)
    return np.array([r, g, b], dtype=np.float32)


def _recombine_luma(
    layer_a: np.ndarray, layer_b: np.ndarray, blended: np.ndarray, amount: float
) -> np.ndarray:
    """Pull a screen-like blend's brightness back toward the average source luma.

    Screen (and similar) blends of two near-identical images run away in
    brightness — screen(x, x) = 2x - x^2, always brighter than x. `amount`
    (0..1) is a dry/wet mix between the raw blend and the fully luma-corrected
    version.
    """
    if amount <= 0.001:
        return blended
    lum_a = _luminance(layer_a)
    lum_b = _luminance(layer_b)
    lum_target = 0.5 * (lum_a + lum_b)
    lum_blended = _luminance(blended)
    ratio = lum_target / (lum_blended + 1e-6)
    if blended.ndim == 3 and ratio.ndim == 2:
        ratio = ratio[:, :, None]
    corrected = np.clip(blended * ratio, 0.0, 1.0)
    return blended * (1.0 - amount) + corrected * amount


# ─── Gaussian blur: GPU (torch) and CPU (cv2) paths ────────────────────────────


def _kernel_size(sigma: float) -> int:
    return int(2 * round(3 * sigma) + 1) | 1


def _gaussian_blur_cpu(img: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(0.1, float(sigma))
    k = _kernel_size(sigma)
    return cv2.GaussianBlur(np.ascontiguousarray(img, dtype=np.float32), (k, k), sigma)


@torch.no_grad()
def _gaussian_blur_gpu(img: np.ndarray, sigma: float, dm) -> np.ndarray:
    sigma = max(0.1, float(sigma))
    k = _kernel_size(sigma)

    t = dm.from_numpy(np.ascontiguousarray(img, dtype=np.float32))
    squeeze_channel = t.ndim == 2
    if squeeze_channel:
        t = t.unsqueeze(-1)
    # (H, W, C) -> (1, C, H, W)
    t = t.permute(2, 0, 1).unsqueeze(0)
    c = t.shape[1]

    coords = torch.arange(k, dtype=torch.float32, device=dm.device) - (k - 1) / 2.0
    kernel1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel1d = kernel1d / kernel1d.sum()

    pad = k // 2
    kx = kernel1d.view(1, 1, 1, k).expand(c, 1, 1, k).contiguous()
    ky = kernel1d.view(1, 1, k, 1).expand(c, 1, k, 1).contiguous()

    t = F.pad(t, (pad, pad, 0, 0), mode="reflect")
    t = F.conv2d(t, kx, groups=c)
    t = F.pad(t, (0, 0, pad, pad), mode="reflect")
    t = F.conv2d(t, ky, groups=c)

    out = t.squeeze(0).permute(1, 2, 0)
    if squeeze_channel:
        out = out.squeeze(-1)
    return out.cpu().numpy().astype(np.float32)


def _gaussian(img: np.ndarray, sigma: float, dm) -> np.ndarray:
    """Dispatch to GPU or CPU Gaussian blur based on device availability and size."""
    if dm.is_gpu and img.size >= GPU_PIXEL_THRESHOLD:
        try:
            return _gaussian_blur_gpu(img, sigma, dm)
        except Exception:
            log.exception("GPU Gaussian blur failed, falling back to CPU")
    return _gaussian_blur_cpu(img, sigma)


# ─── effect implementations (channels-last internally) ────────────────────────


def _fx_orton_glow(img: np.ndarray, p: FXParams, dm) -> np.ndarray:
    """Classic Orton effect: blur + brighten a duplicate, blend back at partial opacity."""
    if p.opacity < 0.001:
        return img
    blurred = _gaussian(img, p.blur_radius, dm)
    boosted = np.clip(blurred * max(0.01, p.glow_brightness), 0.0, 1.0)
    blend_fn = _BLEND_FUNCS.get(p.blend_mode, _blend_screen)
    blended = np.clip(blend_fn(img, boosted), 0.0, 1.0)
    blended = _recombine_luma(img, boosted, blended, p.luma_recovery)

    if p.highlight_protect > 0.0:
        lum = _luminance(img)
        rolloff = np.clip((lum - 0.7) / 0.3, 0.0, 1.0)
        protect = 1.0 - p.highlight_protect * rolloff
        if img.ndim == 3 and protect.ndim == 2:
            protect = protect[:, :, None]
    else:
        protect = 1.0

    eff = p.opacity * protect
    return np.clip(img * (1.0 - eff) + blended * eff, 0.0, 1.0).astype(np.float32)


def _fx_soft_focus(img: np.ndarray, p: FXParams, dm) -> np.ndarray:
    """Gentle diffusion — straight blur/opacity mix, no brightening."""
    if p.opacity < 0.001:
        return img
    blurred = _gaussian(img, p.blur_radius, dm)
    out = img * (1.0 - p.opacity) + blurred * p.opacity
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _fx_bloom(img: np.ndarray, p: FXParams, dm) -> np.ndarray:
    """Isolate bright highlights, blur just those, screen them back — halo around stars/discs."""
    if p.opacity < 0.001:
        return img
    lum = _luminance(img)
    hl_mask = np.clip((lum - p.bloom_threshold) / max(1e-3, 1.0 - p.bloom_threshold), 0.0, 1.0)
    hl_mask3 = hl_mask[..., None] if img.ndim == 3 else hl_mask
    highlights = img * hl_mask3

    blurred = _gaussian(highlights, p.blur_radius, dm)
    boosted = np.clip(blurred * max(0.01, p.bloom_brightness), 0.0, 1.0)
    composite = np.clip(_blend_screen(img, boosted), 0.0, 1.0)
    composite = _recombine_luma(img, boosted, composite, p.luma_recovery)
    out = img * (1.0 - p.opacity) + composite * p.opacity
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _fx_vignette(img: np.ndarray, p: FXParams, dm) -> np.ndarray:
    """Radial darkening toward the frame edges."""
    if p.vignette_amount < 0.001:
        return img
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ny = (yy - cy) / max(cy, 1e-6)
    nx = (xx - cx) / max(cx, 1e-6)
    dist = np.sqrt(nx**2 + ny**2)

    lo = max(1e-3, p.vignette_radius - p.vignette_softness)
    hi = max(lo + 1e-3, p.vignette_radius + p.vignette_softness)
    t = np.clip((dist - lo) / (hi - lo), 0.0, 1.0)
    vig = 1.0 - p.vignette_amount * t
    vig3 = vig[..., None] if img.ndim == 3 else vig
    return np.clip(img * vig3, 0.0, 1.0).astype(np.float32)


def _fx_film_grain(img: np.ndarray, p: FXParams, dm) -> np.ndarray:
    """Add organic monochrome or colour grain. Fixed seed keeps the pattern stable."""
    if p.grain_intensity < 0.001:
        return img
    rng = np.random.RandomState(42)
    h, w = img.shape[:2]

    if p.grain_mono or img.ndim == 2:
        noise = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
        if p.grain_size > 0.01:
            sigma = max(0.3, p.grain_size)
            k = _kernel_size(sigma)
            noise = cv2.GaussianBlur(noise, (k, k), sigma)
            noise = noise / (noise.std() + 1e-6)
        noise3 = noise[..., None] if img.ndim == 3 else noise
    else:
        noise = rng.normal(0.0, 1.0, (h, w, 3)).astype(np.float32)
        if p.grain_size > 0.01:
            sigma = max(0.3, p.grain_size)
            k = _kernel_size(sigma)
            for c in range(3):
                noise[..., c] = cv2.GaussianBlur(noise[..., c], (k, k), sigma)
            noise = noise / (noise.std() + 1e-6)
        noise3 = noise

    out = img + noise3 * p.grain_intensity * 0.15
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _fx_split_tone(img: np.ndarray, p: FXParams, dm) -> np.ndarray:
    """Tint shadows and highlights with independent hues."""
    if p.tone_strength < 0.001 or img.ndim != 3 or img.shape[2] < 3:
        return img
    lum = _luminance(img)
    w_hi = np.clip(lum + p.tone_balance * 0.5, 0.0, 1.0)
    w_lo = 1.0 - w_hi

    shadow_rgb = _hue_to_rgb(p.shadow_hue)
    highlight_rgb = _hue_to_rgb(p.highlight_hue)
    tint = w_lo[..., None] * shadow_rgb + w_hi[..., None] * highlight_rgb

    out = img * (1.0 - p.tone_strength) + np.clip(img * 2.0 * tint, 0.0, 1.0) * p.tone_strength
    return np.clip(out, 0.0, 1.0).astype(np.float32)


_EFFECT_FUNCS: dict[FXEffect, Callable[[np.ndarray, FXParams, object], np.ndarray]] = {
    FXEffect.ORTON_GLOW: _fx_orton_glow,
    FXEffect.SOFT_FOCUS: _fx_soft_focus,
    FXEffect.BLOOM: _fx_bloom,
    FXEffect.VIGNETTE: _fx_vignette,
    FXEffect.FILM_GRAIN: _fx_film_grain,
    FXEffect.SPLIT_TONE: _fx_split_tone,
}


# ─── public entry point ─────────────────────────────────────────────────────────


def apply_fx(
    data: np.ndarray,
    params: FXParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback | None = None,
) -> np.ndarray:
    """Apply an FX effect to an Astraios image.

    Parameters
    ----------
    data : ndarray
        Image data, float32 in [0, 1]. Mono (H, W) or color (C, H, W).
    params : FXParams, optional
        Effect selection and settings. Defaults to Orton Glow.
    mask : Mask, optional
        Processing mask; ``result = processed * mask + original * (1 - mask)``.
    progress : callable, optional
        ``progress(fraction, message)`` callback.

    Returns
    -------
    ndarray
        Processed image, same shape and dtype as ``data``.
    """
    if params is None:
        params = FXParams()
    progress = progress or _noop_progress
    dm = get_device_manager()

    img = np.asarray(data, dtype=np.float32)
    original = img
    is_color = img.ndim == 3

    progress(0.05, f"FX — {params.effect.value}")
    work = np.transpose(img, (1, 2, 0)) if is_color else img

    fn = _EFFECT_FUNCS.get(params.effect)
    if fn is None:
        raise ValueError(f"Unknown FX effect: {params.effect}")

    progress(0.2, "Processing…")
    out_work = fn(work, params, dm)

    out = np.transpose(out_work, (2, 0, 1)) if is_color else out_work
    out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    progress(0.9, "Blending mask…")
    result = apply_mask(original, out, mask)
    progress(1.0, "FX complete")
    return result
