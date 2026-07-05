"""Nebula Flythrough — cinematic zoom/fly-in video renderer.

Renders an MP4 video that flies from a wide view into an astrophoto along a
curved zoom/pan trajectory. Optionally composites an independent stars-only
layer (and a mid layer) over a starless layer so stars appear to have depth
relative to nebulosity (parallax), each layer animating its own zoom/pan path.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro),
Copyright Franklin Marek, GPL-3.0-or-later.
https://github.com/setiastro/setiastrosuitepro
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as func

from astraios.core.device_manager import get_device_manager

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


# ---------------------------------------------------------------------------
# Easing functions  (t in [0,1] -> eased t in [0,1])
# ---------------------------------------------------------------------------


def _ease_linear(t: float) -> float:
    return float(t)


def _ease_in(t: float) -> float:
    return float(t * t * t)


def _ease_out(t: float) -> float:
    t = 1.0 - t
    return float(1.0 - t * t * t)


def _ease_in_out(t: float) -> float:
    if t < 0.5:
        return float(4.0 * t * t * t)
    return float(1.0 - (-2.0 * t + 2.0) ** 3 / 2.0)


EASE_FUNCTIONS: dict[str, Callable[[float], float]] = {
    "Linear": _ease_linear,
    "Ease In": _ease_in,
    "Ease Out": _ease_out,
    "Ease In-Out": _ease_in_out,
}
EASE_PRESETS = tuple(EASE_FUNCTIONS)

# Blend modes available for compositing the stars/mid layer over the base.
LAYER_BLEND_MODES = (
    "Screen",
    "Add",
    "Average",
    "Max",
    "Multiply",
    "Overlay",
    "Soft Light",
    "Hard Light",
    "Difference",
    "Color Dodge",
    "Lighten",
    "Darken",
    "Threshold Mask",  # pixels above `threshold` sit on top of base; below are transparent
)

# Convenience working-resolution presets for UI dropdowns.
WORKING_RES_PRESETS: dict[str, tuple[int, int] | None] = {
    "Full (original)": None,
    "4K  (3840x2160)": (3840, 2160),
    "2K  (2560x1440)": (2560, 1440),
    "HD  (1920x1080)": (1920, 1080),
    "HD  (1280x720)": (1280, 720),
    "SD  (854x480)": (854, 480),
    "SD  (640x360)": (640, 360),
}

# Codec attempt order: (label, fourcc). render_flythrough always falls back to mp4v.
CODEC_OPTIONS: dict[str, str] = {
    "mp4v (MPEG-4, most compatible)": "mp4v",
    "avc1 (H.264, smaller files, needs codec support)": "avc1",
    "XVID (Xvid MPEG-4)": "XVID",
}


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


@dataclass
class LayerFxParams:
    """Optional per-layer optical effects. A magnitude of 0.0 disables an effect."""

    animate_effects: bool = True  # ramp effect strength with zoom velocity instead of constant
    depth_warp: float = 0.0  # luminance-driven per-pixel parallax depth warp strength, 0 disables
    depth_blur_sigma: float = 8.0  # gaussian blur sigma used to smooth the depth map
    depth_invert: bool = False  # invert the depth map (swap near/far)
    radial_stretch: float = 0.0  # radial (barrel/pincushion) edge stretch strength, 0 disables
    zoom_blur: float = 0.0  # warp-speed radial zoom-blur strength, 0 disables
    zoom_blur_samples: int = 12  # number of accumulation samples used for zoom blur
    chroma: float = 0.0  # chromatic aberration strength (R/B channel split), 0 disables
    threshold: float = 0.2  # "Threshold Mask" blend mode: luminance cutoff (only used by that mode)
    feather: float = 0.1  # "Threshold Mask" blend mode: feather width around the cutoff


@dataclass
class LayerTrajectoryParams:
    """Zoom/pan trajectory for one flythrough layer, eased over the clip duration."""

    zoom_start: float = 1.0  # zoom factor at t=0 (1.0 = full frame)
    zoom_end: float = 6.0  # zoom factor at t=1 (higher = deeper zoom-in)
    cx_start: float = 0.5  # crop-center x at t=0, fraction of image width [0,1]
    cy_start: float = 0.5  # crop-center y at t=0, fraction of image height [0,1]
    cx_end: float = 0.5  # crop-center x at t=1, fraction of image width [0,1]
    cy_end: float = 0.5  # crop-center y at t=1, fraction of image height [0,1]
    ease: str = "Ease In-Out"  # easing curve name, one of EASE_PRESETS
    fx: LayerFxParams = field(default_factory=LayerFxParams)  # optional optical effects


@dataclass
class OverlayLayerParams(LayerTrajectoryParams):
    """Trajectory plus how this layer composites onto the layer(s) beneath it."""

    blend_mode: str = "Screen"  # compositing mode, one of LAYER_BLEND_MODES
    opacity: float = 1.0  # compositing opacity, 0..1 (1.0 = fully opaque)


@dataclass
class FlythroughParams:
    """All settings for a rendered Nebula Flythrough video."""

    fps: int = 30  # output frame rate
    duration: float = 10.0  # clip duration in seconds
    out_width: int = 1920  # output video width in pixels
    out_height: int = 1080  # output video height in pixels
    codec: str = "mp4v"  # preferred fourcc; falls back to mp4v if unavailable
    starless: LayerTrajectoryParams = field(default_factory=LayerTrajectoryParams)
    stars: OverlayLayerParams = field(default_factory=OverlayLayerParams)
    mid: OverlayLayerParams = field(
        default_factory=lambda: OverlayLayerParams(zoom_start=1.0, zoom_end=1.0)
    )


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------


def _ensure_hwc3(arr: np.ndarray) -> np.ndarray:
    """Convert an Astraios image (mono (H,W) or color (C,H,W)) to HWC float32 [0,1]."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        hwc = np.stack([a, a, a], axis=-1)
    elif a.ndim == 3:
        c = a.shape[0]
        if c == 1:
            hwc = np.repeat(a[0][:, :, None], 3, axis=2)
        elif c >= 3:
            hwc = a[:3].transpose(1, 2, 0)
        else:
            raise ValueError(f"Unsupported channel count for flythrough layer: {a.shape}")
    else:
        raise ValueError(f"Unsupported image shape for flythrough layer: {a.shape}")
    return np.clip(hwc, 0.0, 1.0).astype(np.float32, copy=False)


def _velocity_ramp(ease_fn: Callable[[float], float], t: float, animate: bool) -> float:
    """Ramp factor in [0,1] proportional to instantaneous zoom velocity at t."""
    if not animate:
        return 1.0
    eps = 0.005
    t_lo = max(0.0, t - eps)
    t_hi = min(1.0, t + eps)
    vel = (ease_fn(t_hi) - ease_fn(t_lo)) / max(t_hi - t_lo, 1e-9)
    return float(np.clip(vel / 3.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Blend modes (CPU / GPU)
# ---------------------------------------------------------------------------


def _blend_layers_cpu(
    base: np.ndarray, layer: np.ndarray, mode: str, opacity: float, **kwargs: float
) -> np.ndarray:
    """Blend `layer` over `base` using `mode` at `opacity`. HxWx3 float32 in [0,1]."""
    b = np.clip(base.astype(np.float32, copy=False), 0.0, 1.0)
    m = np.clip(layer.astype(np.float32, copy=False), 0.0, 1.0)

    if mode == "Screen":
        blended = 1.0 - (1.0 - b) * (1.0 - m)
    elif mode == "Add":
        blended = np.clip(b + m, 0.0, 1.0)
    elif mode == "Average":
        blended = (b + m) * 0.5
    elif mode == "Multiply":
        blended = b * m
    elif mode == "Overlay":
        blended = np.where(b < 0.5, 2.0 * b * m, 1.0 - 2.0 * (1.0 - b) * (1.0 - m))
    elif mode == "Soft Light":
        blended = (1.0 - 2.0 * m) * b * b + 2.0 * m * b
    elif mode == "Hard Light":
        blended = np.where(m < 0.5, 2.0 * b * m, 1.0 - 2.0 * (1.0 - b) * (1.0 - m))
    elif mode == "Difference":
        blended = np.abs(b - m)
    elif mode == "Color Dodge":
        blended = np.where(m >= 1.0, 1.0, np.clip(b / np.maximum(1.0 - m, 1e-7), 0.0, 1.0))
    elif mode in ("Max", "Lighten"):
        blended = np.maximum(b, m)
    elif mode == "Darken":
        blended = np.minimum(b, m)
    elif mode == "Threshold Mask":
        thr = float(kwargs.get("threshold", 0.2))
        feather = max(float(kwargs.get("feather", 0.1)), 1e-4)
        lum = (0.2126 * m[..., 0] + 0.7152 * m[..., 1] + 0.0722 * m[..., 2])[..., None]
        alpha = np.clip((lum - thr) / feather, 0.0, 1.0)
        blended = b * (1.0 - alpha) + m * alpha
    else:
        blended = (b + m) * 0.5

    blended = np.clip(blended, 0.0, 1.0).astype(np.float32)
    if opacity < 1.0:
        blended = b + opacity * (blended - b)
    return np.clip(blended, 0.0, 1.0).astype(np.float32)


def _blend_layers_gpu(
    base_t: torch.Tensor, layer_t: torch.Tensor, mode: str, opacity: float, **kwargs: float
) -> torch.Tensor:
    """GPU equivalent of `_blend_layers_cpu`. Inputs are NCHW tensors."""
    b = torch.clamp(base_t, 0.0, 1.0)
    m = torch.clamp(layer_t, 0.0, 1.0)

    if mode == "Screen":
        blended = 1.0 - (1.0 - b) * (1.0 - m)
    elif mode == "Add":
        blended = torch.clamp(b + m, 0.0, 1.0)
    elif mode == "Average":
        blended = (b + m) * 0.5
    elif mode == "Multiply":
        blended = b * m
    elif mode == "Overlay":
        blended = torch.where(b < 0.5, 2.0 * b * m, 1.0 - 2.0 * (1.0 - b) * (1.0 - m))
    elif mode == "Soft Light":
        blended = (1.0 - 2.0 * m) * b * b + 2.0 * m * b
    elif mode == "Hard Light":
        blended = torch.where(m < 0.5, 2.0 * b * m, 1.0 - 2.0 * (1.0 - b) * (1.0 - m))
    elif mode == "Difference":
        blended = torch.abs(b - m)
    elif mode == "Color Dodge":
        blended = torch.clamp(
            torch.where(m >= 1.0, torch.ones_like(b), b / torch.clamp(1.0 - m, min=1e-7)),
            0.0,
            1.0,
        )
    elif mode in ("Max", "Lighten"):
        blended = torch.maximum(b, m)
    elif mode == "Darken":
        blended = torch.minimum(b, m)
    elif mode == "Threshold Mask":
        thr = float(kwargs.get("threshold", 0.2))
        feather = max(float(kwargs.get("feather", 0.1)), 1e-4)
        lum = 0.2126 * m[:, 0:1] + 0.7152 * m[:, 1:2] + 0.0722 * m[:, 2:3]
        alpha = torch.clamp((lum - thr) / feather, 0.0, 1.0)
        blended = b * (1.0 - alpha) + m * alpha
    else:
        blended = (b + m) * 0.5

    blended = torch.clamp(blended, 0.0, 1.0)
    if opacity < 1.0:
        blended = b + opacity * (blended - b)
    return torch.clamp(blended, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Depth map
# ---------------------------------------------------------------------------


def _build_depth_map(img_hwc: np.ndarray, blur_sigma: float, invert: bool) -> np.ndarray:
    """Luminance-derived, normalized [0,1] depth proxy used for parallax warp."""
    lum = (
        0.299 * img_hwc[:, :, 0] + 0.587 * img_hwc[:, :, 1] + 0.114 * img_hwc[:, :, 2]
    ).astype(np.float32)
    if blur_sigma > 0:
        lum = cv2.GaussianBlur(lum, (0, 0), float(blur_sigma))
    if invert:
        lum = 1.0 - lum
    lo, hi = float(lum.min()), float(lum.max())
    if hi > lo:
        lum = (lum - lo) / (hi - lo)
    return lum.astype(np.float32)


# ---------------------------------------------------------------------------
# CPU zoom-crop (torus wrap boundary — no edge streaks)
# ---------------------------------------------------------------------------


def _zoom_crop_cpu(
    img: np.ndarray, zoom: float, cx_frac: float, cy_frac: float, out_h: int, out_w: int
) -> np.ndarray:
    h, w = img.shape[:2]
    crop_w = max(1, int(round(w / zoom)))
    crop_h = max(1, int(round(h / zoom)))
    cx_px = cx_frac * w
    cy_px = cy_frac * h

    out_ys = np.linspace(cy_px - crop_h / 2.0, cy_px + crop_h / 2.0, out_h, dtype=np.float32)
    out_xs = np.linspace(cx_px - crop_w / 2.0, cx_px + crop_w / 2.0, out_w, dtype=np.float32)
    out_xs = (out_xs % w).reshape(1, -1).repeat(out_h, axis=0)
    out_ys = (out_ys % h).reshape(-1, 1).repeat(out_w, axis=1)
    return cv2.remap(
        img,
        out_xs.astype(np.float32),
        out_ys.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    ).astype(np.float32)


def _zoom_crop_depth_cpu(
    img: np.ndarray,
    depth_map: np.ndarray,
    zoom_base: float,
    depth_strength: float,
    cx_frac: float,
    cy_frac: float,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    if depth_strength < 1e-4:
        return _zoom_crop_cpu(img, zoom_base, cx_frac, cy_frac, out_h, out_w)

    h, w = img.shape[:2]
    cx_px = cx_frac * w
    cy_px = cy_frac * h

    cols = (np.arange(out_w, dtype=np.float32) + 0.5) / out_w - 0.5
    rows = (np.arange(out_h, dtype=np.float32) + 0.5) / out_h - 0.5
    u, v = np.meshgrid(cols, rows)

    crop_w = w / zoom_base
    crop_h = h / zoom_base
    base_src_x = cx_px + u * crop_w
    base_src_y = cy_px + v * crop_h

    sx = (base_src_x % w).astype(np.float32)
    sy = (base_src_y % h).astype(np.float32)
    dm_sampled = cv2.remap(
        depth_map, sx, sy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP
    )

    dx = u * crop_w
    dy = v * crop_h
    dist = np.sqrt(dx * dx + dy * dy)
    safe_dist = np.maximum(dist, 1e-6)
    nx = dx / safe_dist
    ny = dy / safe_dist

    dm_centered = dm_sampled - 0.5
    parallax_px = dm_centered * depth_strength * (zoom_base - 1.0) * 2.0

    final_src_x = ((base_src_x - nx * parallax_px) % w).astype(np.float32)
    final_src_y = ((base_src_y - ny * parallax_px) % h).astype(np.float32)

    return cv2.remap(
        img,
        final_src_x,
        final_src_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# CPU optical effects
# ---------------------------------------------------------------------------


def _radial_stretch_cpu(
    img: np.ndarray, strength: float, cx_frac: float, cy_frac: float
) -> np.ndarray:
    if abs(strength) < 1e-4:
        return img
    h, w = img.shape[:2]
    cx_px, cy_px = cx_frac * w, cy_frac * h
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    dx, dy = xg - cx_px, yg - cy_px
    half_diag = max(1.0, (w * w + h * h) ** 0.5 / 2.0)
    r = np.sqrt(dx * dx + dy * dy) / half_diag
    factor = strength * r * r
    src_x = (xg - dx * factor).astype(np.float32)
    src_y = (yg - dy * factor).astype(np.float32)
    return cv2.remap(
        img, src_x, src_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP
    ).astype(np.float32)


def _zoom_blur_cpu(
    img: np.ndarray, strength: float, cx_frac: float, cy_frac: float, samples: int
) -> np.ndarray:
    if strength < 1e-4:
        return img
    h, w = img.shape[:2]
    max_zoom_spread = 1.0 + strength * 0.25
    acc = np.zeros_like(img, dtype=np.float32)
    total_w = 0.0
    for i in range(samples):
        frac = i / max(1, samples - 1)
        zoom = 1.0 + (max_zoom_spread - 1.0) * frac
        cw = max(1, int(round(w / zoom)))
        ch = max(1, int(round(h / zoom)))
        cx_px, cy_px = cx_frac * w, cy_frac * h
        x0 = max(0, min(int(round(cx_px - cw / 2)), w - cw))
        y0 = max(0, min(int(round(cy_px - ch / 2)), h - ch))
        crop = img[y0 : y0 + ch, x0 : x0 + cw]
        resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        weight = 1.0 - frac * 0.6
        acc += resized * weight
        total_w += weight
    acc /= max(1e-8, total_w)
    blend = strength * 0.7
    return np.clip((1.0 - blend) * img + blend * acc, 0.0, 1.0).astype(np.float32)


def _chroma_cpu(img: np.ndarray, strength: float, cx_frac: float, cy_frac: float) -> np.ndarray:
    if abs(strength) < 1e-4:
        return img
    h, w = img.shape[:2]
    max_shift = strength * 0.03

    def _shift_channel(ch: int, zoom_factor: float) -> np.ndarray:
        cw = max(1, int(round(w / zoom_factor)))
        ch_h = max(1, int(round(h / zoom_factor)))
        cx_px, cy_px = cx_frac * w, cy_frac * h
        x0 = max(0, min(int(round(cx_px - cw / 2)), w - cw))
        y0 = max(0, min(int(round(cy_px - ch_h / 2)), h - ch_h))
        return cv2.resize(
            img[y0 : y0 + ch_h, x0 : x0 + cw, ch], (w, h), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

    out = img.copy()
    r_zoom = 1.0 - max_shift
    b_zoom = 1.0 + max_shift
    if r_zoom > 0.01:
        out[:, :, 0] = _shift_channel(0, r_zoom)
    out[:, :, 2] = _shift_channel(2, b_zoom)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _apply_fx_cpu(
    frame: np.ndarray, ramp: float, cx_frac: float, cy_frac: float, fx: LayerFxParams
) -> np.ndarray:
    out = frame
    rs = fx.radial_stretch * ramp
    if abs(rs) > 1e-4:
        out = _radial_stretch_cpu(out, rs, cx_frac, cy_frac)
    zb = fx.zoom_blur * ramp
    if zb > 1e-4:
        out = _zoom_blur_cpu(out, zb, cx_frac, cy_frac, max(2, int(fx.zoom_blur_samples)))
    ca = fx.chroma * ramp
    if abs(ca) > 1e-4:
        out = _chroma_cpu(out, ca, cx_frac, cy_frac)
    return out


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------


def _hwc_to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(arr)).to(device)
    return t.permute(2, 0, 1).unsqueeze(0).float()


def _tensor_to_hwc(t: torch.Tensor) -> np.ndarray:
    return t.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)


def _depth_to_tensor(dm: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(dm)).to(device).unsqueeze(0).unsqueeze(0).float()


def _wrap_norm_coords(coords: torch.Tensor) -> torch.Tensor:
    """Fold normalised grid_sample coords (range [-1,1]) into a torus wrap."""
    return ((coords + 1.0) % 2.0) - 1.0


def _zoom_crop_gpu(
    t: torch.Tensor, zoom: float, cx_frac: float, cy_frac: float, out_h: int, out_w: int
) -> torch.Tensor:
    _, _, h, w = t.shape
    cx_n = float(cx_frac * 2.0 - 1.0)
    cy_n = float(cy_frac * 2.0 - 1.0)
    hx = 1.0 / max(zoom, 1e-4)
    hy = 1.0 / max(zoom, 1e-4)

    gx = torch.linspace(cx_n - hx, cx_n + hx, out_w, device=t.device, dtype=torch.float32)
    gy = torch.linspace(cy_n - hy, cy_n + hy, out_h, device=t.device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
    grid_x = _wrap_norm_coords(grid_x)
    grid_y = _wrap_norm_coords(grid_y)

    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    return func.grid_sample(t, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def _zoom_crop_depth_gpu(
    t: torch.Tensor,
    depth_t: torch.Tensor | None,
    zoom_base: float,
    depth_strength: float,
    cx_frac: float,
    cy_frac: float,
    out_h: int,
    out_w: int,
) -> torch.Tensor:
    if depth_t is None or depth_strength < 1e-4:
        return _zoom_crop_gpu(t, zoom_base, cx_frac, cy_frac, out_h, out_w)

    _, _, h, w = t.shape
    dev = t.device

    cx_px = cx_frac * w
    cy_px = cy_frac * h
    crop_w = w / zoom_base
    crop_h = h / zoom_base

    cols = (torch.arange(out_w, device=dev, dtype=torch.float32) + 0.5) / out_w - 0.5
    rows = (torch.arange(out_h, device=dev, dtype=torch.float32) + 0.5) / out_h - 0.5
    v_g, u_g = torch.meshgrid(rows, cols, indexing="ij")

    base_src_x = cx_px + u_g * crop_w
    base_src_y = cy_px + v_g * crop_h

    norm_bx = _wrap_norm_coords((base_src_x % w) / max(w - 1, 1) * 2.0 - 1.0)
    norm_by = _wrap_norm_coords((base_src_y % h) / max(h - 1, 1) * 2.0 - 1.0)
    depth_grid = torch.stack([norm_bx, norm_by], dim=-1).unsqueeze(0)
    dm_sampled = (
        func.grid_sample(
            depth_t, depth_grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        .squeeze(0)
        .squeeze(0)
    )

    dx = u_g * crop_w
    dy = v_g * crop_h
    dist = torch.sqrt(dx * dx + dy * dy).clamp(min=1e-6)
    nx = dx / dist
    ny = dy / dist

    dm_centered = dm_sampled - 0.5
    parallax_px = dm_centered * depth_strength * (zoom_base - 1.0) * 2.0

    final_src_x = (base_src_x - nx * parallax_px) % w
    final_src_y = (base_src_y - ny * parallax_px) % h

    norm_fx = _wrap_norm_coords(final_src_x / max(w - 1, 1) * 2.0 - 1.0)
    norm_fy = _wrap_norm_coords(final_src_y / max(h - 1, 1) * 2.0 - 1.0)
    grid = torch.stack([norm_fx, norm_fy], dim=-1).unsqueeze(0)

    return func.grid_sample(t, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def _radial_stretch_gpu(
    t: torch.Tensor, strength: float, cx_frac: float, cy_frac: float
) -> torch.Tensor:
    if abs(strength) < 1e-4:
        return t
    _, _, h, w = t.shape
    dev = t.device
    xs = torch.linspace(-1, 1, w, device=dev, dtype=torch.float32)
    ys = torch.linspace(-1, 1, h, device=dev, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    cx_n = cx_frac * 2.0 - 1.0
    cy_n = cy_frac * 2.0 - 1.0
    dx, dy = grid_x - cx_n, grid_y - cy_n
    half_diag = (2.0**2 + 2.0**2) ** 0.5 / 2.0
    r = torch.sqrt(dx * dx + dy * dy) / half_diag
    factor = strength * r * r
    src_x = grid_x - dx * factor
    src_y = grid_y - dy * factor
    grid = torch.stack([_wrap_norm_coords(src_x), _wrap_norm_coords(src_y)], dim=-1).unsqueeze(0)
    return func.grid_sample(t, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def _zoom_blur_gpu(
    t: torch.Tensor, strength: float, cx_frac: float, cy_frac: float, samples: int
) -> torch.Tensor:
    if strength < 1e-4:
        return t
    _, _, h, w = t.shape
    max_spread = 1.0 + strength * 0.25
    acc = torch.zeros_like(t)
    total_w = 0.0
    for i in range(samples):
        frac = i / max(1, samples - 1)
        zoom = 1.0 + (max_spread - 1.0) * frac
        weight = 1.0 - frac * 0.6
        acc = acc + _zoom_crop_gpu(t, zoom, cx_frac, cy_frac, h, w) * weight
        total_w += weight
    acc = acc / max(1e-8, total_w)
    blend = strength * 0.7
    return torch.clamp((1.0 - blend) * t + blend * acc, 0.0, 1.0)


def _chroma_gpu(t: torch.Tensor, strength: float, cx_frac: float, cy_frac: float) -> torch.Tensor:
    if abs(strength) < 1e-4:
        return t
    _, _, h, w = t.shape
    dev = t.device
    max_shift = strength * 0.03
    r_zoom = 1.0 - max_shift
    b_zoom = 1.0 + max_shift

    def _channel_grid(zoom_factor: float) -> torch.Tensor:
        cw = w / zoom_factor
        ch = h / zoom_factor
        cx_px = cx_frac * w
        cy_px = cy_frac * h
        x0 = float(np.clip(cx_px - cw / 2.0, 0.0, w - cw))
        y0 = float(np.clip(cy_px - ch / 2.0, 0.0, h - ch))
        out_xs = torch.arange(w, device=dev, dtype=torch.float32)
        src_xs = x0 + out_xs * (cw / w)
        norm_xs = (src_xs / max(w - 1, 1)) * 2.0 - 1.0
        out_ys = torch.arange(h, device=dev, dtype=torch.float32)
        src_ys = y0 + out_ys * (ch / h)
        norm_ys = (src_ys / max(h - 1, 1)) * 2.0 - 1.0
        grid_x = norm_xs.unsqueeze(0).expand(h, w)
        grid_y = norm_ys.unsqueeze(1).expand(h, w)
        return torch.stack(
            [_wrap_norm_coords(grid_x), _wrap_norm_coords(grid_y)], dim=-1
        ).unsqueeze(0)

    out = t.clone()
    if r_zoom > 0.01:
        out[:, 0:1] = func.grid_sample(
            t[:, 0:1], _channel_grid(r_zoom), mode="bilinear", padding_mode="zeros",
            align_corners=True,
        )
    out[:, 2:3] = func.grid_sample(
        t[:, 2:3], _channel_grid(b_zoom), mode="bilinear", padding_mode="zeros",
        align_corners=True,
    )
    return torch.clamp(out, 0.0, 1.0)


def _apply_fx_gpu(
    t: torch.Tensor, ramp: float, cx_frac: float, cy_frac: float, fx: LayerFxParams
) -> torch.Tensor:
    out = t
    rs = fx.radial_stretch * ramp
    if abs(rs) > 1e-4:
        out = _radial_stretch_gpu(out, rs, cx_frac, cy_frac)
    zb = fx.zoom_blur * ramp
    if zb > 1e-4:
        out = _zoom_blur_gpu(out, zb, cx_frac, cy_frac, max(2, int(fx.zoom_blur_samples)))
    ca = fx.chroma * ramp
    if abs(ca) > 1e-4:
        out = _chroma_gpu(out, ca, cx_frac, cy_frac)
    return out


# ---------------------------------------------------------------------------
# Per-layer frame combinators
# ---------------------------------------------------------------------------


def _layer_transform(traj: LayerTrajectoryParams, t: float) -> tuple[float, float, float, float]:
    """Returns (zoom, cx, cy, fx_ramp) for this layer at normalised time t."""
    ease_fn = EASE_FUNCTIONS.get(traj.ease, _ease_in_out)
    te = ease_fn(t)
    zoom = traj.zoom_start + (traj.zoom_end - traj.zoom_start) * te
    cx = traj.cx_start + (traj.cx_end - traj.cx_start) * te
    cy = traj.cy_start + (traj.cy_end - traj.cy_start) * te
    ramp = _velocity_ramp(ease_fn, t, traj.fx.animate_effects)
    return zoom, cx, cy, ramp


def _layer_frame_cpu(
    img_hwc: np.ndarray,
    depth_map: np.ndarray | None,
    traj: LayerTrajectoryParams,
    t: float,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    zoom, cx, cy, ramp = _layer_transform(traj, t)
    if depth_map is not None and traj.fx.depth_warp > 1e-4:
        frame = _zoom_crop_depth_cpu(
            img_hwc, depth_map, zoom, traj.fx.depth_warp, cx, cy, out_h, out_w
        )
    else:
        frame = _zoom_crop_cpu(img_hwc, zoom, cx, cy, out_h, out_w)
    return _apply_fx_cpu(frame, ramp, cx, cy, traj.fx)


def _layer_frame_gpu(
    img_t: torch.Tensor,
    depth_t: torch.Tensor | None,
    traj: LayerTrajectoryParams,
    t: float,
    out_h: int,
    out_w: int,
) -> torch.Tensor:
    zoom, cx, cy, ramp = _layer_transform(traj, t)
    if depth_t is not None and traj.fx.depth_warp > 1e-4:
        frame = _zoom_crop_depth_gpu(img_t, depth_t, zoom, traj.fx.depth_warp, cx, cy, out_h, out_w)
    else:
        frame = _zoom_crop_gpu(img_t, zoom, cx, cy, out_h, out_w)
    return _apply_fx_gpu(frame, ramp, cx, cy, traj.fx)


def _render_frame_cpu(
    base_hwc: np.ndarray,
    stars_hwc: np.ndarray | None,
    mid_hwc: np.ndarray | None,
    t: float,
    params: FlythroughParams,
    sl_depth: np.ndarray | None,
    st_depth: np.ndarray | None,
    mid_depth: np.ndarray | None,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    sl_frame = _layer_frame_cpu(base_hwc, sl_depth, params.starless, t, out_h, out_w)

    if mid_hwc is not None:
        mid_frame = _layer_frame_cpu(mid_hwc, mid_depth, params.mid, t, out_h, out_w)
        base = _blend_layers_cpu(sl_frame, mid_frame, params.mid.blend_mode, params.mid.opacity)
    else:
        base = sl_frame

    if stars_hwc is not None:
        st_frame = _layer_frame_cpu(stars_hwc, st_depth, params.stars, t, out_h, out_w)
        result = _blend_layers_cpu(
            base,
            st_frame,
            params.stars.blend_mode,
            params.stars.opacity,
            threshold=params.stars.fx.threshold,
            feather=params.stars.fx.feather,
        )
    else:
        result = base

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _render_frame_gpu(
    base_t: torch.Tensor,
    stars_t: torch.Tensor | None,
    mid_t: torch.Tensor | None,
    t: float,
    params: FlythroughParams,
    sl_depth_t: torch.Tensor | None,
    st_depth_t: torch.Tensor | None,
    mid_depth_t: torch.Tensor | None,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    sl_frame = _layer_frame_gpu(base_t, sl_depth_t, params.starless, t, out_h, out_w)

    if mid_t is not None:
        mid_frame = _layer_frame_gpu(mid_t, mid_depth_t, params.mid, t, out_h, out_w)
        base = _blend_layers_gpu(sl_frame, mid_frame, params.mid.blend_mode, params.mid.opacity)
    else:
        base = sl_frame

    if stars_t is not None:
        st_frame = _layer_frame_gpu(stars_t, st_depth_t, params.stars, t, out_h, out_w)
        result = _blend_layers_gpu(
            base,
            st_frame,
            params.stars.blend_mode,
            params.stars.opacity,
            threshold=params.stars.fx.threshold,
            feather=params.stars.fx.feather,
        )
    else:
        result = base

    return _tensor_to_hwc(torch.clamp(result, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Video writer
# ---------------------------------------------------------------------------


def _open_video_writer(
    path: Path, codec: str, fps: float, size: tuple[int, int]
) -> cv2.VideoWriter:
    tried: list[str] = []
    for cc in (codec, "mp4v"):
        if cc in tried:
            continue
        tried.append(cc)
        try:
            fourcc = cv2.VideoWriter_fourcc(*cc[:4].ljust(4))
        except TypeError:
            continue
        writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not open video writer for {path} (tried codecs: {tried})")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_flythrough(
    data: np.ndarray,
    output_path: str | Path,
    params: FlythroughParams,
    stars_layer: np.ndarray | None = None,
    starless_layer: np.ndarray | None = None,
    mid_layer: np.ndarray | None = None,
    progress: ProgressCallback = _noop_progress,
) -> Path:
    """Render a Nebula Flythrough MP4 to `output_path` and return its Path.

    `data` is used as the base (starless) layer unless `starless_layer` is
    given, in which case `starless_layer` is the base and `data` is ignored.
    `stars_layer` and `mid_layer` are optional additional layers composited
    over the base with independent zoom/pan trajectories for parallax depth.
    All image arrays are Astraios-format float32 arrays: mono (H,W) or color
    (C,H,W).
    """
    base_hwc = _ensure_hwc3(starless_layer if starless_layer is not None else data)
    stars_hwc = _ensure_hwc3(stars_layer) if stars_layer is not None else None
    mid_hwc = _ensure_hwc3(mid_layer) if mid_layer is not None else None

    out_w = max(2, int(params.out_width))
    out_h = max(2, int(params.out_height))
    fps = max(1, int(params.fps))
    n_frames = max(1, int(round(fps * float(params.duration))))

    sl_depth = (
        _build_depth_map(
            base_hwc, params.starless.fx.depth_blur_sigma, params.starless.fx.depth_invert
        )
        if params.starless.fx.depth_warp > 1e-4
        else None
    )
    st_depth = (
        _build_depth_map(stars_hwc, params.stars.fx.depth_blur_sigma, params.stars.fx.depth_invert)
        if stars_hwc is not None and params.stars.fx.depth_warp > 1e-4
        else None
    )
    mid_depth = (
        _build_depth_map(mid_hwc, params.mid.fx.depth_blur_sigma, params.mid.fx.depth_invert)
        if mid_hwc is not None and params.mid.fx.depth_warp > 1e-4
        else None
    )

    device_manager = get_device_manager()
    use_gpu = device_manager.is_gpu
    base_t = stars_t = mid_t = None
    sl_depth_t = st_depth_t = mid_depth_t = None

    if use_gpu:
        try:
            device = device_manager.device
            base_t = _hwc_to_tensor(base_hwc, device)
            stars_t = _hwc_to_tensor(stars_hwc, device) if stars_hwc is not None else None
            mid_t = _hwc_to_tensor(mid_hwc, device) if mid_hwc is not None else None
            sl_depth_t = _depth_to_tensor(sl_depth, device) if sl_depth is not None else None
            st_depth_t = _depth_to_tensor(st_depth, device) if st_depth is not None else None
            mid_depth_t = _depth_to_tensor(mid_depth, device) if mid_depth is not None else None
        except Exception:
            log.exception("Flythrough GPU setup failed, falling back to CPU")
            use_gpu = False

    output_path = Path(output_path)
    writer = _open_video_writer(output_path, params.codec, float(fps), (out_w, out_h))

    try:
        for i in range(n_frames):
            t = i / max(1, n_frames - 1)

            frame_hwc = None
            if use_gpu:
                try:
                    frame_hwc = _render_frame_gpu(
                        base_t, stars_t, mid_t, t, params,
                        sl_depth_t, st_depth_t, mid_depth_t, out_h, out_w,
                    )
                except Exception:
                    log.exception("Flythrough GPU frame %d failed, switching to CPU", i)
                    use_gpu = False

            if frame_hwc is None:
                frame_hwc = _render_frame_cpu(
                    base_hwc, stars_hwc, mid_hwc, t, params,
                    sl_depth, st_depth, mid_depth, out_h, out_w,
                )

            frame_u8 = (np.clip(frame_hwc, 0.0, 1.0) * 255.0).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
            progress((i + 1) / n_frames, f"Frame {i + 1}/{n_frames}")
    finally:
        writer.release()

    return output_path
