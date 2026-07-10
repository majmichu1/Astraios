"""Planetary disc reprojection — equirectangular (lon/lat) map or a
re-oriented orthographic view of a captured disc image.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro),
Copyright Franklin Marek, GPL-3.0-or-later.
https://github.com/setiastro/setiastrosuitepro

Source file ``planetprojection.py`` is a 4400+ line module built mostly
around an interactive 3D/anaglyph "pseudo-surface" viewer (Plotly sphere
meshes, stereo pairs, HTML export). The pieces relevant to *reprojecting a
disc image* — i.e. what this module ports — are:

- ``_planet_centroid_and_area``: Otsu-threshold + largest-connected-component
  disc auto-detection (centroid + area -> radius). Ported unmodified.
- ``_disk_to_equirect_texture``: orthographic disc -> equirectangular
  (lon/lat) texture. Ported unmodified as the ``central_lon_deg`` == 0,
  ``central_lat_deg`` == 0 case (see extension note below).
- ``_sphere_reproject_maps``: yaw-rotate the visible hemisphere about the
  vertical (Y) axis and resample — a re-oriented orthographic view. Ported
  unmodified (``theta_deg`` is the exact source parameter name/unit).

Not ported (out of scope — this module only reprojects a disc, it does not
render 3D scenes): the Plotly sphere/ring mesh builders, HTML export,
anaglyph/stereo-pair compositing, starfield background synthesis, and the
``PlanetProjectionDialog``/``PlanetDiskAdjustDialog``/
``PlanetProjectionPreviewDialog`` Qt widgets (SASpro's own dialog classes —
Astraios has its own dialog in ``ui/dialogs/planet_projection_dialog.py``).

Extension beyond the literal source
------------------------------------
The source's ``_disk_to_equirect_texture`` always centers the map on the
sub-observer point (lon=0, lat=0, equator-on) — it has no "central
longitude/latitude" control. ``PlanetProjectionParams.central_lon_deg`` and
``central_lat_deg`` extend it by rotating the sphere before sampling: the
lon-shift is the same "lon2 = lon + dlon" substitution used verbatim in
``derotate.py``'s ``derotate_stack_lonshift``, and the lat-tilt is the same
body-frame -> camera-frame rotate-around-X used there too (see
``astraios/core/derotate.py``). Both reduce to exactly the source formula
when left at 0 — this is a documented, math-consistent extension, not a
fabricated projection.

GPU/CPU split
-------------
- ``_planet_centroid_and_area`` (auto disc detection): CPU/cv2 (Otsu
  threshold, morphology, connected components). This runs once per image,
  not per pixel or per frame, and OpenCV's Otsu/connectedComponents have no
  ready PyTorch equivalent — CPU is the documented, correct choice here (and
  matches the source, which also uses cv2 for this step).
- The actual disc resample (equirectangular or orthographic re-orientation):
  GPU via ``torch.nn.functional.grid_sample`` with an explicit per-pixel
  flow field (same approach as ``derotate.py``, generalizing
  ``gpu_stars.warp_image_gpu``'s affine-only grid), falling back to
  ``cv2.remap`` (the source's original call) on GPU OOM/error. GPU is
  currently contended by LM Studio on this machine, so no comparative
  benchmark is claimed here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np
import torch
import torch.nn.functional as functional

from astraios.core.derotate import GPU_PIXEL_THRESHOLD
from astraios.core.device_manager import get_device_manager
from astraios.core.masks import Mask

log = logging.getLogger(__name__)

__all__ = ["PlanetProjectionParams", "project_planet"]

ProgressCB = Callable[[float, str], None]

ProjectionType = Literal["equirectangular", "orthographic"]

_CV2_INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
}
_TORCH_INTERP = {
    "nearest": "nearest",
    "linear": "bilinear",
    "cubic": "bicubic",
}


@dataclass
class PlanetProjectionParams:
    """Parameters for reprojecting a planetary/lunar/solar disc image.

    Attributes
    ----------
    projection_type : {"equirectangular", "orthographic"}
        "equirectangular" builds a lon/lat map of the disc (source's
        ``_disk_to_equirect_texture``); "orthographic" produces a
        re-oriented view of the disc itself, yawed by ``theta_deg`` about
        the vertical axis (source's ``_sphere_reproject_maps``).
    cx, cy, r : float or None
        Disc center (pixels) and radius (pixels). ``None`` -> auto-detect
        via ``_planet_centroid_and_area`` (source name/behavior).
    central_lon_deg, central_lat_deg : float
        Extension (see module docstring): rotates the sphere before
        equirectangular sampling. 0/0 reproduces the literal source mapping.
    theta_deg : float
        Orthographic re-orientation yaw, degrees. Source field name.
    tex_h, tex_w : int
        Equirectangular output size (source field names/defaults).
    interpolation : {"nearest", "linear", "cubic"}
        Source used INTER_LINEAR for both the equirect and yaw remaps.
    border_value : float
        Fill value outside the sampled/visible region.
    """

    projection_type: ProjectionType = "equirectangular"
    cx: float | None = None
    cy: float | None = None
    r: float | None = None
    central_lon_deg: float = 0.0
    central_lat_deg: float = 0.0
    theta_deg: float = 0.0
    tex_h: int = 1024
    tex_w: int = 2048
    interpolation: str = "linear"
    border_value: float = 0.0
    mask: Mask | None = field(default=None, repr=False)


def _to_hwc(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Channels-first (C,H,W)/(H,W) -> channels-last (H,W,C) for cv2/source
    math. Returns (hwc_array, was_mono)."""
    if image.ndim == 2:
        return image[:, :, None], True
    return np.ascontiguousarray(np.transpose(image, (1, 2, 0))), False


def _from_hwc(hwc: np.ndarray, was_mono: bool) -> np.ndarray:
    if was_mono:
        return np.ascontiguousarray(hwc[:, :, 0])
    return np.ascontiguousarray(np.transpose(hwc, (2, 0, 1)))


def _planet_centroid_and_area(ch: np.ndarray) -> tuple[float, float, float] | None:
    """Estimate planet centroid (cx, cy) and blob area from a single channel.

    Direct port of ``planetprojection.py``'s ``_planet_centroid_and_area``:
    percentile scaling + Otsu threshold + largest connected component.
    """
    img = ch.astype(np.float32, copy=False)

    p1 = float(np.percentile(img, 1.0))
    p99 = float(np.percentile(img, 99.5))
    if p99 <= p1:
        return None

    scaled = (img - p1) * (255.0 / (p99 - p1))
    scaled = np.clip(scaled, 0, 255).astype(np.uint8)
    scaled = cv2.GaussianBlur(scaled, (0, 0), 1.2)

    _, bw = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)

    num, _labels, stats, cents = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    j = int(np.argmax(areas)) + 1
    area = float(stats[j, cv2.CC_STAT_AREA])
    if area < 200:
        return None

    cx, cy = cents[j]
    return (float(cx), float(cy), float(area))


def _resolve_disc(
    image_hwc: np.ndarray, params: PlanetProjectionParams
) -> tuple[float, float, float]:
    """Return (cx, cy, r), auto-detecting whichever of params.cx/cy/r is None."""
    if params.cx is not None and params.cy is not None and params.r is not None:
        return float(params.cx), float(params.cy), float(params.r)

    lum = image_hwc.mean(axis=2) if image_hwc.shape[2] > 1 else image_hwc[:, :, 0]
    detected = _planet_centroid_and_area(lum)
    if detected is None:
        h, w = image_hwc.shape[:2]
        auto_cx, auto_cy = w * 0.5, h * 0.5
        auto_r = max(8.0, 0.45 * min(h, w))
    else:
        auto_cx, auto_cy, area = detected
        auto_r = max(8.0, float(np.sqrt(area / np.pi)))

    cx = float(params.cx) if params.cx is not None else auto_cx
    cy = float(params.cy) if params.cy is not None else auto_cy
    r = float(params.r) if params.r is not None else auto_r
    return cx, cy, r


def _equirect_maps(
    h: int, w: int, cx: float, cy: float, r: float,
    tex_h: int, tex_w: int, central_lon_rad: float, central_lat_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (map_x, map_y, vis) sampling the disc image into an
    equirectangular (tex_h, tex_w) texture.

    Direct port of ``_disk_to_equirect_texture``'s math when
    central_lon_rad == central_lat_rad == 0 (see module docstring for the
    documented extension when they are nonzero)."""
    lons = np.linspace(-np.pi, np.pi, tex_w, endpoint=False).astype(np.float32)
    lats = np.linspace(+0.5 * np.pi, -0.5 * np.pi, tex_h, endpoint=True).astype(np.float32)
    lon, lat = np.meshgrid(lons, lats)

    lon_shifted = lon - float(central_lon_rad)

    x_body = np.cos(lat) * np.sin(lon_shifted)
    y_body = np.sin(lat)
    z_body = np.cos(lat) * np.cos(lon_shifted)

    phi = float(central_lat_rad)
    cphi, sphi = float(np.cos(phi)), float(np.sin(phi))
    x_cam = x_body
    y_cam = cphi * y_body - sphi * z_body
    z_cam = sphi * y_body + cphi * z_body

    vis = z_cam >= 0.0

    map_x = (cx + r * x_cam).astype(np.float32)
    map_y = (cy + r * y_cam).astype(np.float32)
    return map_x, map_y, vis


def _orthographic_maps(
    h: int, w: int, theta_deg: float, cx: float, cy: float, r: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Direct port of ``_sphere_reproject_maps`` (single-view case: source
    builds a stereo Left/Right pair from +-theta; we expose the single
    signed yaw a caller asks for, which is the ``make(+1.0)`` branch)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (xx - cx) / r
    y = (yy - cy) / r
    rr2 = x * x + y * y
    mask = rr2 <= 1.0

    z = np.zeros_like(x, dtype=np.float32)
    z[mask] = np.sqrt(np.maximum(0.0, 1.0 - rr2[mask])).astype(np.float32)

    a = np.deg2rad(float(theta_deg))
    ca, sa = np.cos(a), np.sin(a)

    x2 = x * ca + z * sa
    y2 = y

    map_x = (cx + r * x2).astype(np.float32)
    map_y = (cy + r * y2).astype(np.float32)
    return map_x, map_y, mask


def _remap_cpu(
    img_hwc: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, valid: np.ndarray,
    *, interp: str, border_value: float,
) -> np.ndarray:
    # cv2.remap wants a plain (H, W) plane for single-channel data, not (H,W,1).
    src = img_hwc[:, :, 0] if img_hwc.shape[2] == 1 else img_hwc
    out = cv2.remap(
        src, map_x, map_y,
        interpolation=_CV2_INTERP[interp],
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(border_value),
    )
    if out.ndim == 2:
        out = out[:, :, None]
    out[~valid] = float(border_value)
    return out


@torch.no_grad()
def _remap_gpu(
    img_hwc: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, valid: np.ndarray,
    *, interp: str, border_value: float,
) -> np.ndarray:
    """torch.grid_sample warp with an explicit per-pixel flow field — see
    derotate.py's ``_remap_gpu`` for the same approach/rationale."""
    dm = get_device_manager()
    out_h, out_w = map_x.shape
    src_h, src_w = img_hwc.shape[:2]
    mode = _TORCH_INTERP[interp]

    chw = np.ascontiguousarray(np.transpose(img_hwc, (2, 0, 1)))  # (C, H, W)
    image_t = dm.from_numpy(chw)

    sx = 2.0 / max(src_w - 1, 1)
    sy = 2.0 / max(src_h - 1, 1)
    grid_x = map_x * sx - 1.0
    grid_y = map_y * sy - 1.0
    grid_np = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)  # (out_h, out_w, 2)
    grid_t = dm.from_numpy(grid_np).unsqueeze(0)

    warped = functional.grid_sample(
        image_t.unsqueeze(0), grid_t, mode=mode, align_corners=True, padding_mode="zeros"
    ).squeeze(0)  # (C, out_h, out_w)

    invalid_t = dm.from_numpy(np.ascontiguousarray(~valid))
    warped = torch.where(invalid_t.unsqueeze(0), torch.full_like(warped, border_value), warped)

    out = warped.cpu().numpy().astype(np.float32)
    del image_t, grid_t, warped, invalid_t
    return np.ascontiguousarray(np.transpose(out, (1, 2, 0)))  # (out_h, out_w, C)


def project_planet(
    image: np.ndarray,
    params: PlanetProjectionParams,
    progress: ProgressCB | None = None,
) -> np.ndarray:
    """Reproject a planetary/lunar/solar disc image.

    ``image`` is float32 [0, 1], mono (H, W) or color (C, H, W). Returns the
    same channel convention: mono in -> mono out, color in -> color out.

    - projection_type="equirectangular": output is (tex_h, tex_w) [-ish,
      channels preserved], a lon/lat map of the visible hemisphere.
    - projection_type="orthographic": output is the same (H, W) as the
      input, yawed by theta_deg about the vertical axis.
    """
    if progress is not None:
        progress(0.0, "Locating disc")

    hwc, was_mono = _to_hwc(image)
    h, w = hwc.shape[:2]
    cx, cy, r = _resolve_disc(hwc, params)

    if progress is not None:
        progress(0.3, f"Projecting ({params.projection_type})")

    if params.projection_type == "equirectangular":
        map_x, map_y, vis = _equirect_maps(
            h, w, cx, cy, r, params.tex_h, params.tex_w,
            np.deg2rad(params.central_lon_deg), np.deg2rad(params.central_lat_deg),
        )
    elif params.projection_type == "orthographic":
        map_x, map_y, vis = _orthographic_maps(h, w, params.theta_deg, cx, cy, r)
    else:
        raise ValueError(f"Unknown projection_type: {params.projection_type!r}")

    dm = get_device_manager()
    out_hwc = None
    # cv2.remap wins this workload at every realistic size (see
    # derotate.GPU_PIXEL_THRESHOLD for the idle-GPU benchmark numbers), so
    # CPU is the primary path and GPU only engages beyond the threshold.
    if dm.is_gpu and map_x.size >= GPU_PIXEL_THRESHOLD:
        try:
            out_hwc = _remap_gpu(
                hwc, map_x, map_y, vis,
                interp=params.interpolation, border_value=params.border_value,
            )
        except (RuntimeError, MemoryError) as exc:
            log.warning("GPU OOM/error during planet projection (%s); falling back to CPU", exc)
            dm.empty_cache()

    if out_hwc is None:
        out_hwc = _remap_cpu(
            hwc, map_x, map_y, vis,
            interp=params.interpolation, border_value=params.border_value,
        )

    if progress is not None:
        progress(0.9, "Finalizing")

    result = _from_hwc(out_hwc, was_mono)

    if params.mask is not None and result.shape[-2:] == image.shape[-2:]:
        result = result * params.mask.data + image * (1.0 - params.mask.data)

    if progress is not None:
        progress(1.0, "Done")

    return result.astype(np.float32)
