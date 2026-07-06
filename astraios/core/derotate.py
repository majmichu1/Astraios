"""Planetary/solar de-rotation — align rotating-body video frames to a common view.

A fast-rotating body (Jupiter, Saturn, the Sun) visibly spins during the
minutes it takes to capture a lucky-imaging SER sequence. Naively stacking
those frames blurs surface detail. De-rotation removes each frame's rotation
relative to a reference time by treating the disc as an orthographically
projected sphere: every image pixel is unprojected to a body-frame
(longitude, latitude), shifted by the rotation elapsed since the reference,
then re-projected back to image pixels and resampled.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro),
Copyright Franklin Marek, GPL-3.0-or-later.
https://github.com/setiastro/setiastrosuitepro

The two functions below (``_build_lonlat_grids`` and the per-frame lon-shift
math in ``_derotate_one``) are a direct, unmodified port of
``derotate.py``'s ``_build_lonlat_grids`` / ``derotate_stack_lonshift``
(same variable names, same trigonometry, same visibility test). Everything
around them (``DerotateParams``, per-frame angle scheduling from SER
timestamps, the GPU/CPU remap dispatch) is new driver code written for
Astraios, since the source file only exposed the bare geometry — the
per-frame stacking driver in SASpro lives elsewhere and was not available to
port.

GPU/CPU split
-------------
- ``_build_lonlat_grids``: kept on CPU/numpy. It runs once per de-rotation
  job (not once per frame) and is pure trig over an (H, W) grid — negligible
  cost, and staying on numpy keeps it a byte-for-byte match of the source.
- The per-frame resample (the expensive part, run once per video frame):
  GPU via ``torch.nn.functional.grid_sample`` (bicubic/bilinear/nearest,
  ``align_corners=True`` to match the convention already used by
  ``gpu_stars.warp_image_gpu``), generalized from an affine grid to an
  arbitrary per-pixel flow field. Falls back to ``cv2.remap`` (the source's
  original call) on GPU OOM/error, mirroring the try/except pattern already
  used in ``stacking.py``'s GPU rejection path. GPU is currently contended by
  LM Studio on this machine, so no comparative benchmark is claimed here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn.functional as functional

from astraios.core.device_manager import get_device_manager
from astraios.core.masks import Mask

log = logging.getLogger(__name__)

__all__ = ["DerotateParams", "derotate_frames"]

ProgressCB = Callable[[float, str], None]

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
class DerotateParams:
    """Parameters for de-rotating a sequence of planetary/solar frames.

    ``cx``, ``cy``, ``r``, ``pole_angle_rad``, ``subobs_lat_rad`` and
    ``border_value`` are exactly the source's ``derotate_stack_lonshift``/
    ``_build_lonlat_grids`` parameter names and units (radians, pixels).

    Rotation can be supplied two ways:
      - ``rotation_rate_deg_per_hour`` + ``frame_times_s`` (seconds elapsed
        since the first frame, e.g. from :meth:`SERFrameReader.read_timestamp`
        converted from .NET ticks): each frame's shift is computed as
        ``dlon = -rotation_rate * (t_i - t_ref)`` — the minus sign undoes the
        apparent forward rotation so every frame is resampled back to how it
        would have looked at the reference time. This scheduling convention
        is Astraios driver code, not part of the ported source (SASpro's
        stacking driver that calls this primitive was not in the provided
        source excerpt).
      - ``per_frame_angles_rad``: an explicit per-frame ``dlon_rad`` value,
        passed straight through to the ported math with no sign change —
        use this when the caller (e.g. a dialog) already computed angles.

    Attributes
    ----------
    cx, cy : float
        Disc center in pixels.
    r : float
        Disc (limb) radius in pixels.
    pole_angle_rad : float
        Rotates image coordinates so the planet's spin axis points "up".
    subobs_lat_rad : float
        Sub-observer latitude (pole tilt toward/away from the viewer).
    rotation_rate_deg_per_hour : float
        Sidereal/synodic rotation rate, degrees per hour. Sign follows the
        body's apparent rotation direction in the image as pole_angle_rad
        defines "up".
    frame_times_s : sequence of float, optional
        Per-frame elapsed time in seconds since the first frame. Required
        (and ignored if absent) unless ``per_frame_angles_rad`` is given.
    per_frame_angles_rad : sequence of float, optional
        Explicit dlon_rad per frame; overrides rate-based scheduling.
    reference_index : int
        Index of the frame treated as the zero-rotation reference.
    interpolation : {"nearest", "linear", "cubic"}
        Resample quality. Source used INTER_CUBIC; default matches that.
    border_value : float
        Fill value for pixels that fall off the visible disc.
    """

    cx: float
    cy: float
    r: float
    pole_angle_rad: float = 0.0
    subobs_lat_rad: float = 0.0
    rotation_rate_deg_per_hour: float = 0.0
    frame_times_s: Sequence[float] | None = None
    per_frame_angles_rad: Sequence[float] | None = None
    reference_index: int = 0
    interpolation: str = "cubic"
    border_value: float = 0.0
    mask: Mask | None = field(default=None, repr=False)


def _build_lonlat_grids(
    h: int,
    w: int,
    cx: float,
    cy: float,
    r: float,
    pole_angle_rad: float,
    subobs_lat_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute lon/lat and visibility mask for an orthographic sphere.

    Direct port of ``derotate.py``'s ``_build_lonlat_grids`` — unmodified
    geometry, same variable names.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xx - cx) / r
    dy = (yy - cy) / r

    ca = float(np.cos(pole_angle_rad))
    sa = float(np.sin(pole_angle_rad))
    dxp = ca * dx - sa * dy
    dyp = sa * dx + ca * dy

    rr2 = dxp * dxp + dyp * dyp
    vis = rr2 <= 1.0

    z0 = np.zeros_like(dxp, dtype=np.float32)
    z0[vis] = np.sqrt(np.maximum(0.0, 1.0 - rr2[vis]))

    phi = float(subobs_lat_rad)
    cphi = float(np.cos(phi))
    sphi = float(np.sin(phi))

    x1 = dxp
    y1 = cphi * dyp + sphi * z0
    z1 = -sphi * dyp + cphi * z0

    lon = np.zeros_like(dxp, dtype=np.float32)
    lat = np.zeros_like(dxp, dtype=np.float32)
    lon[vis] = np.arctan2(x1[vis], z1[vis])
    lat[vis] = np.arcsin(np.clip(y1[vis], -1.0, 1.0))

    return lon, lat, vis


def _lonshift_maps(
    lon: np.ndarray,
    lat: np.ndarray,
    vis: np.ndarray,
    *,
    cx: float,
    cy: float,
    r: float,
    dlon_rad: float,
    pole_angle_rad: float,
    subobs_lat_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (map_x, map_y, valid) for a longitude shift of ``dlon_rad``.

    Direct port of the map-building half of ``derotate.py``'s
    ``derotate_stack_lonshift`` (the remap/channel-loop half is replaced by
    the GPU/CPU dispatch in ``_derotate_one`` below).
    """
    lon2 = lon + float(dlon_rad)

    clat = np.cos(lat)
    xb = clat * np.sin(lon2)
    yb = np.sin(lat)
    zb = clat * np.cos(lon2)

    phi = float(subobs_lat_rad)
    cphi = float(np.cos(phi))
    sphi = float(np.sin(phi))

    xc = xb
    yc = cphi * yb - sphi * zb
    zc = sphi * yb + cphi * zb

    ca = float(np.cos(pole_angle_rad))
    sa = float(np.sin(pole_angle_rad))

    x = ca * xc + sa * yc
    y = -sa * xc + ca * yc

    map_x = (x * r + cx).astype(np.float32)
    map_y = (y * r + cy).astype(np.float32)

    valid = vis & (zc >= 0.0)
    return map_x, map_y, valid


def _remap_cpu(
    img: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    valid: np.ndarray,
    *,
    interp: str,
    border_value: float,
) -> np.ndarray:
    """cv2.remap fallback — same call as the ported source, plane by plane."""
    cv2_interp = _CV2_INTERP[interp]

    def _remap_plane(plane: np.ndarray) -> np.ndarray:
        out = cv2.remap(
            plane, map_x, map_y,
            interpolation=cv2_interp,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float(border_value),
        )
        out[~valid] = float(border_value)
        return out

    if img.ndim == 2:
        return _remap_plane(img)

    # (C, H, W) channels-first -> plane loop -> back to (C, H, W).
    out = np.empty_like(img)
    for c in range(img.shape[0]):
        out[c] = _remap_plane(img[c])
    return out


@torch.no_grad()
def _remap_gpu(
    img: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    valid: np.ndarray,
    *,
    interp: str,
    border_value: float,
) -> np.ndarray:
    """torch.grid_sample warp: generalizes gpu_stars.warp_image_gpu's affine
    grid to an arbitrary per-pixel flow field (map_x, map_y are not affine —
    they come from the sphere unprojection, so affine_grid cannot express
    them; a custom sampling grid is built directly instead)."""
    dm = get_device_manager()
    h, w = map_x.shape
    mode = _TORCH_INTERP[interp]

    mono = img.ndim == 2
    chw = img[None, ...] if mono else img

    image_t = dm.from_numpy(np.ascontiguousarray(chw))  # (C, H, W)

    sx = 2.0 / max(w - 1, 1)
    sy = 2.0 / max(h - 1, 1)
    grid_x = map_x * sx - 1.0
    grid_y = map_y * sy - 1.0
    grid_np = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)  # (H, W, 2)
    grid_t = dm.from_numpy(grid_np).unsqueeze(0)  # (1, H, W, 2)

    warped = functional.grid_sample(
        image_t.unsqueeze(0), grid_t, mode=mode, align_corners=True, padding_mode="zeros"
    ).squeeze(0)  # (C, H, W)

    invalid_t = dm.from_numpy(np.ascontiguousarray(~valid))
    warped = torch.where(invalid_t.unsqueeze(0), torch.full_like(warped, border_value), warped)

    out = warped.cpu().numpy().astype(np.float32)
    del image_t, grid_t, warped, invalid_t
    return out[0] if mono else out


def _derotate_one(
    frame: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    vis: np.ndarray,
    params: DerotateParams,
    dlon_rad: float,
) -> np.ndarray:
    map_x, map_y, valid = _lonshift_maps(
        lon, lat, vis,
        cx=params.cx, cy=params.cy, r=params.r, dlon_rad=dlon_rad,
        pole_angle_rad=params.pole_angle_rad, subobs_lat_rad=params.subobs_lat_rad,
    )

    dm = get_device_manager()
    if dm.is_gpu:
        try:
            return _remap_gpu(
                frame, map_x, map_y, valid,
                interp=params.interpolation, border_value=params.border_value,
            )
        except (RuntimeError, MemoryError) as exc:
            log.warning("GPU OOM/error during de-rotation (%s); falling back to CPU", exc)
            dm.empty_cache()

    return _remap_cpu(
        frame, map_x, map_y, valid,
        interp=params.interpolation, border_value=params.border_value,
    )


def _schedule_dlon(frames: Sequence[np.ndarray], params: DerotateParams) -> list[float]:
    n = len(frames)
    if params.per_frame_angles_rad is not None:
        angles = list(params.per_frame_angles_rad)
        if len(angles) != n:
            raise ValueError(
                f"per_frame_angles_rad has {len(angles)} entries for {n} frames"
            )
        return [float(a) for a in angles]

    if not params.rotation_rate_deg_per_hour:
        return [0.0] * n

    times = params.frame_times_s
    if times is None:
        # No timestamps: assume uniform unit spacing (index == seconds). This
        # only produces a physically meaningful angle if the caller scales
        # rotation_rate_deg_per_hour accordingly; documented in the dataclass.
        times = list(range(n))
    if len(times) != n:
        raise ValueError(f"frame_times_s has {len(times)} entries for {n} frames")

    ref_t = float(times[params.reference_index])
    rate_rad_per_s = np.deg2rad(params.rotation_rate_deg_per_hour) / 3600.0
    return [float(-rate_rad_per_s * (float(t) - ref_t)) for t in times]


def derotate_frames(
    frames: Sequence[np.ndarray],
    params: DerotateParams,
    progress: ProgressCB | None = None,
) -> list[np.ndarray]:
    """De-rotate a sequence of frames to a common (reference-time) view.

    Each frame is float32 [0, 1], mono (H, W) or color (C, H, W). All frames
    must share the same (H, W) — a fixed disc center/radius is assumed for
    the whole sequence, matching a lucky-imaging SER capture where the
    telescope/mount is not tracking planetary rotation.
    """
    if not frames:
        return []

    h, w = frames[0].shape[-2:]
    for i, f in enumerate(frames):
        if f.shape[-2:] != (h, w):
            raise ValueError(f"frame {i} shape {f.shape} does not match frame 0 shape {(h, w)}")

    lon, lat, vis = _build_lonlat_grids(
        h, w, params.cx, params.cy, params.r, params.pole_angle_rad, params.subobs_lat_rad
    )
    dlons = _schedule_dlon(frames, params)

    n = len(frames)
    out: list[np.ndarray] = []
    for i, (frame, dlon) in enumerate(zip(frames, dlons, strict=True)):
        result = _derotate_one(frame, lon, lat, vis, params, dlon)
        if params.mask is not None:
            result = result * params.mask.data + frame * (1.0 - params.mask.data)
        out.append(result.astype(np.float32))
        if progress is not None:
            progress((i + 1) / n, f"De-rotated frame {i + 1}/{n}")

    return out
