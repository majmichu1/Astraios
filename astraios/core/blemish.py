"""Blemish Blaster and Clone Stamp — local spot healing and cloning tools.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro),
Copyright Franklin Marek, GPL-3.0-or-later.
https://github.com/setiastro/setiastrosuitepro

SASpro's ``blemish_blaster.py`` heals a small circular blemish (satellite
trail nick, hot pixel cluster, plane streak, etc.) by sampling six patches
arranged in a hexagon around the click point, keeping the three whose median
value is closest to the target spot's median (to avoid sampling onto another
star or defect), and replacing the blemish with the per-pixel median of those
three patches, feathered at the edge of the brush. ``clone_stamp.py`` paints
one image region onto another with a soft circular brush.

Both are local, small-kernel operations on a user-selected radius (typically
a handful to a few hundred pixels) — there is no benefit to a GPU dispatch
for a single dab, so this module is CPU-only (see ``device_manager`` module
docstring / CLAUDE.md: GPU is reserved for whole-image or large-kernel ops).
The original SASpro implementation used per-pixel Python loops; here the same
math is fully vectorized with NumPy.

Data convention: mono images are ``(H, W)``; color images are ``(C, H, W)``
(channels-first), matching Astraios's convention (SASpro used channels-last
``(H, W, C)`` display buffers internally).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BlemishParams:
    """Parameters for :func:`heal_spot` (Blemish Blaster).

    Attributes:
        radius: Brush radius in pixels.
        feather: Edge softness in ``[0, 1]``. ``0`` = hard-edged disc;
            ``1`` = the correction fades in smoothly all the way from the
            brush edge to its center.
        opacity: Blend strength of the healed value over the original,
            in ``[0, 1]``. ``1.0`` fully replaces; lower values partially
            blend the healed patch back with the original pixels.
        channels: Which channel indices to heal. ``None`` (default) heals
            every channel (all channels for color, the sole plane for mono).
        n_candidates: Number of candidate neighbor patches sampled in a
            hexagonal ring around the click point (SASpro hard-coded 6, at
            60-degree steps).
        n_best: How many of the candidate patches (ranked by how close their
            median is to the target spot's median) are kept and median-blended
            to produce the healed value (SASpro hard-coded 3).
        sample_distance_factor: Distance of each candidate patch's center
            from the click point, expressed as a multiple of ``radius``
            (SASpro hard-coded 1.5).
    """

    radius: int = 12
    feather: float = 0.5
    opacity: float = 1.0
    channels: tuple[int, ...] | None = None
    n_candidates: int = 6
    n_best: int = 3
    sample_distance_factor: float = 1.5


@dataclass
class CloneStampParams:
    """Parameters for :func:`clone_stamp`.

    Attributes:
        radius: Brush radius in pixels.
        feather: Edge softness in ``[0, 1]``. ``0`` = hard-edged disc; ``1``
            = the brush falls off smoothly starting from the very center.
            (Note: this uses SASpro's clone-stamp feather formula, which is
            not the same curve as :attr:`BlemishParams.feather`.)
        opacity: Blend strength of the cloned source over the destination,
            in ``[0, 1]``.
    """

    radius: int = 24
    feather: float = 0.5
    opacity: float = 1.0


def _circle_mask(radius: int, feather: float) -> np.ndarray:
    """Build a ``(2r+1, 2r+1)`` float32 disc mask in ``[0, 1]`` with an
    optional feathered falloff (ported verbatim from SASpro's clone stamp).
    """
    r = int(max(1, radius))
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    d = np.sqrt(x * x + y * y).astype(np.float32)
    m = (d <= r).astype(np.float32)

    if feather <= 0:
        return m

    inner = float(r) * (1.0 - float(np.clip(feather, 0.0, 1.0)))
    inner = max(inner, 0.5)

    fall = np.clip((float(r) - d) / max(1e-6, float(r) - inner), 0.0, 1.0)
    return np.where(d <= inner, 1.0, np.where(d <= r, fall, 0.0)).astype(np.float32)


def _median_circle_2d(
    plane: np.ndarray, cx: int, cy: int, radius: int
) -> float | None:
    """Median of `plane` inside a disc of `radius` centered at (cx, cy).

    Returns ``None`` if the disc doesn't overlap the plane at all.
    """
    h, w = plane.shape
    y0, y1 = max(cy - radius, 0), min(cy + radius + 1, h)
    x0, x1 = max(cx - radius, 0), min(cx + radius + 1, w)
    if y0 >= y1 or x0 >= x1:
        return None

    yy, xx = np.mgrid[y0:y1, x0:x1]
    disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
    vals = plane[y0:y1, x0:x1][disc]
    return float(np.median(vals)) if vals.size else None


def heal_spot(
    data: np.ndarray, x: float, y: float, params: BlemishParams | None = None
) -> np.ndarray:
    """Heal a blemish centered at ``(x, y)`` using surrounding pixels.

    For each pixel within ``radius`` of the click point, the healed value is
    the per-pixel median across the best-matching ``n_best`` neighbor patches
    (out of ``n_candidates`` sampled around the spot); the original and
    healed values are blended by ``opacity`` and a radial feather weight.

    Args:
        data: ``(H, W)`` mono or ``(C, H, W)`` color float32 image in
            ``[0, 1]``.
        x: Click point column (pixel coordinate).
        y: Click point row (pixel coordinate).
        params: Healing parameters. Defaults to :class:`BlemishParams`.

    Returns:
        Healed image, same shape/dtype as `data`, clipped to ``[0, 1]``.
        Returns an unmodified copy if the click point is outside the image
        (never raises for out-of-bounds coordinates).
    """
    if params is None:
        params = BlemishParams()

    data = np.asarray(data, dtype=np.float32)
    mono = data.ndim == 2
    arr = data[None, ...] if mono else data
    c_count, h, w = arr.shape

    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < w and 0 <= yi < h):
        return data.copy()

    r = max(1, int(params.radius))
    channels = params.channels if params.channels is not None else tuple(range(c_count))
    channels = tuple(c for c in channels if 0 <= c < c_count)
    if not channels:
        return data.copy()

    out = arr.copy()

    y0, y1 = max(0, yi - r), min(h, yi + r + 1)
    x0, x1 = max(0, xi - r), min(w, xi + r + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.hypot(xx - xi, yy - yi).astype(np.float32)
    disc = dist <= r

    feather = float(params.feather)
    if feather <= 0:
        weight = np.ones_like(dist)
    else:
        weight = np.clip((r - dist) / (r * feather), 0.0, 1.0)

    # Six (by default) hexagonal candidate neighbor centers, ranked by how
    # close their combined-channel median is to the target spot's median.
    n_ang = max(1, int(params.n_candidates))
    angles = np.deg2rad(np.arange(n_ang) * (360.0 / n_ang))
    offs_x = np.round(np.cos(angles) * r * params.sample_distance_factor).astype(int)
    offs_y = np.round(np.sin(angles) * r * params.sample_distance_factor).astype(int)
    centers = [(xi + int(dx), yi + int(dy)) for dx, dy in zip(offs_x, offs_y, strict=True)]

    def _combined_median(cx: int, cy: int) -> float | None:
        vals = [
            m for c in channels if (m := _median_circle_2d(arr[c], cx, cy, r)) is not None
        ]
        return float(np.median(vals)) if vals else None

    tgt_med = _combined_median(xi, yi)
    tgt_med = tgt_med if tgt_med is not None else 0.0

    diffs = [
        abs((m if (m := _combined_median(cx, cy)) is not None else float("inf")) - tgt_med)
        for cx, cy in centers
    ]
    n_best = max(1, min(int(params.n_best), len(centers)))
    best_idx = np.argsort(diffs)[:n_best]
    sel_centers = [centers[i] for i in best_idx]

    ph, pw = disc.shape
    for c in channels:
        stack = []
        for cx, cy in sel_centers:
            ox, oy = cx - xi, cy - yi
            sy0, sx0 = y0 + oy, x0 + ox
            sy1, sx1 = y1 + oy, x1 + ox

            patch = np.full((ph, pw), np.nan, dtype=np.float32)
            vy0, vy1 = max(0, sy0), min(h, sy1)
            vx0, vx1 = max(0, sx0), min(w, sx1)
            if vy0 < vy1 and vx0 < vx1:
                py0, py1 = vy0 - sy0, vy1 - sy0
                px0, px1 = vx0 - sx0, vx1 - sx0
                patch[py0:py1, px0:px1] = arr[c, vy0:vy1, vx0:vx1]
            stack.append(patch)

        orig_patch = arr[c, y0:y1, x0:x1]
        if stack:
            stacked = np.stack(stack, axis=0)
            with np.errstate(invalid="ignore"):
                healed = np.nanmedian(stacked, axis=0)
            healed = np.where(np.isnan(healed), orig_patch, healed)
        else:
            healed = orig_patch

        blended = (1.0 - params.opacity * weight) * orig_patch + params.opacity * weight * healed
        out[c, y0:y1, x0:x1] = np.where(disc, blended, orig_patch)

    result = out[0] if mono else out
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _blend_clone_inplace_2d(
    plane: np.ndarray,
    tx: int,
    ty: int,
    sx: int,
    sy: int,
    mask_full: np.ndarray,
    r: int,
    opacity: float,
) -> None:
    """In-place clone dab on a single ``(H, W)`` plane (ported from SASpro's
    ``_blend_clone_inplace``, generalized from its channels-last 3-plane form
    to operate one channel at a time).
    """
    h, w = plane.shape

    x0, x1 = tx - r, tx + r + 1
    y0, y1 = ty - r, ty + r + 1

    cx0, cx1 = max(0, x0), min(w, x1)
    cy0, cy1 = max(0, y0), min(h, y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return

    mx0, my0 = cx0 - x0, cy0 - y0
    tw, th = cx1 - cx0, cy1 - cy0

    sx0, sy0 = (sx - r) + mx0, (sy - r) + my0
    sx1, sy1 = sx0 + tw, sy0 + th

    acx0, acy0, acx1, acy1 = cx0, cy0, cx1, cy1

    if sx0 < 0:
        d = -sx0
        sx0 = 0
        acx0 += d
    if sy0 < 0:
        d = -sy0
        sy0 = 0
        acy0 += d
    if sx1 > w:
        d = sx1 - w
        sx1 = w
        acx1 -= d
    if sy1 > h:
        d = sy1 - h
        sy1 = h
        acy1 -= d

    if acx0 >= acx1 or acy0 >= acy1:
        return

    tw, th = acx1 - acx0, acy1 - acy0
    if tw <= 0 or th <= 0:
        return

    mx0, my0 = acx0 - x0, acy0 - y0

    tgt = plane[acy0:acy1, acx0:acx1]
    src = plane[sy0 : sy0 + th, sx0 : sx0 + tw]

    a = (mask_full[my0 : my0 + th, mx0 : mx0 + tw] * float(opacity)).astype(np.float32)
    if a.max() <= 0:
        return

    tgt[:] = (1.0 - a) * tgt + a * src


def clone_stamp(
    data: np.ndarray,
    src_xy: tuple[float, float],
    dst_xy: tuple[float, float],
    params: CloneStampParams | None = None,
) -> np.ndarray:
    """Clone (copy) a circular patch from `src_xy` onto `dst_xy`.

    Args:
        data: ``(H, W)`` mono or ``(C, H, W)`` color float32 image in
            ``[0, 1]``.
        src_xy: ``(x, y)`` source point pixel coordinates.
        dst_xy: ``(x, y)`` destination point pixel coordinates.
        params: Clone parameters. Defaults to :class:`CloneStampParams`.

    Returns:
        Image with the source patch painted at the destination, same
        shape/dtype as `data`, clipped to ``[0, 1]``. Coordinates fully or
        partially outside the image are safely clipped; no exception is
        raised.
    """
    if params is None:
        params = CloneStampParams()

    data = np.asarray(data, dtype=np.float32)
    mono = data.ndim == 2
    arr = (data[None, ...] if mono else data).copy()
    c_count = arr.shape[0]

    r = max(1, int(params.radius))
    mask = _circle_mask(r, params.feather)

    tx, ty = int(round(dst_xy[0])), int(round(dst_xy[1]))
    sx, sy = int(round(src_xy[0])), int(round(src_xy[1]))

    for c in range(c_count):
        _blend_clone_inplace_2d(arr[c], tx, ty, sx, sy, mask, r, params.opacity)

    result = arr[0] if mono else arr
    return np.clip(result, 0.0, 1.0).astype(np.float32)


__all__ = [
    "BlemishParams",
    "CloneStampParams",
    "heal_spot",
    "clone_stamp",
]
