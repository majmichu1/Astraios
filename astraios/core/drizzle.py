"""Drizzle Integration — sub-pixel resolution enhancement during stacking.

Replaces the previous pure-Python pixel loop with vectorized numpy (CPU)
and GPU-accelerated (torch) implementations that are orders of magnitude faster.

GPU path: all pixels of a frame transformed and scattered in one tensor op.
CPU path: vectorized numpy with np.add.at (no Python loops over pixels).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from astraios.core.star_detection import detect_stars, find_transform

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


@dataclass
class DrizzleParams:
    """Parameters for drizzle integration."""

    scale: int = 2           # output scale factor (2 = 2× resolution)
    drop_shrink: float = 0.7  # pixel footprint fraction (0.5–1.0)
    pixel_weight: str = "uniform"  # "uniform" or "gaussian"
    use_gpu: bool = True     # prefer GPU; auto-falls back to CPU


@dataclass
class DrizzleResult:
    """Result of drizzle integration."""

    data: np.ndarray
    weight_map: np.ndarray
    n_frames: int
    output_scale: int


# ---------------------------------------------------------------------------
# Core drizzle implementations
# ---------------------------------------------------------------------------


def _drizzle_frame_numpy(
    image: np.ndarray,
    output: np.ndarray,
    weight_map: np.ndarray,
    transform: np.ndarray | None,
    scale: int,
    drop_shrink: float,
) -> None:
    """Vectorized CPU drizzle for a single frame using numpy.

    Replaces the pure Python double loop — processes all pixels in one
    batch of numpy operations (no Python iteration over pixels).
    """
    is_color = image.ndim == 3
    if is_color:
        n_ch, h, w = image.shape
    else:
        n_ch = 1
        h, w = image.shape

    # Build grid of all input pixel centres: shape (H*W, 2). float32 is plenty
    # for pixel coordinates (< 1e4, resolved to ~1e-3 px) and halves these
    # full-image-sized coordinate buffers (~1.75GB of transients at 73MP).
    iy, ix = np.mgrid[0:h, 0:w]
    ones = np.ones((h, w), dtype=np.float32)
    # Homogeneous coordinates: (3, H*W)
    pts = np.stack([ix.ravel().astype(np.float32),
                    iy.ravel().astype(np.float32),
                    ones.ravel()], axis=0)  # (3, N)

    # Apply affine transform (or identity)
    mat = transform.astype(np.float32) if transform is not None else np.eye(2, 3, dtype=np.float32)

    ref_pts = mat @ pts  # (2, N)
    sx = ref_pts[0]    # x in reference frame
    sy = ref_pts[1]    # y in reference frame

    # Scale to output grid
    ox = sx * scale
    oy = sy * scale

    half_drop = drop_shrink * scale * 0.5
    ox_min = np.floor(ox - half_drop).astype(np.int32)
    ox_max = np.ceil(ox + half_drop).astype(np.int32)
    oy_min = np.floor(oy - half_drop).astype(np.int32)
    oy_max = np.ceil(oy + half_drop).astype(np.int32)

    out_h, out_w = weight_map.shape

    # Vectorised footprint scatter. The original looped over every input pixel
    # (O(H*W) ~ tens of millions of Python iterations) to add each pixel's value
    # into its output footprint rectangle [oy_min:oy_max, ox_min:ox_max]. Instead
    # clamp all footprints at once, then for each (dy, dx) offset within the
    # LARGEST footprint, scatter every pixel that still covers that offset with a
    # single np.add.at. Footprints are tiny (~1-3 px per side for typical
    # drop_shrink/scale), so this is a handful of vectorised scatters. The
    # accumulated output matches the loop to float32 epsilon (only the summation
    # order of overlapping footprints differs); the weight map is exact.
    y0 = np.maximum(0, oy_min)
    x0 = np.maximum(0, ox_min)
    fh = np.minimum(out_h, oy_max) - y0   # clamped footprint height per pixel
    fw = np.minimum(out_w, ox_max) - x0   # clamped footprint width per pixel
    covers = (fh > 0) & (fw > 0)
    if not np.any(covers):
        return

    out_flat = output.reshape(n_ch, -1) if is_color else output.reshape(-1)
    img_flat = image.reshape(n_ch, -1) if is_color else image.reshape(-1)
    w_flat = weight_map.reshape(-1)

    max_fh = int(fh[covers].max())
    max_fw = int(fw[covers].max())
    for dy in range(max_fh):
        for dx in range(max_fw):
            sel = covers & (dy < fh) & (dx < fw)
            if not np.any(sel):
                continue
            tgt = (y0[sel] + dy) * out_w + (x0[sel] + dx)
            if is_color:
                for c in range(n_ch):
                    np.add.at(out_flat[c], tgt, img_flat[c][sel])
            else:
                np.add.at(out_flat, tgt, img_flat[sel])
            np.add.at(w_flat, tgt, 1.0)


@torch.no_grad()
def _drizzle_frame_gpu(
    image: np.ndarray,
    output_t: Any,
    weight_t: Any,
    transform: np.ndarray | None,
    scale: int,
    drop_shrink: float,
) -> None:
    """GPU drizzle for a single frame using torch scatter_add.

    For each input pixel, computes the output bin indices (honouring
    drop_shrink footprint) and uses scatter_add_ to accumulate —
    no Python loops over pixels.
    """
    import torch

    from astraios.core.device_manager import get_device_manager

    dm = get_device_manager()
    device = dm.device

    is_color = image.ndim == 3
    if is_color:
        h, w = image.shape[1], image.shape[2]
    else:
        h, w = image.shape

    out_h, out_w = weight_t.shape

    # Input grid
    iy = torch.arange(h, device=device, dtype=torch.float32)
    ix = torch.arange(w, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(iy, ix, indexing="ij")
    ones = torch.ones_like(grid_x)
    pts = torch.stack([grid_x.flatten(), grid_y.flatten(), ones.flatten()], dim=0)  # (3, N)

    mat = (
        torch.tensor(transform, device=device, dtype=torch.float32)
        if transform is not None
        else torch.eye(2, 3, device=device, dtype=torch.float32)
    )

    ref_pts = mat @ pts  # (2, N)
    ox = ref_pts[0] * scale
    oy = ref_pts[1] * scale

    half_drop = drop_shrink * scale * 0.5
    ox_floor = (ox - half_drop).floor().long()
    ox_ceil  = (ox + half_drop).ceil().long()
    oy_floor = (oy - half_drop).floor().long()
    oy_ceil  = (oy + half_drop).ceil().long()

    # Each input pixel spreads into the up-to-4 integer corners of its footprint
    # [ox_floor, ox_ceil] x [oy_floor, oy_ceil]. Scatter one corner at a time
    # instead of materialising a (C, 4N) "repeat each pixel 4x" tensor (4x the
    # image in VRAM, plus a filtered copy): scatter_add is additive, so the
    # accumulated totals — and the weight counts, including the harmless 4x
    # over-count when the footprint collapses to one bin — are identical, while
    # peak memory stays ~O(N) per corner.
    corner_offsets = (
        (oy_floor, ox_floor),
        (oy_floor, ox_ceil),
        (oy_ceil, ox_floor),
        (oy_ceil, ox_ceil),
    )

    if is_color:
        img_t = dm.from_numpy(image.astype(np.float32))  # (C, H, W)
        img_flat = img_t.reshape(img_t.shape[0], -1)  # (C, N)
        n_ch = img_t.shape[0]
    else:
        img_flat = dm.from_numpy(image.astype(np.float32)).flatten()  # (N,)
        n_ch = 1

    for cy, cx in corner_offsets:
        v = (cx >= 0) & (cx < out_w) & (cy >= 0) & (cy < out_h)
        idx = (cy * out_w + cx)[v]
        if is_color:
            for c in range(n_ch):
                output_t[c].flatten().scatter_add_(0, idx, img_flat[c][v])
        else:
            output_t.flatten().scatter_add_(0, idx, img_flat[v])
        weight_t.flatten().scatter_add_(0, idx, torch.ones(idx.shape[0], device=device))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def drizzle_integrate(
    images: list[np.ndarray],
    transforms: list[np.ndarray | None] | None = None,
    params: DrizzleParams | None = None,
    progress: ProgressCallback | None = None,
) -> DrizzleResult:
    """Integrate multiple images using drizzle for sub-pixel resolution.

    Parameters
    ----------
    images : list[ndarray]
        Input images, shape (H, W) or (C, H, W), float32 in [0, 1].
    transforms : list[ndarray | None], optional
        Pre-computed 2×3 affine transforms. Computed from star matching if None.
    params : DrizzleParams, optional
    progress : callable, optional

    Returns
    -------
    DrizzleResult
    """
    if params is None:
        params = DrizzleParams()
    if progress is None:
        progress = _noop_progress

    if not images:
        raise ValueError("No images provided for drizzle")

    for i, img in enumerate(images):
        if not np.all(np.isfinite(img)):
            n_bad = int(np.sum(~np.isfinite(img)))
            log.warning("Drizzle frame %d: %d NaN/inf values replaced with 0", i, n_bad)
            images[i] = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    ref = images[0]
    is_color = ref.ndim == 3
    if is_color:
        n_ch, h, w = ref.shape
    else:
        h, w = ref.shape
        n_ch = 1

    scale = params.scale
    out_h, out_w = h * scale, w * scale

    # Compute transforms if not provided
    if transforms is None:
        progress(0.0, "Computing registration transforms...")
        transforms = _compute_transforms(images, progress)

    # Decide execution path
    use_gpu = params.use_gpu
    if use_gpu:
        try:
            import torch

            from astraios.core.device_manager import get_device_manager
            dm = get_device_manager()
            use_gpu = dm.device.type != "cpu"
        except Exception:
            use_gpu = False

    if use_gpu:
        import torch

        from astraios.core.device_manager import get_device_manager
        dm = get_device_manager()
        if is_color:
            output_t = torch.zeros(n_ch, out_h, out_w, device=dm.device, dtype=torch.float32)
        else:
            output_t = torch.zeros(out_h, out_w, device=dm.device, dtype=torch.float32)
        weight_t = torch.zeros(out_h, out_w, device=dm.device, dtype=torch.float32)

        for i, (img, transform) in enumerate(zip(images, transforms, strict=True)):
            frac = 0.3 + 0.7 * i / max(len(images) - 1, 1)
            progress(frac, f"Drizzling frame {i + 1}/{len(images)}...")
            if transform is None and i > 0:
                log.warning("Skipping frame %d: no transform", i)
                continue
            _drizzle_frame_gpu(img, output_t, weight_t, transform, scale, params.drop_shrink)

        # Normalize
        valid = weight_t > 0
        if is_color:
            for c in range(n_ch):
                output_t[c][valid] /= weight_t[valid]
        else:
            output_t[valid] /= weight_t[valid]

        result = torch.clamp(output_t, 0, 1).cpu().numpy().astype(np.float32)
        weight_map = weight_t.cpu().numpy().astype(np.float32)

    else:
        # float32 accumulators: drizzle sums dozens of [0,1] frames, so the
        # float32 round-off is ~1e-6 — negligible — and we halve the scaled
        # output buffer (~1.75GB at 73MP, 2x scale).
        if is_color:
            output = np.zeros((n_ch, out_h, out_w), dtype=np.float32)
        else:
            output = np.zeros((out_h, out_w), dtype=np.float32)
        weight_map_f = np.zeros((out_h, out_w), dtype=np.float32)

        for i, (img, transform) in enumerate(zip(images, transforms, strict=True)):
            frac = 0.3 + 0.7 * i / max(len(images) - 1, 1)
            progress(frac, f"Drizzling frame {i + 1}/{len(images)}...")
            if transform is None and i > 0:
                log.warning("Skipping frame %d: no transform", i)
                continue
            _drizzle_frame_numpy(img, output, weight_map_f, transform, scale, params.drop_shrink)

        valid = weight_map_f > 0
        if is_color:
            for c in range(n_ch):
                output[c][valid] /= weight_map_f[valid]
        else:
            output[valid] /= weight_map_f[valid]

        result = np.clip(output, 0, 1).astype(np.float32)
        weight_map = weight_map_f.astype(np.float32)

    progress(1.0, "Drizzle complete")
    return DrizzleResult(
        data=result,
        weight_map=weight_map,
        n_frames=len(images),
        output_scale=scale,
    )


def _compute_transforms(
    images: list[np.ndarray],
    progress: ProgressCallback,
) -> list[np.ndarray | None]:
    """Compute affine transforms by star-matching each frame to the reference."""
    ref_sf = detect_stars(images[0])
    transforms: list[np.ndarray | None] = [np.eye(2, 3, dtype=np.float32)]

    for i in range(1, len(images)):
        frac = 0.3 * i / max(len(images) - 1, 1)
        progress(frac, f"Registering frame {i + 1}/{len(images)}...")
        tgt_sf = detect_stars(images[i])
        t = find_transform(ref_sf, tgt_sf)
        transforms.append(t)

    return transforms
