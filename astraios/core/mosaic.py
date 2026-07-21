"""Mosaic stitching with photometric normalization.

Uses OpenCV for homography computation and image warping.
Supports gradient-matched feathered blending for seamless transitions.

Photometric normalization equalizes brightness across overlapping panels
by computing robust median ratios in overlap regions before blending.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np

from astraios.core.star_detection import detect_stars, find_transform

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


class BlendMethod(Enum):
    FEATHER = auto()
    MULTIBAND = auto()
    AVERAGE = auto()


class NormalizeMethod(Enum):
    NONE = auto()
    LINEAR = auto()
    ROBUST = auto()


@dataclass
class MosaicParams:
    blend_method: BlendMethod = BlendMethod.FEATHER
    feather_width: int = 50
    match_gradient: bool = True
    normalize: NormalizeMethod = NormalizeMethod.ROBUST
    norm_per_channel: bool = True
    clip_negative: bool = True


@dataclass
class MosaicResult:
    data: np.ndarray
    n_panels: int
    output_shape: tuple[int, ...]


def mosaic_stitch(
    panels: list[np.ndarray],
    params: MosaicParams | None = None,
    progress: ProgressCallback | None = None,
) -> MosaicResult:
    """Stitch overlapping panels into a mosaic.

    Parameters
    ----------
    panels : list[ndarray]
        Panel images, shape (H, W) or (C, H, W), float32 in [0, 1].
    params : MosaicParams, optional
        Stitching parameters.
    progress : callable, optional
        Progress callback.

    Returns
    -------
    MosaicResult
        Stitched mosaic.
    """
    if len(panels) < 2:
        raise ValueError("Mosaic stitching requires at least 2 panels")

    if params is None:
        params = MosaicParams()
    if progress is None:
        progress = _noop_progress

    is_color = panels[0].ndim == 3

    progress(0.0, "Computing panel registrations...")
    transforms = _compute_pairwise_transforms(panels, progress)

    progress(0.4, "Computing output canvas...")
    canvas_size, offsets = _compute_canvas(panels, transforms)

    if params.normalize != NormalizeMethod.NONE and len(panels) > 1:
        progress(0.45, "Normalizing photometry...")
        panels = _normalize_photometric(panels, transforms, offsets, canvas_size, params)

    progress(0.5, "Warping and blending panels...")
    if is_color:
        result = np.zeros((panels[0].shape[0], canvas_size[0], canvas_size[1]), dtype=np.float32)
    else:
        result = np.zeros(canvas_size, dtype=np.float32)
    weight_total = np.zeros(canvas_size, dtype=np.float32)

    multiband = params.blend_method == BlendMethod.MULTIBAND
    mb_panels: list[np.ndarray] = []
    mb_weights: list[np.ndarray] = []

    for i, (panel, transform, offset) in enumerate(zip(panels, transforms, offsets, strict=True)):
        frac = 0.5 + 0.5 * i / max(len(panels) - 1, 1)
        progress(frac, f"Blending panel {i + 1}/{len(panels)}...")

        warped, mask = _warp_panel(panel, transform, offset, canvas_size)
        weight = _panel_weight(mask, params)

        if multiband:
            # Multiband needs every panel at once, so hold them until the end.
            mb_panels.append(warped)
            mb_weights.append(weight)
        elif is_color:
            for ch in range(panel.shape[0]):
                result[ch] += warped[ch] * weight
        else:
            result += warped * weight
        weight_total += weight

    if multiband:
        levels = _pyramid_levels(canvas_size)
        if levels < 1:
            log.info("Canvas too small for multiband blending; feathering instead")
            multiband = False
        else:
            progress(0.9, f"Multiband blending ({levels} levels)...")
            if is_color:
                for ch in range(result.shape[0]):
                    result[ch] = _multiband_blend_plane(
                        [p[ch] for p in mb_panels], mb_weights, levels
                    )
            else:
                result = _multiband_blend_plane(mb_panels, mb_weights, levels)
            # Blending already normalized per level; only mask the empty area.
            result = np.where(weight_total > 0, result, 0.0).astype(np.float32)

    if not multiband:
        valid = weight_total > 0
        if is_color:
            for ch in range(result.shape[0]):
                result[ch][valid] /= weight_total[valid]
        else:
            result[valid] /= weight_total[valid]

    if params.clip_negative:
        result = np.maximum(result, 0)
    result = np.clip(result, 0, 1)
    progress(1.0, "Mosaic complete")

    return MosaicResult(
        data=result,
        n_panels=len(panels),
        output_shape=result.shape,
    )


def _compute_pairwise_transforms(
    panels: list[np.ndarray],
    progress: ProgressCallback,
) -> list[np.ndarray]:
    """Compute affine transforms from each panel to panel 0."""
    ref_sf = detect_stars(panels[0])
    transforms = [np.eye(2, 3, dtype=np.float32)]

    for i in range(1, len(panels)):
        frac = 0.4 * i / max(len(panels) - 1, 1)
        progress(frac, f"Registering panel {i + 1}/{len(panels)}...")
        panel_sf = detect_stars(panels[i])
        t = find_transform(ref_sf, panel_sf)
        if t is None:
            log.warning("Failed to register panel %d, using identity", i)
            t = np.eye(2, 3, dtype=np.float32)
        transforms.append(t)

    return transforms


def _compute_canvas(
    panels: list[np.ndarray],
    transforms: list[np.ndarray],
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    """Compute the output canvas size and per-panel offsets."""
    corner_points: list[list[float]] = []
    for panel, transform in zip(panels, transforms, strict=True):
        if panel.ndim == 3:
            h, w = panel.shape[1], panel.shape[2]
        else:
            h, w = panel.shape

        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        for c in corners:
            x = transform[0, 0] * c[0] + transform[0, 1] * c[1] + transform[0, 2]
            y = transform[1, 0] * c[0] + transform[1, 1] * c[1] + transform[1, 2]
            corner_points.append([float(x), float(y)])

    corners_np = np.array(corner_points)
    x_min, y_min = corners_np.min(axis=0)
    x_max, y_max = corners_np.max(axis=0)

    canvas_w = int(np.ceil(x_max - x_min))
    canvas_h = int(np.ceil(y_max - y_min))

    offsets: list[tuple[int, int]] = []
    for _ in transforms:
        ox = -x_min
        oy = -y_min
        offsets.append((int(ox), int(oy)))

    return (canvas_h, canvas_w), offsets


def _warp_panel(
    panel: np.ndarray,
    transform: np.ndarray,
    offset: tuple[int, int],
    canvas_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Warp a panel onto the canvas using its transform + offset."""
    h, w = canvas_size
    is_color = panel.ndim == 3

    t = transform.copy()
    t[0, 2] += offset[0]
    t[1, 2] += offset[1]

    if is_color:
        warped: np.ndarray = np.zeros((panel.shape[0], h, w), dtype=np.float32)
        for ch in range(panel.shape[0]):
            warped[ch] = cv2.warpAffine(
                panel[ch], t, (w, h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        mask = cv2.warpAffine(
            np.ones(panel.shape[1:], dtype=np.float32), t, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    else:
        warped = cv2.warpAffine(
            panel, t, (w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mask = cv2.warpAffine(
            np.ones_like(panel), t, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    return warped, mask


def _compute_blend_weight(
    mask: np.ndarray,
    feather_width: int,
) -> np.ndarray:
    """Compute feathered blend weight from a binary mask."""
    if feather_width <= 0:
        return mask

    binary = (mask > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    weight = np.clip(dist / max(feather_width, 1), 0, 1).astype(np.float32)
    return weight


def _panel_weight(mask: np.ndarray, params: MosaicParams) -> np.ndarray:
    """Per-panel blend weight for the selected blend method.

    AVERAGE uses a flat weight so overlapping panels are simply averaged;
    FEATHER (and MULTIBAND, whose per-level weights come from these) ramps
    the weight up over `feather_width` pixels from the panel edge.
    """
    if params.blend_method == BlendMethod.AVERAGE:
        return (mask > 0.5).astype(np.float32)
    return _compute_blend_weight(mask, params.feather_width)


def _pyramid_levels(shape: tuple[int, int], requested: int = 5) -> int:
    """How many pyramid levels the canvas can support (>=16 px at the top)."""
    smallest = max(min(shape), 1)
    possible = int(np.floor(np.log2(smallest / 16.0))) if smallest >= 32 else 0
    return int(max(0, min(requested, possible)))


def _laplacian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    gaussian = [img.astype(np.float32)]
    for _ in range(levels):
        gaussian.append(cv2.pyrDown(gaussian[-1]))
    pyramid = []
    for i in range(levels):
        up = cv2.pyrUp(gaussian[i + 1], dstsize=(gaussian[i].shape[1], gaussian[i].shape[0]))
        pyramid.append(gaussian[i] - up)
    pyramid.append(gaussian[-1])
    return pyramid


def _gaussian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    pyramid = [img.astype(np.float32)]
    for _ in range(levels):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


def _fill_outside(plane: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Extrapolate a panel past its footprint using the nearest valid pixel.

    A warped panel is zero outside its own footprint. Feeding that straight
    into a Laplacian pyramid puts a cliff at the panel border, and the coarse
    levels smear that black across the seam -- which made multiband blending
    *worse* than feathering for narrow overlaps. Filling with the nearest
    real pixel removes the artificial edge, so only genuine image content
    enters the pyramid; the weight maps still decide what is actually used.
    """
    if valid.all():
        return plane.astype(np.float32)
    if not valid.any():
        return plane.astype(np.float32)
    from scipy import ndimage

    idx = ndimage.distance_transform_edt(~valid, return_distances=False,
                                         return_indices=True)
    return plane[tuple(idx)].astype(np.float32)


def _multiband_blend_plane(
    planes: list[np.ndarray],
    weights: list[np.ndarray],
    levels: int,
) -> np.ndarray:
    """Burt-Adelson multiband blend of N aligned planes with N weight maps.

    Each frequency band is blended with the weight map smoothed to that
    band's scale. Low frequencies therefore cross the seam gradually (hiding
    residual brightness differences between panels) while high frequencies
    stay local, so detail is not smeared the way a wide feather would.
    """
    filled = [_fill_outside(p, w > 0) for p, w in zip(planes, weights, strict=True)]
    laplacians = [_laplacian_pyramid(p, levels) for p in filled]
    gaussians = [_gaussian_pyramid(w, levels) for w in weights]

    blended: list[np.ndarray] = []
    for level in range(levels + 1):
        num = np.zeros_like(laplacians[0][level])
        den = np.zeros_like(gaussians[0][level])
        for lap, gauss in zip(laplacians, gaussians, strict=True):
            num += lap[level] * gauss[level]
            den += gauss[level]
        with np.errstate(invalid="ignore", divide="ignore"):
            blended.append(np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0))

    out = blended[-1]
    for level in range(levels - 1, -1, -1):
        out = cv2.pyrUp(out, dstsize=(blended[level].shape[1], blended[level].shape[0]))
        out = out + blended[level]
    return out.astype(np.float32)


def _normalize_photometric(
    panels: list[np.ndarray],
    transforms: list[np.ndarray],
    offsets: list[tuple[int, int]],
    canvas_size: tuple[int, int],
    params: MosaicParams,
) -> list[np.ndarray]:
    """Normalize panel brightness using overlapping regions on the canvas.

    Warps each panel to the canvas, computes median ratios in overlaps,
    and propagates scale factors from a reference panel.
    """
    n = len(panels)
    is_color = panels[0].ndim == 3
    n_ch = panels[0].shape[0] if is_color else 1

    if is_color and params.norm_per_channel:
        factors: np.ndarray = np.ones((n, n_ch), dtype=np.float32)
    else:
        factors = np.ones(n, dtype=np.float32)

    warped_panels: list[np.ndarray] = []
    warped_masks: list[np.ndarray] = []

    for panel, transform, offset in zip(panels, transforms, offsets, strict=True):
        warped, mask = _warp_panel(panel, transform, offset, canvas_size)
        warped_panels.append(warped)
        warped_masks.append(mask)

    for i in range(n):
        for j in range(i + 1, n):
            overlap_mask = warped_masks[i] * warped_masks[j]
            if overlap_mask.sum() < 100:
                continue

            if is_color and params.norm_per_channel:
                for c in range(n_ch):
                    region_i = warped_panels[i][c][overlap_mask > 0]
                    region_j = warped_panels[j][c][overlap_mask > 0]
                    ratio = _robust_ratio(region_j, region_i, params.normalize)
                    if ratio > 0:
                        factors[j, c] = factors[i, c] * ratio
            else:
                if is_color:
                    region_i = np.mean(warped_panels[i], axis=0)[overlap_mask > 0]
                    region_j = np.mean(warped_panels[j], axis=0)[overlap_mask > 0]
                else:
                    region_i = warped_panels[i][overlap_mask > 0]
                    region_j = warped_panels[j][overlap_mask > 0]
                ratio = _robust_ratio(region_j, region_i, params.normalize)
                if ratio > 0:
                    factors[j] = factors[i] * ratio

    normalized: list[np.ndarray] = []
    for idx, panel in enumerate(panels):
        if is_color and params.norm_per_channel:
            p = panel.copy()
            for c in range(n_ch):
                p[c] = p[c] * factors[idx, c]
        else:
            p = panel * factors[idx]
        normalized.append(p)

    return normalized


def _robust_ratio(
    values_j: np.ndarray,
    values_i: np.ndarray,
    method: NormalizeMethod,
) -> float:
    """Compute robust median ratio between two overlapping regions."""
    valid = (values_i > 1e-6) & (values_j > 1e-6)
    if valid.sum() < 100:
        return 1.0

    ratios = values_j[valid] / values_i[valid]

    if method == NormalizeMethod.ROBUST:
        ratios = np.clip(ratios, 0.1, 10.0)
        return float(np.median(ratios))
    else:
        ratios = np.clip(ratios, 0.01, 100.0)
        return float(np.mean(ratios))
