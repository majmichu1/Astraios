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

from cosmica.core.star_detection import detect_stars, find_transform

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

    for i, (panel, transform, offset) in enumerate(zip(panels, transforms, offsets, strict=True)):
        frac = 0.5 + 0.5 * i / max(len(panels) - 1, 1)
        progress(frac, f"Blending panel {i + 1}/{len(panels)}...")

        warped, mask = _warp_panel(panel, transform, offset, canvas_size)
        weight = _compute_blend_weight(mask, params.feather_width)

        if is_color:
            for ch in range(panel.shape[0]):
                result[ch] += warped[ch] * weight
        else:
            result += warped * weight
        weight_total += weight

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
