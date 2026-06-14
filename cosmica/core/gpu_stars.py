"""GPU-Accelerated Star Detection and Alignment using PyTorch.

Star class is imported from star_detection to avoid duplication.
GPU functions operate on tensors and call back to OpenCV RANSAC
for robust transform estimation.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as functional

from cosmica.core.device_manager import get_device_manager
from cosmica.core.star_detection import Star  # single definition, re-exported here

__all__ = [
    "Star",
    "detect_stars_gpu",
    "match_stars_gpu",
    "estimate_transform_gpu",
    "warp_image_gpu",
    "compose_affine_transforms",
]

log = logging.getLogger(__name__)


@torch.no_grad()
def detect_stars_gpu(
    data: torch.Tensor,
    threshold_sigma: float = 5.0,
    max_stars: int = 500,
    min_distance: int = 5,
) -> list[Star]:
    """Detect stars using GPU max-pooling for local maxima.

    Parameters
    ----------
    data : torch.Tensor
        Input image (H, W) or (C, H, W), float32 in [0, 1].
    threshold_sigma : float
        Detection threshold in sigmas above background.
    max_stars : int
        Maximum number of stars to return (brightest first).
    min_distance : int
        Minimum separation between stars in pixels.
    """
    image = data.mean(dim=0) if data.dim() == 3 else data

    # Noise estimation via median + MAD (same as CPU detect_stars)
    med = torch.median(image)
    mad = torch.median(torch.abs(image - med))
    noise_est = max(mad.item() * 1.4826, 0.01)  # floor prevents near-zero threshold
    thresh = min(med.item() + threshold_sigma * noise_est, 0.95)  # cap matches CPU path

    masked_image = torch.where(image > thresh, image, torch.zeros_like(image))

    # Local maxima via max pooling
    kernel = min_distance
    local_max = functional.max_pool2d(
        masked_image.unsqueeze(0).unsqueeze(0),
        kernel_size=kernel,
        stride=1,
        padding=kernel // 2,
    ).squeeze()

    is_max = (masked_image == local_max) & (masked_image > thresh)
    coords = torch.nonzero(is_max, as_tuple=False)

    if len(coords) == 0:
        return []

    # Collapse each connected cluster of local maxima to ONE detection at its
    # centroid. max_pool flags every pixel of a saturated flat-top plateau as a
    # local maximum, so a bright saturated star would otherwise be reported as
    # hundreds of detections. Labeling the maxima mask and taking the centroid
    # per connected component yields one sub-pixel star per blob — matching the
    # CPU contour-centroid detector and handling saturated stars correctly.
    from scipy import ndimage

    is_max_np = is_max.cpu().numpy()
    image_np = image.detach().cpu().numpy()
    labels, n_comp = ndimage.label(is_max_np)
    if n_comp == 0:
        return []

    comp_ids = range(1, n_comp + 1)
    centroids = ndimage.center_of_mass(is_max_np, labels, comp_ids)  # (row, col) sub-pixel
    cys = np.array([c[0] for c in centroids], dtype=np.float64)
    cxs = np.array([c[1] for c in centroids], dtype=np.float64)
    comp_flux = np.asarray(ndimage.maximum(image_np, labels, comp_ids), dtype=np.float64)

    keep = np.argsort(-comp_flux)[:max_stars]
    cys, cxs, comp_flux = cys[keep], cxs[keep], comp_flux[keep]

    H, W = image.shape
    patch_r = 5
    patch_size = 2 * patch_r + 1

    # Extract all star patches in one GPU operation via unfold (patch centered on
    # the rounded centroid) and estimate FWHM from the half-maximum area.
    padded = functional.pad(
        image.unsqueeze(0).unsqueeze(0),
        (patch_r, patch_r, patch_r, patch_r),
        mode="constant", value=0.0,
    )
    patches = functional.unfold(padded, kernel_size=patch_size, stride=1).squeeze(0)

    ri = np.clip(np.round(cys).astype(np.int64), 0, H - 1)
    ci = np.clip(np.round(cxs).astype(np.int64), 0, W - 1)
    star_idx = torch.from_numpy(ri * W + ci).to(image.device).long()
    peak = torch.from_numpy(comp_flux).to(image.device, dtype=image.dtype)
    star_patches = patches[:, star_idx]
    area = (star_patches > peak.unsqueeze(0) * 0.5).float().sum(dim=0)
    fwhm_vals = (2.0 * (area.clamp(min=1.0) / 3.14159).sqrt()).clamp(1.5, patch_r * 2.0)
    fwhm_cpu = fwhm_vals.cpu().tolist()

    return [
        Star(x=float(cxs[i]), y=float(cys[i]),
             flux=float(comp_flux[i]), fwhm=float(fwhm_cpu[i]))
        for i in range(len(cxs))
    ]


@torch.no_grad()
def match_stars_gpu(
    ref_stars: list[Star],
    target_stars: list[Star],
    max_dist: float,
) -> list[tuple[Star, Star]]:
    """Greedy nearest-neighbour star matching using GPU distance matrix.

    Returns list of (ref_star, target_star) pairs.
    """
    if not ref_stars or not target_stars:
        return []

    dm = get_device_manager()
    ref_pos = torch.tensor([[s.x, s.y] for s in ref_stars], device=dm.device)
    tgt_pos = torch.tensor([[s.x, s.y] for s in target_stars], device=dm.device)

    # Pairwise squared distances (N_ref, N_tgt) — computed on GPU
    diff = ref_pos.unsqueeze(1) - tgt_pos.unsqueeze(0)
    dist_sq = torch.sum(diff**2, dim=2)

    # Only transfer two small vectors (N_ref each) instead of the full matrix
    best_dist_sq, best_j = dist_sq.min(dim=1)
    best_dist_sq_cpu = best_dist_sq.cpu().numpy()
    best_j_cpu = best_j.cpu().numpy().astype(int)

    max_dist_sq = max_dist**2
    ref_order = np.argsort([-s.flux for s in ref_stars])

    matches: list[tuple[Star, Star]] = []
    used_targets: set[int] = set()
    for i in ref_order:
        j = int(best_j_cpu[i])
        if best_dist_sq_cpu[i] < max_dist_sq and j not in used_targets:
            matches.append((ref_stars[i], target_stars[j]))
            used_targets.add(j)

    return matches


@torch.no_grad()
def estimate_transform_gpu(
    matches: list[tuple[Star, Star]],
    ransac_threshold: float = 3.0,
) -> np.ndarray | None:
    """Estimate 2D similarity transform using OpenCV RANSAC for robustness.

    Uses GPU-computed matches (from match_stars_gpu) but runs RANSAC on CPU
    via OpenCV's estimateAffinePartial2D (translation + rotation + uniform scale).

    Parameters
    ----------
    matches : list of (ref_star, target_star) pairs
    ransac_threshold : float
        Max reprojection error in pixels for RANSAC inliers.

    Returns
    -------
    2×3 affine transform matrix (float32) or None if estimation fails.
    """
    if len(matches) < 3:
        return None

    src_pts = np.array([[m[1].x, m[1].y] for m in matches], dtype=np.float32)
    dst_pts = np.array([[m[0].x, m[0].y] for m in matches], dtype=np.float32)

    # OpenCV RANSAC is robust against mismatches (unlike pure least-squares)
    transform, inliers = cv2.estimateAffinePartial2D(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
    )

    if transform is None:
        log.warning("RANSAC transform estimation failed")
        return None

    n_inliers = int(inliers.sum()) if inliers is not None else 0
    log.debug("Transform estimated: %d/%d inliers", n_inliers, len(matches))
    return transform.astype(np.float32)


@torch.no_grad()
def warp_image_gpu(
    image: torch.Tensor,
    matrix: np.ndarray,
    mode: str = "bicubic",
) -> torch.Tensor:
    """Apply 2×3 affine transform to image on GPU.

    Parameters
    ----------
    image : torch.Tensor
        (H, W) or (C, H, W), float32.
    matrix : ndarray
        2×3 affine transform in pixel coordinates.
    mode : str
        Interpolation mode: "bilinear" (fast) or "bicubic" (quality).
    """
    original_2d = False
    if image.dim() == 2:
        image = image.unsqueeze(0)
        original_2d = True
    elif image.dim() != 3:
        raise ValueError(f"Unexpected image dims: {image.dim()}")

    c, h, w = image.shape

    # Pixel-space affine (output → input) to normalized coords for affine_grid.
    sx = 2.0 / max(w - 1, 1)
    sy = 2.0 / max(h - 1, 1)
    m = np.asarray(matrix, dtype=np.float64)
    m3 = np.vstack([m, [0.0, 0.0, 1.0]])
    to_norm = np.array([[sx, 0.0, -1.0], [0.0, sy, -1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    from_norm = np.array(
        [[1.0 / sx, 0.0, 1.0 / sx], [0.0, 1.0 / sy, 1.0 / sy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    norm3 = to_norm @ m3 @ from_norm
    norm_matrix = norm3[:2, :].astype(np.float32)

    theta = torch.from_numpy(norm_matrix).to(image.device).unsqueeze(0)  # [1, 2, 3]
    grid = functional.affine_grid(theta, (1, c, h, w), align_corners=True)
    warped = functional.grid_sample(
        image.unsqueeze(0), grid, mode=mode, align_corners=True, padding_mode="zeros"
    )
    warped = warped.squeeze(0)

    if original_2d:
        warped = warped.squeeze(0)

    return warped


def compose_affine_transforms(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Compose two 2×3 affine transforms: apply m1 first, then m2.

    Works for general 2×3 affine matrices (rotation, scale, shear, translation).
    Returns m_combined such that m_combined(p) = m2(m1(p)).
    """
    # Promote to 3×3 homogeneous
    mat1 = np.vstack([m1, [0, 0, 1]])
    mat2 = np.vstack([m2, [0, 0, 1]])
    mat = mat2 @ mat1
    return mat[:2, :].astype(np.float32)
