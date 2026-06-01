"""Lens distortion correction — barrel/pincushion correction via OpenCV."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from cosmica.core.star_detection import StarField, detect_stars

log = logging.getLogger(__name__)


@dataclass
class LensDistortionParams:
    """Parameters for lens distortion correction.

    Attributes
    ----------
    k1 : float
        Radial distortion coefficient k1.
    k2 : float
        Radial distortion coefficient k2.
    k3 : float
        Radial distortion coefficient k3.
    p1 : float
        Tangential distortion p1.
    p2 : float
        Tangential distortion p2.
    fov : float
        FOV in degrees (0 = auto from focal length).
    focal_length_mm : float
        Focal length (mm).
    sensor_width_mm : float
        Sensor width (mm, default full-frame 36 mm).
    """

    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    fov: float = 0.0
    focal_length_mm: float = 0.0
    sensor_width_mm: float = 36.0

    @property
    def distortion_coeffs(self) -> NDArray:
        """Return the 1x5 OpenCV distortion coefficient vector."""
        return np.array(
            [[self.k1, self.k2, self.p1, self.p2, self.k3]],
            dtype=np.float64,
        )


def _compute_camera_matrix(
    w: int,
    h: int,
    focal_mm: float,
    sensor_width_mm: float,
) -> NDArray:
    """Compute the camera intrinsic matrix.

    Parameters
    ----------
    w : int
        Image width in pixels.
    h : int
        Image height in pixels.
    focal_mm : float
        Focal length in millimetres.
    sensor_width_mm : float
        Sensor width in millimetres.

    Returns
    -------
    NDArray
        3x3 camera matrix.
    """
    focal_px = focal_mm / sensor_width_mm * w if focal_mm > 0 else max(w, h)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    return np.array(
        [[focal_px, 0.0, cx], [0.0, focal_px, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _radial_distortion_map(
    h: int,
    w: int,
    k1: float,
    k2: float,
    k3: float,
) -> tuple[NDArray, NDArray]:
    """Build (map_x, map_y) for remap-based undistortion with radial model.

    Uses the division model: r_corrected = r_distorted / (1 + k1*r^2 + k2*r^4 + k3*r^6).
    This is the inverse of the standard Brown-Conrady forward model, which is
    appropriate for creating a remap from destination → source.

    Parameters
    ----------
    h : int
        Image height.
    w : int
        Image width.
    k1, k2, k3 : float
        Radial distortion coefficients.

    Returns
    -------
    tuple[NDArray, NDArray]
        (map_x, map_y) arrays of shape (h, w) with float32 type.
    """
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    y, x = np.meshgrid(ys, xs, indexing="ij")

    x_norm = (x - cx) / max(w, h)
    y_norm = (y - cy) / max(w, h)
    r2 = x_norm * x_norm + y_norm * y_norm
    r4 = r2 * r2
    r6 = r4 * r2

    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    map_x = (x_norm * radial) * max(w, h) + cx
    map_y = (y_norm * radial) * max(w, h) + cy

    return map_x.astype(np.float32), map_y.astype(np.float32)


def correct_distortion(
    img: NDArray,
    params: LensDistortionParams,
) -> NDArray:
    """Apply lens distortion correction.

    Computes camera matrix from image size + focal length, then
    calls ``cv2.undistort``.

    Parameters
    ----------
    img : NDArray
        (H, W) or (H, W, C) float32/uint16 image.
    params : LensDistortionParams
        Distortion parameters.

    Returns
    -------
    NDArray
        Corrected image, same shape and dtype.
    """
    if img.ndim == 2:
        h, w = img.shape
    elif img.ndim == 3:
        h, w = img.shape[:2]
    else:
        raise ValueError(f"Expected 2D or 3D image, got shape {img.shape}")

    camera_matrix = _compute_camera_matrix(
        w, h, params.focal_length_mm, params.sensor_width_mm,
    )
    dist_coeffs = params.distortion_coeffs

    if np.allclose(dist_coeffs, 0.0):
        return img.copy()

    return cv2.undistort(img, camera_matrix, dist_coeffs)


def estimate_distortion_from_stars(
    img: NDArray,
    known_star_positions: list[tuple[float, float]] | None = None,
) -> LensDistortionParams:
    """Brute-force estimate k1, k2 from radial displacement of detected stars.

    If *known_star_positions* is *None*, extract stars via DAOFind-like peak
    detection.  The search minimises deviation of stars from straight radial
    lines after applying candidate (k1, k2) corrections.

    Parameters
    ----------
    img : NDArray
        (H, W) float32 image.
    known_star_positions : list of (float, float), optional
        Pre-supplied star (x, y) positions.  If *None*, stars are detected
        automatically.

    Returns
    -------
    LensDistortionParams
        Best-guess parameters with only k1, k2 set.
    """
    h, w = img.shape[:2]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    if known_star_positions is not None:
        pts = np.array(known_star_positions, dtype=np.float32)
    else:
        star_field: StarField = detect_stars(
            img, max_stars=200, sigma_threshold=5.0,
        )
        pts = np.array([(s.x, s.y) for s in star_field.stars], dtype=np.float32)

    if len(pts) < 5:
        log.warning("Too few stars (%d) for distortion estimation", len(pts))
        return LensDistortionParams()

    # Normalised coordinates
    norm = max(w, h)
    x_norm = (pts[:, 0] - cx) / norm
    y_norm = (pts[:, 1] - cy) / norm
    r2 = x_norm * x_norm + y_norm * y_norm

    best_score = float("inf")
    best_k1 = 0.0
    best_k2 = 0.0

    k1_range = np.linspace(-0.5, 0.5, 21)
    k2_range = np.linspace(-0.5, 0.5, 21)

    for k1 in k1_range:
        radial = 1.0 + k1 * r2
        x_corr = x_norm * radial * norm + cx
        y_corr = y_norm * radial * norm + cy
        angles = np.arctan2(y_corr - cy, x_corr - cx)
        score = float(np.std(angles))
        if score < best_score:
            best_score = score
            best_k1 = k1

    r4 = r2 * r2
    for k2 in k2_range:
        radial = 1.0 + best_k1 * r2 + k2 * r4
        x_corr = x_norm * radial * norm + cx
        y_corr = y_norm * radial * norm + cy
        angles = np.arctan2(y_corr - cy, x_corr - cx)
        score = float(np.std(angles))
        if score < best_score:
            best_score = score
            best_k2 = k2

    log.info(
        "Estimated distortion: k1=%.4f, k2=%.4f (score=%.6f, stars=%d)",
        best_k1, best_k2, best_score, len(pts),
    )
    return LensDistortionParams(k1=best_k1, k2=best_k2)
