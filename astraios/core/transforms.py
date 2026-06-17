"""Image Transforms — crop, rotate, flip, resize, bin, invert.

Basic geometric and pixel operations. All functions expect float32
images in [0, 1] with shape (H, W) for mono or (C, H, W) for color.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np
import torch.nn.functional as F

from astraios.core.device_manager import get_device_manager

log = logging.getLogger(__name__)


# ---------- Crop ----------


@dataclass
class CropParams:
    x: int = 0
    y: int = 0
    width: int = 0   # 0 = full remaining width
    height: int = 0  # 0 = full remaining height


def crop(image: np.ndarray, params: CropParams | None = None) -> np.ndarray:
    """Crop image to the specified rectangle."""
    if params is None:
        return image.copy()

    if image.ndim == 2:
        h, w = image.shape
    else:
        h, w = image.shape[1], image.shape[2]

    x = max(0, min(params.x, w - 1))
    y = max(0, min(params.y, h - 1))
    cw = params.width if params.width > 0 else w - x
    ch = params.height if params.height > 0 else h - y
    cw = min(cw, w - x)
    ch = min(ch, h - y)

    if image.ndim == 2:
        return image[y : y + ch, x : x + cw].copy()
    return image[:, y : y + ch, x : x + cw].copy()


# ---------- Rotate ----------


class RotateAngle(Enum):
    CW_90 = auto()
    CW_180 = auto()
    CW_270 = auto()
    ARBITRARY = auto()


@dataclass
class RotateParams:
    angle: RotateAngle = RotateAngle.CW_90
    arbitrary_degrees: float = 0.0
    expand: bool = True


def rotate(image: np.ndarray, params: RotateParams | None = None) -> np.ndarray:
    """Rotate image. 90/180/270 use np.rot90; arbitrary uses cv2.warpAffine."""
    if params is None:
        params = RotateParams()

    if params.angle == RotateAngle.CW_90:
        if image.ndim == 2:
            return np.rot90(image, k=-1).copy()
        return np.rot90(image, k=-1, axes=(1, 2)).copy()
    elif params.angle == RotateAngle.CW_180:
        if image.ndim == 2:
            return np.rot90(image, k=2).copy()
        return np.rot90(image, k=2, axes=(1, 2)).copy()
    elif params.angle == RotateAngle.CW_270:
        if image.ndim == 2:
            return np.rot90(image, k=1).copy()
        return np.rot90(image, k=1, axes=(1, 2)).copy()

    # Arbitrary angle
    deg = params.arbitrary_degrees
    if abs(deg) < 0.01:
        return image.copy()

    if image.ndim == 2:
        return _rotate_2d(image, deg, params.expand)

    channels = [_rotate_2d(image[c], deg, params.expand) for c in range(image.shape[0])]
    return np.stack(channels, axis=0)


def _rotate_2d(img: np.ndarray, degrees: float, expand: bool) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), -degrees, 1.0)

    if expand:
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        M[0, 2] += (new_w - w) / 2.0
        M[1, 2] += (new_h - h) / 2.0
        return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)

    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)


# ---------- Flip ----------


class FlipAxis(Enum):
    HORIZONTAL = auto()
    VERTICAL = auto()
    BOTH = auto()


@dataclass
class FlipParams:
    axis: FlipAxis = FlipAxis.HORIZONTAL


def flip(image: np.ndarray, params: FlipParams | None = None) -> np.ndarray:
    """Flip image along the specified axis."""
    if params is None:
        params = FlipParams()

    if params.axis == FlipAxis.HORIZONTAL:
        ax = -1
    elif params.axis == FlipAxis.VERTICAL:
        ax = -2
    else:
        return np.flip(np.flip(image, axis=-1), axis=-2).copy()

    return np.flip(image, axis=ax).copy()


# ---------- Resize / Resample ----------


class InterpolationMethod(Enum):
    NEAREST = auto()
    BILINEAR = auto()
    BICUBIC = auto()
    LANCZOS = auto()


_INTERP_MAP = {
    InterpolationMethod.NEAREST: cv2.INTER_NEAREST,
    InterpolationMethod.BILINEAR: cv2.INTER_LINEAR,
    InterpolationMethod.BICUBIC: cv2.INTER_CUBIC,
    InterpolationMethod.LANCZOS: cv2.INTER_LANCZOS4,
}


@dataclass
class ResizeParams:
    scale: float = 1.0
    target_width: int = 0
    target_height: int = 0
    interpolation: InterpolationMethod = InterpolationMethod.LANCZOS


def resize(image: np.ndarray, params: ResizeParams | None = None) -> np.ndarray:
    """Resize image by scale factor or to target dimensions (GPU-accelerated)."""
    if params is None:
        return image.copy()

    if image.ndim == 2:
        h, w = image.shape
    else:
        h, w = image.shape[1], image.shape[2]

    if params.target_width > 0 and params.target_height > 0:
        new_w, new_h = params.target_width, params.target_height
    elif params.target_width > 0:
        new_w = params.target_width
        new_h = int(h * new_w / max(w, 1))
    elif params.target_height > 0:
        new_h = params.target_height
        new_w = int(w * new_h / max(h, 1))
    else:
        new_w = max(1, int(w * params.scale))
        new_h = max(1, int(h * params.scale))

    dm = get_device_manager()
    if dm.is_gpu and image.shape[0] <= 64:
        mode_map = {
            InterpolationMethod.NEAREST: "nearest",
            InterpolationMethod.BILINEAR: "bilinear",
            InterpolationMethod.BICUBIC: "bicubic",
            InterpolationMethod.LANCZOS: "bicubic",
        }
        align_map = {True: True, False: False, "bicubic": True, "nearest": False}
        torch_mode = mode_map.get(params.interpolation, "bicubic")
        t = dm.from_numpy(image.astype(np.float32)).unsqueeze(0)
        result = F.interpolate(t, size=(new_h, new_w), mode=torch_mode,
                               align_corners=align_map.get(torch_mode, True))
        return dm.to_cpu(result.squeeze(0)).numpy().astype(np.float32)

    interp = _INTERP_MAP.get(params.interpolation, cv2.INTER_LANCZOS4)

    if image.ndim == 2:
        return cv2.resize(image, (new_w, new_h), interpolation=interp).astype(np.float32)

    channels = [
        cv2.resize(image[c], (new_w, new_h), interpolation=interp).astype(np.float32)
        for c in range(image.shape[0])
    ]
    return np.stack(channels, axis=0)


# ---------- Bin ----------


class BinMode(Enum):
    AVERAGE = auto()
    SUM = auto()


@dataclass
class BinParams:
    factor: int = 2
    mode: BinMode = BinMode.AVERAGE


def bin_image(image: np.ndarray, params: BinParams | None = None) -> np.ndarray:
    """Bin image by factor (2x2, 3x3, etc.)."""
    if params is None:
        params = BinParams()

    f = max(2, params.factor)

    if image.ndim == 2:
        return _bin_2d(image, f, params.mode)

    channels = [_bin_2d(image[c], f, params.mode) for c in range(image.shape[0])]
    return np.stack(channels, axis=0)


def _bin_2d(img: np.ndarray, factor: int, mode: BinMode) -> np.ndarray:
    h, w = img.shape
    new_h = (h // factor) * factor
    new_w = (w // factor) * factor
    if new_h != h or new_w != w:
        log.warning("Binning %dx%d: truncating %d rows × %d cols to %d × %d",
                    factor, factor, h, w, new_h, new_w)
    trimmed = img[:new_h, :new_w]
    reshaped = trimmed.reshape(new_h // factor, factor, new_w // factor, factor)
    if mode == BinMode.SUM:
        result = reshaped.sum(axis=(1, 3))
    else:
        result = reshaped.mean(axis=(1, 3))
    return result.astype(np.float32)


# ---------- Invert ----------


def invert(image: np.ndarray) -> np.ndarray:
    """Invert image: result = 1.0 - image."""
    return (1.0 - image).astype(np.float32)
