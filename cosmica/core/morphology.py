"""Morphological Operations — erode, dilate, open, close (GPU-accelerated).

Erosion/dilation run on GPU via F.max_pool2d when available.
Falls back to OpenCV CPU for diamond/cross kernels (no GPU equivalent).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from cosmica.core.device_manager import get_device_manager
from cosmica.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)


class StructuringElement(Enum):
    CIRCLE = auto()
    SQUARE = auto()
    DIAMOND = auto()


class MorphOp(Enum):
    ERODE = auto()
    DILATE = auto()
    OPEN = auto()
    CLOSE = auto()


@dataclass
class MorphologyParams:
    operation: MorphOp = MorphOp.DILATE
    element: StructuringElement = StructuringElement.CIRCLE
    kernel_size: int = 3
    iterations: int = 1


def _get_kernel(element: StructuringElement, size: int) -> np.ndarray:
    size = max(3, size | 1)
    shape_map = {
        StructuringElement.CIRCLE: cv2.MORPH_ELLIPSE,
        StructuringElement.SQUARE: cv2.MORPH_RECT,
        StructuringElement.DIAMOND: cv2.MORPH_CROSS,
    }
    return cv2.getStructuringElement(shape_map[element], (size, size))


def _morphology_gpu(data: np.ndarray, op: MorphOp, kernel_size: int, iterations: int) -> np.ndarray:
    """GPU morphology via max_pool2d (erosion = -max_pool2d(-x))."""
    dm = get_device_manager()
    k = kernel_size
    pad = k // 2

    is_2d = data.ndim == 2
    t = dm.from_numpy(data.astype(np.float32))
    if is_2d:
        t = t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    else:
        t = t.unsqueeze(0)  # (1, C, H, W)

    for _ in range(iterations):
        if op == MorphOp.ERODE:
            t = -F.max_pool2d(-t, k, stride=1, padding=pad)
        elif op == MorphOp.DILATE:
            t = F.max_pool2d(t, k, stride=1, padding=pad)
        elif op == MorphOp.OPEN:
            t = -F.max_pool2d(-t, k, stride=1, padding=pad)
            t = F.max_pool2d(t, k, stride=1, padding=pad)
        elif op == MorphOp.CLOSE:
            t = F.max_pool2d(t, k, stride=1, padding=pad)
            t = -F.max_pool2d(-t, k, stride=1, padding=pad)

    result = dm.to_cpu(t).squeeze().numpy()
    return np.clip(result, 0, 1).astype(np.float32)


def _morphology_cpu(
    data: np.ndarray, kernel: np.ndarray, cv_op: int, iterations: int
) -> np.ndarray:
    """CPU morphology via OpenCV."""
    def _process_channel(ch: np.ndarray) -> np.ndarray:
        return cv2.morphologyEx(ch, cv_op, kernel, iterations=iterations)

    if data.ndim == 2:
        return _process_channel(data)
    result = np.empty_like(data)
    for ch in range(data.shape[0]):
        result[ch] = _process_channel(data[ch])
    return np.clip(result, 0, 1).astype(np.float32)


def morphology_transform(
    data: np.ndarray,
    params: MorphologyParams | None = None,
    mask: Mask | None = None,
) -> np.ndarray:
    if params is None:
        params = MorphologyParams()

    original = data.copy()
    dm = get_device_manager()

    use_gpu = dm.is_gpu and params.element != StructuringElement.DIAMOND

    if use_gpu:
        result = _morphology_gpu(data, params.operation, params.kernel_size, params.iterations)
    else:
        kernel = _get_kernel(params.element, params.kernel_size)
        op_map = {
            MorphOp.ERODE: cv2.MORPH_ERODE,
            MorphOp.DILATE: cv2.MORPH_DILATE,
            MorphOp.OPEN: cv2.MORPH_OPEN,
            MorphOp.CLOSE: cv2.MORPH_CLOSE,
        }
        result = _morphology_cpu(data, kernel, op_map[params.operation], params.iterations)

    return apply_mask(original, result, mask)


def morphology_mask(
    mask: Mask,
    params: MorphologyParams | None = None,
) -> Mask:
    if params is None:
        params = MorphologyParams()

    kernel = _get_kernel(params.element, params.kernel_size)
    op_map = {
        MorphOp.ERODE: cv2.MORPH_ERODE,
        MorphOp.DILATE: cv2.MORPH_DILATE,
        MorphOp.OPEN: cv2.MORPH_OPEN,
        MorphOp.CLOSE: cv2.MORPH_CLOSE,
    }
    cv_op = op_map[params.operation]
    result = cv2.morphologyEx(mask.data, cv_op, kernel, iterations=params.iterations)

    return Mask(
        data=np.clip(result, 0, 1).astype(np.float32),
        name=f"{mask.name} ({params.operation.name})",
        mask_type=mask.mask_type,
    )
