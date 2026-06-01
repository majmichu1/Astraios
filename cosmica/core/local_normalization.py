"""Local normalization for stacking — corrects local background variations between frames."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from cosmica.core.device_manager import get_device_manager

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)


@dataclass
class LocalNormParams:
    kernel_size: int = 51
    sigma: float = 50.0
    clip_limit: float = 0.1


def _gaussian_blur_cpu(image: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    if image.ndim == 2:
        return gaussian_filter(image, sigma=sigma, mode="reflect").astype(np.float32)
    return np.stack([
        gaussian_filter(image[c], sigma=sigma, mode="reflect").astype(np.float32)
        for c in range(image.shape[0])
    ])


def _gaussian_blur_gpu(image: torch.Tensor, sigma: float) -> torch.Tensor:
    import torch.nn.functional as F

    if image.dim() == 2:
        inp = image.unsqueeze(0).unsqueeze(0)
        squeeze = True
    else:
        inp = image.unsqueeze(0)
        squeeze = False

    c = inp.shape[1]
    h = inp.shape[2]
    w = inp.shape[3]

    radius = min(int(sigma * 3), max(1, h // 2 - 1), max(1, w // 2 - 1))
    kernel_size = 2 * radius + 1

    xs = torch.arange(kernel_size, dtype=torch.float32, device=inp.device) - radius
    g = torch.exp(-0.5 * (xs / sigma) ** 2)
    g = g / g.sum()

    kh = g.view(1, 1, 1, kernel_size).expand(c, 1, 1, kernel_size)
    kv = g.view(1, 1, kernel_size, 1).expand(c, 1, kernel_size, 1)

    out = F.conv2d(F.pad(inp, (radius, radius, 0, 0), mode="reflect"), kh, groups=c)
    out = F.conv2d(F.pad(out, (0, 0, radius, radius), mode="reflect"), kv, groups=c)

    if squeeze:
        return out.squeeze(0).squeeze(0)
    return out.squeeze(0)


def local_normalize(
    frames: list[np.ndarray] | np.ndarray,
    reference: np.ndarray | None = None,
    params: LocalNormParams | None = None,
) -> np.ndarray:
    """Apply local normalization to a list of frames.

    Corrects local background variations by computing a low-pass filtered
    version of each frame, then applying a spatially-varying scale+shift
    to match the reference frame's local background.

    Algorithm (per frame):
        1. Compute gaussian_blur(frame) — low-pass approximation of background.
        2. ratio = gaussian_blur(reference) / gaussian_blur(frame)
        3. result = (frame - gaussian_blur(frame)) * ratio + gaussian_blur(reference)

    This removes gradients, vignetting, and illumination differences while
    preserving high-frequency signal (stars, nebula features).

    Args:
        frames: List of (H,W) or (C,H,W) float32 arrays, or an (N,H,W) /
            (N,C,H,W) numpy stack.
        reference: Reference frame (default: median of all frames).
        params: LocalNormParams controlling kernel size, sigma, and clip limit.

    Returns:
        Normalized frames as a numpy stack of same shape as input.
    """
    if params is None:
        params = LocalNormParams()

    if isinstance(frames, np.ndarray) and frames.ndim >= 2:
        n = frames.shape[0]
        if n < 2:
            return frames
        frame_list = [frames[i] for i in range(n)]
    else:
        frame_list = list(frames)
        n = len(frame_list)
        if n < 2:
            return np.array(frame_list, dtype=np.float32)

    log.debug("Local normalization: %d frames, sigma=%.1f, clip_limit=%.3f",
              n, params.sigma, params.clip_limit)

    if reference is None:
        stack = np.stack(frame_list, axis=0)
        reference = np.median(stack, axis=0).astype(np.float32)

    dm = get_device_manager()
    total_pixels = int(np.prod(frame_list[0].shape))
    use_gpu = dm.is_gpu and total_pixels >= 256 * 256

    if use_gpu:
        ref_t = dm.from_numpy(reference.astype(np.float32, copy=True))
        bg_ref = _gaussian_blur_gpu(ref_t, params.sigma)

        result = []
        for frame in frame_list:
            frame_t = dm.from_numpy(frame.astype(np.float32, copy=True))
            bg_frame = _gaussian_blur_gpu(frame_t, params.sigma)
            ratio = bg_ref / bg_frame.clamp(min=1e-8)
            ratio = ratio.clamp(1.0 - params.clip_limit, 1.0 + params.clip_limit)
            norm_t = (frame_t - bg_frame) * ratio + bg_ref
            result.append(dm.to_cpu(norm_t).numpy().astype(np.float32))

        log.debug("Local normalization GPU: %d frames processed", n)
    else:
        bg_ref = _gaussian_blur_cpu(reference, params.sigma)
        result = []
        for frame in frame_list:
            bg_frame = _gaussian_blur_cpu(frame, params.sigma)
            ratio = bg_ref / np.maximum(bg_frame, 1e-8)
            ratio = np.clip(ratio, 1.0 - params.clip_limit, 1.0 + params.clip_limit)
            norm = (frame - bg_frame) * ratio + bg_ref
            result.append(norm.astype(np.float32))

        log.debug("Local normalization CPU: %d frames processed", n)

    return np.array(result, dtype=np.float32)
