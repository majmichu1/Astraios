"""Resampling for registration: one warp, same kernel on the GPU and the CPU.

Until 2026-09 the GPU alignment paths resampled with ``grid_sample`` bicubic
while the CPU paths used OpenCV Lanczos-4, so the same stack differed by
hardware, and neither clamped the result. On a like-for-like test against
Siril 1.4.4 (same 50 subs, same reference, same stacking settings) the two
stacks had the same star peaks; Siril's was ~10% quieter at the single-pixel
scale and ~1.5% softer (FWHM 3.87 vs 3.81 px), the usual sharpness/noise
trade-off of the resampling kernel, and equal from 4x4 binning up. Clamping
limits every interpolated value to the range of the source pixels around the
sample point, which removes the ringing overshoot around stars (Siril's own
frames still reach 1.07 on saturated stars) and keeps a high-order kernel from
amplifying noise.

Kernels
-------
``bilinear``
    Softest and fastest; noise factor ~0.63 per frame on white noise at
    random sub-pixel phases, but it blurs stars slightly.
``bicubic``
    Keys cubic (a = -0.75), the same kernel in OpenCV and torch; ~0.83.
``lanczos4``
    8x8-tap windowed sinc, what Siril and PixInsight register with; ~0.90
    unclamped, ~0.87 clamped, sharpest.

Both devices implement every kernel: cv2.warpAffine on the CPU, torch on the
GPU (``grid_sample`` for bilinear/bicubic, a 64-tap gather for Lanczos-4).
``clamp`` is defined identically on both: the value is limited to the
[min, max] of the 3x3 source neighbourhood of the nearest source pixel.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional

INTERPOLATIONS = ("bilinear", "bicubic", "lanczos4")
DEFAULT_INTERPOLATION = "bicubic"

_CV2_FLAGS = {"bilinear": 1, "bicubic": 2, "lanczos4": 4}  # cv2.INTER_LINEAR/CUBIC/LANCZOS4


def _check(interpolation: str) -> str:
    key = str(interpolation).lower()
    if key not in INTERPOLATIONS:
        raise ValueError(f"interpolation must be one of {INTERPOLATIONS}, got {interpolation!r}")
    return key


def invert_affine(m: np.ndarray) -> np.ndarray:
    """Invert a 2x3 affine (forward source->destination) to destination->source."""
    m3 = np.vstack([np.asarray(m, dtype=np.float64), [0.0, 0.0, 1.0]])
    return np.linalg.inv(m3)[:2]


# --------------------------------------------------------------------------- CPU


def warp_affine_cpu(
    image: np.ndarray,
    transform: np.ndarray,
    out_shape: tuple[int, int],
    interpolation: str = DEFAULT_INTERPOLATION,
    clamp: bool = True,
) -> np.ndarray:
    """Warp ``image`` ((H, W) or (C, H, W)) by the *forward* 2x3 ``transform``.

    Output is ``out_shape`` = (height, width), float32, zero outside the
    source footprint (the stacker treats those zeros as missing data).
    """
    import cv2

    key = _check(interpolation)
    h, w = int(out_shape[0]), int(out_shape[1])
    mat = np.asarray(transform, dtype=np.float64)
    flag = _CV2_FLAGS[key]

    def one(ch: np.ndarray) -> np.ndarray:
        src = np.ascontiguousarray(ch, dtype=np.float32)
        out = cv2.warpAffine(
            src, mat, (w, h), flags=flag, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        )
        if clamp and key != "bilinear":
            k = np.ones((3, 3), np.uint8)
            lo = cv2.warpAffine(cv2.erode(src, k), mat, (w, h), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            hi = cv2.warpAffine(cv2.dilate(src, k), mat, (w, h), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            np.clip(out, lo, hi, out=out)
        return out

    if image.ndim == 3:
        return np.stack([one(image[c]) for c in range(image.shape[0])], axis=0)
    return one(image)


# --------------------------------------------------------------------------- GPU


def _source_coords(
    m_inv: torch.Tensor, h: int, w: int, y0: int, y1: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Source (u, v) for output pixels of rows y0:y1, from the inverse affine."""
    dev = m_inv.device
    xs = torch.arange(w, device=dev, dtype=torch.float32)
    ys = torch.arange(y0, y1, device=dev, dtype=torch.float32)
    u = m_inv[0, 0] * xs[None, :] + m_inv[0, 1] * ys[:, None] + m_inv[0, 2]
    v = m_inv[1, 0] * xs[None, :] + m_inv[1, 1] * ys[:, None] + m_inv[1, 2]
    return u, v


def _grid(u: torch.Tensor, v: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Normalised grid for grid_sample(align_corners=True): pixel x -> 2x/(W-1) - 1."""
    gx = u * (2.0 / max(w - 1, 1)) - 1.0
    gy = v * (2.0 / max(h - 1, 1)) - 1.0
    return torch.stack((gx, gy), dim=-1)[None]


def _sample(img4: torch.Tensor, grid: torch.Tensor, mode: str) -> torch.Tensor:
    return functional.grid_sample(
        img4, grid, mode=mode, padding_mode="zeros", align_corners=True
    )[0]


def _lanczos_weights(frac: torch.Tensor, ks: torch.Tensor, a: int) -> torch.Tensor:
    d = frac[..., None] - ks
    w = torch.where(d.abs() < 1e-6, torch.ones_like(d), torch.sinc(d) * torch.sinc(d / a))
    w = w * (d.abs() < a)
    return w / w.sum(-1, keepdim=True)


@torch.no_grad()
def warp_affine_gpu(
    image: torch.Tensor,
    transform_inv: np.ndarray | torch.Tensor,
    out_shape: tuple[int, int] | None = None,
    interpolation: str = DEFAULT_INTERPOLATION,
    clamp: bool = True,
    rows_per_chunk: int = 256,
) -> torch.Tensor:
    """Warp ``image`` ((H, W) or (C, H, W) tensor) by the *inverse* 2x3 affine.

    ``transform_inv`` maps output pixel coordinates to source coordinates,
    the convention of ``gpu_stars.warp_image_gpu``. Output is float32 on the
    same device, ``out_shape`` = (height, width) or the input shape, zero
    outside the source footprint. Lanczos-4 is a 64-tap gather processed in
    row chunks so a 24 MP colour frame never needs more than a few hundred
    MB of scratch.
    """
    key = _check(interpolation)
    squeeze = image.dim() == 2
    img = image[None] if squeeze else image
    if img.dim() != 3:
        raise ValueError(f"expected (H, W) or (C, H, W), got {tuple(image.shape)}")
    img = img.float()
    c, h, w = img.shape
    oh, ow = (h, w) if out_shape is None else (int(out_shape[0]), int(out_shape[1]))
    m_inv = torch.as_tensor(np.asarray(transform_inv, dtype=np.float32), device=img.device)
    img4 = img[None]

    if key in ("bilinear", "bicubic"):
        u, v = _source_coords(m_inv, oh, ow, 0, oh)
        out = _sample(img4, _grid(u, v, h, w), key)
    else:
        a = 4
        ks = torch.arange(-a + 1, a + 1, device=img.device, dtype=torch.float32)
        sx, sy = 2.0 / max(w - 1, 1), 2.0 / max(h - 1, 1)
        out = torch.empty((c, oh, ow), device=img.device, dtype=torch.float32)
        for y0 in range(0, oh, rows_per_chunk):
            y1 = min(oh, y0 + rows_per_chunk)
            u, v = _source_coords(m_inv, oh, ow, y0, y1)
            u0, v0 = torch.floor(u), torch.floor(v)
            wu = _lanczos_weights(u - u0, ks, a)  # (rows, W, 8)
            wv = _lanczos_weights(v - v0, ks, a)
            gx0 = u0 * sx - 1.0
            gy0 = v0 * sy - 1.0
            acc = torch.zeros((c, y1 - y0, ow), device=img.device, dtype=torch.float32)
            for j in range(2 * a):
                gy = gy0 + float(ks[j]) * sy
                row = torch.zeros_like(acc)
                for i in range(2 * a):
                    grid = torch.stack((gx0 + float(ks[i]) * sx, gy), dim=-1)[None]
                    row += _sample(img4, grid, "nearest") * wu[..., i][None]
                acc += row * wv[..., j][None]
            out[:, y0:y1] = acc

    if clamp and key != "bilinear":
        u, v = _source_coords(m_inv, oh, ow, 0, oh)
        grid = _grid(u, v, h, w)
        lo = _sample(-functional.max_pool2d(-img4, 3, 1, 1), grid, "nearest")
        hi = _sample(functional.max_pool2d(img4, 3, 1, 1), grid, "nearest")
        out = torch.minimum(torch.maximum(out, lo), hi)

    return out[0] if squeeze else out
