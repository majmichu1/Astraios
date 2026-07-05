"""Multi-Frame Deconvolution — joint Richardson-Lucy sharpening across a stack.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro), Copyright
Franklin Marek, GPL-3.0-or-later. https://github.com/setiastro/setiastrosuitepro

Unlike single-frame deconvolution, this jointly solves for one sharp latent
image ``x`` that explains an entire stack of *registered* frames, each with
its own (independently measured) point-spread function. Every iteration
accumulates a numerator/denominator across all frames before applying a
single multiplicative update to ``x`` — frames with a tighter PSF or a
cleaner background pull harder on the result, which is what lets multi-frame
deconvolution converge to a sharper answer than deconvolving (or simply
averaging) any single frame.

Deviations from the original SASpro implementation (documented for anyone
diffing behavior against the reference tool):

* All convolutions run through PyTorch FFTs on ``get_device_manager()``'s
  device (CPU or GPU) instead of switching between a NumPy-FFT path and a
  Torch path depending on whether Torch is installed — Astraios always has
  Torch available, so a single code path suffices. This mirrors
  ``astraios.core.deconvolution``'s ``_padded_kernel_fft`` circular-FFT
  convolution rather than SASpro's linear ("SAME"-padded) FFT convolution;
  for PSF kernels much smaller than the image (the normal case) the two are
  visually identical and the circular form is cheaper.
* Frames are processed one at a time per iteration (never stacked into one
  big batch tensor), which is what keeps VRAM use bounded regardless of how
  many frames are in the set — this is the same strategy SASpro's CUDA
  variant uses to stay OOM-safe on consumer GPUs.
* PSF estimation reuses ``astraios.core.star_detection.detect_stars`` instead
  of SASpro's SEP-based star finder.
* Per-frame "rejection" masks use the Astraios masking convention (1 = use
  this pixel, 0 = exclude it) rather than SASpro's internal inverted
  convention, for consistency with the rest of the codebase.
* The super-resolution (drizzle-like) PSF is produced with a direct
  Kronecker upsample of the native PSF rather than SASpro's iterative
  gradient-descent PSF-fit — equivalent under the same block-average
  down-sampling forward model, without needing a nested optimizer.
* The final result is clipped to ``[0, 1]`` to satisfy Astraios's image-data
  invariant (SASpro leaves it unclipped above 1).
* SASpro's disk-memmap streaming, multi-process PSF-estimation pool, and
  FITS/XISF I/O layers are omitted — this module operates purely on
  in-memory registered frames, as required by the rest of Astraios's core
  processing modules.

The robustly-weighted multiplicative update, kappa clamp, relaxation
(damping), and early-stop convergence check are kept numerically faithful
to the original.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from astropy.stats import SigmaClip
from scipy.ndimage import gaussian_filter as _gaussian_filter
from scipy.ndimage import shift as _ndi_shift

from astraios.core.device_manager import get_device_manager
from astraios.core.masks import Mask, apply_mask
from astraios.core.star_detection import detect_stars

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]

_EPS = 1e-6


def _noop_progress(fraction: float, message: str) -> None:
    pass


@dataclass
class MultiFrameDeconvParams:
    """Settings for joint multi-frame Richardson-Lucy deconvolution.

    Attributes
    ----------
    iterations : int
        Maximum number of RL iterations to run.
    min_iterations : int
        Minimum iterations before early-stop is allowed to trigger.
    kappa : float
        Clamps the per-pixel multiplicative update to ``[1/kappa, kappa]``
        each iteration, limiting overshoot/ringing.
    relaxation : float
        Damping factor blending the raw update into the estimate
        (1.0 = full step, no damping).
    rho : str
        Residual loss used to weight pixels: ``"huber"`` (robust, default)
        or ``"l2"`` (classic, unweighted Richardson-Lucy).
    huber_delta : float
        Huber transition point, in image-intensity units. Negative means
        "auto": ``|huber_delta|`` times the residual's robust RMS (MAD-based),
        re-estimated periodically per frame.
    seed_mode : str
        How the initial estimate ``x0`` is built: ``"robust"`` (sigma-clipped
        mean across frames, astropy ``SigmaClip``), ``"median"``, ``"mean"``,
        or ``"integrated"`` (use ``seed_image`` verbatim).
    seed_image : ndarray or None
        Externally supplied seed image; required when ``seed_mode="integrated"``.
    sigma_clip : float
        Sigma threshold used by ``seed_mode="robust"``.
    color_mode : str
        ``"luma"`` solves a single shared-luminance channel (fastest, no
        color deconvolved separately); ``"perchannel"`` solves every RGB
        channel independently with the same per-frame PSF.
    psf_ksize : int or None
        Force an odd PSF kernel size in pixels. ``None`` auto-sizes from the
        measured stellar FWHM.
    psf_star_max : int
        Maximum number of stars used to build each frame's empirical PSF.
    psf_det_sigma : float
        Star-detection threshold (MAD-sigma units) used when measuring
        per-frame PSFs.
    psf_max_eccentricity : float
        Reject elongated/trailed stars above this roundness when fitting a PSF.
    psf_soften_sigma : float
        Gaussian pre-blur (pixels) applied to each fitted PSF to suppress
        pixel-grid noise in the empirical kernel.
    use_variance_maps : bool
        Weight residuals by each frame's estimated background-noise variance,
        so cleaner frames pull harder on the result.
    frame_masks : list of ndarray or None
        Optional per-frame weighting masks, one per frame, ``(H, W)`` float32
        in ``[0, 1]`` (1 = use this pixel, 0 = exclude it) — e.g. to reject
        satellite trails or plane streaks local to one frame.
    rejection_strength : float
        Blend factor between ignoring ``frame_masks`` entirely (0.0) and
        fully applying them (1.0); values in between blend two independent
        solves.
    super_resolution : int
        Integer output upsampling factor (drizzle-like); 1 = native resolution.
    sr_sigma : float
        Reserved for compatibility with the SASpro SR-PSF fit; unused by the
        direct Kronecker PSF lift used here.
    low_vram : bool
        Release the GPU cache after every frame instead of only between
        iterations — slower, but keeps peak VRAM lower on constrained GPUs.
    force_cpu : bool
        Disable the GPU path even if one is available.
    early_stop : bool
        Stop iterating once the update size and relative change both plateau.
    early_stop_tol_update : float
        Minimum meaningful median multiplicative-update size (plateau floor).
    early_stop_tol_relchange : float
        Minimum meaningful relative pixel change (plateau floor).
    early_stop_frac : float
        Fraction of the first iteration's update/change used as an additional
        adaptive tolerance (plateau relative to how much the solve moved
        initially).
    early_stop_patience : int
        Consecutive plateaued iterations required before stopping.
    """

    iterations: int = 20
    min_iterations: int = 3
    kappa: float = 2.0
    relaxation: float = 0.7
    rho: str = "huber"
    huber_delta: float = -1.5
    seed_mode: str = "robust"
    seed_image: np.ndarray | None = None
    sigma_clip: float = 5.0
    color_mode: str = "luma"
    psf_ksize: int | None = None
    psf_star_max: int = 80
    psf_det_sigma: float = 6.0
    psf_max_eccentricity: float = 0.5
    psf_soften_sigma: float = 0.25
    use_variance_maps: bool = False
    frame_masks: list[np.ndarray] | None = field(default=None)
    rejection_strength: float = 1.0
    super_resolution: int = 1
    sr_sigma: float = 1.1
    low_vram: bool = False
    force_cpu: bool = False
    early_stop: bool = True
    early_stop_tol_update: float = 2e-4
    early_stop_tol_relchange: float = 5e-4
    early_stop_frac: float = 0.40
    early_stop_patience: int = 2


@dataclass
class MultiFrameDeconvResult:
    """Extra diagnostics alongside the deconvolved image."""

    image: np.ndarray
    psfs: list[np.ndarray]
    used_iterations: int
    early_stopped: bool


# ---------------------------------------------------------------------------
# Early-stop convergence check (ported from SASpro's EarlyStopper)
# ---------------------------------------------------------------------------


class _EarlyStopper:
    """Detects when the multiplicative update has plateaued.

    Tracks an EMA of the per-iteration update magnitude (``um``) and relative
    change (``rc``); stops once both have shrunk below an adaptive tolerance
    for ``patience`` consecutive iterations.
    """

    def __init__(
        self,
        tol_upd_floor: float,
        tol_rel_floor: float,
        early_frac: float,
        patience: int,
        min_iters: int,
        ema_alpha: float = 0.5,
    ) -> None:
        self.tol_upd_floor = float(tol_upd_floor)
        self.tol_rel_floor = float(tol_rel_floor)
        self.early_frac = float(early_frac)
        self.ema_alpha = float(ema_alpha)
        self.patience = int(patience)
        self.min_iters = int(min_iters)

        self._initialized = False
        self.ema_um: float = 0.0
        self.ema_rc: float = 0.0
        self.base_um: float = 0.0
        self.base_rc: float = 0.0
        self.early_cnt = 0

    def step(self, it: int, um: float, rc: float) -> bool:
        um = float(um)
        rc = float(rc)

        if it == 1 or not self._initialized:
            self.ema_um = um
            self.ema_rc = rc
            self.base_um = um
            self.base_rc = rc
            self._initialized = True
        else:
            a = self.ema_alpha
            self.ema_um = a * um + (1.0 - a) * self.ema_um
            self.ema_rc = a * rc + (1.0 - a) * self.ema_rc

        b_um = self.base_um if self.base_um > 0 else um
        b_rc = self.base_rc if self.base_rc > 0 else rc

        tol_um = max(self.tol_upd_floor, self.early_frac * b_um)
        tol_rc = max(self.tol_rel_floor, self.early_frac * b_rc)

        small = (self.ema_um < tol_um) or (self.ema_rc < tol_rc)

        if small and it >= self.min_iters:
            self.early_cnt += 1
        else:
            self.early_cnt = 0

        return self.early_cnt >= self.patience


# ---------------------------------------------------------------------------
# PSF estimation and utilities
# ---------------------------------------------------------------------------


def _to_luma(frame: np.ndarray) -> np.ndarray:
    """Reduce a (C, H, W) or (H, W) frame to a single luminance plane."""
    if frame.ndim == 2:
        return frame.astype(np.float32, copy=False)
    if frame.shape[0] == 3:
        r, g, b = frame[0], frame[1], frame[2]
        luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return np.asarray(luma, dtype=np.float32)
    return np.asarray(frame.mean(axis=0), dtype=np.float32)


def _normalize_psf(psf: np.ndarray) -> np.ndarray:
    psf = np.maximum(psf, 0.0).astype(np.float32, copy=False)
    total = float(psf.sum())
    if not np.isfinite(total) or total <= 1e-8:
        # Degenerate kernel (e.g. all-zero) — fall back to a delta so the
        # solver still has a valid, sum-1 PSF to work with.
        out = np.zeros_like(psf)
        c = tuple(s // 2 for s in out.shape)
        out[c] = 1.0
        return out
    return (psf / total).astype(np.float32, copy=False)


def _soften_psf(psf: np.ndarray, sigma_px: float) -> np.ndarray:
    if sigma_px <= 0:
        return psf
    return _normalize_psf(_gaussian_filter(psf, sigma=float(sigma_px)))


def _gaussian_psf(fwhm_px: float, ksize: int) -> np.ndarray:
    sigma = max(fwhm_px, 1.0) / 2.3548
    r = (ksize - 1) // 2
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    g = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return _normalize_psf(g.astype(np.float32))


def _auto_ksize_from_fwhm(fwhm_px: float, kmin: int = 11, kmax: int = 51) -> int:
    """Choose an odd kernel size covering roughly +/-4 sigma."""
    import math

    sigma = max(fwhm_px, 1.0) / 2.3548
    r = int(math.ceil(4.0 * sigma))
    k = 2 * r + 1
    k = max(kmin, min(k, kmax))
    return k if k % 2 == 1 else k + 1


def _clamp_ksize_to_image(ksize: int, h: int, w: int) -> int:
    """Never let a PSF kernel be as large as (or larger than) the image."""
    limit = max(3, min(h, w) - 1)
    k = min(ksize, limit)
    return k if k % 2 == 1 else max(3, k - 1)


def _flip_kernel(psf: np.ndarray) -> np.ndarray:
    # Torch dislikes negative-stride views, so force a fresh contiguous copy.
    return np.flip(np.flip(psf, -1), -2).copy()


def _cutout_subpixel(gray: np.ndarray, cy: float, cx: float, ksize: int) -> np.ndarray | None:
    h, w = gray.shape
    iy, ix = int(round(cy)), int(round(cx))
    half = ksize // 2
    y0, y1 = iy - half, iy + half + 1
    x0, x1 = ix - half, ix + half + 1
    if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
        return None
    patch = gray[y0:y1, x0:x1].astype(np.float32, copy=True)
    dy, dx = cy - iy, cx - ix
    if abs(dy) > 1e-3 or abs(dx) > 1e-3:
        patch = _ndi_shift(patch, shift=(dy, dx), order=3, mode="nearest")
    return patch


def estimate_frame_psf(gray: np.ndarray, params: MultiFrameDeconvParams) -> np.ndarray:
    """Build an empirical PSF for one frame from its brightest, roundest stars.

    Falls back to an analytic Gaussian PSF (sized from the median detected
    stellar FWHM, or a generic default) when no usable stars are found.
    """
    h, w = gray.shape
    field = detect_stars(
        gray, max_stars=max(200, params.psf_star_max * 3), sigma_threshold=params.psf_det_sigma
    )

    candidates = [
        s
        for s in field.stars
        if s.roundness <= params.psf_max_eccentricity and 0.02 < s.flux < 0.98
    ]
    if not candidates:
        candidates = [s for s in field.stars if s.flux < 0.98]

    fwhms = [s.fwhm for s in candidates if s.fwhm > 0]
    fwhm_med = float(np.median(fwhms)) if fwhms else 3.0

    ksize = params.psf_ksize if params.psf_ksize else _auto_ksize_from_fwhm(fwhm_med)
    ksize = _clamp_ksize_to_image(ksize, h, w)

    patches = []
    for s in sorted(candidates, key=lambda s: -s.flux)[: params.psf_star_max]:
        patch = _cutout_subpixel(gray, s.y, s.x, ksize)
        if patch is None:
            continue
        total = float(patch.sum())
        if total <= 0:
            continue
        patches.append(patch / total)

    if not patches:
        log.debug("multiframe_deconv: no usable stars, falling back to Gaussian PSF")
        psf = _gaussian_psf(fwhm_med, ksize)
    else:
        psf = np.median(np.stack(patches, axis=0), axis=0).astype(np.float32)
        psf = _normalize_psf(psf)

    return _soften_psf(psf, params.psf_soften_sigma)


def _lift_psf_for_super_resolution(psf: np.ndarray, r: int) -> np.ndarray:
    """Lift a native-resolution PSF onto an r-times finer output grid.

    Uses a direct Kronecker upsample (each native pixel becomes an r x r
    block) rather than SASpro's iterative gradient-descent PSF fit — under
    the block-average down-sampling forward model used below, this is the
    equivalent operator without the extra optimization.
    """
    if r <= 1:
        return psf
    lifted = np.kron(psf, np.ones((r, r), dtype=np.float32))
    return _normalize_psf(lifted)


# ---------------------------------------------------------------------------
# Frame preparation
# ---------------------------------------------------------------------------


def _as_frame_list(frames: Sequence[np.ndarray] | np.ndarray) -> list[np.ndarray]:
    if isinstance(frames, np.ndarray):
        if frames.ndim not in (3, 4):
            raise ValueError(
                "A stacked frames array must be (N, H, W) or (N, C, H, W), "
                f"got ndim={frames.ndim}"
            )
        frame_list = [np.asarray(frames[i], dtype=np.float32) for i in range(frames.shape[0])]
    else:
        frame_list = [np.asarray(f, dtype=np.float32) for f in frames]

    if not frame_list:
        raise ValueError("multiframe_deconvolve requires at least one frame")

    ref_shape = frame_list[0].shape
    for i, f in enumerate(frame_list):
        if f.shape != ref_shape:
            raise ValueError(
                f"Frame {i} has shape {f.shape}, expected {ref_shape} — "
                "all frames must already be registered to an identical shape"
            )
    return frame_list


def _prepare_frames(frame_list: list[np.ndarray], color_mode: str) -> list[np.ndarray]:
    """Coerce every frame to (C, H, W) float32, C=1 for luma mode."""
    out = []
    for f in frame_list:
        if color_mode == "luma":
            out.append(_to_luma(f)[None, ...])
        elif f.ndim == 2:
            out.append(f[None, ...])
        else:
            out.append(f.astype(np.float32, copy=False))

    channel_counts = {a.shape[0] for a in out}
    if len(channel_counts) > 1:
        raise ValueError(f"Mixed channel counts across frames: {sorted(channel_counts)}")
    return out


def _build_seed(prepared: list[np.ndarray], params: MultiFrameDeconvParams) -> np.ndarray:
    mode = params.seed_mode.lower().strip()
    stack = np.stack(prepared, axis=0)  # (N, C, H, W)

    if mode == "integrated":
        if params.seed_image is None:
            raise ValueError("seed_mode='integrated' requires params.seed_image")
        seed = np.asarray(params.seed_image, dtype=np.float32)
        if seed.ndim == 2:
            seed = seed[None, ...]
        if seed.shape[0] != stack.shape[2] and stack.shape[2] == 1:
            seed = _to_luma(seed)[None, ...]
        if seed.shape[-2:] != stack.shape[-2:]:
            raise ValueError(
                f"seed_image spatial shape {seed.shape[-2:]} does not match "
                f"frame shape {stack.shape[-2:]}"
            )
        return seed.astype(np.float32, copy=False)

    if mode == "median":
        seed = np.median(stack, axis=0)
    elif mode == "mean":
        seed = np.mean(stack, axis=0)
    elif mode == "robust":
        sigclip = SigmaClip(sigma=params.sigma_clip, maxiters=5)
        clipped = sigclip(stack, axis=0)
        fallback = np.mean(stack, axis=0)
        seed = np.ma.mean(clipped, axis=0)
        seed = seed.filled(fallback) if np.ma.is_masked(seed) else np.asarray(seed)
    else:
        raise ValueError(f"Unknown seed_mode: {params.seed_mode!r}")

    return np.ascontiguousarray(seed, dtype=np.float32)


def _upsample_seed_np(seed: np.ndarray, r: int) -> np.ndarray:
    if r <= 1:
        return seed
    scaled = seed / float(r * r)
    return np.stack(
        [np.kron(scaled[c], np.ones((r, r), dtype=np.float32)) for c in range(seed.shape[0])],
        axis=0,
    )


def _prepare_frame_masks(
    frame_masks: list[np.ndarray] | None, n_frames: int, hw: tuple[int, int]
) -> list[np.ndarray | None]:
    if frame_masks is None:
        return [None] * n_frames
    if len(frame_masks) != n_frames:
        raise ValueError(
            f"frame_masks has {len(frame_masks)} entries, expected {n_frames} (one per frame)"
        )
    out: list[np.ndarray | None] = []
    for m in frame_masks:
        if m is None:
            out.append(None)
            continue
        mm = np.asarray(m, dtype=np.float32)
        if mm.ndim == 3:
            mm = mm[0]
        if mm.shape != hw:
            raise ValueError(f"frame mask shape {mm.shape} does not match frame shape {hw}")
        out.append(np.clip(mm, 0.0, 1.0))
    return out


def _estimate_frame_variance(gray: np.ndarray) -> float:
    med = float(np.median(gray))
    mad = float(np.median(np.abs(gray - med))) + 1e-6
    return float((1.4826 * mad) ** 2)


# ---------------------------------------------------------------------------
# Torch FFT convolution (circular, matches astraios.core.deconvolution style)
# ---------------------------------------------------------------------------


def _padded_kernel_fft(kernel: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """rfft2 of ``kernel`` zero-padded to (h, w) and centred at the origin."""
    kh, kw = kernel.shape
    padded = torch.zeros(h, w, device=kernel.device, dtype=kernel.dtype)
    padded[:kh, :kw] = kernel
    padded = torch.roll(padded, (-(kh // 2), -(kw // 2)), dims=(0, 1))
    return torch.fft.rfft2(padded)


def _fft_convolve(image: torch.Tensor, kernel_fft: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """FFT-convolve (C, H, W) or (H, W) ``image`` with a precomputed kernel FFT."""
    return torch.fft.irfft2(torch.fft.rfft2(image) * kernel_fft, s=(h, w))


def _downsample_avg_t(x: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 1:
        return x
    *lead, h, w = x.shape
    hs, ws = (h // r) * r, (w // r) * r
    x = x[..., :hs, :ws]
    x = x.reshape(*lead, hs // r, r, ws // r, r)
    return x.mean(dim=(-3, -1))


def _upsample_sum_t(x: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 1:
        return x
    return x.repeat_interleave(r, dim=-2).repeat_interleave(r, dim=-1)


def _weight_map_t(
    y: torch.Tensor,
    pred: torch.Tensor,
    huber_delta: float,
    var: torch.Tensor | None,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Huber-weighted precision map: psi(r)/r * 1/(var+eps) * mask."""
    r = y - pred

    if huber_delta < 0:
        med = torch.median(r)
        mad = torch.median(torch.abs(r - med)) + 1e-6
        delta = float(-huber_delta) * float(torch.clamp(1.4826 * mad, min=1e-6))
    else:
        delta = float(huber_delta)

    absr = torch.abs(r)
    if delta > 0.0:
        delta_t = torch.tensor(delta, dtype=r.dtype, device=r.device)
        psi_over_r = torch.where(absr <= delta_t, torch.ones_like(r), delta_t / (absr + _EPS))
    else:
        psi_over_r = torch.ones_like(r)

    if var is None:
        medv = torch.median(r)
        madv = torch.median(torch.abs(r - medv)) + 1e-6
        v = torch.clamp((1.4826 * madv) ** 2, min=1e-8)
    else:
        v = var if var.dim() == r.dim() else var.unsqueeze(0)
        v = torch.clamp(v, min=1e-8)

    w = psi_over_r / (v + _EPS)

    if mask is not None:
        m = mask if mask.dim() == w.dim() else mask.unsqueeze(0)
        w = w * m

    return torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Core RL solver
# ---------------------------------------------------------------------------


def _run_solver(
    x_seed: torch.Tensor,
    frame_tensors: list[torch.Tensor],
    psf_ffts: list[torch.Tensor],
    psf_t_ffts: list[torch.Tensor],
    mask_tensors: list[torch.Tensor | None],
    var_tensors: list[torch.Tensor | None],
    params: MultiFrameDeconvParams,
    r: int,
    progress: ProgressCallback,
    progress_range: tuple[float, float],
) -> tuple[torch.Tensor, int, bool]:
    x = x_seed.clone()
    num = torch.zeros_like(x)
    den = torch.zeros_like(x)
    hs, ws = x.shape[-2:]
    n_frames = len(frame_tensors)

    rho_is_l2 = params.rho.lower() == "l2"
    local_delta = 0.0 if rho_is_l2 else params.huber_delta

    early = (
        _EarlyStopper(
            tol_upd_floor=params.early_stop_tol_update,
            tol_rel_floor=params.early_stop_tol_relchange,
            early_frac=params.early_stop_frac,
            patience=params.early_stop_patience,
            min_iters=params.min_iterations,
        )
        if params.early_stop
        else None
    )

    dm = get_device_manager()
    max_iters = max(1, int(params.iterations))
    used_iters = max_iters
    early_stopped = False
    p0, p1 = progress_range

    for it in range(1, max_iters + 1):
        num.zero_()
        den.zero_()

        for i in range(n_frames):
            y = frame_tensors[i]
            mt = mask_tensors[i]
            vt = var_tensors[i]

            if r > 1:
                pred_super = _fft_convolve(x, psf_ffts[i], hs, ws)
                pred = _downsample_avg_t(pred_super, r)
                del pred_super
            else:
                pred = _fft_convolve(x, psf_ffts[i], hs, ws)

            wmap = _weight_map_t(y, pred, local_delta, vt, mt)
            wy = wmap * y
            wp = wmap * pred
            if r > 1:
                wy = _upsample_sum_t(wy, r)
                wp = _upsample_sum_t(wp, r)

            num += _fft_convolve(wy, psf_t_ffts[i], hs, ws)
            den += _fft_convolve(wp, psf_t_ffts[i], hs, ws)

            del pred, wmap, wy, wp
            if params.low_vram and dm.is_gpu:
                dm.empty_cache()

        neutral = (den.abs() < 1e-12 + _EPS) & (num.abs() < 1e-12)
        den = den + _EPS
        ratio = num / den
        ratio = torch.where(neutral, torch.ones_like(ratio), ratio)
        kappa = max(params.kappa, 1.0 + 1e-6)
        ratio = torch.clamp(ratio, 1.0 / kappa, kappa)

        um = float(torch.median(torch.abs(ratio - 1.0)).item())
        x_next = torch.clamp(x * ratio, min=0.0)
        rc = float(
            (
                torch.median(torch.abs(x_next - x)) / (torch.median(torch.abs(x)) + 1e-8)
            ).item()
        )

        frac = p0 + (p1 - p0) * (it / max_iters)
        progress(frac, f"Multi-frame deconvolution: iteration {it}/{max_iters}")

        if early is not None and early.step(it, um, rc):
            x = x_next
            used_iters = it
            early_stopped = True
            break

        x = x * (1.0 - params.relaxation) + params.relaxation * x_next

    return x, used_iters, early_stopped


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def multiframe_deconvolve(
    frames: Sequence[np.ndarray] | np.ndarray,
    params: MultiFrameDeconvParams | None = None,
    psfs: Sequence[np.ndarray] | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback | None = None,
) -> np.ndarray:
    """Jointly deconvolve a stack of registered frames into one sharp image.

    Parameters
    ----------
    frames : list of ndarray, or ndarray
        Registered (already aligned, identical shape) frames. Either a list
        of ``(H, W)``/``(C, H, W)`` arrays, or a single stacked array shaped
        ``(N, H, W)``/``(N, C, H, W)``.
    params : MultiFrameDeconvParams, optional
        Solver settings. Defaults to a moderate 20-iteration Huber solve.
    psfs : list of ndarray, optional
        One PSF kernel per frame (odd-sized, 2D). When omitted, a PSF is
        estimated per frame from its detected stars (see
        ``estimate_frame_psf``).
    mask : Mask, optional
        Final protect mask, blended in as
        ``result = deconvolved * mask + seed * (1 - mask)`` against the
        multi-frame seed image, at the (possibly super-resolved) output
        resolution.
    progress : callable, optional
        ``progress(fraction, message)`` callback.

    Returns
    -------
    ndarray
        The deconvolved image, float32 in ``[0, 1]``, shape ``(H, W)`` (mono
        or ``color_mode="luma"``) or ``(C, H, W)`` (``color_mode="perchannel"``
        on color input). If ``super_resolution > 1``, spatial dimensions are
        scaled by that factor.
    """
    params = params or MultiFrameDeconvParams()
    progress = progress or _noop_progress

    color_mode = params.color_mode.lower().strip()
    if color_mode not in ("luma", "perchannel"):
        raise ValueError(f"color_mode must be 'luma' or 'perchannel', got {params.color_mode!r}")
    if params.rho.lower() not in ("huber", "l2"):
        raise ValueError(f"rho must be 'huber' or 'l2', got {params.rho!r}")

    progress(0.0, "Preparing frames")
    frame_list = _as_frame_list(frames)
    prepared = _prepare_frames(frame_list, color_mode)
    n_frames = len(prepared)
    channels, native_h, native_w = prepared[0].shape

    dm = get_device_manager()
    device = torch.device("cpu") if params.force_cpu else dm.device

    # ---- PSFs -------------------------------------------------------
    if psfs is None:
        psf_list = []
        for i, frame in enumerate(prepared):
            gray = frame[0]
            psf_list.append(estimate_frame_psf(gray, params))
            progress(0.02 + 0.13 * (i + 1) / n_frames, f"Estimating PSF {i + 1}/{n_frames}")
    else:
        psf_list = list(psfs)
        if len(psf_list) != n_frames:
            raise ValueError(f"Got {len(psf_list)} psfs for {n_frames} frames")
        psf_list = [_normalize_psf(np.asarray(p, dtype=np.float32)) for p in psf_list]

    r = max(1, int(params.super_resolution))
    if r > 1:
        psf_list = [_lift_psf_for_super_resolution(p, r) for p in psf_list]
    flip_list = [_flip_kernel(p) for p in psf_list]

    # ---- Seed ---------------------------------------------------------
    progress(0.16, "Building seed image")
    seed = _build_seed(prepared, params)
    if r > 1:
        seed = _upsample_seed_np(seed, r)
    hs, ws = seed.shape[-2:]

    # ---- Masks / variance ----------------------------------------------
    mask_arrays = _prepare_frame_masks(params.frame_masks, n_frames, (native_h, native_w))
    if params.use_variance_maps:
        var_arrays: list[np.ndarray | None] = [
            np.full((native_h, native_w), _estimate_frame_variance(f[0]), dtype=np.float32)
            for f in prepared
        ]
    else:
        var_arrays = [None] * n_frames

    def _solve_on(device_: torch.device) -> tuple[torch.Tensor, int, bool]:
        x_t = torch.from_numpy(np.ascontiguousarray(seed, dtype=np.float32)).to(device_)
        frame_tensors = [
            torch.from_numpy(np.ascontiguousarray(f, dtype=np.float32)).to(device_)
            for f in prepared
        ]
        var_tensors: list[torch.Tensor | None] = [
            None if v is None else torch.from_numpy(v).to(device_) for v in var_arrays
        ]
        psf_ffts = [
            _padded_kernel_fft(torch.from_numpy(k).to(device_), hs, ws) for k in psf_list
        ]
        psf_t_ffts = [
            _padded_kernel_fft(torch.from_numpy(k).to(device_), hs, ws) for k in flip_list
        ]

        no_reject_masks: list[torch.Tensor | None] = [None] * n_frames
        has_masks = any(m is not None for m in mask_arrays)

        if not has_masks or params.rejection_strength <= 0.0:
            x_final, used_iters, early_stopped = _run_solver(
                x_t, frame_tensors, psf_ffts, psf_t_ffts, no_reject_masks, var_tensors,
                params, r, progress, (0.20, 0.95),
            )
        elif params.rejection_strength >= 1.0:
            full_masks: list[torch.Tensor | None] = [
                None if m is None else torch.from_numpy(m).to(device_) for m in mask_arrays
            ]
            x_final, used_iters, early_stopped = _run_solver(
                x_t, frame_tensors, psf_ffts, psf_t_ffts, full_masks, var_tensors,
                params, r, progress, (0.20, 0.95),
            )
        else:
            full_masks = [
                None if m is None else torch.from_numpy(m).to(device_) for m in mask_arrays
            ]
            x0, u0, e0 = _run_solver(
                x_t.clone(), frame_tensors, psf_ffts, psf_t_ffts, no_reject_masks, var_tensors,
                params, r, progress, (0.20, 0.57),
            )
            x1, u1, e1 = _run_solver(
                x_t.clone(), frame_tensors, psf_ffts, psf_t_ffts, full_masks, var_tensors,
                params, r, progress, (0.57, 0.95),
            )
            strength = params.rejection_strength
            x_final = x0 * (1.0 - strength) + x1 * strength
            used_iters = max(u0, u1)
            early_stopped = bool(e0 and e1)

        return x_final, used_iters, early_stopped

    try:
        x_final, used_iters, early_stopped = _solve_on(device)
    except RuntimeError as exc:
        if device.type == "cpu":
            raise
        log.warning("multiframe_deconvolve: GPU solve failed (%s), retrying on CPU", exc)
        dm.empty_cache()
        x_final, used_iters, early_stopped = _solve_on(torch.device("cpu"))

    log.info(
        "multiframe_deconvolve: %d frame(s), %d iteration(s) used%s",
        n_frames, used_iters, " (early stop)" if early_stopped else "",
    )

    progress(0.97, "Finalizing")
    result = x_final.detach().cpu().numpy().astype(np.float32)
    result = np.clip(result, 0.0, 1.0)
    if result.shape[0] == 1:
        result = result[0]

    if mask is not None:
        reference = seed if seed.shape[0] > 1 else seed[0]
        reference = np.clip(reference, 0.0, 1.0).astype(np.float32)
        result = apply_mask(reference, result, mask)

    progress(1.0, "Done")
    return result
