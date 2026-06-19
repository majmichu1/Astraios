"""AI Denoise — deep learning noise reduction using a trained N2S U-Net.

Inference strategy: J-invariant prediction with signal-aware blending.

The model was trained with Noise2Self (N2S): masked pixels → predict full image.
Correct inference = J-invariant: for each pixel, predict it from surrounding
context (it was masked during training). We do this by averaging many masked
forward passes and only collecting predictions at masked positions.

Additionally, a signal-protection blend preserves bright structures (stars,
nebula cores) and applies denoising primarily to background sky regions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from astraios.ai.model_manager import ModelType, get_model_manager
from astraios.ai.models.unet import UNet
from astraios.core.device_manager import get_device_manager
from astraios.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


@dataclass
class AIDenoiseParams:
    """Parameters for AI denoising."""

    strength: float = 1.0        # 0–1, overall blend: original vs denoised
    protect_stars: float = 0.8   # 0–1, how much to protect bright signal
    protect_threshold: float = 0.15  # pixels above this are treated as "signal"
    n_passes: int = 16           # J-invariant passes (more = smoother result)
    mask_ratio: float = 0.15     # must match training mask ratio
    tile_size: int = 256         # must match training patch size
    overlap: int = 32


def _load_trained_model(prefer_best: bool = True) -> UNet | None:
    """Load the best available trained N2S model from disk."""
    from pathlib import Path

    models_dir = Path(__file__).resolve().parent.parent / "models"

    candidates = []
    # A denoise model the user set in Preferences takes priority.
    from astraios.core.user_paths import model_override

    user_model = model_override("denoise")
    if user_model is not None:
        candidates.append(user_model)
    if prefer_best:
        candidates.append(models_dir / "best_n2s_model.pt")
    # Try checkpoints newest first (up to epoch 30)
    for i in range(30, 0, -1):
        candidates.append(models_dir / f"checkpoint_epoch_{i}.pt")
    candidates.append(models_dir / "cosmica_denoise_n2s_v1.pt")

    for path in candidates:
        if not path.exists():
            continue
        try:
            raw = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(raw, dict) and "model_state_dict" in raw:
                cfg = raw.get("config", {})
                bf = cfg.get("base_features", 32)
                depth = cfg.get("depth", 4)
                model = UNet(in_channels=1, out_channels=1, base_features=bf, depth=depth)
                state_dict = raw["model_state_dict"]
                # Strip DenoiseUNet wrapper prefix if present
                if all(k.startswith("unet.") for k in state_dict):
                    state_dict = {k.removeprefix("unet."): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                log.info("Loaded AI denoise model: %s (val_loss=%.3e)", path.name, raw.get("val_loss", float("nan")))
            else:
                # Plain state dict (v1 format, depth=3)
                model = UNet(in_channels=1, out_channels=1, base_features=32, depth=3)
                model.load_state_dict(raw)
                log.info("Loaded AI denoise model (plain state dict): %s", path.name)
            model.eval()
            return model
        except Exception as e:
            log.debug("Could not load %s: %s", path.name, e)
            continue

    # Try model manager cache
    try:
        mm = get_model_manager()
        model_path = mm.get_cache_path(ModelType.DENOISE)
        if model_path and model_path.exists():
            raw = torch.load(model_path, map_location="cpu", weights_only=True)
            model = UNet(in_channels=1, out_channels=1, base_features=32, depth=4)
            state = raw["model_state_dict"] if isinstance(raw, dict) else raw
            model.load_state_dict(state)
            model.eval()
            return model
    except Exception as e:
        log.debug("Model manager load failed: %s", e)

    return None


def _jinvariant_tile(
    tile: torch.Tensor,
    model: torch.nn.Module,
    n_passes: int,
    mask_ratio: float,
) -> torch.Tensor:
    """J-invariant inference on a single (1,1,H,W) tile tensor already on device."""
    accum = torch.zeros_like(tile)
    count = torch.zeros_like(tile)
    with torch.no_grad():
        for _ in range(n_passes):
            mask = torch.rand_like(tile) < mask_ratio
            masked = tile.clone()
            masked[mask] = 0.0
            pred = model(masked)
            accum[mask] += pred[mask]
            count[mask] += 1
    result = tile.clone()
    valid = count > 0
    result[valid] = (accum[valid] / count[valid]).clamp(0, 1)
    return result


def _jinvariant_channel(
    data: np.ndarray,
    model: torch.nn.Module,
    params: AIDenoiseParams,
    device: torch.device,
    progress: ProgressCallback,
) -> np.ndarray:
    """J-invariant inference on a single 2D channel (H, W) with tiled processing.

    Splits large images into overlapping tiles so inference fits in available
    VRAM even while training is running in parallel.
    """
    h, w = data.shape
    tile_size = params.tile_size
    overlap = params.overlap

    # If image fits comfortably, process in one shot
    if h <= tile_size and w <= tile_size:
        x = torch.from_numpy(data.copy()).unsqueeze(0).unsqueeze(0).to(device)
        model.eval()
        result = _jinvariant_tile(x, model, params.n_passes, params.mask_ratio)
        return result.squeeze().cpu().numpy()

    # Tiled processing with overlap blending
    model.eval()
    output = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    ys = list(range(0, h - tile_size + 1, tile_size - overlap))
    if ys[-1] + tile_size < h:
        ys.append(h - tile_size)
    xs = list(range(0, w - tile_size + 1, tile_size - overlap))
    if xs[-1] + tile_size < w:
        xs.append(w - tile_size)

    total_tiles = len(ys) * len(xs)
    tile_idx = 0

    for y0 in ys:
        for x0 in xs:
            y1, x1 = y0 + tile_size, x0 + tile_size
            patch = data[y0:y1, x0:x1].copy()
            t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
            result_t = _jinvariant_tile(t, model, params.n_passes, params.mask_ratio)
            result_np = result_t.squeeze().cpu().numpy()

            # Cosine blending window to avoid seam artefacts
            win_y = np.hanning(tile_size).astype(np.float32)
            win_x = np.hanning(tile_size).astype(np.float32)
            win = np.outer(win_y, win_x) + 1e-6

            output[y0:y1, x0:x1] += result_np * win
            weight[y0:y1, x0:x1] += win

            tile_idx += 1
            progress(tile_idx / total_tiles,
                     f"J-invariant tile {tile_idx}/{total_tiles}…")

            # Free tile memory explicitly
            del t, result_t

    # Free GPU cache once after all tiles
    get_device_manager().empty_cache()

    return (output / weight).clip(0, 1)


def _signal_blend(
    original: np.ndarray,
    denoised: np.ndarray,
    params: AIDenoiseParams,
) -> np.ndarray:
    """Blend denoised result with original, protecting bright signal.

    Pixels above protect_threshold are blended back toward the original
    proportional to their brightness, preventing star/nebula destruction.
    """
    # Soft protection weight: 0 at background, 1 at bright signal
    signal_weight = np.clip(
        (original - params.protect_threshold) / max(params.protect_threshold, 1e-6),
        0, 1
    ) * params.protect_stars  # scale by protection strength

    # Final: denoised in dark regions, original in bright regions
    blended = denoised * (1.0 - signal_weight) + original * signal_weight

    # Global strength blend
    if params.strength < 1.0:
        blended = original * (1.0 - params.strength) + blended * params.strength

    return blended.astype(np.float32)


def ai_denoise(
    data: np.ndarray,
    model: torch.nn.Module | None = None,
    params: AIDenoiseParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback | None = None,
) -> np.ndarray:
    """Denoise using J-invariant N2S inference with signal protection.

    Falls back to wavelet denoise if no model is available.
    """
    if params is None:
        params = AIDenoiseParams()
    if progress is None:
        progress = _noop_progress

    if model is None:
        model = _load_trained_model()

    if model is None:
        log.info("No AI model available — falling back to wavelet denoise")
        return _traditional_denoise(data, params, mask, progress)

    dm = get_device_manager()
    device = dm.device
    model = model.to(device)
    original = data.copy()

    if data.ndim == 2:
        denoised = _jinvariant_channel(data, model, params, device, progress)
        result = _signal_blend(data, denoised, params)
    else:
        n_ch = data.shape[0]
        result = np.empty_like(data)
        for ch in range(n_ch):
            def ch_progress(frac, msg, c=ch):
                progress((c + frac) / n_ch, f"Ch {c+1}/{n_ch}: {msg}")
            denoised_ch = _jinvariant_channel(data[ch], model, params, device, ch_progress)
            result[ch] = _signal_blend(data[ch], denoised_ch, params)

    result = np.clip(result, 0, 1).astype(np.float32)
    return apply_mask(original, result, mask)


def _traditional_denoise(
    data: np.ndarray,
    params: AIDenoiseParams,
    mask: Mask | None,
    progress: ProgressCallback,
) -> np.ndarray:
    from astraios.core.denoise import DenoiseParams, denoise
    wp = DenoiseParams(
        method="wavelet",
        strength=max(0.1, params.strength * 0.8),
        detail_preservation=0.7,
        wavelet="db4",
        wavelet_levels=4,
    )
    return denoise(data, params=wp, mask=mask, progress=progress)
