"""AI Super-Resolution — upscale astro images with learned detail synthesis.

Uses Real-ESRGAN architecture with astro-specific pre-processing.
Falls back to classic interpolation if no model is available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from astraios.core.device_manager import get_device_manager

log = logging.getLogger(__name__)

MODEL_DIR = Path.home() / ".astraios" / "models" / "super_resolution"
MODEL_URLS = {
    "real_esrgan_x2.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.0/RealESRGAN_x2.pth",
    "real_esrgan_x4.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.0/RealESRGAN_x4.pth",
}


@dataclass
class SuperResParams:
    scale: int = 2
    model: str = "real_esrgan"
    tile_size: int = 512
    pre_denoise: bool = True


def upscale(
    image: NDArray,
    params: SuperResParams | None = None,
) -> NDArray:
    """Upscale an astro image using AI super-resolution.

    Args:
        image: (H, W) or (C, H, W) float32 in [0, 1].
        params: Super-resolution parameters.

    Returns:
        Upscaled image: (scale*H, scale*W) or (C, scale*H, scale*W).
    """
    if params is None:
        params = SuperResParams()

    if params.model == "bicubic":
        return _upscale_classic(image, params.scale, "bicubic")
    elif params.model == "lanczos":
        return _upscale_classic(image, params.scale, "lanczos")
    else:
        return _upscale_ai(image, params)


def _upscale_classic(
    image: NDArray, scale: int, interpolation: str,
) -> NDArray:
    """Classic interpolation upscaling (no AI)."""
    import cv2

    interp_map = {
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    method = interp_map.get(interpolation, cv2.INTER_CUBIC)

    if image.ndim == 3:
        c, h, w = image.shape
        out = np.empty((c, h * scale, w * scale), dtype=image.dtype)
        for ch in range(c):
            out[ch] = cv2.resize(image[ch], (w * scale, h * scale), interpolation=method)
        return out
    else:
        h, w = image.shape
        return cv2.resize(image, (w * scale, h * scale), interpolation=method)


def _upscale_ai(
    image: NDArray,
    params: SuperResParams,
) -> NDArray:
    """AI-based super-resolution using Real-ESRGAN."""
    dm = get_device_manager()
    img = image.astype(np.float32)

    if params.pre_denoise:
        from astraios.core.denoise import DenoiseParams, denoise
        img = denoise(img, DenoiseParams(strength=0.3))

    model = _load_model(params.scale)
    if model is None:
        log.warning("AI super-resolution model not found, falling back to Lanczos")
        return _upscale_classic(img, params.scale, "lanczos")

    try:
        import torch

        device = dm.device
        model = model.to(device)
        use_tiles = params.tile_size > 0

        if img.ndim == 3:
            c, h, w = img.shape
            result = np.empty((c, h * params.scale, w * params.scale), dtype=np.float32)
            for ch in range(c):
                tensor = torch.from_numpy(img[ch][None, None, :, :]).float().to(device)
                if use_tiles and max(h, w) > params.tile_size:
                    up = _tiled_inference(model, tensor, params.tile_size, device, params.scale)
                else:
                    with torch.no_grad():
                        up = model(tensor)
                result[ch] = up[0, 0].cpu().numpy().astype(np.float32)
        else:
            h, w = img.shape
            tensor = torch.from_numpy(img[None, None, :, :]).float().to(device)
            if use_tiles and max(h, w) > params.tile_size:
                up = _tiled_inference(model, tensor, params.tile_size, device, params.scale)
            else:
                with torch.no_grad():
                    up = model(tensor)
            result = up[0, 0].cpu().numpy().astype(np.float32)

        del tensor, up
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return np.clip(result, 0, 1).astype(image.dtype)

    except Exception as e:
        log.warning("AI super-resolution failed (%s), falling back to Lanczos", e)
        return _upscale_classic(img, params.scale, "lanczos")


def _load_model(scale: int) -> Any | None:
    """Load Real-ESRGAN model from local cache, downloading if needed."""
    model_name = f"real_esrgan_x{scale}.pth"
    model_path = MODEL_DIR / model_name

    if not model_path.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        url = MODEL_URLS.get(model_name)
        if url:
            log.info("Downloading super-resolution model: %s", url)
            try:
                import urllib.request
                urllib.request.urlretrieve(url, model_path)
                log.info("Model downloaded: %s", model_path)
            except Exception as e:
                log.warning("Failed to download model: %s", e)
                return None
        else:
            log.warning("No model URL for scale %d", scale)
            return None

    try:
        from collections import OrderedDict

        import torch

        state = torch.load(model_path, map_location="cpu", weights_only=True)

        class _SimpleUpsampler(torch.nn.Module):
            def __init__(self, scale):
                super().__init__()
                self.scale = scale

            def forward(self, x):
                return torch.nn.functional.interpolate(
                    x, scale_factor=self.scale, mode="bicubic", align_corners=False
                )

        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            model = RRDBNet(
                num_in_ch=1, num_out_ch=1, num_feat=64,
                num_block=23, num_grow_ch=32, scale=scale,
            )
            if isinstance(state, OrderedDict):
                new_state = OrderedDict()
                for k, v in state.items():
                    new_state[k.replace("module.", "")] = v
                missing, unexpected = model.load_state_dict(new_state, strict=False)
                if missing:
                    log.debug("Missing keys in model: %d", len(missing))
            return model
        except ImportError:
            log.debug("basicsr not installed, using simple upsampler")
            return _SimpleUpsampler(scale)

    except Exception as e:
        log.warning("Failed to load model: %s", e)
        return None


def _tiled_inference(model, tensor, tile_size: int, device, scale: int):
    """Run inference in tiles to reduce VRAM usage.

    ``scale`` is the model's upscale factor; it was hardcoded to 2, which
    corrupted x4 super-resolution (each tile was cropped/placed at 2x into a
    half-size buffer).
    """
    import torch

    _, _, h, w = tensor.shape
    out_h, out_w = h * scale, w * scale
    output = torch.zeros((1, 1, out_h, out_w), device=device)
    count = torch.zeros((1, 1, out_h, out_w), device=device)

    overlap = tile_size // 4
    stride = tile_size - overlap

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1 = y
            x1 = x
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)

            tile = tensor[:, :, y1:y2, x1:x2]

            pad_h = max(0, tile_size - (y2 - y1))
            pad_w = max(0, tile_size - (x2 - x1))
            if pad_h > 0 or pad_w > 0:
                # replicate, not reflect: reflect raises when the pad exceeds
                # the tile dimension, which happens for narrow edge tiles
                # (e.g. 1000px image, 512px tiles -> 232px edge tile).
                tile = torch.nn.functional.pad(tile, (0, pad_w, 0, pad_h), mode="replicate")

            with torch.no_grad():
                out_tile = model(tile)

            out_tile = out_tile[:, :, :(y2 - y1) * scale, :(x2 - x1) * scale]

            out_y1 = y * scale
            out_x1 = x * scale
            oy, ox = out_tile.shape[2], out_tile.shape[3]
            output[:, :, out_y1:out_y1 + oy, out_x1:out_x1 + ox] += out_tile
            count[:, :, out_y1:out_y1 + oy, out_x1:out_x1 + ox] += 1

    output = output / count.clamp(min=1)
    return output


__all__ = ["upscale", "SuperResParams"]
