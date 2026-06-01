from __future__ import annotations

import logging
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from cosmica.core.device_manager import get_device_manager

log = logging.getLogger(__name__)

MODEL_DIR = Path.home() / ".cosmica" / "models" / "cosmic_clarity"
RELEASE_BASE = "https://github.com/setiastro/cosmicclarity/releases/download/Windows"

MODEL_URLS: dict[str, str] = {}

MIN_PIXELS_GPU = 256 * 256


@dataclass
class CosmicClarityParams:
    model: str = "denoise"
    strength: float = 1.0
    tile_size: int = 512
    keep_original_size: bool = True


def apply(
    image: NDArray,
    params: CosmicClarityParams | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> NDArray:
    if params is None:
        params = CosmicClarityParams()

    model = _load_model(params.model)
    if model is None:
        log.warning("CosmicClarity model '%s' not available, returning original", params.model)
        return image.copy()

    dm = get_device_manager()
    device = dm.device if image.size >= MIN_PIXELS_GPU else "cpu"
    img = image.astype(np.float32)

    if img.ndim == 3:
        tensor = torch.from_numpy(img[None, :, :, :]).float().to(device)
    else:
        tensor = torch.from_numpy(img[None, None, :, :]).float().to(device)

    model = model.to(device)
    model.eval()

    use_tiles = params.tile_size > 0 and max(tensor.shape[2:]) > params.tile_size

    with torch.no_grad():
        if use_tiles:
            output = _tiled_inference(model, tensor, params.tile_size, device)
        else:
            output = model(tensor)
            if isinstance(output, (list, tuple)):
                output = output[0]

    if img.ndim == 3:
        result = output[0].cpu().numpy().astype(np.float32)
    else:
        result = output[0, 0].cpu().numpy().astype(np.float32)

    if params.strength < 1.0:
        result = img * (1.0 - params.strength) + result * params.strength

    del tensor, output
    if device == "cuda":
        torch.cuda.empty_cache()

    return np.clip(result, 0, 1).astype(image.dtype)


KNOWN_MODELS = {
    "denoise": "deep_denoise_cnn_AI3_6.pth",
    "sharpen": "deep_sharp_stellar_cnn_AI3_5s.pth",
    "satellite": "satelliteremovalAI3.5.pth",
    "darkstar": "darkstar_v2.1.pth",
}


def _load_model(model_name: str):
    model_path = MODEL_DIR / f"{model_name}.pt"
    if model_path.exists():
        try:
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            return _build_and_load(state)
        except Exception as e:
            log.warning("Failed to load cached model %s: %s", model_name, e)
            return None

    actual_name = KNOWN_MODELS.get(model_name)
    if actual_name:
        actual_path = MODEL_DIR / actual_name
        if not actual_path.exists():
            url = f"{RELEASE_BASE}/{actual_name}"
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            try:
                log.info("Downloading CosmicClarity model %s from %s", actual_name, url)
                urllib.request.urlretrieve(url, actual_path)
                log.info("Model saved: %s", actual_path)
            except Exception as e:
                log.warning(
                    "Download failed: %s. "
                    "Download manually from %s and place in %s",
                    e, url, MODEL_DIR,
                )
                return None
        try:
            state = torch.load(actual_path, map_location="cpu", weights_only=True)
            return _build_and_load(state)
        except Exception as e:
            log.warning("Failed to load model %s: %s", actual_name, e)
            return None

    log.warning(
        "Unknown model '%s'. Known: %s. "
        "Place a .pt file named '%s.pt' in %s, "
        "or download the actual models from %s",
        model_name,
        ", ".join(KNOWN_MODELS),
        model_name,
        MODEL_DIR,
        RELEASE_BASE,
    )
    return None


def _build_and_load(state):
    model = _CosmicNet()
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            log.debug("Model missing keys: %d", len(missing))
        if unexpected:
            log.debug("Model unexpected keys: %d", len(unexpected))
        return model
    except Exception as e:
        log.warning("State dict incompatible with _CosmicNet: %s", e)
        return None


class _CosmicNet(torch.nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=64):
        super().__init__()
        self.enc1 = _ConvBlock(in_ch, features)
        self.enc2 = _ConvBlock(features, features * 2, stride=2)
        self.enc3 = _ConvBlock(features * 2, features * 4, stride=2)
        self.dec3 = _ConvBlock(features * 4 + features * 2, features * 2)
        self.dec2 = _ConvBlock(features * 2 + features, features)
        self.dec1 = torch.nn.Conv2d(features, out_ch, 1)
        self.up = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d3 = self.dec3(torch.cat([self.up(e3), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e1], dim=1))
        return self.dec1(d2)


class _ConvBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


def _tiled_inference(model, tensor, tile_size, device):
    _, c, h, w = tensor.shape
    out = torch.zeros_like(tensor)
    count = torch.zeros_like(tensor)

    overlap = tile_size // 4
    stride = tile_size - overlap

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1, x1 = y, x
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)
            tile = tensor[:, :, y1:y2, x1:x2]
            pad_h = max(0, tile_size - (y2 - y1))
            pad_w = max(0, tile_size - (x2 - x1))
            if pad_h > 0 or pad_w > 0:
                tile = F.pad(tile, (0, pad_w, 0, pad_h), mode="reflect")
            out_tile = model(tile)[:, :, :y2 - y1, :x2 - x1]
            out[:, :, y1:y2, x1:x2] += out_tile
            count[:, :, y1:y2, x1:x2] += 1

    return out / count.clamp(min=1)
