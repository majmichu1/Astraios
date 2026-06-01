"""Tiled Inference — run AI models on large images by processing tiles with overlap.

Handles splitting an image into overlapping patches, running inference on each,
and blending the results back together with smooth overlap transitions.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import torch

from cosmica.core.device_manager import get_device_manager

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


def tiled_inference(
    data: np.ndarray,
    model: torch.nn.Module,
    tile_size: int = 512,
    overlap: int = 64,
    batch_size: int = 1,
    progress: ProgressCallback | None = None,
) -> np.ndarray:
    """Run a model on an image using tiled inference with overlap blending.

    Parameters
    ----------
    data : ndarray
        Single-channel 2D array (H, W), float32 in [0, 1].
    model : nn.Module
        Trained model expecting (B, 1, H, W) input.
    tile_size : int
        Tile size in pixels.
    overlap : int
        Overlap between adjacent tiles.
    batch_size : int
        Number of tiles to process at once.
    progress : callable, optional
        Progress callback.

    Returns
    -------
    ndarray
        Processed image, same shape as input.
    """
    if progress is None:
        progress = _noop_progress

    dm = get_device_manager()
    device = dm.device
    original_device = next(model.parameters()).device
    model = model.to(device)
    h, w = data.shape

    # Compute tile positions (deduplicated at generation)
    stride = tile_size - overlap
    tiles = []
    seen = set()
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)
            key = (y_start, x_start, y_end, x_end)
            if key not in seen:
                seen.add(key)
                tiles.append(key)

    n_tiles = len(tiles)
    log.debug("Tiled inference: %d tiles of %dx%d with %dpx overlap", n_tiles, tile_size, tile_size, overlap)

    # Output and weight buffers
    output = np.zeros((h, w), dtype=np.float64)
    weight = np.zeros((h, w), dtype=np.float64)

    # Blending weight: cosine ramp at edges
    blend = _create_blend_weight(tile_size, overlap)

    model.eval()
    with torch.no_grad():
        for i in range(0, n_tiles, batch_size):
            batch_tiles = tiles[i : i + batch_size]
            frac = i / max(n_tiles, 1)
            progress(frac, f"Processing tile {i + 1}/{n_tiles}...")

            # Extract and stack tiles
            batch_data = []
            for y_start, x_start, y_end, x_end in batch_tiles:
                tile = data[y_start:y_end, x_start:x_end]
                # Pad if smaller than tile_size
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    padded = np.zeros((tile_size, tile_size), dtype=np.float32)
                    padded[: tile.shape[0], : tile.shape[1]] = tile
                    tile = padded
                batch_data.append(tile)

            batch_t = torch.from_numpy(np.stack(batch_data)).unsqueeze(1).to(device)
            result_t = model(batch_t)
            results = result_t.squeeze(1).cpu().numpy()

            # Scatter results back
            for j, (y_start, x_start, y_end, x_end) in enumerate(batch_tiles):
                th = y_end - y_start
                tw = x_end - x_start
                result_tile = results[j, :th, :tw]
                blend_tile = blend[:th, :tw]
                output[y_start:y_end, x_start:x_end] += result_tile * blend_tile
                weight[y_start:y_end, x_start:x_end] += blend_tile

    # Normalize
    valid = weight > 0
    output[valid] /= weight[valid]

    # Restore model to original device
    model = model.to(original_device)

    progress(1.0, "Tiled inference complete")
    return np.clip(output, 0, 1).astype(np.float32)


def _create_blend_weight(tile_size: int, overlap: int) -> np.ndarray:
    """Create a 2D cosine blend weight for tile overlap blending."""
    w = np.ones(tile_size, dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0, np.pi / 2, overlap, dtype=np.float32)
        cos_ramp = np.sin(ramp) ** 2  # smooth 0->1 ramp
        w[:overlap] = cos_ramp
        w[-overlap:] = cos_ramp[::-1]
    return np.outer(w, w)
