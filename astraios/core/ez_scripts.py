"""EZ Script Suite — one-click processing presets for common workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from astraios.core.abe import ABEParams, abe_extract
from astraios.core.background import BackgroundParams, extract_background
from astraios.core.background_neutralization import (
    BackgroundNeutralizationParams,
    background_neutralization,
)
from astraios.core.color_calibration import ColorCalibrationParams, color_calibrate
from astraios.core.color_tools import SCNRParams
from astraios.core.color_tools import scnr as apply_scnrg
from astraios.core.cosmetic import CosmeticParams, cosmetic_correction
from astraios.core.deconvolution import DeconvolutionParams, richardson_lucy
from astraios.core.denoise import DenoiseParams, denoise
from astraios.core.filters import UnsharpMaskParams, unsharp_mask
from astraios.core.histogram_transform import HistogramTransformParams, histogram_transform
from astraios.core.stretch import ArcsinhStretchParams, arcsinh_stretch

log = logging.getLogger(__name__)


@dataclass
class EZPreset:
    """A named preset with a processing pipeline."""

    name: str
    description: str
    steps: list[dict] = field(default_factory=list)


REGISTRY: dict[str, EZPreset] = {}


def register(preset: EZPreset) -> None:
    REGISTRY[preset.name] = preset


def list_presets() -> list[str]:
    return list(REGISTRY.keys())


def run_preset(
    img: NDArray,
    preset_name: str,
    progress_callback=None,
) -> NDArray:
    """Run a named preset on an image.

    Args:
        img: (H, W) or (H, W, C) float32.
        preset_name: Name of the preset to run.
        progress_callback: Optional callable(progress: float, message: str).

    Returns:
        Processed image.
    """
    preset = REGISTRY.get(preset_name)
    if preset is None:
        msg = f"Unknown preset: {preset_name}"
        raise ValueError(msg)

    data = img.astype(np.float32).copy()
    n = len(preset.steps)

    for i, step in enumerate(preset.steps):
        name = step.get("name", f"step_{i}")
        params = step.get("params", {})
        if progress_callback:
            progress_callback((i + 1) / n, name)
        log.info("EZ: %s — %s", preset_name, name)

        if name == "cosmetic":
            if isinstance(params, dict):
                params = CosmeticParams(**params)
            data = cosmetic_correction(data, params).data
        elif name == "background":
            if isinstance(params, dict):
                params = BackgroundParams(**params)
            bg_model = extract_background(data, params)
            data = data - bg_model
        elif name == "neutralize":
            if isinstance(params, dict):
                params = BackgroundNeutralizationParams(**params)
            data = background_neutralization(data, params)
        elif name == "color_calibrate":
            if isinstance(params, dict):
                params = ColorCalibrationParams(**params)
            result = color_calibrate(data, params)
            if isinstance(result, np.ndarray):
                data = result
            else:
                data = result.data
        elif name == "deconvolution":
            if isinstance(params, dict):
                params = DeconvolutionParams(**params)
            data = richardson_lucy(data, params)
        elif name == "denoise":
            if isinstance(params, dict):
                params = DenoiseParams(**params)
            data = denoise(data, params)
        elif name == "unsharp_mask":
            if isinstance(params, dict):
                params = UnsharpMaskParams(**params)
            data = unsharp_mask(data, params)
        elif name == "stretch":
            if isinstance(params, dict):
                params = ArcsinhStretchParams(**params)
            data = arcsinh_stretch(data, params)
        elif name == "histogram":
            if isinstance(params, dict):
                params = HistogramTransformParams(**params)
            data = histogram_transform(data, params)
        elif name == "scnrg":
            if isinstance(params, dict):
                params = SCNRParams(**params)
            data = apply_scnrg(data, params)
        elif name == "abe":
            if isinstance(params, dict):
                params = ABEParams(**params)
            data = abe_extract(data, params)
            if isinstance(data, tuple):
                data = data[0]
        else:
            log.warning("EZ: unknown step %s, skipping", name)

    return data


def _stretch_params() -> dict:
    return {
        "stretch_factor": 5.0,
        "black_point": 0.001,
    }


# ── Built-in presets ──────────────────────────────────────────────

register(EZPreset(
    name="OSC Quick Processing",
    description="One-click processing for one-shot-color (OSC) images",
    steps=[
        {"name": "cosmetic", "params": {}},
        {"name": "background", "params": {}},
        {"name": "neutralize", "params": {}},
        {"name": "color_calibrate", "params": {}},
        {"name": "denoise", "params": {}},
        {"name": "unsharp_mask", "params": {}},
        {"name": "stretch", "params": _stretch_params()},
    ],
))

register(EZPreset(
    name="Narrowband Processing",
    description="For narrowband (Ha/OIII/SII) monochrome images",
    steps=[
        {"name": "cosmetic", "params": {}},
        {"name": "background", "params": {}},
        {"name": "denoise", "params": {}},
        {"name": "unsharp_mask", "params": {}},
        {"name": "stretch", "params": _stretch_params()},
    ],
))

register(EZPreset(
    name="Deep Sky Minimal",
    description="Minimal processing — denoise + stretch only",
    steps=[
        {"name": "denoise", "params": {}},
        {"name": "stretch", "params": _stretch_params()},
    ],
))

register(EZPreset(
    name="Luminance Processing",
    description="For Luminance (L) channel — includes deconvolution",
    steps=[
        {"name": "cosmetic", "params": {}},
        {"name": "background", "params": {}},
        {"name": "deconvolution", "params": {"iterations": 30}},
        {"name": "denoise", "params": {}},
        {"name": "histogram", "params": {}},
    ],
))

register(EZPreset(
    name="Full Processing with ABE",
    description="Complete processing — includes Automatic Background Extraction",
    steps=[
        {"name": "abe", "params": {}},
        {"name": "cosmetic", "params": {}},
        {"name": "neutralize", "params": {}},
        {"name": "color_calibrate", "params": {}},
        {"name": "denoise", "params": {}},
        {"name": "unsharp_mask", "params": {}},
        {"name": "stretch", "params": _stretch_params()},
    ],
))

register(EZPreset(
    name="Starless Processing",
    description="Denoise background, stretch starless (requires star mask)",
    steps=[
        {"name": "background", "params": {}},
        {"name": "denoise", "params": {}},
        {"name": "stretch", "params": _stretch_params()},
    ],
))
