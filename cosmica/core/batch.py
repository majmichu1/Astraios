"""Batch Processing — apply processing pipelines to multiple images.

Supports defining ordered processing steps and applying them to a batch
of input files with parallel execution support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from cosmica.core.image_io import ImageData, load_image, save_image

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


@dataclass
class PipelineStep:
    """A single processing step in a batch pipeline."""

    tool_name: str  # registered tool name (e.g., "auto_stretch", "denoise")
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    mask_name: str | None = None  # saved mask name to protect/reveal areas


@dataclass
class Pipeline:
    """An ordered list of processing steps."""

    name: str = "Untitled Pipeline"
    steps: list[PipelineStep] = field(default_factory=list)

    def add_step(self, tool_name: str, params: dict[str, Any] | None = None) -> PipelineStep:
        step = PipelineStep(tool_name=tool_name, params=params or {})
        self.steps.append(step)
        return step

    def remove_step(self, index: int):
        if 0 <= index < len(self.steps):
            self.steps.pop(index)

    def move_step(self, from_index: int, to_index: int):
        if 0 <= from_index < len(self.steps) and 0 <= to_index < len(self.steps):
            step = self.steps.pop(from_index)
            self.steps.insert(to_index, step)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "steps": [
                {
                    "tool_name": s.tool_name,
                    "params": s.params,
                    "enabled": s.enabled,
                    "mask_name": s.mask_name,
                }
                for s in self.steps
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Pipeline:
        pipeline = cls(name=d.get("name", "Untitled"))
        for step_d in d.get("steps", []):
            step = PipelineStep(
                tool_name=step_d["tool_name"],
                params=step_d.get("params", {}),
                enabled=step_d.get("enabled", True),
                mask_name=step_d.get("mask_name"),
            )
            pipeline.steps.append(step)
        return pipeline


# Tool registry: maps tool names to processing functions
_TOOL_REGISTRY: dict[str, Callable] = {}


def register_tool(name: str, func: Callable):
    """Register a processing function for use in batch pipelines."""
    _TOOL_REGISTRY[name] = func


def get_registered_tools() -> dict[str, Callable]:
    """Get all registered tools (registering the built-in defaults on first use)."""
    if not _TOOL_REGISTRY:
        _register_default_tools()
    return dict(_TOOL_REGISTRY)


def _register_default_tools():
    """Register all built-in processing tools."""
    import dataclasses as _dc

    def _p(cls, kw):
        """Build a Params dataclass from a flat dict, ignoring unknown keys."""
        known = {f.name for f in _dc.fields(cls)}
        filtered = {k: v for k, v in kw.items() if k in known}
        return cls(**filtered) if filtered else None

    from cosmica.core.background import extract_background
    from cosmica.core.banding import BandingParams, banding_reduction
    from cosmica.core.color_tools import ColorAdjustParams, SCNRParams, color_adjust, scnr
    from cosmica.core.cosmetic import cosmetic_correction
    from cosmica.core.denoise import DenoiseParams, denoise
    from cosmica.core.filters import (
        ConvolutionKernel,
        ConvolutionParams,
        UnsharpMaskParams,
        convolve,
        unsharp_mask,
    )
    from cosmica.core.frequency_separation import (
        FrequencySeparationParams,
        SeparationMethod,
        frequency_separation,
    )
    from cosmica.core.histogram_transform import HistogramTransformParams, histogram_transform
    from cosmica.core.local_contrast import LocalContrastParams, local_contrast_enhance
    from cosmica.core.morphology import (
        MorphologyParams,
        MorphOp,
        StructuringElement,
        morphology_transform,
    )
    from cosmica.core.star_stretch import StarStretchParams, star_stretch
    from cosmica.core.stretch import (
        ArcsinhStretchParams,
        GHSParams,
        StatisticalStretchParams,
        StretchParams,
        arcsinh_stretch,
        auto_stretch,
        generalized_hyperbolic_stretch,
        statistical_stretch,
    )
    from cosmica.core.transforms import invert
    from cosmica.core.wavelets import WaveletParams, wavelet_sharpen

    register_tool("auto_stretch", lambda data, **kw: auto_stretch(data, _p(StretchParams, kw)))
    register_tool("ghs", lambda data, **kw: generalized_hyperbolic_stretch(
        data, _p(GHSParams, kw) if kw else None
    ))
    register_tool("background_extraction", lambda data, **kw: extract_background(data)[0])
    register_tool("cosmetic_correction", lambda data, **kw: cosmetic_correction(data).data)
    register_tool(
        "banding_reduction", lambda data, **kw: banding_reduction(data, _p(BandingParams, kw))
    )
    register_tool(
        "histogram_transform",
        lambda data, **kw: histogram_transform(data, _p(HistogramTransformParams, kw)),
    )
    register_tool("scnr", lambda data, **kw: scnr(data, _p(SCNRParams, kw)))
    register_tool(
        "color_adjust", lambda data, **kw: color_adjust(data, _p(ColorAdjustParams, kw))
    )
    register_tool("denoise", lambda data, **kw: denoise(data, _p(DenoiseParams, kw)))
    register_tool(
        "local_contrast",
        lambda data, **kw: local_contrast_enhance(data, _p(LocalContrastParams, kw)),
    )
    register_tool(
        "wavelet_sharpen", lambda data, **kw: wavelet_sharpen(data, _p(WaveletParams, kw))
    )
    register_tool(
        "unsharp_mask", lambda data, **kw: unsharp_mask(data, _p(UnsharpMaskParams, kw))
    )
    register_tool("invert", lambda data, **kw: invert(data))
    register_tool(
        "arcsinh_stretch", lambda data, **kw: arcsinh_stretch(data, _p(ArcsinhStretchParams, kw))
    )
    register_tool(
        "statistical_stretch",
        lambda data, **kw: statistical_stretch(data, _p(StatisticalStretchParams, kw)),
    )
    register_tool(
        "star_stretch", lambda data, **kw: star_stretch(data, _p(StarStretchParams, kw))
    )

    def _frequency_separation_tool(
        data, method="SUBTRACT", sigma=5.0, hf_boost=1.0, lf_smooth=0.0, **kw
    ):
        m = SeparationMethod[method] if isinstance(method, str) else method
        params = FrequencySeparationParams(
            sigma=sigma, method=m, hf_boost=hf_boost, lf_smooth=lf_smooth
        )
        return frequency_separation(data, params)

    register_tool("frequency_separation", _frequency_separation_tool)

    def _morphology_tool(
        data, operation="ERODE", kernel_size=3, iterations=1, element="CIRCLE", **kw
    ):
        # StructuringElement members are CIRCLE / SQUARE / DIAMOND. Accept the
        # common alias "DISK" for CIRCLE rather than raising KeyError.
        if isinstance(element, str):
            element = "CIRCLE" if element.upper() == "DISK" else element.upper()
            element = StructuringElement[element]
        if isinstance(operation, str):
            operation = MorphOp[operation.upper()]
        params = MorphologyParams(
            operation=operation,
            element=element,
            kernel_size=kernel_size,
            iterations=iterations,
        )
        return morphology_transform(data, params)

    register_tool("morphology", _morphology_tool)

    def _convolve_tool(data, kernel="GAUSSIAN", radius=2.0, amount=1.0, **kw):
        k = ConvolutionKernel[kernel.upper()] if isinstance(kernel, str) else kernel
        return convolve(data, ConvolutionParams(kernel=k, radius=radius, amount=amount))

    register_tool("convolve", _convolve_tool)


def apply_pipeline_to_image(
    data: np.ndarray,
    pipeline: Pipeline,
    progress: ProgressCallback | None = None,
    masks: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Apply a pipeline to a single image.

    Parameters
    ----------
    data : ndarray
        Image data, float32 in [0, 1].
    pipeline : Pipeline
        Processing pipeline to apply.
    progress : callable, optional
        Progress callback ``(fraction, message)``.
    masks : dict, optional
        Named masks ``{name: (H, W) float32 array}`` for mask-aware steps.

    Returns
    -------
    ndarray
        Processed image.
    """
    if not _TOOL_REGISTRY:
        _register_default_tools()

    if progress is None:
        progress = _noop_progress

    from cosmica.core.masks import apply_mask

    enabled = [s for s in pipeline.steps if s.enabled]
    n = len(enabled)
    result = data.copy()
    for i, step in enumerate(enabled):
        progress(i / max(n, 1), f"Step {i + 1}/{n}: {step.tool_name}")
        func = _TOOL_REGISTRY.get(step.tool_name)
        if func is None:
            log.warning("Unknown tool: %s, skipping", step.tool_name)
            continue
        log.info("Applying: %s", step.tool_name)
        processed = func(result, **step.params)
        if step.mask_name and masks and step.mask_name in masks:
            processed = apply_mask(result, processed, masks[step.mask_name])
            log.info("  masked with '%s'", step.mask_name)
        result = processed

    progress(1.0, "Pipeline complete")
    return result


@dataclass
class BatchResult:
    """Result of batch processing."""

    n_processed: int
    n_failed: int
    output_paths: list[Path]
    errors: list[str]


def batch_process(
    input_paths: list[Path],
    pipeline: Pipeline,
    output_dir: Path,
    output_format: str = "fits",
    progress: ProgressCallback | None = None,
) -> BatchResult:
    """Apply a pipeline to multiple input files.

    Parameters
    ----------
    input_paths : list[Path]
        Input image file paths.
    pipeline : Pipeline
        Processing pipeline to apply.
    output_dir : Path
        Directory to save processed images.
    output_format : str
        Output format ("fits" or "xisf").
    progress : callable, optional
        Progress callback.

    Returns
    -------
    BatchResult
        Processing results.
    """
    if progress is None:
        progress = _noop_progress

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    errors: list[str] = []
    n_processed = 0

    for i, path in enumerate(input_paths):
        frac = i / max(len(input_paths), 1)
        progress(frac, f"Processing {path.name} ({i + 1}/{len(input_paths)})...")

        try:
            img = load_image(str(path))
            processed = apply_pipeline_to_image(img.data, pipeline)

            out_name = f"{path.stem}_processed.{output_format}"
            out_path = output_dir / out_name
            save_image(ImageData(data=processed, header=img.header), out_path)

            output_paths.append(out_path)
            n_processed += 1
        except Exception as e:
            log.error("Error processing %s: %s", path.name, e)
            errors.append(f"{path.name}: {e}")

    progress(1.0, f"Batch complete: {n_processed}/{len(input_paths)} processed")

    return BatchResult(
        n_processed=n_processed,
        n_failed=len(errors),
        output_paths=output_paths,
        errors=errors,
    )
