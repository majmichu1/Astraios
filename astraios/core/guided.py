"""Guided processing — a fixed, correct workflow presented one step at a time.

Astraios has a large toolbox, and a large toolbox is exactly what makes
beginners bounce off software like this: a dozen tabs and a hundred sliders
with no indication of what to touch first. This module encodes the answer to
"what do I do, and in what order?" as data: an ordered list of steps, each
with a plain-language explanation, a couple of controls that matter, and an
automatic suggestion derived from the image itself.

The order is the professional one, not an arbitrary one. Everything that
belongs on linear data (gradient removal, colour calibration, sharpening,
noise reduction) happens before the stretch; everything that only makes sense
on stretched data (saturation) comes after. Doing colour calibration after a
stretch, for instance, gives a measurably worse result, so the wizard simply
does not offer that mistake.

No Qt here on purpose: the sequence, the suggestions and the application of
each step are plain functions over arrays, so they can be tested and reused
(scripting, batch) without a UI.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


def is_color(image: np.ndarray) -> bool:
    """True for a (3, H, W) colour image."""
    return image.ndim == 3 and image.shape[0] >= 3


def _luma(image: np.ndarray) -> np.ndarray:
    return image.mean(axis=0) if image.ndim == 3 else image


def looks_linear(image: np.ndarray) -> bool:
    """Heuristic: unstretched data sits almost entirely near black.

    A stacked, unstretched frame has a median far below its peak; a stretched
    one has its histogram pushed up into the midtones. Used to decide whether
    the stretch step still needs doing, and to warn when someone feeds in an
    image that has already been processed elsewhere.
    """
    lum = _luma(image)
    finite = lum[np.isfinite(lum)]
    if finite.size == 0:
        return False
    return float(np.median(finite)) < 0.10


@dataclass
class GuidedControl:
    """One user-facing knob for a step.

    Deliberately few per step: the point of the wizard is that a beginner can
    accept the suggestion and move on, with one or two dials to nudge if the
    preview does not look right.
    """

    key: str
    label: str
    minimum: float
    maximum: float
    step: float
    decimals: int
    summary: str
    higher: str
    lower: str


@dataclass
class GuidedStep:
    """A single stage of the guided workflow."""

    step_id: str
    title: str
    summary: str
    detail: str
    controls: list[GuidedControl]
    apply_fn: Callable[[np.ndarray, dict[str, Any]], np.ndarray]
    suggest_fn: Callable[[np.ndarray], dict[str, Any]]
    color_only: bool = False
    default_skipped: bool = False

    def applies_to(self, image: np.ndarray) -> bool:
        """False when the step is meaningless for this image (e.g. colour
        balance on a mono frame), so the wizard can skip it silently rather
        than showing a control that does nothing."""
        if self.color_only and not is_color(image):
            return False
        return True

    def suggest(self, image: np.ndarray) -> dict[str, Any]:
        """Automatic starting parameters, measured from the image."""
        try:
            return dict(self.suggest_fn(image))
        except Exception:
            log.debug("Guided: suggestion failed for %s", self.step_id, exc_info=True)
            return {c.key: c.minimum for c in self.controls}

    def apply(self, image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        """Run this step. Returns a new array; never mutates the input."""
        out = self.apply_fn(image, params)
        return np.clip(np.nan_to_num(out), 0.0, 1.0).astype(np.float32, copy=False)


# ── Step implementations ──────────────────────────────────────────────
# Each is a thin adapter over an existing core tool. Imports stay inside the
# functions so that importing this module (e.g. to render the step list in the
# UI) does not drag in torch and the whole processing stack.


def _apply_trim(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    percent = float(params.get("percent", 0.0))
    if percent <= 0:
        return image
    h, w = image.shape[-2], image.shape[-1]
    dy = int(round(h * percent / 100.0))
    dx = int(round(w * percent / 100.0))
    # Never trim the frame out of existence.
    dy = min(dy, max(0, (h - 8) // 2))
    dx = min(dx, max(0, (w - 8) // 2))
    if dy == 0 and dx == 0:
        return image
    if image.ndim == 3:
        return image[:, dy:h - dy, dx:w - dx]
    return image[dy:h - dy, dx:w - dx]


def _suggest_trim(image: np.ndarray) -> dict[str, Any]:
    return {"percent": 0.0}


def _apply_gradient(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    from astraios.core.abe import ABEParams, abe_extract

    degree = int(round(float(params.get("degree", 2))))
    result = abe_extract(image, ABEParams(polynomial_degree=degree))
    return result[0] if isinstance(result, tuple) else result


def _suggest_gradient(image: np.ndarray) -> dict[str, Any]:
    return {"degree": 2}


def _apply_color(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    from astraios.core.color_calibration import ColorCalibrationParams, color_calibrate

    result = color_calibrate(
        image, ColorCalibrationParams(neutralize_background=True)
    )
    out = result.data

    amount = float(params.get("green_amount", 0.0))
    if amount > 0:
        from astraios.core.color_tools import SCNRParams, scnr

        out = scnr(out, SCNRParams(amount=amount))
    return out


def _suggest_color(image: np.ndarray) -> dict[str, Any]:
    """Suggest green removal only when there is actually a green excess.

    A broadband OSC frame under light pollution usually has one; a narrowband
    palette deliberately does not, and stripping green there would wreck it.
    """
    if not is_color(image):
        return {"green_amount": 0.0}
    means = [float(np.nanmean(image[c])) for c in range(3)]
    r, g, b = means
    neutral = (r + b) / 2.0
    if neutral <= 1e-6:
        return {"green_amount": 0.0}
    excess = (g - neutral) / neutral
    return {"green_amount": float(np.clip(excess * 2.0, 0.0, 0.8))}


def _apply_sharpen(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    from astraios.core.filters import UnsharpMaskParams, unsharp_mask

    return unsharp_mask(
        image,
        UnsharpMaskParams(
            radius=float(params.get("radius", 1.5)),
            amount=float(params.get("amount", 0.4)),
        ),
    )


def _suggest_sharpen(image: np.ndarray) -> dict[str, Any]:
    return {"radius": 1.5, "amount": 0.4}


def _apply_denoise(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    from astraios.core.denoise import DenoiseParams, denoise

    return denoise(
        image, DenoiseParams(strength=float(params.get("strength", 0.4)))
    )


def _suggest_denoise(image: np.ndarray) -> dict[str, Any]:
    """Scale the strength to the noise actually present.

    Uses a robust MAD estimate of the background so a clean stack is not
    smoothed as hard as a single noisy sub.
    """
    lum = _luma(image)
    finite = lum[np.isfinite(lum)]
    if finite.size == 0:
        return {"strength": 0.4}
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    sigma = mad * 1.4826
    # sigma ~0.002 (clean) -> gentle; ~0.02 (noisy) -> strong.
    strength = float(np.clip((sigma - 0.001) / 0.02, 0.15, 0.85))
    return {"strength": round(strength, 2)}


def _apply_stretch(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    from astraios.core.stretch import StatisticalStretchParams, statistical_stretch

    return statistical_stretch(
        image,
        StatisticalStretchParams(
            target_median=float(params.get("brightness", 0.25)),
            linked=bool(params.get("linked", True)),
        ),
    )


def _suggest_stretch(image: np.ndarray) -> dict[str, Any]:
    return {"brightness": 0.25, "linked": True}


def _apply_saturation(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    from astraios.core.color_tools import ColorAdjustParams, color_adjust

    boost = float(params.get("saturation", 0.0))
    if abs(boost) < 1e-6:
        return image
    return color_adjust(image, ColorAdjustParams(saturation=boost))


def _suggest_saturation(image: np.ndarray) -> dict[str, Any]:
    return {"saturation": 15.0}


# ── The workflow ──────────────────────────────────────────────────────

def build_workflow() -> list[GuidedStep]:
    """The ordered guided workflow.

    Returned fresh each call so a caller can filter it for a given image
    without mutating shared state.
    """
    return [
        GuidedStep(
            step_id="trim",
            title="Trim the edges",
            summary="Cuts off the ragged border that stacking leaves behind.",
            detail=(
                "Frames never line up perfectly, so the outer edge of a stack "
                "is built from fewer exposures and looks noisy or streaky. "
                "Trimming a little off every side removes it. Leave at 0 if "
                "your edges already look clean."
            ),
            controls=[
                GuidedControl(
                    "percent", "Trim", 0.0, 15.0, 0.5, 1,
                    "How much to cut from each side, as a percentage of the frame.",
                    "Removes more of the border, at the cost of field of view.",
                    "Keeps the full frame; 0 trims nothing.",
                ),
            ],
            apply_fn=_apply_trim,
            suggest_fn=_suggest_trim,
            default_skipped=True,
        ),
        GuidedStep(
            step_id="gradient",
            title="Remove the gradient",
            summary="Flattens the uneven glow from light pollution and the Moon.",
            detail=(
                "Skyglow is rarely even across the frame, so one corner ends up "
                "brighter than the other. This models that slow background ramp "
                "and subtracts it, which is what lets the later stretch bring up "
                "faint detail instead of just amplifying the gradient."
            ),
            controls=[
                GuidedControl(
                    "degree", "Complexity", 1, 5, 1, 0,
                    "How complicated a background shape to model.",
                    "Follows more complex gradients, but can start eating "
                    "large faint nebulosity.",
                    "A gentler, safer flattening. Start here.",
                ),
            ],
            apply_fn=_apply_gradient,
            suggest_fn=_suggest_gradient,
        ),
        GuidedStep(
            step_id="color",
            title="Balance the colour",
            summary="Makes the sky neutral and the stars their true colours.",
            detail=(
                "Cameras and filters do not weight red, green and blue equally, "
                "and light pollution adds its own cast. This measures the frame "
                "and corrects it. It runs before the stretch on purpose: colour "
                "measured on linear data is honest, measured after a stretch it "
                "is not."
            ),
            controls=[
                GuidedControl(
                    "green_amount", "Remove green cast", 0.0, 1.0, 0.05, 2,
                    "Broadband images under light pollution often skew green.",
                    "Removes the cast more completely.",
                    "Leaves the colour as measured; 0 disables it. Keep at 0 "
                    "for narrowband palettes, where green is meaningful.",
                ),
            ],
            apply_fn=_apply_color,
            suggest_fn=_suggest_color,
            color_only=True,
        ),
        GuidedStep(
            step_id="sharpen",
            title="Sharpen the detail",
            summary="Recovers fine structure softened by the atmosphere.",
            detail=(
                "Seeing and tracking smear fine detail. A modest sharpen brings "
                "it back. Done here, while the data is still linear, it behaves "
                "predictably; done after a hard stretch it mostly amplifies "
                "noise. Keep it subtle: if stars grow dark rings, back off."
            ),
            controls=[
                GuidedControl(
                    "amount", "Strength", 0.0, 1.5, 0.05, 2,
                    "How hard the sharpening pushes.",
                    "Crisper detail, but dark or bright rings can appear "
                    "around stars.",
                    "A gentle lift; 0 leaves the image untouched.",
                ),
                GuidedControl(
                    "radius", "Detail size", 0.5, 5.0, 0.1, 1,
                    "The size of the structures being sharpened, in pixels.",
                    "Targets broader structure such as dust lanes and arms.",
                    "Targets the very finest detail and star cores.",
                ),
            ],
            apply_fn=_apply_sharpen,
            suggest_fn=_suggest_sharpen,
        ),
        GuidedStep(
            step_id="denoise",
            title="Reduce the noise",
            summary="Smooths grain in the background without flattening detail.",
            detail=(
                "The suggested strength is measured from your own data, so a "
                "clean stack is treated gently and a noisy one more firmly. "
                "Watch the background in the preview: you want grain gone but "
                "faint structure still visible."
            ),
            controls=[
                GuidedControl(
                    "strength", "Strength", 0.0, 1.0, 0.05, 2,
                    "How aggressively noise is smoothed away.",
                    "A cleaner background, but faint detail can go plastic.",
                    "Keeps more real detail and more grain with it.",
                ),
            ],
            apply_fn=_apply_denoise,
            suggest_fn=_suggest_denoise,
        ),
        GuidedStep(
            step_id="stretch",
            title="The big stretch",
            summary="Turns the dark linear stack into a visible picture.",
            detail=(
                "A stacked frame is almost black because the camera records "
                "light linearly while your eye does not. This is the step that "
                "makes the target appear. Everything above happens first "
                "precisely so this stretch lifts real signal rather than "
                "gradients and noise."
            ),
            controls=[
                GuidedControl(
                    "brightness", "Brightness", 0.05, 0.60, 0.01, 2,
                    "How bright the background sky ends up.",
                    "Lifts more faint signal into view, but flattens contrast "
                    "and shows more noise.",
                    "A darker, punchier sky with less faint detail.",
                ),
            ],
            apply_fn=_apply_stretch,
            suggest_fn=_suggest_stretch,
        ),
        GuidedStep(
            step_id="saturation",
            title="Make it beautiful",
            summary="Deepens the colour now that the image is stretched.",
            detail=(
                "Stretching washes colour out, so a final saturation lift puts "
                "it back: red in emission nebulae, blue in reflection nebulae "
                "and hot stars. This comes last because saturating linear data "
                "would do almost nothing."
            ),
            controls=[
                GuidedControl(
                    "saturation", "Colour", -50.0, 80.0, 1.0, 0,
                    "How strong the colours are.",
                    "Richer, more vivid colour, until star cores clip to "
                    "solid hues.",
                    "Muted colour; negative values go toward grey.",
                ),
            ],
            apply_fn=_apply_saturation,
            suggest_fn=_suggest_saturation,
            color_only=True,
        ),
    ]


def workflow_for(image: np.ndarray) -> list[GuidedStep]:
    """The steps that make sense for this particular image."""
    return [s for s in build_workflow() if s.applies_to(image)]


@dataclass
class GuidedRun:
    """Result of running the whole workflow non-interactively."""

    image: np.ndarray
    applied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def run_workflow(
    image: np.ndarray,
    overrides: dict[str, dict[str, Any]] | None = None,
    skip: set[str] | None = None,
    progress: ProgressCallback = _noop_progress,
) -> GuidedRun:
    """Run every applicable step with its suggested (or overridden) settings.

    This is the wizard's "just do it for me" path, and it is also what the
    tests exercise: the UI only ever adds preview and the choice to skip.
    """
    overrides = overrides or {}
    skip = set(skip or ())
    steps = workflow_for(image)
    # Copy rather than asarray. Every step currently returns a fresh array, so
    # this fixes no present bug, but asarray does not copy when the input is
    # already float32 -- which is the normal case here -- so `out` would alias
    # the caller's image. One future step doing an in-place write would then
    # silently rewrite the user's loaded data, including when they back out of
    # the wizard without applying anything. One copy is cheap next to the six
    # intermediates the workflow already allocates.
    out = np.array(image, dtype=np.float32, copy=True)
    run = GuidedRun(image=out)

    for i, step in enumerate(steps):
        progress(i / max(len(steps), 1), step.title)
        if step.step_id in skip or (step.default_skipped and step.step_id not in overrides):
            run.skipped.append(step.step_id)
            continue
        params = step.suggest(out)
        params.update(overrides.get(step.step_id, {}))
        try:
            out = step.apply(out, params)
            run.applied.append(step.step_id)
        except Exception:
            log.warning("Guided: step %s failed, skipping", step.step_id, exc_info=True)
            run.skipped.append(step.step_id)

    progress(1.0, "Finished")
    run.image = out
    return run


__all__ = [
    "GuidedControl",
    "GuidedStep",
    "GuidedRun",
    "build_workflow",
    "workflow_for",
    "run_workflow",
    "is_color",
    "looks_linear",
]
