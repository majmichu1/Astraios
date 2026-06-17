"""Macro Scripting — record and replay processing operations.

Provides a MacroRecorder that captures processing steps as they happen,
and a MacroPlayer that replays them on the current image.
Macros are stored as Pipeline objects and can be saved/loaded as JSON.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from astraios.core.batch import Pipeline, PipelineStep, apply_pipeline_to_image

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


@dataclass
class MacroRecorder:
    """Records processing operations as PipelineSteps.

    Usage:
        recorder = MacroRecorder()
        recorder.start()
        recorder.record_step("auto_stretch", {"midtone": 0.25})
        recorder.record_step("scnr", {"amount": 0.8})
        macro = recorder.stop()
        save_macro(macro, Path("my_macro.json"))
    """

    _recording: bool = False
    _pipeline: Pipeline = field(default_factory=lambda: Pipeline(name="Recorded Macro"))

    def start(self, name: str = "Recorded Macro"):
        """Start recording a new macro."""
        self._recording = True
        self._pipeline = Pipeline(name=name)
        log.info("Macro recording started: %s", name)

    def stop(self) -> Pipeline:
        """Stop recording and return the captured pipeline."""
        self._recording = False
        n = len(self._pipeline.steps)
        log.info("Macro recording stopped: %d steps captured", n)
        return self._pipeline

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def step_count(self) -> int:
        return len(self._pipeline.steps)

    def record_step(
        self,
        tool_name: str,
        params: dict[str, Any] | None = None,
        mask_name: str | None = None,
    ):
        """Record a processing step if currently recording.

        Parameters
        ----------
        tool_name : str
            Registered tool name.
        params : dict, optional
            Tool parameters.
        mask_name : str, optional
            Name of a saved mask to apply with this step.
        """
        if not self._recording:
            return
        step = PipelineStep(
            tool_name=tool_name, params=params or {}, mask_name=mask_name,
        )
        self._pipeline.steps.append(step)
        log.debug("Recorded step: %s %s (mask=%s)", tool_name, params or {}, mask_name)

    def discard(self):
        """Discard the current recording."""
        self._recording = False
        self._pipeline = Pipeline(name="Recorded Macro")
        log.info("Macro recording discarded")


def play_macro(
    data: np.ndarray,
    macro: Pipeline,
    progress: ProgressCallback | None = None,
    masks: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Replay a macro (pipeline) on image data.

    Parameters
    ----------
    data : ndarray
        Image data, float32 in [0, 1].
    macro : Pipeline
        The macro/pipeline to replay.
    progress : callable, optional
        Progress callback.
    masks : dict, optional
        Named masks ``{name: (H, W) float32 array}`` for mask-aware steps.

    Returns
    -------
    ndarray
        Processed image after all macro steps.
    """
    if progress is None:
        progress = _noop_progress

    n_steps = sum(1 for s in macro.steps if s.enabled)
    if n_steps == 0:
        log.warning("Macro has no enabled steps")
        return data.copy()

    progress(0.0, f"Playing macro: {macro.name} ({n_steps} steps)")
    log.info("Playing macro: %s (%d steps)", macro.name, n_steps)

    result = apply_pipeline_to_image(data, macro, progress=progress, masks=masks)

    progress(1.0, "Macro playback complete")
    return result


def save_macro(macro: Pipeline, path: Path) -> None:
    """Save a macro/pipeline to a JSON file.

    Parameters
    ----------
    macro : Pipeline
        The macro to save.
    path : Path
        Output file path (.json).
    """
    data = macro.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    log.info("Macro saved: %s (%d steps)", path, len(macro.steps))


def load_macro(path: Path) -> Pipeline:
    """Load a macro/pipeline from a JSON file.

    Parameters
    ----------
    path : Path
        Input file path (.json).

    Returns
    -------
    Pipeline
        Loaded macro.
    """
    with open(path) as f:
        data = json.load(f)
    macro = Pipeline.from_dict(data)
    log.info("Macro loaded: %s (%d steps)", macro.name, len(macro.steps))
    return macro
