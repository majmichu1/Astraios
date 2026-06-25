"""Processing history: a non-destructive, linear, replayable pipeline.

Each step records the canonical tool name and its parameters (plus an optional
named mask), not pixel data. The history re-evaluates from the base image
through the enabled steps to reproduce the image at any stage, so steps can be
toggled, deleted, reordered, or have their parameters edited and everything
downstream recomputes.

Re-running a step goes through the shared tool registry (``astraios.core.batch``),
the same path macros and batch pipelines use, so one step definition drives all
three. A step whose ``tool_name`` is empty is display-only (it is shown in the
history but cannot be replayed); see ``is_replayable``.

The class is named ``ProcessingGraph`` for backwards compatibility, but the
model is a straight pipeline: ``base -> step[0] -> step[1] -> ... -> result``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from numpy.typing import NDArray

log = logging.getLogger(__name__)


def _params_hash(params: dict[str, Any]) -> str:
    """Stable hash of a params dict, robust to nested/unhashable values.

    ``frozenset(params.items())`` blows up on list/dict values; JSON with
    ``default=str`` handles enums, paths, arrays-as-lists, etc.
    """
    try:
        return json.dumps(params, sort_keys=True, default=str)
    except Exception:
        return repr(sorted(params.items(), key=lambda kv: kv[0]))


@dataclass
class HistoryStep:
    """A single non-destructive step in the processing history."""

    tool_name: str                       # canonical registry name; "" = display-only
    params: dict[str, Any] = field(default_factory=dict)
    display_name: str = ""               # human label for the list
    mask_name: str | None = None
    enabled: bool = True
    locked: bool = False                 # protected from delete/reorder

    _cache_output: NDArray | None = field(default=None, repr=False)
    _cache_key: str | None = field(default=None, repr=False)

    @property
    def label(self) -> str:
        return self.display_name or self.tool_name.replace("_", " ").title() or "Step"

    @property
    def replayable(self) -> bool:
        return bool(self.tool_name)

    def _invalidate(self) -> None:
        self._cache_output = None
        self._cache_key = None


# Type of the re-execution callback: (tool_name, params, image) -> image
ProcessFn = Callable[[str, dict[str, Any], NDArray], NDArray | None]


@dataclass
class ProcessingGraph:
    """A linear, non-destructive, replayable processing history."""

    base_image: NDArray | None = field(default=None, repr=False)
    steps: list[HistoryStep] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    #  Base + recording                                                   #
    # ------------------------------------------------------------------ #
    def set_base(self, image: NDArray) -> None:
        """Set the base image the history is applied to. Invalidates all caches."""
        self.base_image = image.copy()
        for s in self.steps:
            s._invalidate()

    def record(
        self,
        tool_name: str,
        params: dict[str, Any] | None = None,
        display_name: str = "",
        mask_name: str | None = None,
    ) -> int:
        """Append a step to the history. Returns its index.

        ``tool_name`` should be a canonical registry name so the step can be
        replayed; pass ``""`` for a display-only marker. The newly appended
        step starts with the latest pixels (its cache is filled lazily on the
        next evaluate), so recording never triggers a recompute by itself.
        """
        step = HistoryStep(
            tool_name=tool_name,
            params=dict(params or {}),
            display_name=display_name,
            mask_name=mask_name,
        )
        self.steps.append(step)
        return len(self.steps) - 1

    # ------------------------------------------------------------------ #
    #  Mutation (each invalidates this step and everything downstream)    #
    # ------------------------------------------------------------------ #
    def _valid(self, index: int) -> bool:
        return 0 <= index < len(self.steps)

    def _invalidate_from(self, index: int) -> None:
        for i in range(max(0, index), len(self.steps)):
            self.steps[i]._invalidate()

    def remove(self, index: int) -> bool:
        if not self._valid(index) or self.steps[index].locked:
            return False
        self.steps.pop(index)
        self._invalidate_from(index)
        return True

    def move(self, src: int, dst: int) -> bool:
        if not self._valid(src) or not self._valid(dst) or src == dst:
            return False
        if self.steps[src].locked:
            return False
        step = self.steps.pop(src)
        self.steps.insert(dst, step)
        self._invalidate_from(min(src, dst))
        return True

    def set_enabled(self, index: int, enabled: bool) -> None:
        if self._valid(index) and self.steps[index].enabled != enabled:
            self.steps[index].enabled = enabled
            self._invalidate_from(index)

    def set_locked(self, index: int, locked: bool) -> None:
        if self._valid(index):
            self.steps[index].locked = locked

    def update_params(self, index: int, params: dict[str, Any]) -> None:
        if self._valid(index):
            self.steps[index].params = dict(params)
            self._invalidate_from(index)

    def clear(self) -> None:
        self.steps.clear()

    # ------------------------------------------------------------------ #
    #  Evaluation                                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _key(step: HistoryStep) -> str:
        return f"{step.tool_name}|{step.mask_name}|{_params_hash(step.params)}"

    def is_replayable(self, up_to: int | None = None) -> bool:
        """True if every enabled step up to *up_to* has a replayable tool name."""
        last = self._last_index(up_to)
        return all(
            s.replayable for s in self.steps[: last + 1] if s.enabled
        )

    def _last_index(self, up_to: int | None) -> int:
        if up_to is None:
            return len(self.steps) - 1
        return max(-1, min(up_to, len(self.steps) - 1))

    def evaluate(
        self,
        up_to: int | None = None,
        process_fn: ProcessFn | None = None,
    ) -> NDArray | None:
        """Reproduce the image after applying steps ``0..up_to`` (inclusive).

        Walks from the base image, applying each enabled step through
        ``process_fn``. Results are cached per step keyed on (tool, mask,
        params); a step recomputes only when its key changed or an upstream
        step recomputed. Returns a fresh array (never an aliased cache buffer),
        or ``None`` if there is no base image or a step fails.
        """
        if self.base_image is None:
            return None
        last = self._last_index(up_to)
        if last < 0:
            return self.base_image.copy()

        current = self.base_image
        upstream_changed = False

        for i in range(last + 1):
            step = self.steps[i]
            if not step.enabled:
                continue
            key = self._key(step)
            cache_ok = (
                not upstream_changed
                and step._cache_output is not None
                and step._cache_key == key
            )
            if cache_ok:
                current = step._cache_output
                continue
            if process_fn is None or not step.tool_name:
                # Display-only / no executor: cannot recompute, pass pixels through.
                continue
            try:
                out = process_fn(step.tool_name, step.params, current)
            except Exception as e:  # noqa: BLE001 - one bad step must not crash the view
                log.error("History step %d (%s) failed: %s", i, step.tool_name, e)
                return None
            if out is None:
                continue
            step._cache_output = out
            step._cache_key = key
            current = out
            upstream_changed = True

        # Never hand back a cache buffer (or the base) the caller could mutate.
        return current.copy()

    # ------------------------------------------------------------------ #
    #  Introspection / export / persistence                              #
    # ------------------------------------------------------------------ #
    def list_steps(self) -> list[dict]:
        """Rows for the history UI, in order."""
        rows = []
        for i, s in enumerate(self.steps):
            rows.append(
                {
                    "index": i,
                    "label": s.label,
                    "tool_name": s.tool_name,
                    "enabled": s.enabled,
                    "locked": s.locked,
                    "replayable": s.replayable,
                    "mask_name": s.mask_name,
                    "params": s.params,
                }
            )
        return rows

    def to_pipeline(self, name: str = "History"):
        """Export the enabled, replayable steps as a macro Pipeline."""
        from astraios.core.scripting import Pipeline, PipelineStep

        steps = [
            PipelineStep(tool_name=s.tool_name, params=dict(s.params), mask_name=s.mask_name)
            for s in self.steps
            if s.enabled and s.replayable
        ]
        return Pipeline(name=name, steps=steps)

    def to_dict(self) -> dict:
        return {
            "version": 2,
            "steps": [
                {
                    "tool_name": s.tool_name,
                    "params": s.params,
                    "display_name": s.display_name,
                    "mask_name": s.mask_name,
                    "enabled": s.enabled,
                    "locked": s.locked,
                }
                for s in self.steps
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> ProcessingGraph:
        graph = cls()
        # v2 (linear). Older v1 graphs (node dict) are display-only history, so
        # we keep their step names but drop the broken DAG wiring/params.
        if "steps" in data:
            for sd in data["steps"]:
                graph.steps.append(
                    HistoryStep(
                        tool_name=sd.get("tool_name", ""),
                        params=sd.get("params", {}),
                        display_name=sd.get("display_name", ""),
                        mask_name=sd.get("mask_name"),
                        enabled=sd.get("enabled", True),
                        locked=sd.get("locked", False),
                    )
                )
        elif "nodes" in data:
            for nid, nd in data["nodes"].items():
                if nid == data.get("root_id", "base"):
                    continue
                graph.steps.append(
                    HistoryStep(
                        tool_name="",  # legacy nodes were never replayable
                        params=nd.get("params", {}),
                        display_name=str(nd.get("process_name", "")),
                        enabled=nd.get("enabled", True),
                        locked=nd.get("locked", False),
                    )
                )
        return graph


__all__ = ["ProcessingGraph", "HistoryStep", "ProcessFn"]
