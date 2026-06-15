"""Per-target processing recipes.

After plate solving identifies the target, the Smart Processor resolves a
**recipe** — a set of processing instructions — and merges it into the target's
``processing_hints`` so the planner applies the right pipeline for *that kind of
object* (and, where curated, *that specific object*).

Resolution is layered, later layers win:

    1. defaults (empty — heuristics take over for anything unspecified)
    2. TYPE recipe        (recipes.json ``types`` keyed on object_type)
    3. catalog hints      (the target's own ``processing_hints``)
    4. named override     (recipes.json ``targets`` keyed on the target id)

Recipes reuse the planner's existing hint vocabulary (``stretch``,
``ha_dominant``, ``bg_sensitive``, ``hdr_merge_recommended``, …) so no planner
change is needed to honour them, plus a few processor-level knobs:
``chroma_strength`` (float), ``star_reduction`` (0..1), ``use_starnet`` (bool),
``noise_reduction`` (minimal|moderate|strong).

Because recipes are keyed on *type*, every object resolves — you never need to
have imaged it. Fully best-effort: a missing/broken recipes.json just means the
heuristic defaults are used.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

__all__ = ["RecipeBook", "get_recipe_book"]

_RECIPES_JSON = Path(__file__).resolve().parent.parent / "resources" / "recipes.json"


class RecipeBook:
    """Loads recipes.json and resolves merged hints for a target."""

    def __init__(self, path: Path | None = None) -> None:
        self._types: dict[str, dict[str, Any]] = {}
        self._targets: dict[str, dict[str, Any]] = {}
        self._loaded = False
        self._path = path or _RECIPES_JSON

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            with open(self._path, encoding="utf-8") as fh:
                raw = json.load(fh)
            self._types = {k: v for k, v in raw.get("types", {}).items() if isinstance(v, dict)}
            # Index target overrides case-insensitively, normalising spaces.
            for tid, rec in raw.get("targets", {}).items():
                if isinstance(rec, dict):
                    self._targets[_norm(tid)] = rec
            log.info("Loaded recipes: %d types, %d named targets",
                     len(self._types), len(self._targets))
        except Exception as exc:  # missing / malformed → heuristics only
            log.info("No usable recipes.json (%s) — using heuristic defaults", exc)

    def resolve(
        self,
        object_type: str | None,
        target_id: str | None,
        catalog_hints: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Return the merged hints for a target (type → catalog → named)."""
        self._ensure_loaded()
        merged: dict[str, Any] = {}
        if object_type and object_type in self._types:
            merged.update(self._types[object_type])
        if catalog_hints:
            merged.update(catalog_hints)
        if target_id:
            override = self._targets.get(_norm(target_id))
            if override:
                merged.update(override)
        return merged

    def has_named(self, target_id: str | None) -> bool:
        self._ensure_loaded()
        return bool(target_id) and _norm(target_id) in self._targets


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "")


_BOOK: RecipeBook | None = None


def get_recipe_book() -> RecipeBook:
    """Process-wide singleton recipe book."""
    global _BOOK
    if _BOOK is None:
        _BOOK = RecipeBook()
    return _BOOK
