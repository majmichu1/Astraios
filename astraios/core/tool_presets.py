"""Tool Presets — save/recall named parameter sets for any processing tool.

Ported from Seti Astro Suite Pro's parameter save/recall feature (added to
its Statistical Stretch tool in v1.18.12, Copyright Franklin Marek,
GPL-3.0-or-later, https://github.com/setiastro/setiastrosuitepro) — but
implemented as a *general* mechanism here rather than being wired to a
single tool, since every Astraios tool already captures its settings as a
plain ``*Params`` dataclass (see e.g. `astraios/core/pedestal.py`,
`astraios/core/stretch.py`). Any tool can save/recall presets by passing its
own ``tool_name`` and a params dict.

Storage: one JSON file per tool at ``~/.astraios/presets/<tool_name>.json``,
mapping preset name -> encoded params dict. Encoding reuses
`astraios.core.processing_graph._encode_params` / `_decode_params` — the
same enum-tagging scheme already used to make processing-history steps
JSON-safe and replayable — so a preset containing enum fields (e.g. a
``DenoiseMethod`` member) round-trips exactly instead of silently degrading
to a string.

Writes are atomic (write to a temp file, then `os.replace`) so a crash or
concurrent write can't leave a truncated/corrupt preset file behind.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from astraios.core.processing_graph import _decode_params, _encode_params

log = logging.getLogger(__name__)

PRESETS_DIR = Path.home() / ".astraios" / "presets"

# Characters kept as-is in a tool_name when deriving a filename; everything
# else is replaced with "_" so arbitrary tool names can't escape PRESETS_DIR
# or collide with filesystem-reserved characters.
_SAFE_CHARS = "-_."


def _sanitize_tool_name(tool_name: str) -> str:
    if not tool_name or not str(tool_name).strip():
        raise ValueError("tool_name must be a non-empty string")
    return "".join(c if c.isalnum() or c in _SAFE_CHARS else "_" for c in str(tool_name))


def _preset_file(tool_name: str) -> Path:
    return PRESETS_DIR / f"{_sanitize_tool_name(tool_name)}.json"


def _load_all(tool_name: str) -> dict[str, Any]:
    """Return the raw (still enum-encoded) preset dict for a tool, or {} if absent/corrupt."""
    path = _preset_file(tool_name)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Corrupt or unreadable preset file %s (%s); treating as empty.", path, exc)
        return {}
    if not isinstance(raw, dict):
        log.warning("Preset file %s did not contain a JSON object; treating as empty.", path)
        return {}
    return raw


def _save_all(tool_name: str, data: dict[str, Any]) -> None:
    path = _preset_file(tool_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.replace(path)


def save_preset(tool_name: str, preset_name: str, params: dict[str, Any]) -> None:
    """Save (or overwrite) a named preset of `params` for `tool_name`.

    Parameters
    ----------
    tool_name : str
        Identifies which tool's preset file to write to (e.g. "statistical_stretch").
    preset_name : str
        The name under which to store `params`; overwrites any existing
        preset of the same name for this tool.
    params : dict
        JSON-encodable parameter values; enum members are supported and
        round-trip exactly via `load_preset`.
    """
    if not preset_name or not str(preset_name).strip():
        raise ValueError("preset_name must be a non-empty string")
    data = _load_all(tool_name)
    data[preset_name] = _encode_params(dict(params))
    _save_all(tool_name, data)


def load_preset(tool_name: str, preset_name: str) -> dict[str, Any]:
    """Load a previously-saved preset, decoding any tagged enum members.

    Raises
    ------
    KeyError
        If `tool_name` has no preset named `preset_name` (including when the
        tool has no preset file at all).
    """
    data = _load_all(tool_name)
    if preset_name not in data:
        raise KeyError(f"No preset {preset_name!r} for tool {tool_name!r}")
    decoded = _decode_params(data[preset_name])
    if not isinstance(decoded, dict):
        raise ValueError(f"Preset {preset_name!r} for tool {tool_name!r} is not a params dict")
    return decoded


def list_presets(tool_name: str) -> list[str]:
    """Return the sorted preset names saved for `tool_name` (empty list if none)."""
    return sorted(_load_all(tool_name).keys())


def delete_preset(tool_name: str, preset_name: str) -> None:
    """Delete a preset if it exists; a no-op if the tool or preset is missing."""
    data = _load_all(tool_name)
    if preset_name in data:
        del data[preset_name]
        _save_all(tool_name, data)
