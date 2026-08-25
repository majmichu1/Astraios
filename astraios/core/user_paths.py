"""User-configured paths for external tools and AI models.

Set in Preferences, these let a user point Astraios at models and binaries they
already have (a StarNet executable, a denoise/sharpen model, a Cosmic Clarity
model folder) instead of re-downloading them. It also means Astraios never has
to bundle or redistribute third-party models: it simply uses what the user
points it at.

Reads are best-effort. If the setting is unset, the path doesn't exist, or Qt
isn't available (e.g. a headless test), every accessor returns ``None`` and the
caller falls back to its previous behaviour.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "starnet_binary",
    "model_override",
    "cosmic_clarity_dir",
    "astap_command",
    "default_dir",
]


def _get(key: str) -> str | None:
    """Read ``models/<key>`` from the Astraios QSettings, or None."""
    try:
        from PyQt6.QtCore import QSettings

        val = QSettings("Astraios", "Astraios").value(f"models/{key}")
    except Exception:
        return None
    if val is None:
        return None
    text = str(val).strip()
    return text or None


def starnet_binary() -> Path | None:
    """Path to a user-provided StarNet executable, if set and present."""
    raw = _get("starnet_path")
    if raw:
        p = Path(raw)
        if p.is_file():
            return p
    return None


def model_override(kind: str) -> Path | None:
    """Path to a user-provided model weight file for ``kind``.

    ``kind`` is ``"denoise"`` or ``"sharpen"``. Returns None when unset or the
    file is missing.
    """
    raw = _get(f"{kind}_model")
    if raw:
        p = Path(raw)
        if p.is_file():
            return p
    return None


def _setting(key: str) -> str | None:
    """Read any Astraios QSettings key as stripped text, or None."""
    try:
        from PyQt6.QtCore import QSettings
        val = QSettings("Astraios", "Astraios").value(key)
    except Exception:
        return None
    if val is None:
        return None
    text = str(val).strip()
    return text or None


def astap_command() -> str | None:
    """The ASTAP executable to run: the one set in Preferences if it exists,
    else whatever is on PATH, else None.

    ASTAP installs outside PATH on Windows and macOS, which used to make the
    offline solver silently unavailable there.
    """
    import shutil

    raw = _setting("platesolver/astap_path")
    if raw and Path(raw).is_file():
        return raw
    return shutil.which("astap_cli") or shutil.which("astap")


def default_dir(kind: str) -> str:
    """Starting folder for file dialogs: ``kind`` is ``"import"`` or ``"export"``.

    Returns "" (the platform default) when the preference is unset or the
    folder no longer exists.
    """
    raw = _setting(f"paths/default_{kind}_dir")
    return raw if raw and Path(raw).is_dir() else ""


def cosmic_clarity_dir() -> Path | None:
    """Folder holding the user's own Cosmic Clarity model files, if set."""
    raw = _get("cosmic_clarity_dir")
    if raw:
        p = Path(raw)
        if p.is_dir():
            return p
    return None
