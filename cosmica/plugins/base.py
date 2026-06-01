"""Plugin system base classes for Cosmica."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

PLUGIN_DIRS = [
    Path.home() / ".cosmica" / "plugins",
    Path(__file__).parent / "builtin",
]


class CosmicaProcess:
    """A single process/tool provided by a plugin.

    Subclass this and override apply() and/or interface().
    """

    identifier: str = "base_process"
    name: str = "Base Process"
    description: str = ""
    version: str = "1.0.0"

    def apply(self, image: np.ndarray, params: dict[str, Any] | None = None) -> np.ndarray:
        raise NotImplementedError

    def interface(self, parent=None):
        return None  # optional QWidget for settings


class CosmicaPlugin:
    """A plugin that provides one or more processes."""

    name: str = ""
    version: str = "1.0.0"
    author: str = ""
    description: str = ""
    processes: list[type[CosmicaProcess]] = []

    def register(self):
        for proc_cls in self.processes:
            proc = proc_cls()
            registry[proc.identifier] = proc
            log.info("Plugin %s registered process: %s", self.name, proc.identifier)


registry: dict[str, CosmicaProcess] = {}


def register_process(proc: CosmicaProcess):
    registry[proc.identifier] = proc
    log.info("Registered process: %s (%s)", proc.identifier, proc.name)


def get_process(identifier: str) -> CosmicaProcess | None:
    return registry.get(identifier)


def list_processes() -> list[str]:
    return list(registry.keys())


def scan_plugins():
    """Scan plugin directories and register found plugins."""
    for plugin_dir in PLUGIN_DIRS:
        if not plugin_dir.exists():
            plugin_dir.mkdir(parents=True, exist_ok=True)
            continue
        for path in plugin_dir.iterdir():
            if path.suffix == ".py" and not path.name.startswith("_"):
                _load_plugin_from_file(path)
            elif path.is_dir() and (path / "plugin.toml").exists():
                _load_plugin_from_dir(path)


def _load_plugin_from_file(path: Path):
    """Load a single .py file as a plugin."""
    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            return
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, CosmicaPlugin)
                and attr is not CosmicaPlugin
            ):
                plugin = attr()
                plugin.register()
                log.info("Loaded plugin: %s from %s", plugin.name, path)
    except Exception as e:
        log.warning("Failed to load plugin %s: %s", path, e)


def _load_plugin_from_dir(path: Path):
    """Load a plugin from a directory with plugin.toml manifest."""
    try:
        import tomllib

        manifest = tomllib.loads((path / "plugin.toml").read_text())
        main_file = path / (manifest.get("main", "main") + ".py")
        if main_file.exists():
            _load_plugin_from_file(main_file)
    except Exception as e:
        log.warning("Failed to load plugin dir %s: %s", path, e)
