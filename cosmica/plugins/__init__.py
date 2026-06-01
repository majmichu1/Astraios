"""Cosmica plugin system."""

from cosmica.plugins.base import (
    CosmicaPlugin,
    CosmicaProcess,
    get_process,
    list_processes,
    register_process,
    registry,
    scan_plugins,
)

__all__ = [
    "CosmicaPlugin",
    "CosmicaProcess",
    "get_process",
    "list_processes",
    "register_process",
    "registry",
    "scan_plugins",
]
