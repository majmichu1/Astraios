"""Astraios plugin system."""

from astraios.plugins.base import (
    AstraiosPlugin,
    AstraiosProcess,
    get_process,
    list_processes,
    register_process,
    registry,
    scan_plugins,
)

__all__ = [
    "AstraiosPlugin",
    "AstraiosProcess",
    "get_process",
    "list_processes",
    "register_process",
    "registry",
    "scan_plugins",
]
