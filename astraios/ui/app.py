"""Application bootstrap — QApplication setup, theme, splash, and launch."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from PyQt6.QtCore import QEvent
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import QApplication, QComboBox, QSplashScreen

import astraios
from astraios.ui.theme import DARK_THEME

log = logging.getLogger(__name__)


class _AstraiosApp(QApplication):
    """QApplication subclass that blocks scroll-wheel on all QComboBox instances."""

    def notify(self, obj, event):
        if event.type() == QEvent.Type.Wheel and isinstance(obj, QComboBox):
            return True
        return super().notify(obj, event)


def _migrate_legacy_settings() -> None:
    """Copy settings from the old "Cosmica" org/app into "Astraios" once.

    Best-effort and idempotent: only runs when the Astraios store is empty and a
    legacy Cosmica store exists, so it never clobbers newer values.
    """
    from PyQt6.QtCore import QSettings

    new = QSettings("Astraios", "Astraios")
    if new.allKeys():
        return  # already populated (migrated before, or fresh values written)
    old = QSettings("Cosmica", "Cosmica")
    keys = old.allKeys()
    if not keys:
        return  # nothing to migrate — genuinely new install
    for key in keys:
        new.setValue(key, old.value(key))
    new.sync()
    logging.getLogger(__name__).info(
        "Migrated %d setting(s) from the former 'Cosmica' configuration", len(keys)
    )


# Libraries that log at DEBUG on import and drown out our own messages.
# numcodecs alone emits one line per registered codec at every startup.
_NOISY_LIBRARIES = (
    "numcodecs",
    "matplotlib",
    "PIL",
    "asyncio",
    "urllib3",
    "fsspec",
    "dask",
    "h5py",
)


def _configure_logging(argv: list[str]) -> None:
    """Set up logging: our own messages at INFO, third-party noise at WARNING.

    Debug logging is opt-in via ``--debug`` or ``ASTRAIOS_DEBUG=1``. Running
    the root logger at DEBUG (as this used to) meant every dependency logged
    its internals on every launch, which buried real warnings and cost time
    formatting messages nobody reads.
    """
    debug = "--debug" in argv or os.environ.get("ASTRAIOS_DEBUG", "") not in ("", "0")

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Our own package follows the requested level; chatty dependencies never
    # drop below WARNING, even in debug runs, unless explicitly asked for.
    logging.getLogger("astraios").setLevel(logging.DEBUG if debug else logging.INFO)
    if not (debug and os.environ.get("ASTRAIOS_DEBUG_LIBS", "") not in ("", "0")):
        for name in _NOISY_LIBRARIES:
            logging.getLogger(name).setLevel(logging.WARNING)


def run_application(argv: list[str] | None = None) -> int:
    """Initialize and run the Astraios application."""
    if argv is None:
        argv = sys.argv

    _configure_logging(argv)

    app = _AstraiosApp(argv)
    app.setApplicationName(astraios.__app_name__)
    app.setApplicationVersion(astraios.__version__)
    app.setOrganizationName("Astraios")

    # One-time settings migration from the project's former name ("Cosmica").
    # Carries over the astrometry.net API key, equipment, and preferences so the
    # rename doesn't silently wipe a returning user's saved configuration.
    _migrate_legacy_settings()

    # Set application icon
    icon_path = Path(__file__).resolve().parent.parent / "resources" / "icons" / "astraios.svg"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    # Set default font — Space Grotesk first, then fallbacks
    for family in ("Space Grotesk", "Inter", "Segoe UI", "Roboto", "Ubuntu"):
        font = QFont(family, 13)
        if font.exactMatch():
            break
    app.setFont(font)

    # Apply dark theme
    app.setStyleSheet(DARK_THEME)

    # Show splash screen
    splash = QSplashScreen()
    splash.showMessage(
        f"  {astraios.__app_name__} v{astraios.__version__}\n\n  Loading...",
        alignment=0x0004 | 0x0080,  # AlignCenter
    )
    splash.setStyleSheet(
        "QSplashScreen { background-color: #0d1117; color: #e6edf3; "
        "border: 1px solid #30363d; border-radius: 8px; font-size: 16px; }"
    )
    splash.resize(380, 180)
    splash.show()
    app.processEvents()

    # Import here to avoid circular imports
    from astraios.ui.main_window import MainWindow

    window = MainWindow()
    window.show()
    splash.finish(window)

    log.info("Astraios %s started", astraios.__version__)
    return app.exec()
