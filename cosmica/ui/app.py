"""Application bootstrap — QApplication setup, theme, splash, and launch."""

from __future__ import annotations

import logging
import sys

from PyQt6.QtCore import QEvent
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication, QComboBox, QSplashScreen

import cosmica
from cosmica.ui.theme import DARK_THEME

log = logging.getLogger(__name__)


class _CosmicaApp(QApplication):
    """QApplication subclass that blocks scroll-wheel on all QComboBox instances."""

    def notify(self, obj, event):
        if event.type() == QEvent.Type.Wheel and isinstance(obj, QComboBox):
            return True
        return super().notify(obj, event)


def run_application(argv: list[str] | None = None) -> int:
    """Initialize and run the Cosmica application."""
    if argv is None:
        argv = sys.argv

    # Set up root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    app = _CosmicaApp(argv)
    app.setApplicationName(cosmica.__app_name__)
    app.setApplicationVersion(cosmica.__version__)
    app.setOrganizationName("Cosmica")

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
        f"  {cosmica.__app_name__} v{cosmica.__version__}\n\n  Loading...",
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
    from cosmica.ui.main_window import MainWindow

    window = MainWindow()
    window.show()
    splash.finish(window)

    log.info("Cosmica %s started", cosmica.__version__)
    return app.exec()
