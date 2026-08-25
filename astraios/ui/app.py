"""Application bootstrap — QApplication setup, theme, splash, and launch."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from PyQt6.QtCore import QEvent, QLocale, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialogButtonBox,
    QPushButton,
    QSplashScreen,
)

import astraios
from astraios.ui.theme import DARK_THEME, load_bundled_fonts

log = logging.getLogger(__name__)


class _AstraiosApp(QApplication):
    """QApplication with two app-wide widget rules.

    * Scroll-wheel events never reach a QComboBox, so scrolling a panel
      cannot silently change a setting the pointer happens to pass over.
    * A push button is a dialog's default button only when the dialog says
      so. Qt's own rule is that any focused ``autoDefault`` button becomes
      the default, and the theme draws the default button in the accent
      colour: clicking "Browse..." would turn it green and make Enter fire
      it, and a dialog whose primary action starts disabled would hand the
      accent to whichever button got focus first. Buttons inside a
      QDialogButtonBox keep Qt's behaviour, since the box manages its own
      default.
    """

    def notify(self, obj, event):
        etype = event.type()
        if etype == QEvent.Type.Wheel and isinstance(obj, QComboBox):
            return True
        if (
            etype == QEvent.Type.Polish
            and isinstance(obj, QPushButton)
            and obj.autoDefault()
            and not obj.isDefault()
            and not isinstance(obj.parent(), QDialogButtonBox)
        ):
            obj.setAutoDefault(False)
        return super().notify(obj, event)


class _UpdateCheckThread(QThread):
    """Checks GitHub Releases off the GUI thread — the network call must
    not block startup. Notify-only: the user still downloads and installs
    manually."""

    update_available = pyqtSignal(object)

    def run(self):
        try:
            from astraios.updater.auto_updater import AutoUpdater
            info = AutoUpdater().check_for_updates()
            if info.available:
                self.update_available.emit(info)
        except Exception:
            log.debug("Update check failed", exc_info=True)


def _notify_update(window, info) -> None:
    from PyQt6.QtWidgets import QMessageBox

    where = info.download_url or "the GitHub Releases page"
    QMessageBox.information(
        window,
        f"Update available: v{info.latest_version}",
        f"Astraios {info.latest_version} is available "
        f"(you are running {info.current_version}).\n\nDownload: {where}",
    )


def _schedule_update_check(window) -> None:
    def _start_check():
        checker = _UpdateCheckThread(window)
        checker.update_available.connect(lambda info: _notify_update(window, info))
        checker.finished.connect(checker.deleteLater)
        checker.start()

    QTimer.singleShot(5000, _start_check)


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

    # Preferences that must be known before any GPU work starts. The device
    # is chosen once, when the device manager is first used, so "Use GPU"
    # has to be turned into an environment variable before that happens.
    from PyQt6.QtCore import QSettings

    _settings = QSettings("Astraios", "Astraios")
    if str(_settings.value("processing/use_gpu", "true")).lower() in ("false", "0"):
        os.environ["ASTRAIOS_FORCE_CPU"] = "1"
    _check_updates = str(_settings.value("update/check_on_startup", "true")).lower() not in (
        "false", "0"
    )

    # Set application icon
    icon_path = Path(__file__).resolve().parent.parent / "resources" / "icons" / "astraios.svg"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    # Numbers everywhere in this UI are written with a decimal point: slider
    # readouts, FITS headers, pixel math, the histogram statistics. Without
    # this a Polish or German system rendered "3,0" in every spin box beside a
    # "3.0" slider label, and typed values with a point were rejected.
    QLocale.setDefault(QLocale.c())

    # The design's fonts ship with the app (resources/fonts, SIL OFL), so the
    # interface looks the same on every machine instead of falling through the
    # fallback list to whatever the OS has.
    loaded = load_bundled_fonts()
    if loaded:
        log.debug("Loaded bundled fonts: %s", ", ".join(loaded))
    font = QFont("Space Grotesk" if "Space Grotesk" in loaded else "Inter")
    font.setPixelSize(12)
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
    if _check_updates:
        _schedule_update_check(window)
    return app.exec()
