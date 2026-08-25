"""Session-wide test setup.

Chiefly: give the test run its own QSettings store instead of the developer's
real one.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _qsettings_sandbox(tmp_path_factory):
    """Redirect QSettings to a temp dir and mark the first-run welcome as seen.

    Two separate problems, one fix.

    The blocking one: ``MainWindow.__init__`` arms
    ``QTimer.singleShot(500, self._maybe_first_run_welcome)``, and on a machine
    that has never run Astraios that ends at ``QMessageBox.exec()`` -- a modal
    dialog, waiting for a click that a headless runner will never produce. The
    timer outlives the test that built the window, so it fires later, during
    whichever test pytest-qt happens to be processing events for. That is
    exactly how it presented: the suite reached 92%, the welcome fired during
    the setup of a test in tests/test_ui, and the job sat there until the
    timeout. It never reproduced on a developer machine, because a developer
    has launched the app once and their settings already say the welcome has
    been shown. Only a clean environment -- that is, only CI -- ever hit it.

    The quieter one: without this, the tests read and write the real user's
    Astraios settings. A test run could change how the application behaves for
    the person who ran it, and tests would pass or fail depending on settings
    left behind by an earlier run.

    Session-scoped and autouse so it is in place before any window is built.
    """
    try:
        from PyQt6.QtCore import QSettings
    except ImportError:  # pragma: no cover - PyQt6 is a hard dependency
        yield
        return

    settings_dir = tmp_path_factory.mktemp("qsettings")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(settings_dir)
    )

    settings = QSettings("Astraios", "Astraios")
    settings.setValue("ui/welcome_shown", True)
    settings.sync()

    yield
