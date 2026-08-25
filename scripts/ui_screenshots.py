"""Render the Astraios UI offscreen for visual review.

Writes PNGs of the main window (idle, and with a synthetic star field
loaded, on every Tools tab) and of every dialog that can be constructed
without arguments, using the same fonts, locale and stylesheet the real
application applies. Runs headless, so it works on a CI runner or over SSH::

    QT_QPA_PLATFORM=offscreen poetry run python scripts/ui_screenshots.py out_dir

The point is to look at the result, not to assert on it: a stylesheet
change that reads fine in code can still clip a label, double an arrow or
lose a control, and only a picture shows that.
"""

from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PyQt6.QtCore import QLocale, QSettings  # noqa: E402
from PyQt6.QtGui import QFont  # noqa: E402
from PyQt6.QtWidgets import QTabWidget  # noqa: E402


def _synthetic_field(path: Path) -> Path:
    """A 1400x900 RGB star field with a ring nebula, written as FITS."""
    from astropy.io import fits

    rng = np.random.default_rng(3)
    h, w = 900, 1400
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.full((3, h, w), 0.02, np.float32)
    img += rng.normal(0, 0.004, (3, h, w)).astype(np.float32)
    r = np.hypot(yy - h * 0.47, xx - w * 0.48)
    ring = np.exp(-((r - 110) ** 2) / (2 * 28**2)) * 0.35 + np.exp(-(r**2) / (2 * 140**2)) * 0.12
    img[0] += ring * 0.45
    img[1] += ring * 0.75
    img[2] += ring * 1.0
    for _ in range(1800):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        amp = rng.pareto(2.2) * 0.02 + 0.01
        sig = rng.uniform(1.0, 2.2)
        x0, y0 = int(x), int(y)
        sl = (slice(max(0, y0 - 8), y0 + 9), slice(max(0, x0 - 8), x0 + 9))
        g = np.exp(-((yy[sl] - y) ** 2 + (xx[sl] - x) ** 2) / (2 * sig**2)) * amp
        tint = rng.choice([(1.0, 1.0, 1.0), (0.8, 0.9, 1.0), (1.0, 0.85, 0.7)])
        for c in range(3):
            img[c][sl] += g * tint[c]
    hdu = fits.PrimaryHDU(np.clip(img, 0, 1).astype(np.float32))
    hdu.header["OBJECT"] = "NGC 7662"
    hdu.writeto(path, overwrite=True)
    return path


def main(out_dir: str) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from astraios.ui import app as appmod

    app = appmod._AstraiosApp([])
    # Sandboxed settings so the welcome dialog and saved layouts stay out.
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(out / "qs"))
    s = QSettings("Astraios", "Astraios")
    s.setValue("ui/welcome_shown", True)
    s.sync()
    QLocale.setDefault(QLocale.c())
    appmod.load_bundled_fonts()
    font = QFont("Space Grotesk")
    font.setPixelSize(12)
    app.setFont(font)
    app.setStyleSheet(appmod.DARK_THEME)

    from astraios.ui.main_window import MainWindow

    win = MainWindow()
    win.resize(2000, 1125)
    win.show()
    for _ in range(5):
        app.processEvents()
    win.grab().save(str(out / "main_idle.png"))

    fits_path = _synthetic_field(out / "synthetic_field.fits")
    win._load_frame(str(fits_path))
    t0 = time.time()
    while win._current_image is None and time.time() - t0 < 30:
        app.processEvents()
        time.sleep(0.02)
    for _ in range(20):
        app.processEvents()

    tabs = max(win.findChildren(QTabWidget), key=lambda t: t.count())
    for i in range(tabs.count()):
        tabs.setCurrentIndex(i)
        app.processEvents()
        app.processEvents()
        name = tabs.tabText(i).split(" ", 1)[-1].replace(" ", "_").replace("/", "-")
        win.grab().save(str(out / f"main_tab{i}_{name}.png"))
    win.close()

    import astraios.ui.dialogs as dialogs_pkg

    img = np.clip(np.random.default_rng(0).random((3, 64, 64)) * 0.4 + 0.05, 0, 1)
    img = img.astype(np.float32)
    candidates = [
        {}, {"parent": None}, {"image_data": img}, {"current_image": img},
        {"image": img}, {"base_image": img}, {"frame_paths": []}, {"paths": []},
    ]
    n = 0
    for mod in sorted(pkgutil.iter_modules(dialogs_pkg.__path__), key=lambda m: m.name):
        m = importlib.import_module(f"astraios.ui.dialogs.{mod.name}")
        for name, obj in inspect.getmembers(m, inspect.isclass):
            if not name.endswith("Dialog") or obj.__module__ != m.__name__:
                continue
            dlg = None
            for kw in candidates:
                try:
                    dlg = obj(**kw)
                    break
                except TypeError:
                    continue
                except Exception as exc:  # a construction bug, worth seeing
                    print(f"FAIL {name}: {type(exc).__name__}: {exc}")
                    break
            if dlg is None:
                continue
            dlg.show()
            app.processEvents()
            app.processEvents()
            dlg.grab().save(str(out / f"dlg_{name}.png"))
            dlg.close()
            dlg.deleteLater()
            app.processEvents()
            n += 1
    print(f"wrote {tabs.count() + 1} main-window views and {n} dialogs to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "ui_screenshots"))
