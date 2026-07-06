"""Planet projection dialog.

Drives astraios.core.planet_projection (equirectangular lon/lat map, or a
re-oriented orthographic view, of a planetary/lunar/solar disc), ported
from Seti Astro Suite Pro (GPL-3.0, Franklin Marek). See
astraios/core/planet_projection.py for the ported geometry, the
central-lon/lat extension, and the GPU/CPU dispatch notes.
"""

from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from astraios.ui.widgets.ui_kit import help_dot, param_help

log = logging.getLogger(__name__)


class _Worker(QThread):
    progress = pyqtSignal(float, str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, image: np.ndarray, params):
        super().__init__()
        self._image = image
        self._params = params

    def run(self):
        try:
            from astraios.core.planet_projection import project_planet

            def _prog(frac: float, message: str):
                self.progress.emit(frac, message)

            result = project_planet(self._image, self._params, progress=_prog)
            self.done.emit(result)
        except Exception as exc:
            log.exception("Planet projection failed")
            self.failed.emit(str(exc))


class PlanetProjectionDialog(QDialog):
    """Reproject the current planetary/lunar/solar disc image."""

    result_ready = pyqtSignal(object)

    def __init__(self, image: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Planet Projection")
        self.setMinimumWidth(460)
        self._image = image
        self._worker: _Worker | None = None

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Reprojects a planetary/lunar/solar disc: either flattens it "
            "into an equirectangular longitude/latitude map, or re-orients "
            "the disc itself by yawing the view around the vertical axis."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        lay.addWidget(intro)

        self._type_combo = QComboBox()
        self._type_combo.addItems(
            ["Equirectangular (lon/lat map)", "Orthographic (re-oriented view)"]
        )
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        type_row = QHBoxLayout()
        type_row.addWidget(self._type_combo)
        type_row.addWidget(help_dot(param_help(
            "Projection type.",
            how="<b>Equirectangular</b> unwraps the visible hemisphere into "
                "a flat lon/lat texture, useful for mapping surface "
                "features. <b>Orthographic</b> keeps the disc's round "
                "appearance but yaws it as if seen from a different "
                "longitude.",
        )))
        type_row.addStretch()
        lay.addLayout(type_row)

        disc = QGroupBox("Disc Geometry")
        dform = QFormLayout(disc)
        self._auto_detect = QCheckBox("Auto-detect disc (center + radius)")
        self._auto_detect.setChecked(True)
        self._auto_detect.toggled.connect(self._on_auto_toggled)
        dform.addRow(self._auto_detect)
        self._cx = QDoubleSpinBox()
        self._cx.setRange(0.0, 100000.0)
        self._cy = QDoubleSpinBox()
        self._cy.setRange(0.0, 100000.0)
        self._r = QDoubleSpinBox()
        self._r.setRange(1.0, 100000.0)
        self._r.setValue(100.0)
        h, w = image.shape[-2:]
        self._cx.setValue(w / 2.0)
        self._cy.setValue(h / 2.0)
        self._r.setValue(0.45 * min(h, w))
        dform.addRow("Center X (px)", self._row(self._cx, param_help(
            "Disc center, X pixel coordinate. Ignored while auto-detect is on.",
        )))
        dform.addRow("Center Y (px)", self._row(self._cy, param_help(
            "Disc center, Y pixel coordinate. Ignored while auto-detect is on.",
        )))
        dform.addRow("Radius (px)", self._row(self._r, param_help(
            "Disc (limb) radius in pixels. Ignored while auto-detect is on.",
        )))
        lay.addWidget(disc)
        self._disc_group = disc
        self._on_auto_toggled(True)

        self._equirect_group = QGroupBox("Equirectangular")
        eform = QFormLayout(self._equirect_group)
        self._central_lon = QDoubleSpinBox()
        self._central_lon.setRange(-180.0, 180.0)
        self._central_lon.setSuffix(" deg")
        eform.addRow("Central longitude", self._row(self._central_lon, param_help(
            "Which body longitude appears at the map's horizontal center.",
            how="Rotates the sphere before sampling — the same longitude "
                "shift used by planetary de-rotation, applied once instead "
                "of per-frame.",
            default="0 = the longitude that was facing the observer.",
        )))
        self._central_lat = QDoubleSpinBox()
        self._central_lat.setRange(-90.0, 90.0)
        self._central_lat.setSuffix(" deg")
        eform.addRow("Central latitude", self._row(self._central_lat, param_help(
            "Sub-observer latitude (pole tilt) used when building the map.",
            default="0 = equator-on.",
        )))
        self._tex_h = QSpinBox()
        self._tex_h.setRange(16, 8192)
        self._tex_h.setValue(1024)
        eform.addRow("Map height (px)", self._row(self._tex_h, param_help(
            "Output map height in pixels (spans -90..+90 deg latitude).",
        )))
        self._tex_w = QSpinBox()
        self._tex_w.setRange(16, 16384)
        self._tex_w.setValue(2048)
        eform.addRow("Map width (px)", self._row(self._tex_w, param_help(
            "Output map width in pixels (spans the full 360 deg of longitude).",
        )))
        lay.addWidget(self._equirect_group)

        self._ortho_group = QGroupBox("Orthographic")
        oform = QFormLayout(self._ortho_group)
        self._theta = QDoubleSpinBox()
        self._theta.setRange(-89.0, 89.0)
        self._theta.setSuffix(" deg")
        oform.addRow("Yaw", self._row(self._theta, param_help(
            "Rotates the visible hemisphere about the vertical axis.",
            higher="Shows more of the limb on one side, as if the body had "
                   "rotated further before capture.",
            lower="Rotates the other way.",
            default="0 = unchanged view.",
        )))
        lay.addWidget(self._ortho_group)

        opts = QGroupBox("Resample")
        oform2 = QFormLayout(opts)
        self._interp = QComboBox()
        self._interp.addItems(["Nearest", "Linear", "Cubic"])
        self._interp.setCurrentText("Linear")
        oform2.addRow("Interpolation", self._row(self._interp, param_help(
            "Resample quality for the sphere warp.",
            higher="Cubic is sharpest but can ring slightly at the limb.",
            lower="Nearest is fastest and ring-free but blocky.",
        )))
        self._border = QDoubleSpinBox()
        self._border.setRange(0.0, 1.0)
        self._border.setSingleStep(0.05)
        oform2.addRow("Border value", self._row(self._border, param_help(
            "Fill value for pixels outside the visible/sampled region.",
        )))
        lay.addWidget(opts)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        lay.addWidget(self._progress)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        btns = QHBoxLayout()
        self._run_btn = QPushButton("Project")
        self._run_btn.clicked.connect(self._run)
        btns.addWidget(self._run_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btns.addWidget(close_btn)
        lay.addLayout(btns)

        self._on_type_changed(0)

    @staticmethod
    def _row(widget, tip: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(widget)
        row.addWidget(help_dot(tip))
        row.addStretch()
        return row

    def _on_auto_toggled(self, checked: bool):
        self._cx.setEnabled(not checked)
        self._cy.setEnabled(not checked)
        self._r.setEnabled(not checked)

    def _on_type_changed(self, index: int):
        is_equirect = index == 0
        self._equirect_group.setVisible(is_equirect)
        self._ortho_group.setVisible(not is_equirect)

    def _run(self):
        from astraios.core.planet_projection import PlanetProjectionParams

        interp_map = {"Nearest": "nearest", "Linear": "linear", "Cubic": "cubic"}
        auto = self._auto_detect.isChecked()
        params = PlanetProjectionParams(
            projection_type="equirectangular" if self._type_combo.currentIndex() == 0
            else "orthographic",
            cx=None if auto else float(self._cx.value()),
            cy=None if auto else float(self._cy.value()),
            r=None if auto else float(self._r.value()),
            central_lon_deg=float(self._central_lon.value()),
            central_lat_deg=float(self._central_lat.value()),
            theta_deg=float(self._theta.value()),
            tex_h=int(self._tex_h.value()),
            tex_w=int(self._tex_w.value()),
            interpolation=interp_map[self._interp.currentText()],
            border_value=float(self._border.value()),
        )

        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status.setText("Projecting...")
        self._worker = _Worker(self._image, params)
        self._worker.progress.connect(
            lambda f, m: (self._progress.setValue(int(f * 100)), self._status.setText(m))
        )
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_done(self, result):
        self._progress.setValue(100)
        if not isinstance(result, np.ndarray):
            self._status.setText("Projection produced no image.")
            self._run_btn.setEnabled(True)
            return
        self._status.setText("Done.")
        self.result_ready.emit(result)
        self.accept()

    def _on_fail(self, msg: str):
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._status.setText(f"Failed: {msg}")
