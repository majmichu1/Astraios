"""Nebula Flythrough dialog — render a cinematic zoom video from the image.

The rendering core is ported from Seti Astro Suite Pro (GPL-3.0, Franklin
Marek); this dialog drives astraios.core.flythrough.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from astraios.ui.widgets.ui_kit import help_dot, param_help

log = logging.getLogger(__name__)


class _RenderWorker(QThread):
    progress = pyqtSignal(float, str)
    finished_ok = pyqtSignal(object)  # Path
    failed = pyqtSignal(str)

    def __init__(self, data, output_path, params, stars_layer, starless_layer):
        super().__init__()
        self._data = data
        self._output_path = output_path
        self._params = params
        self._stars = stars_layer
        self._starless = starless_layer

    def run(self):
        try:
            from astraios.core.flythrough import render_flythrough
            out = render_flythrough(
                self._data, self._output_path, self._params,
                stars_layer=self._stars, starless_layer=self._starless,
                progress=lambda f, m: self.progress.emit(f, m),
            )
            self.finished_ok.emit(out)
        except Exception as exc:
            log.exception("Flythrough render failed")
            self.failed.emit(str(exc))


class FlythroughDialog(QDialog):
    """Render a fly-into-the-nebula MP4 from the current image."""

    def __init__(self, image_data: np.ndarray, parent=None,
                 stars_layer: np.ndarray | None = None):
        super().__init__(parent)
        self.setWindowTitle("Nebula Flythrough")
        self.setMinimumWidth(460)
        self._data = image_data
        self._stars_layer = stars_layer
        self._worker: _RenderWorker | None = None

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Renders a cinematic zoom video that flies into the image. "
            "If a star layer was extracted (StarNet), it is composited "
            "with parallax for a depth effect."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        lay.addWidget(intro)

        vid = QGroupBox("Video")
        form = QFormLayout(vid)
        self._duration = QDoubleSpinBox()
        self._duration.setRange(1.0, 120.0)
        self._duration.setValue(8.0)
        self._duration.setSuffix(" s")
        form.addRow("Duration", self._row(self._duration, param_help(
            "Length of the rendered video.",
            higher="A longer, slower flight through the same zoom range.",
            lower="A shorter, snappier flight.",
        )))
        self._fps = QSpinBox()
        self._fps.setRange(10, 60)
        self._fps.setValue(30)
        form.addRow("Frame rate", self._row(self._fps, param_help(
            "Frames rendered per second of video.",
            higher="Smoother motion, but roughly proportionally longer to "
                   "render.",
            lower="Choppier motion, faster to render.",
            default="30 is smooth; 60 doubles render time.",
        )))
        self._res_combo = QComboBox()
        self._res_combo.addItems(["1920x1080", "1280x720", "3840x2160", "1080x1920 (vertical)"])
        form.addRow("Resolution", self._row(self._res_combo,
                                            "Output video size. Vertical is for "
                                            "phone reels and shorts."))
        lay.addWidget(vid)

        cam = QGroupBox("Camera")
        cform = QFormLayout(cam)
        self._zoom_start = QDoubleSpinBox()
        self._zoom_start.setRange(0.5, 20.0)
        self._zoom_start.setSingleStep(0.1)
        self._zoom_start.setValue(1.0)
        cform.addRow("Zoom start", self._row(self._zoom_start, param_help(
            "Magnification at the start of the flight.",
            higher="Starts already zoomed in closer to the target.",
            lower="Starts further out; 1.0 shows the whole image.",
        )))
        self._zoom_end = QDoubleSpinBox()
        self._zoom_end.setRange(0.5, 20.0)
        self._zoom_end.setSingleStep(0.1)
        self._zoom_end.setValue(3.0)
        cform.addRow("Zoom end", self._row(self._zoom_end, param_help(
            "Magnification at the end of the flight.",
            higher="Ends zoomed in tighter on the target.",
            lower="Ends further back, closer to the start's framing.",
            default="3.0 ends three times closer than the start.",
        )))
        self._ease_combo = QComboBox()
        self._ease_combo.addItems(["Ease In-Out", "Linear", "Ease In", "Ease Out"])
        cform.addRow("Motion", self._row(self._ease_combo,
                                         "Ease In-Out accelerates gently and "
                                         "slows into the target, like a real "
                                         "camera move."))
        self._cx = QDoubleSpinBox()
        self._cx.setRange(0.0, 1.0)
        self._cx.setSingleStep(0.05)
        self._cx.setValue(0.5)
        self._cy = QDoubleSpinBox()
        self._cy.setRange(0.0, 1.0)
        self._cy.setSingleStep(0.05)
        self._cy.setValue(0.5)
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("x"))
        target_row.addWidget(self._cx)
        target_row.addWidget(QLabel("y"))
        target_row.addWidget(self._cy)
        target_row.addWidget(help_dot(
            "Where the camera flies to, as a fraction of the image "
            "(0.5, 0.5 is the center)."))
        cform.addRow("Target", target_row)
        lay.addWidget(cam)

        fx = QGroupBox("Effects")
        fform = QFormLayout(fx)
        self._parallax_check = QCheckBox("Star parallax (needs extracted stars)")
        self._parallax_check.setChecked(self._stars_layer is not None)
        self._parallax_check.setEnabled(self._stars_layer is not None)
        self._parallax_check.setToolTip(
            "<qt>Moves the stars slightly faster than the nebula so the "
            "flight has 3D depth. Extract stars first (Star Removal) to "
            "enable.</qt>")
        fform.addRow(self._parallax_check)
        self._zoom_blur_check = QCheckBox("Warp-speed zoom blur")
        self._zoom_blur_check.setToolTip(
            "<qt>Adds streaking toward the edges while the camera moves "
            "fast, like jumping to hyperspace.</qt>")
        fform.addRow(self._zoom_blur_check)
        lay.addWidget(fx)

        out_row = QHBoxLayout()
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("Choose output .mp4 file...")
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse)
        out_row.addWidget(self._out_edit, 1)
        out_row.addWidget(btn_browse)
        lay.addLayout(out_row)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        lay.addWidget(self._progress)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        btns = QHBoxLayout()
        self._render_btn = QPushButton("Render Video")
        self._render_btn.clicked.connect(self._render)
        btns.addWidget(self._render_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btns.addWidget(close_btn)
        lay.addLayout(btns)

    @staticmethod
    def _row(widget, tip: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(widget)
        row.addWidget(help_dot(tip))
        row.addStretch()
        return row

    def _browse(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Flythrough Video", "flythrough.mp4", "MP4 Video (*.mp4)"
        )
        if path:
            if not path.lower().endswith(".mp4"):
                path += ".mp4"
            self._out_edit.setText(path)

    def _render(self):
        out = self._out_edit.text().strip()
        if not out:
            self._status.setText("Choose an output file first.")
            return
        from astraios.core.flythrough import FlythroughParams, LayerTrajectoryParams

        res_text = self._res_combo.currentText().split(" ")[0]
        w, h = (int(v) for v in res_text.split("x"))
        traj = dict(
            zoom_start=float(self._zoom_start.value()),
            zoom_end=float(self._zoom_end.value()),
            cx_start=0.5, cy_start=0.5,
            cx_end=float(self._cx.value()), cy_end=float(self._cy.value()),
            ease=self._ease_combo.currentText(),
        )
        params = FlythroughParams(
            fps=int(self._fps.value()),
            duration=float(self._duration.value()),
            out_width=w, out_height=h,
        )
        params.starless = LayerTrajectoryParams(**traj)
        if self._zoom_blur_check.isChecked():
            params.starless.fx.zoom_blur = 0.5
        stars = None
        if self._parallax_check.isChecked() and self._stars_layer is not None:
            stars = self._stars_layer
            # Stars zoom slightly harder for parallax depth
            params.stars.zoom_start = traj["zoom_start"]
            params.stars.zoom_end = traj["zoom_end"] * 1.12
            params.stars.cx_end = traj["cx_end"]
            params.stars.cy_end = traj["cy_end"]
            params.stars.ease = traj["ease"]

        self._render_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status.setText("Rendering...")

        self._worker = _RenderWorker(
            self._data, Path(out), params, stars,
            self._data if stars is not None else None,
        )
        self._worker.progress.connect(
            lambda f, m: (self._progress.setValue(int(f * 100)),
                          self._status.setText(m)))
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_done(self, path):
        self._render_btn.setEnabled(True)
        self._progress.setValue(100)
        self._status.setText(f"Saved: {path}")

    def _on_fail(self, msg: str):
        self._render_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status.setText(f"Render failed: {msg}")
