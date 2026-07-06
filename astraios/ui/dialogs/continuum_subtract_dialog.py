"""Continuum Subtraction dialog.

Drives astraios.core.continuum_subtract (isolate narrowband emission by
subtracting a scaled continuum frame), ported from Seti Astro Suite Pro
(GPL-3.0, Franklin Marek).
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
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from astraios.ui.widgets.ui_kit import help_dot

log = logging.getLogger(__name__)


class _Worker(QThread):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, nb, cont, params):
        super().__init__()
        self._nb, self._cont, self._params = nb, cont, params

    def run(self):
        try:
            from astraios.core.continuum_subtract import subtract_continuum
            self.done.emit(subtract_continuum(self._nb, self._cont, self._params))
        except Exception as exc:
            log.exception("Continuum subtraction failed")
            self.failed.emit(str(exc))


class ContinuumSubtractDialog(QDialog):
    """Subtract a scaled continuum/broadband frame from a narrowband image."""

    result_ready = pyqtSignal(object)

    def __init__(self, narrowband: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Continuum Subtraction")
        self.setMinimumWidth(440)
        self._nb = narrowband
        self._cont: np.ndarray | None = None
        self._worker: _Worker | None = None

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Isolates narrowband emission (e.g. H-alpha) by subtracting a "
            "scaled broadband/continuum frame: stars cancel out while the "
            "nebula stays. The current image is the narrowband input."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        lay.addWidget(intro)

        cont_row = QHBoxLayout()
        self._cont_edit = QLineEdit()
        self._cont_edit.setPlaceholderText("Load the continuum / broadband frame...")
        self._cont_edit.setReadOnly(True)
        browse = QPushButton("Load...")
        browse.clicked.connect(self._browse)
        cont_row.addWidget(self._cont_edit, 1)
        cont_row.addWidget(browse)
        lay.addLayout(cont_row)

        form = QFormLayout()
        self._method = QComboBox()
        self._method.addItems(["Star-based (automatic)", "Manual scale"])
        form.addRow(*self._r("Scaling", self._method,
                    "Star-based estimates the continuum scale automatically "
                    "from stars so they subtract cleanly. Manual uses your "
                    "fixed factor below."))
        self._scale = QDoubleSpinBox()
        self._scale.setRange(0.0, 5.0)
        self._scale.setSingleStep(0.05)
        self._scale.setValue(0.8)
        form.addRow(*self._r("Scale factor", self._scale,
                    "How much continuum to subtract in Manual mode (and the "
                    "starting point for Star-based). Raise if stars leave "
                    "bright residuals, lower if the nebula goes too dark."))
        self._pedestal = QCheckBox("Match background level")
        self._pedestal.setChecked(True)
        self._pedestal.setToolTip(
            "<qt>Align the two frames' background brightness before "
            "subtracting so the result doesn't come out too dark or "
            "clipped.</qt>")
        form.addRow(self._pedestal)
        self._normalize = QCheckBox("Match overall gain")
        self._normalize.setChecked(True)
        self._normalize.setToolTip(
            "<qt>Scale the continuum to the narrowband's overall level "
            "(robust median/MAD match) before subtracting.</qt>")
        form.addRow(self._normalize)
        lay.addLayout(form)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        lay.addWidget(self._progress)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        btns = QHBoxLayout()
        self._apply_btn = QPushButton("Subtract Continuum")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._apply)
        btns.addWidget(self._apply_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btns.addWidget(close_btn)
        lay.addLayout(btns)

    @staticmethod
    def _r(label, widget, tip):
        row = QHBoxLayout()
        row.addWidget(widget)
        row.addWidget(help_dot(tip))
        row.addStretch()
        return label, row

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Continuum Frame", "",
            "Images (*.fit *.fits *.fts *.xisf *.tif *.tiff *.png)"
        )
        if not path:
            return
        from astraios.core.image_io import load_image
        try:
            img = load_image(path)
        except Exception as exc:
            self._status.setText(f"Could not load: {exc}")
            return
        cont = img.data
        if cont.shape[-2:] != self._nb.shape[-2:]:
            self._status.setText(
                f"Continuum size {cont.shape[-2:]} does not match the "
                f"narrowband {self._nb.shape[-2:]}."
            )
            return
        self._cont = cont
        from pathlib import Path
        self._cont_edit.setText(Path(path).name)
        self._apply_btn.setEnabled(True)
        self._status.setText("")

    def _apply(self):
        from astraios.core.continuum_subtract import ContinuumSubtractParams

        if self._cont is None:
            return
        params = ContinuumSubtractParams(
            scaling_method=("manual" if self._method.currentIndex() == 1
                            else "star_based"),
            scale_factor=float(self._scale.value()),
            background_pedestal=self._pedestal.isChecked(),
            normalize_gain=self._normalize.isChecked(),
        )
        self._apply_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)  # busy
        self._status.setText("Subtracting...")
        self._worker = _Worker(self._nb, self._cont, params)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_done(self, result):
        self._progress.setVisible(False)
        if not isinstance(result, np.ndarray):
            self._status.setText("No result produced.")
            self._apply_btn.setEnabled(True)
            return
        self.result_ready.emit(result)
        self.accept()

    def _on_fail(self, msg):
        self._progress.setVisible(False)
        self._apply_btn.setEnabled(True)
        self._status.setText(f"Failed: {msg}")
